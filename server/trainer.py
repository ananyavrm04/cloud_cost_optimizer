"""
Training orchestrator. Runs the LLM agent against a fresh CloudCostEnvironment
instance over multiple episodes. Persists per-episode metrics to JSONL for the
dashboard.

Storage layout (in repo root, gitignored):
  training_runs/
    <run_id>.jsonl         (one EpisodeRecord per line)
    <run_id>.meta.json     (RunSummary, updated live during the run)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .agent import (
    DecideRequest,
    decide,
    record_mistake,
    clear_reflection,
)

log = logging.getLogger("cco.trainer")

RUNS_DIR = Path("training_runs")
RUNS_DIR.mkdir(exist_ok=True)

# Cap simultaneous background runs so the endpoint can't be used to spawn a flood.
MAX_CONCURRENT_RUNS = int(os.getenv("CCO_MAX_CONCURRENT_RUNS", "2"))

# In-memory cache so polling is fast and doesn't always hit disk
_active_runs: dict[str, "RunSummary"] = {}


# ── schemas ─────────────────────────────────────────────────────────────────────
class StartTrainingRequest(BaseModel):
    episodes: int = Field(default=30, ge=1, le=200)
    max_steps_per_episode: int = Field(default=8, ge=1, le=50)
    reflection_enabled: bool = False
    run_name: Optional[str] = None


class StartTrainingResponse(BaseModel):
    run_id: str
    started_at: float
    status: str = "running"


class EpisodeRecord(BaseModel):
    episode: int
    total_reward: float
    steps: int
    action_distribution: dict[str, int]
    avg_confidence: float
    fallback_count: int
    steps_data: list[dict[str, Any]]


class RunSummary(BaseModel):
    run_id: str
    started_at: float
    finished_at: Optional[float] = None
    episodes_completed: int = 0
    total_episodes: int
    reflection_enabled: bool = False
    run_name: Optional[str] = None
    status: str = "running"
    avg_reward: float = 0.0
    rewards: list[float] = Field(default_factory=list)
    avg_confidence_per_episode: list[float] = Field(default_factory=list)
    fallback_rate_per_episode: list[float] = Field(default_factory=list)


# ── persistence ─────────────────────────────────────────────────────────────────
def _meta_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.meta.json"


def _episodes_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.jsonl"


def _save_meta(summary: "RunSummary") -> None:
    _meta_path(summary.run_id).write_text(summary.model_dump_json(indent=2))


# ── env interaction (defensive — handles different env API shapes) ──────────────
def _create_env():
    """Create a fresh CloudCostEnvironment instance for training."""
    try:
        from .cloud_cost_environment import CloudCostEnvironment
    except ImportError:
        from server.cloud_cost_environment import CloudCostEnvironment
    return CloudCostEnvironment()


def _create_action(action_type: str):
    """Construct a CloudCostAction object. Try multiple import paths."""
    CloudCostAction = None
    for import_attempt in (
        "from ..models import CloudCostAction",
        "from models import CloudCostAction",
    ):
        try:
            exec(import_attempt, globals())
            CloudCostAction = globals()["CloudCostAction"]
            break
        except ImportError:
            continue
    if CloudCostAction is None:
        raise ImportError("Could not import CloudCostAction from models")
    return CloudCostAction(action_type=action_type)


def _obs_to_dict(obs: Any) -> dict:
    """Convert env observation to a JSON-serializable dict."""
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "__dict__"):
        return {k: v for k, v in vars(obs).items() if not k.startswith("_")}
    return {"raw": str(obs)}


def _parse_step_result(result: Any) -> tuple[Any, float, bool, bool]:
    """Parse env.step() return — handles Gymnasium (5-tuple, 4-tuple) and object forms."""
    if isinstance(result, tuple):
        if len(result) >= 5:
            return result[0], float(result[1] or 0), bool(result[2]), bool(result[3])
        if len(result) == 4:
            return result[0], float(result[1] or 0), bool(result[2]), False
        if len(result) >= 2:
            return result[0], float(result[1] or 0), False, False
        return result[0], 0.0, False, False
    # Object form (CloudCostObservation might carry reward/done attributes)
    reward = float(getattr(result, "reward", 0) or 0)
    done = bool(getattr(result, "done", False))
    truncated = bool(getattr(result, "truncated", False))
    return result, reward, done, truncated


# ── episode runner ──────────────────────────────────────────────────────────────
async def _run_episode(env, req: StartTrainingRequest, episode_idx: int) -> EpisodeRecord:
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    history: list[dict] = []
    steps_data: list[dict] = []
    total_reward = 0.0
    confidences: list[float] = []
    fallback_count = 0
    action_counter: Counter = Counter()
    bad_decisions: list[dict] = []

    for step in range(req.max_steps_per_episode):
        decide_req = DecideRequest(
            observation=_obs_to_dict(obs),
            step=step,
            history=history,
            enable_reflection=req.reflection_enabled,
        )
        try:
            decision = await decide(decide_req)
        except Exception as e:
            log.error("decide() failed (ep %d step %d): %s", episode_idx, step, e)
            break

        confidences.append(decision.confidence)
        if decision.fallback_triggered:
            fallback_count += 1
        action_counter[decision.action_type] += 1

        try:
            action = _create_action(decision.action_type)
            step_result = env.step(action)
            next_obs, reward, done, truncated = _parse_step_result(step_result)
        except Exception as e:
            log.error("env.step() failed (ep %d step %d): %s", episode_idx, step, e)
            break

        total_reward += reward
        steps_data.append({
            "step": step,
            "action": decision.action_type,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "model": decision.model_used,
            "fallback": decision.fallback_triggered,
            "reward": reward,
            "latency_ms": decision.latency_ms,
        })
        history.append({
            "step": step,
            "action_type": decision.action_type,
            "reasoning": decision.reasoning,
        })

        if reward < 0:
            bad_decisions.append({
                "observation": _obs_to_dict(obs),
                "action_type": decision.action_type,
                "reasoning": decision.reasoning,
                "reward": reward,
            })

        obs = next_obs
        if done or truncated:
            break

    if req.reflection_enabled:
        for bad in bad_decisions:
            record_mistake(bad["observation"], bad["action_type"], bad["reasoning"], bad["reward"])

    return EpisodeRecord(
        episode=episode_idx,
        total_reward=total_reward,
        steps=len(steps_data),
        action_distribution=dict(action_counter),
        avg_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        fallback_count=fallback_count,
        steps_data=steps_data,
    )


# ── public API ──────────────────────────────────────────────────────────────────
async def start_training(req: StartTrainingRequest) -> StartTrainingResponse:
    """Start a background training run. Returns immediately with run_id."""
    active = sum(1 for s in _active_runs.values() if s.status == "running")
    if active >= MAX_CONCURRENT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent training runs ({active}/{MAX_CONCURRENT_RUNS}). Wait for one to finish.",
        )
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    summary = RunSummary(
        run_id=run_id,
        started_at=time.time(),
        total_episodes=req.episodes,
        reflection_enabled=req.reflection_enabled,
        run_name=req.run_name,
    )
    _active_runs[run_id] = summary
    _save_meta(summary)
    _episodes_path(run_id).touch()

    if req.reflection_enabled:
        clear_reflection()

    asyncio.create_task(_background_train(req, summary))
    return StartTrainingResponse(run_id=run_id, started_at=summary.started_at, status="running")


async def _background_train(req: StartTrainingRequest, summary: RunSummary) -> None:
    env = _create_env()
    episodes_file = _episodes_path(summary.run_id)
    try:
        for ep in range(req.episodes):
            try:
                record = await _run_episode(env, req, ep)
            except Exception as e:
                log.error("episode %d crashed: %s", ep, e)
                continue
            with episodes_file.open("a") as f:
                f.write(record.model_dump_json() + "\n")
            summary.episodes_completed = ep + 1
            summary.rewards.append(record.total_reward)
            summary.avg_reward = sum(summary.rewards) / len(summary.rewards)
            summary.avg_confidence_per_episode.append(record.avg_confidence)
            summary.fallback_rate_per_episode.append(
                record.fallback_count / max(record.steps, 1)
            )
            _save_meta(summary)
        summary.status = "completed"
    except Exception as e:
        log.exception("training run %s failed: %s", summary.run_id, e)
        summary.status = "failed"
    finally:
        summary.finished_at = time.time()
        _save_meta(summary)


def get_run_summary(run_id: str) -> Optional[RunSummary]:
    if run_id in _active_runs:
        return _active_runs[run_id]
    p = _meta_path(run_id)
    if p.exists():
        return RunSummary.model_validate_json(p.read_text())
    return None


def list_runs() -> list[RunSummary]:
    out = []
    for f in sorted(RUNS_DIR.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(RunSummary.model_validate_json(f.read_text()))
        except Exception:
            continue
    return out


def get_run_episodes(run_id: str) -> list[dict]:
    p = _episodes_path(run_id)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
