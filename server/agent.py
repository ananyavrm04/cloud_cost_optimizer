"""
LLM-powered autonomous agent for cloud cost optimization.

Tiered routing:
    Primary  : groq/llama-3.3-70b-versatile   (free, ~200ms, 30 RPM upstream)
    Fallback : gemini/gemini-2.0-flash         (free, ~500ms, 15 RPM upstream)
    Heuristic: round-robin (only if both LLMs fail entirely)

Reflection learning (optional, used by trainer):
    When enable_reflection=True, the agent's prompt includes recent mistakes
    (decisions that earned negative rewards) as anti-examples to learn from.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any

import litellm
from pydantic import BaseModel, Field

log = logging.getLogger("cco.agent")

PRIMARY_MODEL          = os.getenv("CCO_PRIMARY_MODEL",  "groq/llama-3.3-70b-versatile")
FALLBACK_MODEL         = os.getenv("CCO_FALLBACK_MODEL", "gemini/gemini-2.0-flash")
RATE_LIMIT_RPM         = int(os.getenv("CCO_RATE_LIMIT_RPM", "10"))
CONFIDENCE_THRESHOLD   = float(os.getenv("CCO_CONFIDENCE_FALLBACK", "0.7"))
LLM_TIMEOUT_SEC        = float(os.getenv("CCO_LLM_TIMEOUT", "15"))
REFLECTION_MAX_ITEMS   = int(os.getenv("CCO_REFLECTION_MAX_ITEMS", "10"))

VALID_ACTIONS = {"terminate", "resize", "switch_pricing", "skip"}


# ── schemas ─────────────────────────────────────────────────────────────────────
class DecideRequest(BaseModel):
    observation: dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    history: list[dict[str, Any]] = Field(default_factory=list)
    enable_reflection: bool = False


class DecideResponse(BaseModel):
    action_type: str
    reasoning: str
    confidence: float
    model_used: str
    fallback_triggered: bool = False
    latency_ms: int = 0


# ── reflection memory ───────────────────────────────────────────────────────────
# Stores recent bad decisions (negative-reward outcomes). The trainer feeds these
# in between episodes; the agent reads them when enable_reflection=True.
_reflection_memory: list[dict] = []


def record_mistake(observation: dict, action_type: str, reasoning: str, reward: float) -> None:
    """Record a penalized decision so future prompts can learn from it."""
    _reflection_memory.append({
        "observation_summary": json.dumps(observation, default=str)[:280],
        "action": action_type,
        "reasoning": reasoning[:160],
        "reward": float(reward),
    })
    while len(_reflection_memory) > REFLECTION_MAX_ITEMS:
        _reflection_memory.pop(0)


def clear_reflection() -> None:
    """Reset reflection memory (called at the start of each independent training run)."""
    _reflection_memory.clear()


def reflection_size() -> int:
    return len(_reflection_memory)


# ── per-provider rate limiters ──────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int, name: str = ""):
        self.rpm = max(1, rpm)
        self.window = 60.0
        self.name = name
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._calls and self._calls[0] < now - self.window:
                self._calls.popleft()
            if len(self._calls) >= self.rpm:
                wait = self.window - (now - self._calls[0]) + 0.05
                log.info("rate_limit %s: sleeping %.2fs", self.name, wait)
                await asyncio.sleep(wait)
                now = time.monotonic()
                while self._calls and self._calls[0] < now - self.window:
                    self._calls.popleft()
            self._calls.append(time.monotonic())


_primary_limiter  = RateLimiter(RATE_LIMIT_RPM, name="primary")
_fallback_limiter = RateLimiter(RATE_LIMIT_RPM, name="fallback")


# ── prompts ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an autonomous cloud cost optimization agent. Choose the next action for a \
cloud infrastructure environment with resources having cost, utilization, and \
SLA-criticality attributes.

Allowed actions (exactly one):
  - terminate      : safely remove a clearly idle resource (low risk, small savings)
  - resize         : downsize an oversized instance (medium risk, larger savings)
  - switch_pricing : convert on-demand to reserved (low risk, recurring savings)
  - skip           : preserve a critical resource (no change)

Priority order (NEVER violate):
  1. SLA safety        — never compromise uptime
  2. Dependency safety — never break a resource others depend on
  3. Cost reduction    — only after safety is guaranteed

Critical resources are typically named: web-server-*, db-*, auth-service, api-gateway, \
load-balancer. Be very cautious about terminating these even if utilization is low.

Confidence guidance:
  - 0.9+    : Clear-cut decision, no ambiguity, safety verified
  - 0.7-0.9 : Reasonable decision, minor uncertainty
  - <0.7    : Significant uncertainty (escalated to a stronger model)

Respond with ONLY a JSON object (no markdown, no preamble):
{
  "action_type": "terminate" | "resize" | "switch_pricing" | "skip",
  "reasoning": "<one concise sentence — why this action is safe AND beneficial>",
  "confidence": <float between 0.0 and 1.0>
}\
"""


def _build_user_prompt(observation: dict, step: int, history: list[dict], reflection: bool) -> str:
    parts: list[str] = []

    if reflection and _reflection_memory:
        recent = _reflection_memory[-3:]
        bad_lines = [
            f"  - You chose `{m['action']}` (reasoning: {m['reasoning']}) → reward {m['reward']:.2f} (PENALTY)"
            for m in recent
        ]
        parts.append(
            "PAST MISTAKES TO AVOID (these decisions were penalized — do NOT repeat similar patterns):\n"
            + "\n".join(bad_lines)
            + "\n"
        )

    obs_json = json.dumps(observation, indent=2, default=str, sort_keys=True)
    parts.append(f"Current infrastructure observation:\n```json\n{obs_json}\n```")
    parts.append(f"\nStep: {step}")

    if history:
        hist_lines = [
            f"  step {h.get('step', '?')}: {h.get('action_type', '?')} — {h.get('reasoning', '')[:80]}"
            for h in history[-5:]
        ]
        parts.append("\nRecent actions in this episode:\n" + "\n".join(hist_lines))

    parts.append("\nDecide the next action. Respond with the JSON object only.")
    return "\n".join(parts)


# ── core LLM call ───────────────────────────────────────────────────────────────
async def _call_model(
    model: str,
    limiter: RateLimiter,
    observation: dict,
    step: int,
    history: list[dict],
    reflection: bool,
) -> dict:
    await limiter.acquire()
    resp = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(observation, step, history, reflection)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=400,
        timeout=LLM_TIMEOUT_SEC,
    )
    raw = resp.choices[0].message.content or "{}"
    return json.loads(raw)


def _validate(raw: Any) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("LLM response is not a JSON object")
    action = str(raw.get("action_type", "")).strip().lower().replace("-", "_")
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action_type: {action!r}")
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(raw.get("reasoning", "No reasoning provided")).strip()[:280]
    return {"action_type": action, "reasoning": reasoning, "confidence": confidence}


def _heuristic(step: int, why: str) -> DecideResponse:
    actions = ["resize", "terminate", "switch_pricing", "skip"]
    return DecideResponse(
        action_type=actions[step % len(actions)],
        reasoning=f"LLM unavailable ({why}); applying round-robin heuristic.",
        confidence=0.3,
        model_used="heuristic",
        fallback_triggered=True,
    )


async def decide(req: DecideRequest) -> DecideResponse:
    t0 = time.monotonic()
    fallback_triggered = False
    model_used = PRIMARY_MODEL

    try:
        raw = await _call_model(
            PRIMARY_MODEL, _primary_limiter,
            req.observation, req.step, req.history, req.enable_reflection,
        )
        decision = _validate(raw)
    except Exception as e:
        log.warning("primary LLM failed: %s: %s", type(e).__name__, e)
        fallback_triggered = True
        model_used = FALLBACK_MODEL
        try:
            raw = await _call_model(
                FALLBACK_MODEL, _fallback_limiter,
                req.observation, req.step, req.history, req.enable_reflection,
            )
            decision = _validate(raw)
        except Exception as e2:
            log.error("fallback LLM also failed: %s: %s", type(e2).__name__, e2)
            heur = _heuristic(req.step, f"{type(e).__name__}/{type(e2).__name__}")
            heur.latency_ms = int((time.monotonic() - t0) * 1000)
            return heur

    if (not fallback_triggered) and decision["confidence"] < CONFIDENCE_THRESHOLD:
        log.info("primary confidence %.2f < %.2f, escalating", decision["confidence"], CONFIDENCE_THRESHOLD)
        try:
            raw = await _call_model(
                FALLBACK_MODEL, _fallback_limiter,
                req.observation, req.step, req.history, req.enable_reflection,
            )
            decision = _validate(raw)
            fallback_triggered = True
            model_used = FALLBACK_MODEL
        except Exception as e:
            log.warning("escalation failed: %s", e)

    return DecideResponse(
        action_type=decision["action_type"],
        reasoning=decision["reasoning"],
        confidence=decision["confidence"],
        model_used=model_used,
        fallback_triggered=fallback_triggered,
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
