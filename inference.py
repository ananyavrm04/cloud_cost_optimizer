import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import requests

from client import CloudCostEnv
from models import CloudCostAction

load_dotenv()


def _parse_scalar(value: str):
    v = value.strip().strip("'\"")
    if not v:
        return ""
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _load_config_yaml(path: str = "config.yaml") -> dict:
    # Enhancement #55: optional simple config.yaml support.
    # Supported format: flat key: value pairs.
    p = Path(path)
    if not p.exists():
        return {}
    result: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        result[key.strip()] = _parse_scalar(value)
    return result


CONFIG = _load_config_yaml()


def _cfg(name: str, default=None):
    # env var override > config.yaml > default
    if name in os.environ:
        return os.environ[name]
    if name in CONFIG:
        return CONFIG[name]
    return default

# Validator-aligned env handling:
# - API_BASE_URL and API_KEY must come from injected env vars
# - MODEL_NAME keeps a local-friendly default
SUBMISSION_MODE = str(_cfg("SUBMISSION_MODE", "1")).strip().lower() in {"1", "true", "yes"}
if SUBMISSION_MODE and ("API_BASE_URL" not in os.environ or "API_KEY" not in os.environ):
    raise RuntimeError("Missing required env vars: API_BASE_URL and API_KEY")

API_BASE_URL = str(_cfg("API_BASE_URL", "https://router.huggingface.co/v1"))
MODEL_NAME = str(_cfg("MODEL_NAME", "openai/gpt-oss-20b:fastest"))
API_KEY = str(_cfg("API_KEY", os.environ.get("HF_TOKEN", "")))
ENV_URL = str(_cfg("ENV_URL", "http://localhost:7860"))
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = int(_cfg("MAX_STEPS", 100))
LLM_TIMEOUT_SECONDS = float(_cfg("LLM_TIMEOUT_SECONDS", 20))
LLM_MAX_RETRIES = int(_cfg("LLM_MAX_RETRIES", 2))
HISTORY_WINDOW = int(_cfg("HISTORY_WINDOW", 5))
LLM_FAILURE_THRESHOLD = int(_cfg("LLM_FAILURE_THRESHOLD", 3))
NEGATIVE_STREAK_THRESHOLD = int(_cfg("NEGATIVE_STREAK_THRESHOLD", 3))
ENV_HEALTH_RETRIES = int(_cfg("ENV_HEALTH_RETRIES", 5))
ENV_HEALTH_BACKOFF_SECONDS = float(_cfg("ENV_HEALTH_BACKOFF_SECONDS", 1.0))

llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a cloud cost optimization agent.
Reduce monthly cloud cost while preserving uptime (>= 99.9%).

You can choose one action:
- terminate
- resize
- switch_pricing
- skip

Rules:
- Never act on non-running resources.
- Prefer safe cost reductions first.
- Avoid touching resources that have downstream dependencies when safer options exist.

Examples:
1) If a non-critical compute node has cpu=2 and mem=5 with no dependents,
   return: {"action_type":"terminate","resource_id":"idle-web-1","new_size":"","new_pricing":""}
2) If a non-critical large node has cpu=22 and no dependents,
   return: {"action_type":"resize","resource_id":"oversized-compute-1","new_size":"medium","new_pricing":""}
3) If a resource is eligible_for_reserved and currently on_demand,
   return: {"action_type":"switch_pricing","resource_id":"reserved-eligible-1","new_size":"","new_pricing":"reserved"}

Reply with JSON only:
{
  "action_type": "terminate",
  "resource_id": "server-1",
  "new_size": "",
  "new_pricing": ""
}
"""

VALID_ACTIONS = {"terminate", "resize", "switch_pricing", "skip"}
SIZE_ORDER = ["small", "medium", "large", "xlarge"]
VALID_PRICING = {"on_demand", "reserved", "spot"}
LLM_METRICS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _empty_action() -> dict:
    return {"action_type": "skip", "resource_id": "", "new_size": "", "new_pricing": ""}


def _build_dependents_graph(running_resources: list[dict]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for r in running_resources:
        graph.setdefault(r["id"], [])
    for r in running_resources:
        for dep in r.get("dependencies", []):
            graph.setdefault(dep, [])
            graph[dep].append(r["id"])
    return graph


def _dependency_depths(running_resources: list[dict]) -> dict[str, int]:
    graph = _build_dependents_graph(running_resources)
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def dfs(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return 0
        visiting.add(node)
        depth = 0
        for child in graph.get(node, []):
            depth = max(depth, 1 + dfs(child))
        visiting.remove(node)
        memo[node] = depth
        return depth

    for rid in graph:
        dfs(rid)
    return memo


def _next_downsize(size: str) -> str:
    if size not in SIZE_ORDER:
        return "small"
    idx = SIZE_ORDER.index(size)
    if idx <= 0:
        return size
    return SIZE_ORDER[idx - 1]


def _is_non_downsize(current_size: str, requested_size: str) -> bool:
    # Enhancement #74: explicitly block equal-size and upsize proposals.
    if current_size not in SIZE_ORDER or requested_size not in SIZE_ORDER:
        return True
    return SIZE_ORDER.index(requested_size) >= SIZE_ORDER.index(current_size)


def _pricing_multiplier(pricing: str) -> float:
    if pricing == "spot":
        return 0.3
    if pricing == "reserved":
        return 0.6
    return 1.0


def _size_multiplier(size: str) -> float:
    mapping = {"small": 0.25, "medium": 0.5, "large": 1.0, "xlarge": 2.0}
    return mapping.get(size, 1.0)


def _resource_type_risk_weight(resource_type: str) -> float:
    # Type-aware safety profile (Enhancement #25).
    if resource_type == "database":
        return 2.0
    if resource_type == "storage":
        return 1.4
    return 1.0


def _action_risk_score(action_data: dict, resource: dict, dep_depth: int, uptime_margin: float) -> float:
    # Risk score surfaced for ranking and prompt context (Enhancement #26).
    risk = 0.0
    risk += _resource_type_risk_weight(resource.get("type", "compute"))
    if resource.get("is_critical", False):
        risk += 3.0
    risk += dep_depth * 0.8
    if action_data.get("action_type") == "terminate":
        risk += 1.5
    elif action_data.get("action_type") == "resize":
        risk += 0.8
    else:
        risk += 0.2
    if uptime_margin < 0.2:
        risk += 1.5
    return risk


def _expected_savings(action_data: dict, resource: dict) -> float:
    cost = float(resource.get("cost_per_month", 0.0))
    action_type = action_data.get("action_type", "")
    if action_type == "terminate":
        return cost
    if action_type == "resize":
        current_size = resource.get("size", "")
        new_size = action_data.get("new_size", "")
        size_drop = _size_multiplier(current_size) - _size_multiplier(new_size)
        return max(0.0, cost * size_drop / max(_size_multiplier(current_size), 1e-9))
    if action_type == "switch_pricing":
        current_pricing = resource.get("pricing", "on_demand")
        new_pricing = action_data.get("new_pricing", "")
        current_mult = _pricing_multiplier(current_pricing)
        new_mult = _pricing_multiplier(new_pricing)
        if new_mult >= current_mult:
            return 0.0
        return cost * (1.0 - (new_mult / current_mult))
    return 0.0


def _summarize_observation_diff(prev_obs, curr_obs) -> str:
    # Compact state-diff for prompt context (Enhancement #42).
    if prev_obs is None:
        return "initial step"
    prev_map = {r.get("id"): r for r in prev_obs.resources}
    curr_map = {r.get("id"): r for r in curr_obs.resources}
    terminated = 0
    resized = 0
    pricing_changed = 0
    for rid, prev_r in prev_map.items():
        curr_r = curr_map.get(rid)
        if not curr_r:
            continue
        if prev_r.get("status") == "running" and curr_r.get("status") == "terminated":
            terminated += 1
        if prev_r.get("size") != curr_r.get("size"):
            resized += 1
        if prev_r.get("pricing") != curr_r.get("pricing"):
            pricing_changed += 1
    cost_delta = curr_obs.current_monthly_cost - prev_obs.current_monthly_cost
    uptime_delta = curr_obs.current_uptime - prev_obs.current_uptime
    return (
        f"delta_cost={cost_delta:.2f}, delta_uptime={uptime_delta:.4f}, "
        f"terminated={terminated}, resized={resized}, repriced={pricing_changed}"
    )


def _action_key(action_data: dict) -> tuple:
    return (
        action_data.get("action_type", ""),
        action_data.get("resource_id", ""),
        action_data.get("new_size", ""),
        action_data.get("new_pricing", ""),
    )


def _compress_resources_for_prompt(running_resources: list[dict], dependency_depth: dict[str, int]) -> list[str]:
    groups: dict[tuple, list[dict]] = {}
    for r in running_resources:
        key = (
            r.get("type", ""),
            r.get("size", ""),
            r.get("pricing", ""),
            r.get("is_critical", False),
            r.get("eligible_for_reserved", False),
            dependency_depth.get(r["id"], 0),
        )
        groups.setdefault(key, []).append(r)

    lines: list[str] = []
    grouped_items = list(groups.items())
    grouped_items.sort(
        key=lambda kv: sum(float(x.get("cost_per_month", 0.0)) for x in kv[1]),
        reverse=True,
    )
    for key, members in grouped_items:
        total_cost = sum(float(m.get("cost_per_month", 0.0)) for m in members)
        avg_cpu = sum(float(m.get("cpu_usage_avg", 0.0)) for m in members) / max(1, len(members))
        avg_mem = sum(float(m.get("mem_usage_avg", 0.0)) for m in members) / max(1, len(members))
        ids = [m.get("id", "") for m in members]
        sample = ", ".join(ids[:5])
        if len(ids) > 5:
            sample += f" (+{len(ids) - 5} more)"
        lines.append(
            f"- cluster count={len(members)} type={key[0]} size={key[1]} pricing={key[2]} "
            f"critical={key[3]} eligible_reserved={key[4]} dep_depth={key[5]} "
            f"avg_cpu={avg_cpu:.1f} avg_mem={avg_mem:.1f} total_cost={total_cost:.2f} ids=[{sample}]"
        )
    return lines


def build_user_prompt(observation, step_history: list[str], steps_remaining: int, observation_diff: str) -> str:
    running_resources = [r for r in observation.resources if r.get("status") == "running"]
    # Enhancement #50: prioritize high-cost resources first.
    running_resources.sort(key=lambda r: float(r.get("cost_per_month", 0.0)), reverse=True)
    dependency_depth = _dependency_depths(running_resources)
    max_dep_depth = max(dependency_depth.values(), default=0)
    uptime_margin = observation.current_uptime - observation.sla_target
    savings_abs = max(0.0, observation.original_monthly_cost - observation.current_monthly_cost)
    savings_pct = (savings_abs / max(observation.original_monthly_cost, 1e-9)) * 100.0

    # Enhancement #46: compress prompt when resource count is large.
    if len(running_resources) > 40:
        resources = _compress_resources_for_prompt(running_resources, dependency_depth)
    else:
        resources = []
        for r in running_resources:
            resources.append(
                f"- id={r['id']} size={r['size']} cpu={r['cpu_usage_avg']:.1f} "
                f"mem={r['mem_usage_avg']:.1f} cost={r['cost_per_month']:.2f} "
                f"critical={r['is_critical']} deps={r['dependencies']} "
                f"dep_depth={dependency_depth.get(r['id'], 0)} "
                f"pricing={r['pricing']} eligible_reserved={r['eligible_for_reserved']}"
            )

    history_block = "\n".join(step_history) if step_history else "(none)"
    return (
        f"Current cost: {observation.current_monthly_cost:.2f}\n"
        f"Original cost: {observation.original_monthly_cost:.2f}\n"
        f"Savings so far: {savings_abs:.2f} ({savings_pct:.2f}%)\n"
        f"Uptime: {observation.current_uptime:.3f} target={observation.sla_target}\n"
        f"Uptime margin: {uptime_margin:.3f}\n"
        f"Max dependency depth: {max_dep_depth}\n"
        f"Steps remaining: {steps_remaining}\n"
        f"Feedback: {observation.step_feedback}\n"
        f"Observation diff: {observation_diff}\n"
        f"Recent history:\n{history_block}\n"
        f"Running resources:\n" + "\n".join(resources)
    )


def heuristic_action(observation, blocked_actions: set[tuple]) -> dict:
    running = [r for r in observation.resources if r.get("status") == "running"]
    if not running:
        return _empty_action()

    dependency_depth = _dependency_depths(running)

    uptime_margin = observation.current_uptime - observation.sla_target
    candidates: list[tuple[float, dict]] = []

    # Enhancement #61: adaptive thresholds based on live progress.
    original = max(float(observation.original_monthly_cost or 0.0), 1.0)
    current = float(observation.current_monthly_cost or 0.0)
    savings_ratio = max(0.0, min(1.0, (original - current) / original))
    terminate_cpu_threshold = 5.0
    resize_cpu_threshold = 35.0
    if savings_ratio < 0.20:
        terminate_cpu_threshold = 8.0
        resize_cpu_threshold = 45.0
    elif savings_ratio < 0.35:
        terminate_cpu_threshold = 6.5
        resize_cpu_threshold = 40.0

    # Candidate: terminate
    for r in running:
        candidate = ("terminate", r["id"], "", "")
        if (
            r.get("cpu_usage_avg", 100.0) < terminate_cpu_threshold
            and r.get("mem_usage_avg", 100.0) < 10.0
            and not r.get("is_critical", True)
            and dependency_depth.get(r["id"], 0) == 0
            and candidate not in blocked_actions
        ):
            action = {
                "action_type": "terminate",
                "resource_id": r["id"],
                "new_size": "",
                "new_pricing": "",
            }
            risk = _action_risk_score(action, r, dependency_depth.get(r["id"], 0), uptime_margin)
            gain = _expected_savings(action, r)
            score = gain / (1.0 + risk)
            candidates.append((score, action))

    # Candidate: resize
    for r in running:
        size = r.get("size", "")
        new_size = _next_downsize(size)
        candidate = ("resize", r["id"], new_size, "")
        if (
            size in ("medium", "large", "xlarge")
            and r.get("cpu_usage_avg", 100.0) < resize_cpu_threshold
            and not r.get("is_critical", False)
            and dependency_depth.get(r["id"], 0) == 0
            and new_size != size
            and candidate not in blocked_actions
        ):
            action = {
                "action_type": "resize",
                "resource_id": r["id"],
                "new_size": new_size,
                "new_pricing": "",
            }
            risk = _action_risk_score(action, r, dependency_depth.get(r["id"], 0), uptime_margin)
            gain = _expected_savings(action, r)
            score = gain / (1.0 + risk)
            candidates.append((score, action))

    # Candidate: pricing switch
    for r in running:
        candidate = ("switch_pricing", r["id"], "", "reserved")
        if (
            r.get("eligible_for_reserved", False)
            and r.get("pricing") != "reserved"
            and candidate not in blocked_actions
        ):
            action = {
                "action_type": "switch_pricing",
                "resource_id": r["id"],
                "new_size": "",
                "new_pricing": "reserved",
            }
            risk = _action_risk_score(action, r, dependency_depth.get(r["id"], 0), uptime_margin)
            gain = _expected_savings(action, r)
            score = gain / (1.0 + risk)
            candidates.append((score, action))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return _empty_action()


def normalize_action(action: dict, observation, blocked_actions: set[tuple]) -> dict:
    running_resources = [r for r in observation.resources if r.get("status") == "running"]
    resource_map = {r.get("id"): r for r in running_resources}
    dependency_depth = _dependency_depths(running_resources)

    if not isinstance(action, dict):
        return heuristic_action(observation, blocked_actions)

    action_type = action.get("action_type", "")
    if action_type not in VALID_ACTIONS:
        return heuristic_action(observation, blocked_actions)

    if action_type == "skip":
        return _empty_action()

    resource_id = action.get("resource_id", "")
    if resource_id not in resource_map:
        return heuristic_action(observation, blocked_actions)

    normalized = {
        "action_type": action_type,
        "resource_id": resource_id,
        "new_size": action.get("new_size", ""),
        "new_pricing": action.get("new_pricing", ""),
    }

    if _action_key(normalized) in blocked_actions:
        return heuristic_action(observation, blocked_actions)

    resource = resource_map[resource_id]

    # Dependency graph safety guard
    if action_type in {"terminate", "resize"} and dependency_depth.get(resource_id, 0) > 0:
        return heuristic_action(observation, blocked_actions)

    # Semantic validation
    if action_type == "resize":
        current_size = resource.get("size", "")
        requested_size = normalized.get("new_size", "")
        if _is_non_downsize(current_size, requested_size):
            return heuristic_action(observation, blocked_actions)
    elif action_type == "switch_pricing":
        requested_pricing = normalized.get("new_pricing", "")
        current_pricing = resource.get("pricing", "")
        if requested_pricing not in VALID_PRICING:
            return heuristic_action(observation, blocked_actions)
        if requested_pricing == current_pricing:
            return heuristic_action(observation, blocked_actions)
        if requested_pricing == "reserved" and not resource.get("eligible_for_reserved", False):
            return heuristic_action(observation, blocked_actions)

    return normalized


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    payload = match.group()
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}


def _llm_request(messages: list[dict]) -> tuple[dict, dict]:
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=200,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    usage = getattr(response, "usage", None)
    usage_dict = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    LLM_METRICS["calls"] += 1
    LLM_METRICS["prompt_tokens"] += usage_dict["prompt_tokens"]
    LLM_METRICS["completion_tokens"] += usage_dict["completion_tokens"]
    LLM_METRICS["total_tokens"] += usage_dict["total_tokens"]
    raw = (response.choices[0].message.content or "").strip()
    return _extract_json(raw), usage_dict


def call_llm(user_prompt: str) -> tuple[dict, bool]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max(1, LLM_MAX_RETRIES)):
        try:
            parsed, _ = _llm_request(messages)
            if parsed:
                return parsed, True

            # Deterministic parse recovery prompt
            repair_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY valid JSON with keys: action_type, resource_id, "
                        "new_size, new_pricing. No extra text."
                    ),
                }
            ]
            parsed, _ = _llm_request(repair_messages)
            if parsed:
                return parsed, True
        except Exception as exc:
            # Enhancement #73: bounded exponential backoff on rate limit.
            status_code = getattr(exc, "status_code", None)
            response = getattr(exc, "response", None)
            if status_code is None and response is not None:
                status_code = getattr(response, "status_code", None)
            if status_code == 429 and attempt < max(1, LLM_MAX_RETRIES) - 1:
                time.sleep(min(2.0, 0.5 * (2 ** attempt)))
            continue

    # Graceful degradation trigger
    return {}, False


def compute_score(state) -> float:
    # Keep scores strictly inside (0, 1) even after 4-decimal formatting.
    # If eps is too small (e.g. 1e-6), `:.4f` can still print 1.0000 or 0.0000.
    eps = 1e-3
    if not state.optimal_savings or state.optimal_savings <= 0:
        return eps
    score = state.total_savings / state.optimal_savings
    score = max(eps, min(1.0 - eps, score))
    if state.sla_violated:
        score = min(score, 0.3)
    return max(eps, min(1.0 - eps, score))


def wait_for_env_health() -> None:
    """Enhancement #99: retry environment health probe before running an episode."""
    health_url = f"{ENV_URL.rstrip('/')}/health"
    last_error: str = ""
    for attempt in range(1, max(1, ENV_HEALTH_RETRIES) + 1):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        if attempt < ENV_HEALTH_RETRIES:
            time.sleep(min(5.0, ENV_HEALTH_BACKOFF_SECONDS * (2 ** (attempt - 1))))
    raise RuntimeError(f"Environment health check failed at {health_url}: {last_error}")


def run_task(task_id: str) -> tuple[float, int]:
    print(f"[START] task={task_id}", flush=True)
    wait_for_env_health()
    with CloudCostEnv(base_url=ENV_URL).sync() as env:
        reset_result = env.reset(task_id=task_id)
        observation = (
            reset_result.observation
            if hasattr(reset_result, "observation")
            else reset_result
        )

        steps_executed = 0
        step_history: deque[str] = deque(maxlen=max(1, HISTORY_WINDOW))
        blocked_actions: set[tuple] = set()
        consecutive_negative = 0
        consecutive_llm_failures = 0
        llm_unavailable = False
        sla_damaged = False
        last_failed_action_key: tuple | None = None
        last_observation = None
        last_observation_diff = "initial step"

        for step_idx in range(1, MAX_STEPS + 1):
            steps_remaining = MAX_STEPS - step_idx + 1
            state_snapshot = env.state()
            projected_if_skip = compute_score(state_snapshot)
            # Early completion on full optimization (Enhancement #21).
            if (
                state_snapshot.optimal_savings
                and state_snapshot.optimal_savings > 0
                and state_snapshot.total_savings >= state_snapshot.optimal_savings
            ):
                break
            if state_snapshot.sla_violated:
                sla_damaged = True

            # Enhancement #63: once SLA is violated, switch to damage-control.
            if sla_damaged or observation.current_uptime < observation.sla_target:
                action_data = _empty_action()
            elif consecutive_negative >= NEGATIVE_STREAK_THRESHOLD:
                action_data = _empty_action()
            elif llm_unavailable:
                action_data = heuristic_action(observation, blocked_actions)
            else:
                raw_action, llm_ok = call_llm(
                    build_user_prompt(
                        observation=observation,
                        step_history=list(step_history)
                        + [
                            (
                                "progress total_savings="
                                f"{state_snapshot.total_savings:.2f} optimal_savings={state_snapshot.optimal_savings:.2f} "
                                f"savings_progress={projected_if_skip:.4f}"
                            )
                        ],
                        steps_remaining=steps_remaining,
                        observation_diff=last_observation_diff,
                    )
                )
                if llm_ok:
                    consecutive_llm_failures = 0
                    action_data = normalize_action(raw_action, observation, blocked_actions)
                else:
                    consecutive_llm_failures += 1
                    if consecutive_llm_failures >= LLM_FAILURE_THRESHOLD:
                        llm_unavailable = True
                    action_data = heuristic_action(observation, blocked_actions)

            action = CloudCostAction(
                action_type=action_data.get("action_type", "skip"),
                resource_id=action_data.get("resource_id", ""),
                new_size=action_data.get("new_size", ""),
                new_pricing=action_data.get("new_pricing", ""),
            )
            current_action_key = (action.action_type, action.resource_id, action.new_size, action.new_pricing)
            # Enhancement #45: block immediate repeat of the same failed action.
            if (
                last_failed_action_key is not None
                and current_action_key == last_failed_action_key
                and action.action_type != "skip"
            ):
                blocked_actions.add(current_action_key)
                fallback_data = heuristic_action(observation, blocked_actions)
                action = CloudCostAction(
                    action_type=fallback_data.get("action_type", "skip"),
                    resource_id=fallback_data.get("resource_id", ""),
                    new_size=fallback_data.get("new_size", ""),
                    new_pricing=fallback_data.get("new_pricing", ""),
                )
                current_action_key = (action.action_type, action.resource_id, action.new_size, action.new_pricing)
            # Enhancement #48: avoid clearly non-positive expected reward actions.
            if action.action_type != "skip":
                resource_map = {r.get("id"): r for r in observation.resources if r.get("status") == "running"}
                resource = resource_map.get(action.resource_id)
                if resource is not None:
                    predicted = _expected_savings(
                        {
                            "action_type": action.action_type,
                            "resource_id": action.resource_id,
                            "new_size": action.new_size,
                            "new_pricing": action.new_pricing,
                        },
                        resource,
                    )
                    if predicted <= 0:
                        action = CloudCostAction(**_empty_action())

            result = env.step(action)
            last_observation = observation
            observation = result.observation
            steps_executed = step_idx
            last_observation_diff = _summarize_observation_diff(last_observation, observation)

            blocked_actions.add((action.action_type, action.resource_id, action.new_size, action.new_pricing))
            step_history.append(
                f"step={step_idx} action={action.action_type} target={action.resource_id or '-'} "
                f"reward={result.reward:.4f} projected_if_skip={projected_if_skip:.4f} "
                f"feedback={observation.step_feedback}"
            )

            if result.reward < 0:
                consecutive_negative += 1
                last_failed_action_key = current_action_key
            else:
                consecutive_negative = 0
                last_failed_action_key = None

            print(
                f"[STEP] task={task_id} step={step_idx} action={action.action_type} "
                f"target={action.resource_id or '-'} reward={result.reward:.4f} done={result.done}",
                flush=True,
            )
            if result.done:
                break

        score = compute_score(env.state())
        print(f"[END] task={task_id} score={score:.4f} steps={steps_executed}", flush=True)
        return score, steps_executed


if __name__ == "__main__":
    mode = "llm-required"
    print(f"[INFO] mode={mode}", flush=True)
    scores: dict[str, float] = {}
    per_task_steps: dict[str, int] = {}
    for task_id in ["easy", "medium", "hard"]:
        score, steps = run_task(task_id)
        scores[task_id] = score
        per_task_steps[task_id] = steps
    print(
        f"[SUMMARY] easy={scores['easy']:.4f} medium={scores['medium']:.4f} hard={scores['hard']:.4f}",
        flush=True,
    )
    # Enhancement #94: standardized benchmark export.
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "submission_mode": SUBMISSION_MODE,
        "scores": scores,
        "llm_usage": dict(LLM_METRICS),
        "steps": per_task_steps,
    }
    (artifacts_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
