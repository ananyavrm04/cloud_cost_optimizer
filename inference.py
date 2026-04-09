import json
import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from client import CloudCostEnv
from models import CloudCostAction

load_dotenv()

# Validator-aligned env handling:
# - defaults only for API_BASE_URL and MODEL_NAME
# - no default for HF_TOKEN
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b:fastest")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
USE_LLM = os.environ.get("USE_LLM", "0").strip().lower() in {"1", "true", "yes"}
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS = 100

if USE_LLM and not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required when USE_LLM=1.")

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if USE_LLM else None

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


def build_user_prompt(observation) -> str:
    resources = []
    for r in observation.resources:
        if r.get("status") != "running":
            continue
        resources.append(
            f"- id={r['id']} size={r['size']} cpu={r['cpu_usage_avg']:.1f} "
            f"mem={r['mem_usage_avg']:.1f} cost={r['cost_per_month']:.2f} "
            f"critical={r['is_critical']} deps={r['dependencies']} "
            f"pricing={r['pricing']} eligible_reserved={r['eligible_for_reserved']}"
        )
    return (
        f"Current cost: {observation.current_monthly_cost:.2f}\n"
        f"Original cost: {observation.original_monthly_cost:.2f}\n"
        f"Uptime: {observation.current_uptime:.3f} target={observation.sla_target}\n"
        f"Feedback: {observation.step_feedback}\n"
        f"Running resources:\n" + "\n".join(resources)
    )


def _next_downsize(size: str) -> str:
    if size not in SIZE_ORDER:
        return "small"
    idx = SIZE_ORDER.index(size)
    if idx <= 0:
        return size
    return SIZE_ORDER[idx - 1]


def heuristic_action(observation) -> dict:
    running = [r for r in observation.resources if r.get("status") == "running"]
    if not running:
        return {"action_type": "skip", "resource_id": "", "new_size": "", "new_pricing": ""}

    dependents = set()
    for r in running:
        for dep in r.get("dependencies", []):
            dependents.add(dep)

    # 1) Safe terminate: idle + non-critical + no dependents
    for r in running:
        if (
            r.get("cpu_usage_avg", 100.0) < 5.0
            and r.get("mem_usage_avg", 100.0) < 10.0
            and not r.get("is_critical", True)
            and r.get("id") not in dependents
        ):
            return {
                "action_type": "terminate",
                "resource_id": r["id"],
                "new_size": "",
                "new_pricing": "",
            }

    # 2) Resize down conservative
    for r in running:
        size = r.get("size", "")
        if (
            size in ("medium", "large", "xlarge")
            and r.get("cpu_usage_avg", 100.0) < 35.0
            and not r.get("is_critical", False)
        ):
            new_size = _next_downsize(size)
            if new_size != size:
                return {
                    "action_type": "resize",
                    "resource_id": r["id"],
                    "new_size": new_size,
                    "new_pricing": "",
                }

    # 3) Pricing switch
    for r in running:
        if r.get("eligible_for_reserved", False) and r.get("pricing") != "reserved":
            return {
                "action_type": "switch_pricing",
                "resource_id": r["id"],
                "new_size": "",
                "new_pricing": "reserved",
            }

    return {"action_type": "skip", "resource_id": "", "new_size": "", "new_pricing": ""}


def normalize_action(action: dict, observation) -> dict:
    if not isinstance(action, dict):
        return heuristic_action(observation)
    action_type = action.get("action_type", "")
    if action_type not in VALID_ACTIONS:
        return heuristic_action(observation)
    if action_type == "skip":
        return {"action_type": "skip", "resource_id": "", "new_size": "", "new_pricing": ""}
    resource_id = action.get("resource_id", "")
    running_ids = {r.get("id") for r in observation.resources if r.get("status") == "running"}
    if resource_id not in running_ids:
        return heuristic_action(observation)
    return {
        "action_type": action_type,
        "resource_id": resource_id,
        "new_size": action.get("new_size", ""),
        "new_pricing": action.get("new_pricing", ""),
    }


def call_llm(user_prompt: str) -> dict:
    if llm is None:
        return {}
    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    raw = (response.choices[0].message.content or "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def compute_score(state) -> float:
    if not state.optimal_savings or state.optimal_savings <= 0:
        return 0.0
    score = state.total_savings / state.optimal_savings
    score = max(0.0, min(1.0, score))
    if state.sla_violated:
        score = min(score, 0.3)
    return score


def run_task(task_id: str) -> float:
    print(f"[START] task={task_id}", flush=True)
    with CloudCostEnv(base_url=ENV_URL).sync() as env:
        reset_result = env.reset(task_id=task_id)
        observation = (
            reset_result.observation
            if hasattr(reset_result, "observation")
            else reset_result
        )
        steps_executed = 0
        for step_idx in range(1, MAX_STEPS + 1):
            raw_action = call_llm(build_user_prompt(observation))
            action_data = normalize_action(raw_action, observation)
            action = CloudCostAction(
                action_type=action_data.get("action_type", "skip"),
                resource_id=action_data.get("resource_id", ""),
                new_size=action_data.get("new_size", ""),
                new_pricing=action_data.get("new_pricing", ""),
            )
            result = env.step(action)
            observation = result.observation
            steps_executed = step_idx
            print(
                f"[STEP] task={task_id} step={step_idx} action={action.action_type} "
                f"target={action.resource_id or '-'} reward={result.reward:.4f} done={result.done}",
                flush=True,
            )
            if result.done:
                break
        score = compute_score(env.state())
        print(f"[END] task={task_id} score={score:.4f} steps={steps_executed}", flush=True)
        return score


if __name__ == "__main__":
    mode = "LLM+heuristic" if USE_LLM else "heuristic-only"
    print(f"[INFO] mode={mode}", flush=True)
    scores: dict[str, float] = {}
    for task_id in ["easy", "medium", "hard"]:
        score = run_task(task_id)
        scores[task_id] = score
    print(
        f"[SUMMARY] easy={scores['easy']:.4f} medium={scores['medium']:.4f} hard={scores['hard']:.4f}",
        flush=True,
    )
