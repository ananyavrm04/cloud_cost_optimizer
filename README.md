---
title: Cloud Cost Optimizer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
- openenv
---

# Cloud Cost Optimizer - OpenEnv Environment

## Overview

**Cloud Cost Optimizer** is an AI-powered, simulated cloud infrastructure environment where agents learn to reduce monthly cloud costs while maintaining Service Level Agreement (SLA) uptime guarantees.

### The Problem

Cloud infrastructure is often over-provisioned and inefficiently configured:
- Idle servers continue to drain the budget
- Instances are oversized for actual workload demands
- Pricing plans aren't optimized for usage patterns
- Critical dependencies make optimization risky

### The Solution

The agent's job is to:
1. **Identify waste** - Find idle servers, oversized instances, and suboptimal pricing
2. **Take actions** - Terminate, resize, or reprice resources
3. **Maintain uptime** - Never violate the 99.9% SLA target
4. **Maximize savings** - Achieve the highest cost reduction possible

---

## What We Actually Built

This project is not just a basic "call LLM and take action" loop. We built it as a reliable optimization system that can survive real hackathon validator conditions and still produce clean, interpretable output.

The agent reasons step-by-step over cloud resources, but it is constrained by safety and consistency checks so it does not blindly optimize cost at the expense of uptime. We added dependency-aware guardrails, action normalization, and risk-aware ranking so high-impact but risky moves are filtered or deprioritized.

We also focused heavily on reliability under real-world failure modes. If the LLM response is malformed, the parser recovers deterministically. If API calls fail or rate-limit, retries and graceful fallback logic prevent crashes. If the environment is unavailable, health probes with retry/backoff prevent invalid runs before they start.

The current pipeline also includes a sliding multi-turn LLM context window, prompt-response caching, reconnect-safe environment calls with action replay, task trace checkpointing, and optimal-path diff artifacts for post-run analysis.

A major submission requirement was structured stdout parsing. Our inference pipeline emits strict `[START]`, `[STEP]`, and `[END]` blocks with flush-safe logging, so validators can parse every episode deterministically. We also enforce score output strictly inside `(0,1)` to satisfy validator range constraints.

For engineering quality, we added task schema validation, runtime environment-state invariants, CI automation, and dockerized end-to-end testing with a mock OpenAI-compatible LLM server. This makes the repo reproducible for both reviewers and deployment.

---

## Action Space

The agent can take exactly one of these four actions per step:

### 1. `terminate`
**Shut down a resource entirely.**

- **Use when:** CPU usage is very low (< 5%), memory is minimal (< 10%), the resource is not critical, and no other resources depend on it
- **Effect:** Resource is removed; cost drops to $0/month for that resource

**Example:**
```json
{
  "action_type": "terminate",
  "resource_id": "idle-web-server-1",
  "new_size": "",
  "new_pricing": ""
}
```

### 2. `resize`
**Downsize an instance to a smaller size.**

- **Use when:** CPU usage is moderate (< 30%), memory is underutilized, resource can fit in a smaller size class
- **Allowed transitions:** `large` -> `medium`, `medium` -> `small`, `xlarge` -> `large` or `medium` or `small`
- **NOT allowed:** Upward resizing

**Example:**
```json
{
  "action_type": "resize",
  "resource_id": "oversized-compute-1",
  "new_size": "medium",
  "new_pricing": ""
}
```

**Size cost multipliers:**
- `small`: 1.0x base (e.g., $120/mo)
- `medium`: 2.0x base (e.g., $240/mo)
- `large`: 4.0x base (e.g., $480/mo)
- `xlarge`: 8.0x base (e.g., $960/mo)

### 3. `switch_pricing`
**Change pricing plan for a resource.**

- **Use when:** Resource is marked `eligible_for_reserved` and has stable, predictable usage
- **Available options:** `reserved` (60% of on-demand), `spot` (40% of on-demand)

**Example:**
```json
{
  "action_type": "switch_pricing",
  "resource_id": "stable-compute-1",
  "new_size": "",
  "new_pricing": "reserved"
}
```

**Pricing cost multipliers:**
- `on_demand`: 1.0x (baseline)
- `reserved`: 0.6x (40% discount)
- `spot`: 0.4x (60% discount)

### 4. `skip`
**Signal that optimization is complete.**

```json
{
  "action_type": "skip",
  "resource_id": "",
  "new_size": "",
  "new_pricing": ""
}
```

---

## Observation Space

After each `reset()` or `step()`, the agent receives a `CloudCostObservation`:

```python
{
  "resources": [...],
  "current_monthly_cost": 5680.0,
  "original_monthly_cost": 8500.0,
  "current_uptime": 99.99,
  "sla_target": 99.9,
  "budget": 0.0,
  "step_feedback": "Action X succeeded with savings of $Y"
}
```

---

## Reward Structure

### Episode Score

```
score = agent_savings / optimal_savings
```

- **Score capped at 0.3** if SLA was ever violated during the episode

---

## Task Descriptions

### Easy Task (`task_id="easy"`)
- 10 compute servers total
- 6 clearly idle (CPU < 5%)
- 4 active servers
- No dependencies, no critical resources
- **Optimal savings:** ~$2,880/month

### Medium Task (`task_id="medium"`)
- 30 resources: compute, storage, database
- 8 idle servers, 7 oversized, 2 reserved-eligible
- Some critical resources and dependencies
- **Optimal savings:** ~$9,648/month

### Hard Task (`task_id="hard"`)
- 60 resources
- 10 idle workers, 12 oversized, 5 reserved-eligible
- 14 critical resources in dependency chains
- **Optimal savings:** ~$13,920/month

---

## Setup & Installation

### Local Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the environment server:**
   ```bash
   python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
   ```

3. **Run the inference agent:**
   ```bash
   export API_BASE_URL="https://your-llm-api/v1"
   export MODEL_NAME="your-model-name"
   export API_KEY="your-proxy-api-key"
   export ENV_URL="http://localhost:7860"
   
   python inference.py
   ```

### Configuration

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint (required) |
| `API_KEY` | API key for the above endpoint (required) |
| `SUBMISSION_MODE` | `1/true` enforces strict validator-safe env requirements (default enabled) |
| `MODEL_NAME` | Model identifier (required) |
| `PROMPT_VERSION` | Prompt file version from `prompts/<version>.txt` (default `v1`) |
| `ENV_URL` | Server URL; defaults to `http://localhost:7860` (optional) |
| `ENV_HEALTH_RETRIES` | Health probe retries before each task (optional, default `5`) |
| `ENV_HEALTH_BACKOFF_SECONDS` | Exponential backoff base for health probe (optional, default `1.0`) |
| `FALLBACK_MODEL_NAME` | Optional backup model for automatic fallback if primary model fails/parses badly |
| `MODEL_FALLBACKS` | Optional comma-separated extra fallback models after `MODEL_NAME` and `FALLBACK_MODEL_NAME` |
| `NOOP_STREAK_THRESHOLD` | Auto-skip after repeated unchanged observations (optional, default `3`) |
| `USE_CLIENT_RECONNECT_WRAPPER` | Enable client-level reconnect/replay wrapper (optional, default enabled) |
| `INTERACTIVE_MODE` | Human-in-the-loop confirmations for each proposed action (optional, default off) |
| `ENSEMBLE_VOTES` | Number of LLM vote samples per step (optional, default `1`) |
| `ENSEMBLE_TEMPERATURE` | Sampling temperature when ensemble voting is enabled |
| `WARM_START_STEPS` | Bounded warm-start seeded actions at beginning of each task (optional, default `0`) |
| `EMIT_PROJECTED_LOGS` | Emit `[PROJECTED]` lines with projected score-if-skip each step (optional, default off) |
| `PROMPT_COST_PER_1K_TOKENS` | Prompt token price for cost accounting (optional, default `0`) |
| `COMPLETION_COST_PER_1K_TOKENS` | Completion token price for cost accounting (optional, default `0`) |

Optional file-based config is also supported via `config.yaml` (see `config.yaml.example`).
Environment variables always take precedence over config file values.

---

## Baseline Scores

| Task | Baseline score (heuristic agent) |
|------|----------------------------------|
| easy | 0.92 |
| medium | 0.68 |
| hard | 0.44 |

---

## Running Tests

```bash
python -m pytest tests/test_env.py -v
```

### Local Pre-Submit Validator

```bash
python scripts/pre_submit_check.py
```

Checks:
- required env vars
- required structured stdout blocks
- score range `(0,1)` for all tasks

### One-Command Full Pre-Submit (Windows PowerShell)

```powershell
./scripts/full_presubmit.ps1
```

This runs:
- `python inference.py`
- `python scripts/pre_submit_check.py`

### Prompt Version Comparison

```bash
python scripts/compare_prompts.py
```

Optional versions:

```bash
python scripts/compare_prompts.py v1 v2
```

### Interactive Demo Mode

```bash
python inference.py --interactive
```

### Dockerized End-to-End Check

```powershell
./scripts/run_e2e_docker.ps1
```

This starts:
- environment server container
- mock OpenAI-compatible LLM container
- inference runner container

---

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
- `ruff` lint (non-blocking)
- `mypy` type-check (non-blocking)
- `pytest tests/test_env.py` (blocking)

---

## Artifacts

After `python inference.py`, benchmark metrics are exported to:

- `artifacts/benchmark_report.json`

Includes:
- task scores
- token usage totals
- model/base URL metadata
- prompt version, per-model token usage, and estimated LLM cost (USD)
- per-task action success stats and transfer-learning hints
- per-step traces in `artifacts/traces/`
- estimated-optimal diff reports in `artifacts/optimal_diff/`

### Structured Stdout Tags

- `[START]`: task start marker for parser
- `[STEP]`: step-by-step action, reward, done signal
- `[END]`: per-task score and steps
- `[SUMMARY]`: final task score summary
- `[TOKENS]`: optional per-step token usage (`EMIT_TOKEN_LOGS=1`)
- `[PROJECTED]`: optional projected score-if-skip (`EMIT_PROJECTED_LOGS=1`)
- `[STATS]`: per-task action success statistics
- `[COVERAGE]`: explored action-space ratio per task

---

## File Structure

```
cloud-cost-optimizer/
├── openenv.yaml                       # OpenEnv specification
├── pyproject.toml                     # Package metadata
├── requirements.txt                   # Python dependencies
├── inference.py                       # Agent script
├── README.md                          # This file
├── __init__.py                        # Package exports
├── models.py                          # Pydantic data models
├── client.py                          # EnvClient for server connection
│
├── server/
│   ├── __init__.py
│   ├── cloud_cost_environment.py      # Core environment logic
│   ├── app.py                         # FastAPI application
│   └── Dockerfile                     # Docker build configuration
│
├── tasks/
│   ├── easy.json                      # Easy task (10 servers)
│   ├── medium.json                    # Medium task (30 servers)
│   └── hard.json                      # Hard task (60 servers)
│
└── tests/
    └── test_env.py                    # Unit tests
```

---

## License

This project is provided as-is for the OpenEnv hackathon.
