# Cloud Cost Optimizer - Functionalities

This document summarizes the implemented functionalities of the project for review and submission.

## 1. Environment and Task System

- Simulated cloud cost optimization environment with three tasks:
  - `easy`
  - `medium`
  - `hard`
- Task configurations are stored in `tasks/`.
- Environment supports reset/step lifecycle via OpenEnv-compatible server/client flow.

## 2. Supported Agent Actions

The agent can issue exactly these actions:

- `terminate`
- `resize`
- `switch_pricing`
- `skip`

Each action is represented through `CloudCostAction` and validated before execution.

## 3. Observation-Driven Agent Loop

For each task:

1. Reset environment
2. Read current observation
3. Build prompt from observation
4. Get action from LLM
5. Normalize/validate action
6. Execute action with `step`
7. Repeat until `done=True` or max steps

## 4. LLM Integration (Validator-Compatible)

- Uses OpenAI-compatible client.
- Required runtime variables:
  - `API_BASE_URL`
  - `API_KEY`
- Model is configurable via:
  - `MODEL_NAME`
- Environment endpoint configurable via:
  - `ENV_URL`

## 5. Action Safety and Normalization

- Invalid or malformed LLM outputs are handled safely.
- Unsupported action types are replaced by safe normalized decisions.
- Resource IDs are validated against currently running resources.
- `skip` is normalized into canonical empty-target format.

## 6. Cost Optimization Heuristics (Fallback Logic)

Normalization fallback logic includes:

- Idle non-critical resource termination candidates
- Conservative downsize candidates for oversized resources
- Pricing switch for reserved-eligible resources
- Safe `skip` when no valid optimization remains

## 7. Structured Stdout Logs (Phase-2 Format)

`inference.py` prints parser-friendly structured blocks to stdout:

- `[START] task=...`
- `[STEP] task=... step=... action=... target=... reward=... done=...`
- `[END] task=... score=... steps=...`
- `[SUMMARY] easy=... medium=... hard=...`

All critical prints are flushed (`flush=True`) for reliable capture.

## 8. Scoring Logic

- Score is computed from:
  - `total_savings / optimal_savings`
- SLA violation penalty is applied.
- Final score is clamped strictly inside open interval `(0, 1)`.
- Clamp is rounding-safe for 4-decimal output to avoid printing `0.0000` or `1.0000`.

## 9. Deployment and Runtime

- FastAPI server implementation is available under `server/`.
- Dockerized runtime supported.
- Root-level Dockerfile and HF Space metadata are included.
- OpenEnv validation compatibility is maintained through `openenv.yaml`.

## 10. Repository Structure Highlights

- `inference.py` - agent execution script
- `server/app.py` - API server entrypoint
- `server/cloud_cost_environment.py` - core environment logic
- `models.py` - shared models
- `client.py` - environment client interface
- `tasks/` - task definitions
- `tests/` - test cases
- `README.md` - setup and usage guide

## 11. Submission-Relevant Compliance Points

- Uses injected proxy configuration (`API_BASE_URL`, `API_KEY`) for LLM calls.
- Emits required structured output blocks for validator parsing.
- Avoids out-of-range boundary scores (`0` or `1`) in task outputs.
- Provides deployable Hugging Face Space compatible repository layout.

