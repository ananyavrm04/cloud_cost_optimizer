---
title: Cloud Cost Optimizer
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
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
- `small`: 0.25x base (e.g., $120/mo)
- `medium`: 0.5x base (e.g., $240/mo)
- `large`: 1.0x base (e.g., $480/mo)
- `xlarge`: 2.0x base (e.g., $960/mo)

### 3. `switch_pricing`
**Change pricing plan for a resource.**

- **Use when:** Resource is marked `eligible_for_reserved` and has stable, predictable usage
- **Available options:** `reserved` (60% of on-demand), `spot` (30% of on-demand)

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
- `spot`: 0.3x (70% discount)

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
   export HF_TOKEN="your-hf-token"
   export ENV_URL="http://localhost:7860"
   
   python inference.py
   ```

### Configuration

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint (required) |
| `MODEL_NAME` | Model identifier (required) |
| `HF_TOKEN` | Hugging Face API token (required) |
| `ENV_URL` | Server URL; defaults to `http://localhost:7860` (optional) |
| `USE_LLM` | `1` to enable LLM calls; default `0` (heuristic-only, no credit spend) |

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
