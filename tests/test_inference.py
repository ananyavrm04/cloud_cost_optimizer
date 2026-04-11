"""
Unit tests for inference agent helpers.
Run:
    python -m pytest tests/test_inference.py -q
"""

import os
from types import SimpleNamespace

import pytest

# Ensure inference module can import without mandatory submission env vars.
os.environ.setdefault("SUBMISSION_MODE", "0")
os.environ.setdefault("API_BASE_URL", "http://localhost:9999/v1")
os.environ.setdefault("API_KEY", "test-key")

import inference  # noqa: E402


def _obs(resources):
    return SimpleNamespace(
        resources=resources,
        current_uptime=100.0,
        sla_target=99.9,
        original_monthly_cost=1000.0,
        current_monthly_cost=900.0,
    )


def test_compute_score_strict_bounds():
    state_hi = SimpleNamespace(total_savings=1e9, optimal_savings=1.0, sla_violated=False)
    state_lo = SimpleNamespace(total_savings=0.0, optimal_savings=100.0, sla_violated=False)
    hi = inference.compute_score(state_hi)
    lo = inference.compute_score(state_lo)
    assert 0.0 < hi < 1.0
    assert 0.0 < lo < 1.0
    assert hi <= 0.999
    assert lo >= 0.001


def test_normalize_action_blocks_resize_upsize():
    observation = _obs(
        [
            {
                "id": "r1",
                "status": "running",
                "size": "small",
                "cpu_usage_avg": 10.0,
                "mem_usage_avg": 10.0,
                "is_critical": False,
                "dependencies": [],
                "pricing": "on_demand",
                "eligible_for_reserved": False,
                "type": "compute",
                "cost_per_month": 100.0,
            }
        ]
    )
    action = {"action_type": "resize", "resource_id": "r1", "new_size": "xlarge", "new_pricing": ""}
    normalized = inference.normalize_action(action, observation, blocked_actions=set())
    assert normalized["action_type"] != "resize"


def test_normalize_action_blocks_dependency_resize():
    observation = _obs(
        [
            {
                "id": "db",
                "status": "running",
                "size": "xlarge",
                "cpu_usage_avg": 10.0,
                "mem_usage_avg": 10.0,
                "is_critical": False,
                "dependencies": [],
                "pricing": "on_demand",
                "eligible_for_reserved": False,
                "type": "database",
                "cost_per_month": 500.0,
            },
            {
                "id": "api",
                "status": "running",
                "size": "large",
                "cpu_usage_avg": 20.0,
                "mem_usage_avg": 20.0,
                "is_critical": True,
                "dependencies": ["db"],
                "pricing": "on_demand",
                "eligible_for_reserved": False,
                "type": "compute",
                "cost_per_month": 300.0,
            },
        ]
    )
    action = {"action_type": "resize", "resource_id": "db", "new_size": "large", "new_pricing": ""}
    normalized = inference.normalize_action(action, observation, blocked_actions=set())
    assert normalized["resource_id"] != "db" or normalized["action_type"] != "resize"


def test_enforce_prompt_budget(monkeypatch):
    monkeypatch.setattr(inference, "PROMPT_MAX_CHARS", 50)
    raw = "x" * 200
    capped = inference._enforce_prompt_budget(raw)
    assert len(capped) <= 50
    assert "[TRUNCATED_FOR_BUDGET]" in capped


def test_dynamic_warm_start_uses_observation_ids():
    resources = [
        {
            "id": "idle-1",
            "status": "running",
            "size": "large",
            "cpu_usage_avg": 1.0,
            "mem_usage_avg": 2.0,
            "is_critical": False,
            "dependencies": [],
            "pricing": "on_demand",
            "eligible_for_reserved": False,
            "type": "compute",
            "cost_per_month": 300.0,
        },
        {
            "id": "busy-1",
            "status": "running",
            "size": "large",
            "cpu_usage_avg": 80.0,
            "mem_usage_avg": 70.0,
            "is_critical": False,
            "dependencies": [],
            "pricing": "on_demand",
            "eligible_for_reserved": False,
            "type": "compute",
            "cost_per_month": 300.0,
        },
    ]
    obs = _obs(resources)
    old_steps = inference.WARM_START_STEPS
    try:
        inference.WARM_START_STEPS = 3
        plan = inference._build_warm_start_plan(obs)
    finally:
        inference.WARM_START_STEPS = old_steps
    assert all(item["resource_id"] in {"idle-1", "busy-1"} for item in plan)
    assert all(item["action_type"] in {"terminate", "resize"} for item in plan)

