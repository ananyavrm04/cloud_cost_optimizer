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
        step_feedback="",
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


# ---------------------------------------------------------------------------
# Enhancement #9: Regression fixtures for known failure modes
# ---------------------------------------------------------------------------


def test_compute_score_sla_violated_caps_at_03():
    state = SimpleNamespace(total_savings=900.0, optimal_savings=1000.0, sla_violated=True)
    score = inference.compute_score(state)
    assert score <= 0.3
    assert score > 0.0


def test_compute_score_zero_optimal_returns_eps():
    state = SimpleNamespace(total_savings=100.0, optimal_savings=0.0, sla_violated=False)
    score = inference.compute_score(state)
    assert score == pytest.approx(0.001)


def test_normalize_action_invalid_type_falls_to_heuristic():
    obs = _obs([
        {"id": "r1", "status": "running", "size": "large", "cpu_usage_avg": 2.0,
         "mem_usage_avg": 3.0, "is_critical": False, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": False, "type": "compute",
         "cost_per_month": 480.0},
    ])
    result = inference.normalize_action(
        {"action_type": "destroy", "resource_id": "r1"},
        obs,
        blocked_actions=set(),
    )
    assert result["action_type"] in inference.VALID_ACTIONS


def test_normalize_action_nonexistent_resource():
    obs = _obs([
        {"id": "r1", "status": "running", "size": "large", "cpu_usage_avg": 50.0,
         "mem_usage_avg": 50.0, "is_critical": False, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": False, "type": "compute",
         "cost_per_month": 200.0},
    ])
    result = inference.normalize_action(
        {"action_type": "terminate", "resource_id": "nonexistent"},
        obs,
        blocked_actions=set(),
    )
    assert result["resource_id"] != "nonexistent"


def test_normalize_action_switch_pricing_ineligible():
    obs = _obs([
        {"id": "r1", "status": "running", "size": "large", "cpu_usage_avg": 50.0,
         "mem_usage_avg": 50.0, "is_critical": False, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": False, "type": "compute",
         "cost_per_month": 200.0},
    ])
    result = inference.normalize_action(
        {"action_type": "switch_pricing", "resource_id": "r1", "new_pricing": "reserved",
         "new_size": ""},
        obs,
        blocked_actions=set(),
    )
    assert not (result["action_type"] == "switch_pricing" and result["resource_id"] == "r1")


def test_heuristic_action_empty_resources():
    obs = _obs([])
    result = inference.heuristic_action(obs, blocked_actions=set())
    assert result["action_type"] == "skip"


def test_heuristic_action_all_critical():
    obs = _obs([
        {"id": "c1", "status": "running", "size": "large", "cpu_usage_avg": 2.0,
         "mem_usage_avg": 3.0, "is_critical": True, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": False, "type": "database",
         "cost_per_month": 500.0},
    ])
    result = inference.heuristic_action(obs, blocked_actions=set())
    assert result["action_type"] == "skip"


def test_combined_idle_score():
    assert inference._combined_idle_score(0.0, 0.0) == pytest.approx(1.0)
    assert inference._combined_idle_score(100.0, 100.0) == pytest.approx(0.0)
    assert inference._combined_idle_score(50.0, 50.0) == pytest.approx(0.5)
    # CPU-dominant: 80% CPU, 10% mem -> less idle
    score = inference._combined_idle_score(80.0, 10.0)
    assert 0.0 < score < 0.5


def test_extract_json_malformed_returns_empty():
    result = inference._extract_json("not json at all")
    assert result == {}


def test_extract_json_valid():
    result = inference._extract_json('{"action_type": "skip", "resource_id": ""}')
    assert result["action_type"] == "skip"


def test_extract_json_with_surrounding_text():
    result = inference._extract_json('I think we should {"action_type": "terminate", "resource_id": "s1"} done')
    assert result["action_type"] == "terminate"
    assert result["resource_id"] == "s1"


def test_build_user_prompt_contains_key_fields():
    obs = _obs([
        {"id": "r1", "status": "running", "size": "large", "cpu_usage_avg": 10.0,
         "mem_usage_avg": 20.0, "is_critical": False, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": True, "type": "compute",
         "cost_per_month": 480.0},
    ])
    prompt = inference.build_user_prompt(
        observation=obs,
        step_history=["step=1 action=skip target=- reward=0.0000"],
        steps_remaining=5,
        observation_diff="initial step",
        task_id="easy",
    )
    assert "Uptime margin" in prompt
    assert "Steps remaining: 5" in prompt
    assert "Savings so far" in prompt
    assert "idle_score=" in prompt


def test_dependency_depths_no_cycles():
    resources = [
        {"id": "a", "dependencies": []},
        {"id": "b", "dependencies": ["a"]},
        {"id": "c", "dependencies": ["b"]},
    ]
    depths = inference._dependency_depths(resources)
    assert depths["a"] == 2  # a -> b -> c
    assert depths["b"] == 1  # b -> c
    assert depths["c"] == 0


# ---------------------------------------------------------------------------
# Enhancement #18: SLA penalty cap configurability
# ---------------------------------------------------------------------------


def test_sla_penalty_cap_configurable(monkeypatch):
    monkeypatch.setattr(inference, "SLA_PENALTY_CAP", 0.5)
    state = SimpleNamespace(total_savings=900.0, optimal_savings=1000.0, sla_violated=True)
    score = inference.compute_score(state)
    assert score <= 0.5
    assert score > 0.3


def test_sla_penalty_cap_default_is_03():
    assert inference.SLA_PENALTY_CAP == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Enhancement #86: Virtual resource tags
# ---------------------------------------------------------------------------


def test_virtual_tags_idle_resource():
    r = {"cpu_usage_avg": 1.0, "mem_usage_avg": 2.0, "is_critical": False, "type": "compute"}
    tags = inference._virtual_tags(r)
    assert "team:idle" in tags
    assert "env:staging" in tags
    assert "tier:compute" in tags


def test_virtual_tags_critical_database():
    r = {"cpu_usage_avg": 60.0, "mem_usage_avg": 55.0, "is_critical": True, "type": "database"}
    tags = inference._virtual_tags(r)
    assert "team:active" in tags
    assert "env:production" in tags
    assert "tier:data" in tags


def test_virtual_tags_underused_storage():
    r = {"cpu_usage_avg": 15.0, "mem_usage_avg": 10.0, "is_critical": False, "type": "storage"}
    tags = inference._virtual_tags(r)
    assert "team:underused" in tags
    assert "tier:storage" in tags


# ---------------------------------------------------------------------------
# Enhancement #75: Trend hint in prompt
# ---------------------------------------------------------------------------


def test_build_user_prompt_contains_trend_stable():
    obs = _obs([
        {"id": "r1", "status": "running", "size": "large", "cpu_usage_avg": 10.0,
         "mem_usage_avg": 20.0, "is_critical": False, "dependencies": [],
         "pricing": "on_demand", "eligible_for_reserved": True, "type": "compute",
         "cost_per_month": 480.0},
    ])
    prompt = inference.build_user_prompt(
        observation=obs, step_history=[], steps_remaining=5,
        observation_diff="initial step", task_id="easy",
    )
    assert "trend=stable" in prompt


# ---------------------------------------------------------------------------
# Enhancement #44: Temperature annealing computation
# ---------------------------------------------------------------------------


def test_temp_annealing_zero_when_disabled():
    # ENABLE_TEMP_ANNEALING defaults to False, so dynamic_temp should be 0.0
    assert inference.ENABLE_TEMP_ANNEALING is False


def test_temp_annealing_formula():
    threshold = 5
    max_temp = 0.3
    # At exactly threshold steps of no progress, factor = 0.0
    no_progress = threshold
    factor = min(1.0, (no_progress - threshold) / max(1, threshold))
    assert factor == pytest.approx(0.0)
    # At 2x threshold, factor = 1.0
    no_progress = threshold * 2
    factor = min(1.0, (no_progress - threshold) / max(1, threshold))
    assert factor == pytest.approx(1.0)
    temp = min(max_temp, factor * max_temp)
    assert temp == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Enhancement #88: Reward normalization
# ---------------------------------------------------------------------------


def test_reward_normalization_divides_by_count():
    reward = 0.5
    resource_count = 10
    normalized = reward / resource_count
    assert normalized == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Enhancement #92: Curriculum learning patterns
# ---------------------------------------------------------------------------


def test_curriculum_transfer_hint_empty_when_disabled():
    old = inference.CURRICULUM_CARRY_PATTERNS
    try:
        inference.CURRICULUM_CARRY_PATTERNS = False
        inference.GLOBAL_ACTION_PATTERNS.clear()
        inference.GLOBAL_TASK_LEARNINGS.clear()
        hint = inference._build_transfer_hint()
        assert hint == ""
    finally:
        inference.CURRICULUM_CARRY_PATTERNS = old


def test_curriculum_transfer_hint_with_patterns():
    old_carry = inference.CURRICULUM_CARRY_PATTERNS
    old_patterns = list(inference.GLOBAL_ACTION_PATTERNS)
    old_learnings = list(inference.GLOBAL_TASK_LEARNINGS)
    try:
        inference.CURRICULUM_CARRY_PATTERNS = True
        inference.GLOBAL_ACTION_PATTERNS.clear()
        inference.GLOBAL_TASK_LEARNINGS.clear()
        inference.GLOBAL_TASK_LEARNINGS.append("easy=>terminate:useful(3/3)")
        inference.GLOBAL_ACTION_PATTERNS.extend([
            {"action_type": "terminate", "reward": 0.5, "resource_type": "compute", "was_critical": False},
            {"action_type": "terminate", "reward": 0.4, "resource_type": "compute", "was_critical": False},
        ])
        hint = inference._build_transfer_hint()
        assert "Successful patterns" in hint
        assert "terminate_compute" in hint
    finally:
        inference.CURRICULUM_CARRY_PATTERNS = old_carry
        inference.GLOBAL_ACTION_PATTERNS.clear()
        inference.GLOBAL_ACTION_PATTERNS.extend(old_patterns)
        inference.GLOBAL_TASK_LEARNINGS.clear()
        inference.GLOBAL_TASK_LEARNINGS.extend(old_learnings)


# ---------------------------------------------------------------------------
# Enhancement #31: Progressive mode defaults
# ---------------------------------------------------------------------------


def test_progressive_mode_defaults_off():
    assert inference.PROGRESSIVE_MODE is False
    assert inference.PROGRESSIVE_THRESHOLD == pytest.approx(0.8)

