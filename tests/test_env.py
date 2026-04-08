"""
Tests for the Cloud Cost Optimizer Environment.
Run: python -m pytest tests/ -v
"""

import json
import pytest
from pathlib import Path

from models import CloudCostAction, CloudCostObservation, CloudCostState, Resource
from server.cloud_cost_environment import CloudCostEnvironment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TASKS_DIR = Path(__file__).parent.parent / "tasks"

MOCK_EASY = {
    "resources": [
        {
            "id": "s1", "name": "Idle Server", "type": "compute",
            "size": "large", "cpu_usage_avg": 2.0, "mem_usage_avg": 3.0,
            "cost_per_month": 480.0, "is_critical": False,
            "dependencies": [], "pricing": "on_demand",
            "eligible_for_reserved": False, "status": "running",
        },
        {
            "id": "s2", "name": "Active Server", "type": "compute",
            "size": "medium", "cpu_usage_avg": 75.0, "mem_usage_avg": 60.0,
            "cost_per_month": 200.0, "is_critical": True,
            "dependencies": [], "pricing": "on_demand",
            "eligible_for_reserved": True, "status": "running",
        },
        {
            "id": "s3", "name": "Dep Server", "type": "compute",
            "size": "small", "cpu_usage_avg": 5.0, "mem_usage_avg": 5.0,
            "cost_per_month": 60.0, "is_critical": False,
            "dependencies": ["s1"], "pricing": "on_demand",
            "eligible_for_reserved": False, "status": "running",
        },
    ],
    "optimal_savings": 480.0,
    "sla_target": 99.9,
    "initial_uptime": 100.0,
}


@pytest.fixture(autouse=True)
def write_mock_task(tmp_path, monkeypatch):
    """Write mock task JSON and patch the tasks directory."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "easy.json").write_text(json.dumps(MOCK_EASY))

    # Patch _load_task to use tmp tasks dir
    original_load = CloudCostEnvironment._load_task

    def patched_load(self, task_id):
        path = tasks / f"{task_id}.json"
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        resources = [Resource(**r) for r in data["resources"]]
        return (
            resources,
            float(data.get("optimal_savings", 0.0)),
            float(data.get("sla_target", 99.9)),
            float(data.get("initial_uptime", 100.0)),
        )

    monkeypatch.setattr(CloudCostEnvironment, "_load_task", patched_load)


@pytest.fixture
def env():
    e = CloudCostEnvironment()
    e.reset(task_id="easy")
    return e


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self):
        e = CloudCostEnvironment()
        obs = e.reset(task_id="easy")
        assert isinstance(obs, CloudCostObservation)

    def test_reset_initialises_cost(self):
        e = CloudCostEnvironment()
        obs = e.reset(task_id="easy")
        assert obs.original_monthly_cost == 480.0 + 200.0 + 60.0

    def test_reset_all_resources_running(self):
        e = CloudCostEnvironment()
        obs = e.reset(task_id="easy")
        for r in obs.resources:
            assert r["status"] == "running"

    def test_reset_clears_state(self):
        e = CloudCostEnvironment()
        e.reset(task_id="easy")
        e.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        e.reset(task_id="easy")
        assert e.state.total_savings == 0.0
        assert e.state.step_count == 0
        assert e.state.sla_violated is False

    def test_reset_done_false(self):
        e = CloudCostEnvironment()
        obs = e.reset(task_id="easy")
        assert obs.done is False


# ---------------------------------------------------------------------------
# terminate
# ---------------------------------------------------------------------------

class TestTerminate:
    def test_terminate_idle_server(self, env):
        obs = env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        assert obs.reward > 0
        assert env.state.total_savings == 480.0

    def test_terminate_updates_status(self, env):
        env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        resource = next(r for r in env._resources if r.id == "s1")
        assert resource.status == "terminated"
        assert resource.cost_per_month == 0.0

    def test_terminate_critical_blocked(self, env):
        obs = env.step(CloudCostAction(action_type="terminate", resource_id="s2"))
        assert obs.reward < 0
        assert env.state.total_savings == 0.0

    def test_terminate_already_terminated(self, env):
        env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        obs = env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        assert obs.reward == -0.1

    def test_terminate_unknown_id(self, env):
        obs = env.step(CloudCostAction(action_type="terminate", resource_id="nonexistent"))
        assert obs.reward == -0.1

    def test_terminate_with_broken_dep(self, env):
        # s3 depends on s1 — terminating s1 breaks s3's dependency
        obs = env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        # reward should be reduced by 0.2 per broken dep
        assert obs.reward < (480.0 / 740.0)  # less than pure cost_saved_ratio

    def test_terminate_increments_step_count(self, env):
        env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        assert env.state.step_count == 1


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------

class TestResize:
    def test_resize_down_saves_money(self, env):
        obs = env.step(CloudCostAction(
            action_type="resize", resource_id="s1", new_size="small"
        ))
        assert obs.reward > 0
        resource = next(r for r in env._resources if r.id == "s1")
        assert resource.size == "small"
        # large(4x) -> small(1x): base = 480/4 = 120, new = 120*1 = 120, saved = 360
        assert env.state.total_savings == pytest.approx(360.0)

    def test_resize_same_size_rejected(self, env):
        obs = env.step(CloudCostAction(
            action_type="resize", resource_id="s1", new_size="large"
        ))
        assert obs.reward == pytest.approx(-0.05)

    def test_resize_invalid_size_rejected(self, env):
        obs = env.step(CloudCostAction(
            action_type="resize", resource_id="s1", new_size="superlarge"
        ))
        assert obs.reward == pytest.approx(-0.1)

    def test_resize_updates_status(self, env):
        env.step(CloudCostAction(action_type="resize", resource_id="s1", new_size="small"))
        resource = next(r for r in env._resources if r.id == "s1")
        assert resource.status == "resized"


# ---------------------------------------------------------------------------
# switch_pricing
# ---------------------------------------------------------------------------

class TestSwitchPricing:
    def test_switch_eligible_to_reserved(self, env):
        obs = env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s2", new_pricing="reserved"
        ))
        assert obs.reward > 0
        resource = next(r for r in env._resources if r.id == "s2")
        assert resource.pricing == "reserved"
        # 200 * 0.60 = 120, saved = 80
        assert env.state.total_savings == pytest.approx(80.0)

    def test_switch_non_eligible_blocked(self, env):
        obs = env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s1", new_pricing="reserved"
        ))
        assert obs.reward == pytest.approx(-0.1)

    def test_switch_same_pricing_rejected(self, env):
        # "on_demand" is not in VALID_PRICING {"reserved", "spot"} so action is rejected with -0.1
        obs = env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s2", new_pricing="on_demand"
        ))
        assert obs.reward == pytest.approx(-0.1)

    def test_switch_reserved_twice_rejected(self, env):
        # Switch to reserved first, then try again — same pricing = -0.05
        env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s2", new_pricing="reserved"
        ))
        obs = env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s2", new_pricing="reserved"
        ))
        assert obs.reward == pytest.approx(-0.05)

    def test_switch_invalid_pricing_rejected(self, env):
        obs = env.step(CloudCostAction(
            action_type="switch_pricing", resource_id="s2", new_pricing="free"
        ))
        assert obs.reward == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# skip
# ---------------------------------------------------------------------------

class TestSkip:
    def test_skip_ends_episode(self, env):
        obs = env.step(CloudCostAction(action_type="skip"))
        assert obs.done is True

    def test_skip_reward_zero(self, env):
        obs = env.step(CloudCostAction(action_type="skip"))
        assert obs.reward == pytest.approx(0.0)

    def test_skip_after_savings(self, env):
        env.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        obs = env.step(CloudCostAction(action_type="skip"))
        assert obs.done is True


# ---------------------------------------------------------------------------
# invalid action type
# ---------------------------------------------------------------------------

class TestInvalidAction:
    def test_unknown_action_type(self, env):
        obs = env.step(CloudCostAction(action_type="destroy", resource_id="s1"))
        assert obs.reward == pytest.approx(-0.1)
        assert obs.done is False


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

class TestState:
    def test_state_returns_cloud_cost_state(self, env):
        assert isinstance(env.state, CloudCostState)

    def test_state_task_id(self, env):
        assert env.state.task_id == "easy"

    def test_state_step_count_increments(self, env):
        env.step(CloudCostAction(action_type="skip"))
        assert env.state.step_count == 1

    def test_state_sla_not_violated_initially(self, env):
        assert env.state.sla_violated is False


# ---------------------------------------------------------------------------
# uptime & SLA
# ---------------------------------------------------------------------------

class TestUptimeAndSLA:
    def test_initial_uptime_100(self, env):
        obs = env._make_observation(done=False, reward=0.0, feedback="")
        assert obs.current_uptime == pytest.approx(100.0)

    def test_terminate_critical_reduces_uptime(self):
        # Directly set a critical resource to terminated and check uptime drops
        e = CloudCostEnvironment()
        e.reset(task_id="easy")
        # s2 is critical — forcibly mark it terminated to test _calculate_uptime
        e._resources[1].status = "terminated"
        assert e._calculate_uptime() < 100.0

    def test_sla_violated_flag_set(self):
        e = CloudCostEnvironment()
        e.reset(task_id="easy")
        # Set SLA target above 100% so any uptime triggers violation,
        # then forcibly terminate a critical resource and call step to trigger check
        e._sla_target = 100.1
        e._initial_uptime = 100.0
        # Terminate s1 (non-critical) — uptime stays at 100.0 < 100.1 → SLA violated
        e._resources[0].is_critical = False
        e.step(CloudCostAction(action_type="terminate", resource_id="s1"))
        assert e.state.sla_violated is True
