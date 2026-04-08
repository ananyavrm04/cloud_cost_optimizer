"""
Cloud Cost Optimizer Environment — core logic.
"""

import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
try:
    from ..models import CloudCostAction, CloudCostObservation, CloudCostState, Resource
except (ImportError, ModuleNotFoundError):
    from models import CloudCostAction, CloudCostObservation, CloudCostState, Resource


# Cost multipliers per size tier
SIZE_MULTIPLIERS: dict[str, float] = {
    "small": 1.0,
    "medium": 2.0,
    "large": 4.0,
    "xlarge": 8.0,
}

# Fraction of on_demand cost after switching pricing
PRICING_MULTIPLIERS: dict[str, float] = {
    "on_demand": 1.0,
    "reserved": 0.60,   # 40% discount
    "spot": 0.40,       # 60% discount
}

VALID_ACTIONS = {"terminate", "resize", "switch_pricing", "skip"}
VALID_SIZES = set(SIZE_MULTIPLIERS.keys())
VALID_PRICING = {"reserved", "spot"}


class CloudCostEnvironment(Environment):
    """Simulated cloud infrastructure environment for cost optimization."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = CloudCostState(episode_id=str(uuid4()), step_count=0)
        self._resources: list[Resource] = []
        self._original_cost: float = 0.0
        self._max_steps: int = 0
        self._sla_target: float = 99.9
        self._initial_uptime: float = 100.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy",
        **kwargs,
    ) -> CloudCostObservation:
        """Reset the environment for the given task."""
        resources, optimal_savings, sla_target, initial_uptime = self._load_task(task_id)

        self._resources = resources
        self._original_cost = sum(r.cost_per_month for r in self._resources)
        self._max_steps = max(1, len(self._resources) * 2)
        self._sla_target = sla_target
        self._initial_uptime = initial_uptime

        self._state = CloudCostState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            total_savings=0.0,
            sla_violated=False,
            optimal_savings=optimal_savings,
        )

        return self._make_observation(done=False, reward=0.0, feedback="Episode started.")

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: CloudCostAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CloudCostObservation:
        """Execute one action and return the resulting observation."""
        self._state.step_count += 1

        # Validate action type
        if action.action_type not in VALID_ACTIONS:
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Invalid action_type '{action.action_type}'. Must be one of {VALID_ACTIONS}.",
            )

        # Handle skip — episode ends
        if action.action_type == "skip":
            return self._make_observation(
                done=True,
                reward=0.0,
                feedback="Agent signalled done.",
            )

        # All other actions require a valid resource
        resource = self._find_resource(action.resource_id)
        if resource is None:
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Resource '{action.resource_id}' not found.",
            )

        if resource.status == "terminated":
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Resource '{action.resource_id}' is already terminated.",
            )

        # Dispatch to action handler
        if action.action_type == "terminate":
            return self._handle_terminate(resource)
        elif action.action_type == "resize":
            return self._handle_resize(resource, action.new_size)
        elif action.action_type == "switch_pricing":
            return self._handle_switch_pricing(resource, action.new_pricing)

        # Should never reach here
        return self._make_observation(done=False, reward=-0.1, feedback="Unknown error.")

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> CloudCostState:
        return self._state

    # ------------------------------------------------------------------
    # action handlers
    # ------------------------------------------------------------------

    def _handle_terminate(self, resource: Resource) -> CloudCostObservation:
        broken_deps = self._count_broken_deps(resource.id)
        cost_saved = resource.cost_per_month

        resource.status = "terminated"
        resource.cost_per_month = 0.0

        self._state.total_savings += cost_saved
        cost_saved_ratio = cost_saved / self._original_cost if self._original_cost > 0 else 0.0

        current_uptime = self._calculate_uptime()
        sla_penalty = max(0.0, self._sla_target - current_uptime)
        if sla_penalty > 0:
            self._state.sla_violated = True

        reward = cost_saved_ratio - (sla_penalty * 10) - (0.2 * broken_deps)

        done = self._check_done()
        feedback = (
            f"Terminated '{resource.id}'. Saved ${cost_saved:.2f}/mo."
            + (f" Broke {broken_deps} dependency chain(s)." if broken_deps else "")
            + (" Critical resource terminated." if resource.is_critical else "")
            + (f" SLA warning: uptime={current_uptime:.3f}%." if sla_penalty > 0 else "")
        )
        return self._make_observation(done=done, reward=reward, feedback=feedback)

    def _handle_resize(self, resource: Resource, new_size: str) -> CloudCostObservation:
        if new_size not in VALID_SIZES:
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Invalid size '{new_size}'. Must be one of {VALID_SIZES}.",
            )
        if new_size == resource.size:
            return self._make_observation(
                done=False,
                reward=-0.05,
                feedback=f"Resource '{resource.id}' is already size '{new_size}'.",
            )

        old_size = resource.size
        old_cost = resource.cost_per_month
        base_cost = old_cost / SIZE_MULTIPLIERS[resource.size]
        new_cost = base_cost * SIZE_MULTIPLIERS[new_size]
        cost_saved = old_cost - new_cost

        resource.size = new_size
        resource.cost_per_month = new_cost
        resource.status = "resized"

        self._state.total_savings += cost_saved
        cost_saved_ratio = cost_saved / self._original_cost if self._original_cost > 0 else 0.0

        current_uptime = self._calculate_uptime()
        sla_penalty = max(0.0, self._sla_target - current_uptime)
        if sla_penalty > 0:
            self._state.sla_violated = True

        reward = cost_saved_ratio - (sla_penalty * 10)

        done = self._check_done()
        feedback = (
            f"Resized '{resource.id}' from {old_size} to {new_size}. "
            f"Saved ${cost_saved:.2f}/mo."
        ) if cost_saved >= 0 else (
            f"Resized '{resource.id}' from {old_size} to {new_size} - cost increased by ${-cost_saved:.2f}/mo."
        )
        return self._make_observation(done=done, reward=reward, feedback=feedback)

    def _handle_switch_pricing(self, resource: Resource, new_pricing: str) -> CloudCostObservation:
        if new_pricing not in VALID_PRICING:
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Invalid pricing '{new_pricing}'. Must be one of {VALID_PRICING}.",
            )
        if not resource.eligible_for_reserved:
            return self._make_observation(
                done=False,
                reward=-0.1,
                feedback=f"Resource '{resource.id}' is not eligible for pricing changes.",
            )
        if new_pricing == resource.pricing:
            return self._make_observation(
                done=False,
                reward=-0.05,
                feedback=f"Resource '{resource.id}' is already on '{new_pricing}' pricing.",
            )

        old_cost = resource.cost_per_month
        # Convert to on_demand baseline first, then apply new multiplier
        on_demand_base = old_cost / PRICING_MULTIPLIERS[resource.pricing]
        new_cost = on_demand_base * PRICING_MULTIPLIERS[new_pricing]
        cost_saved = old_cost - new_cost

        resource.pricing = new_pricing
        resource.cost_per_month = new_cost

        self._state.total_savings += cost_saved
        cost_saved_ratio = cost_saved / self._original_cost if self._original_cost > 0 else 0.0

        current_uptime = self._calculate_uptime()
        sla_penalty = max(0.0, self._sla_target - current_uptime)
        if sla_penalty > 0:
            self._state.sla_violated = True

        reward = cost_saved_ratio - (sla_penalty * 10)

        done = self._check_done()
        feedback = (
            f"Switched '{resource.id}' to {new_pricing} pricing. Saved ${cost_saved:.2f}/mo."
        )
        return self._make_observation(done=done, reward=reward, feedback=feedback)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _find_resource(self, resource_id: str) -> Resource | None:
        for r in self._resources:
            if r.id == resource_id:
                return r
        return None

    def _count_broken_deps(self, terminated_id: str) -> int:
        """Count running resources that depend on the resource being terminated."""
        count = 0
        for r in self._resources:
            if r.status == "running" and terminated_id in r.dependencies:
                count += 1
        return count

    def _calculate_uptime(self) -> float:
        """
        Calculate current uptime.
        Starts from task's initial_uptime (100.0 for easy/medium, 99.92 for hard).
        Each terminated critical resource reduces uptime by 0.05%.
        This ensures hard task (init=99.92, SLA=99.9) fails SLA after 1 critical termination.
        """
        terminated_critical = sum(
            1 for r in self._resources
            if r.status == "terminated" and r.is_critical
        )
        uptime = self._initial_uptime - (terminated_critical * 0.05)
        return max(0.0, uptime)

    def _check_done(self) -> bool:
        """Episode ends when max-step budget is exhausted."""
        return self._state.step_count >= self._max_steps

    def _make_observation(
        self,
        done: bool,
        reward: float,
        feedback: str,
    ) -> CloudCostObservation:
        current_cost = sum(
            r.cost_per_month for r in self._resources if r.status != "terminated"
        )
        return CloudCostObservation(
            resources=[r.model_dump() for r in self._resources],
            current_monthly_cost=round(current_cost, 2),
            original_monthly_cost=round(self._original_cost, 2),
            sla_target=self._sla_target,
            current_uptime=self._calculate_uptime(),
            budget=0.0,
            step_feedback=feedback,
            done=done,
            reward=reward,
        )

    def _load_task(self, task_id: str) -> tuple[list[Resource], float, float, float]:
        """Load task JSON and return (resources, optimal_savings, sla_target, initial_uptime)."""
        tasks_dir = Path(__file__).parent.parent / "tasks"
        path = tasks_dir / f"{task_id}.json"
        with open(path) as f:
            data = json.load(f)
        resources = [Resource(**r) for r in data["resources"]]
        optimal_savings = float(data.get("optimal_savings", 0.0))
        sla_target = float(data.get("sla_target", 99.9))
        initial_uptime = float(data.get("initial_uptime", 100.0))
        return resources, optimal_savings, sla_target, initial_uptime
