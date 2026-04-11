"""
Data models for the Cloud Cost Optimizer Environment.
"""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class Resource(BaseModel):
    """Represents a single cloud resource in the simulated infrastructure."""

    id: str
    name: str
    type: str  # "compute" | "storage" | "database"
    size: str  # "small" | "medium" | "large" | "xlarge"
    cpu_usage_avg: float  # 0-100 percent
    mem_usage_avg: float  # 0-100 percent
    cost_per_month: float  # USD
    is_critical: bool
    dependencies: List[str] = Field(default_factory=list)  # list of resource IDs
    pricing: str = "on_demand"  # "on_demand" | "reserved" | "spot"
    eligible_for_reserved: bool = False
    status: str = "running"  # "running" | "terminated" | "resized"


class CloudCostAction(Action):
    """Action sent by the agent to the Cloud Cost Optimizer environment."""

    action_type: str = Field(
        ..., description="terminate | resize | switch_pricing | skip"
    )
    resource_id: str = Field(default="", description="ID of the target resource")
    new_size: str = Field(
        default="", description="For resize: small | medium | large | xlarge"
    )
    new_pricing: str = Field(
        default="", description="For switch_pricing: reserved | spot"
    )


class CloudCostObservation(Observation):
    """Observation returned to the agent after each reset or step."""

    schema_version: str = "1.0"
    resources: List[Dict[str, Any]] = Field(default_factory=list)
    current_monthly_cost: float = 0.0
    original_monthly_cost: float = 0.0
    sla_target: float = 99.9
    current_uptime: float = 100.0
    budget: float = 0.0
    step_feedback: str = ""


class CloudCostState(State):
    """Internal state of the environment (extends base State)."""

    # State already has: episode_id (Optional[str]), step_count (int)
    task_id: str = ""
    total_savings: float = 0.0
    sla_violated: bool = False
    optimal_savings: float = 0.0
