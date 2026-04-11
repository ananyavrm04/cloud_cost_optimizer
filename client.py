"""
Cloud Cost Optimizer Environment Client.

Usage:
    with CloudCostEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="easy")
        result = env.step(CloudCostAction(action_type="skip"))
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import CloudCostAction, CloudCostObservation, CloudCostState


class CloudCostEnv(EnvClient[CloudCostAction, CloudCostObservation, CloudCostState]):
    """WebSocket client for the Cloud Cost Optimizer environment."""

    def _step_payload(self, action: CloudCostAction) -> Dict:
        """Serialize action to dict for the WebSocket step message."""
        return {
            "action_type": action.action_type,
            "resource_id": action.resource_id,
            "new_size": action.new_size,
            "new_pricing": action.new_pricing,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CloudCostObservation]:
        """Deserialize server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = CloudCostObservation(
            schema_version=obs_data.get("schema_version", "1.0"),
            resources=obs_data.get("resources", []),
            current_monthly_cost=obs_data.get("current_monthly_cost", 0.0),
            original_monthly_cost=obs_data.get("original_monthly_cost", 0.0),
            sla_target=obs_data.get("sla_target", 99.9),
            current_uptime=obs_data.get("current_uptime", 100.0),
            step_feedback=obs_data.get("step_feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CloudCostState:
        """Deserialize state response into CloudCostState."""
        return CloudCostState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            total_savings=payload.get("total_savings", 0.0),
            sla_violated=payload.get("sla_violated", False),
            optimal_savings=payload.get("optimal_savings", 0.0),
        )
