"""
Cloud Cost Optimizer Environment Client.

Usage:
    with CloudCostEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="easy")
        result = env.step(CloudCostAction(action_type="skip"))
"""

import time
from typing import Callable, Dict, List, Optional

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


class ReconnectingCloudCostEnv:
    """
    Reconnecting wrapper at client layer (Enhancement #58).

    It retries reset/step/state calls by reconnecting and replaying executed actions.
    """

    def __init__(self, base_url: str, retries: int = 2, backoff_seconds: float = 0.2):
        self.base_url = base_url
        self.retries = max(1, retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._ctx = None
        self._env = None
        self._task_id: Optional[str] = None
        self._replay_actions: List[CloudCostAction] = []

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self) -> None:
        if self._ctx is None:
            return
        try:
            self._ctx.__exit__(None, None, None)
        finally:
            self._ctx = None
            self._env = None

    def _connect(self) -> None:
        self.close()
        self._ctx = CloudCostEnv(base_url=self.base_url).sync()
        self._env = self._ctx.__enter__()

    def _recover_and_replay(self) -> None:
        self._connect()
        if not self._task_id:
            return
        self._env.reset(task_id=self._task_id)
        for action in self._replay_actions:
            self._env.step(action)

    def _with_retry(self, fn: Callable):
        last_error = None
        for attempt in range(self.retries):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries - 1:
                    break
                time.sleep(self.backoff_seconds * (2 ** attempt))
                self._recover_and_replay()
        raise last_error

    def reset(self, task_id: str, seed: int | None = None):
        def call():
            self._task_id = task_id
            self._replay_actions = []
            kwargs = {"task_id": task_id}
            if seed is not None:
                kwargs["seed"] = seed
            return self._env.reset(**kwargs)

        return self._with_retry(call)

    def step(self, action: CloudCostAction, **kwargs):
        def call():
            result = self._env.step(action, **kwargs)
            self._replay_actions.append(action)
            return result

        return self._with_retry(call)

    def state(self):
        return self._with_retry(lambda: self._env.state())
