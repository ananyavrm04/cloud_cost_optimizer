"""
Optional integration tests: real uvicorn server + websocket client round-trip.
Run manually:
    RUN_INTEGRATION=1 python -m pytest tests/test_integration_roundtrip.py -q
"""

import os
import subprocess
import sys
import time

import pytest
import requests

from client import CloudCostEnv
from models import CloudCostAction


def _wait_until_healthy(base_url: str, attempts: int = 30) -> bool:
    for _ in range(attempts):
        try:
            response = requests.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="module")
def live_server():
    if os.getenv("RUN_INTEGRATION", "0") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 to run integration round-trip tests.")

    host = "127.0.0.1"
    port = 8871
    base_url = f"http://{host}:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_until_healthy(base_url), "Server did not become healthy in time."
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.integration
def test_health_endpoint(live_server):
    response = requests.get(f"{live_server}/health", timeout=2.0)
    assert response.status_code == 200


@pytest.mark.integration
def test_reset_state_step_flow(live_server):
    with CloudCostEnv(base_url=live_server).sync() as env:
        reset_result = env.reset(task_id="easy")
        obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result
        assert obs.done is False
        state = env.state()
        assert state.task_id == "easy"
        assert state.step_count == 0

        step_result = env.step(CloudCostAction(action_type="skip"))
        assert step_result.done is True


@pytest.mark.integration
def test_invalid_action_reports_error_code(live_server):
    with CloudCostEnv(base_url=live_server).sync() as env:
        env.reset(task_id="easy")
        step_result = env.step(CloudCostAction(action_type="resize", resource_id="missing-id", new_size="small"))
        assert step_result.done is False
        assert step_result.reward < 0
        assert step_result.observation.metadata.get("error_code") == "ERR_RESOURCE_NOT_FOUND"


@pytest.mark.integration
def test_step_after_done_remains_done(live_server):
    with CloudCostEnv(base_url=live_server).sync() as env:
        env.reset(task_id="easy")
        done_result = env.step(CloudCostAction(action_type="skip"))
        assert done_result.done is True
        late_result = env.step(CloudCostAction(action_type="terminate", resource_id="web-server-1"))
        assert late_result.done is True
