"""
Optional integration test: real uvicorn server + websocket client round-trip.
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


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION", "0") != "1",
    reason="Set RUN_INTEGRATION=1 to run integration round-trip test.",
)
def test_client_server_roundtrip():
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
        healthy = False
        for _ in range(30):
            try:
                response = requests.get(f"{base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(0.3)

        assert healthy, "Server did not become healthy in time."

        with CloudCostEnv(base_url=base_url).sync() as env:
            reset_result = env.reset(task_id="easy")
            obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result
            assert obs.done is False
            step_result = env.step(CloudCostAction(action_type="skip"))
            assert step_result.done is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
