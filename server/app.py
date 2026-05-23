"""
FastAPI application for the Cloud Cost Optimizer Environment.

Usage (dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

Usage (prod / Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from dotenv import load_dotenv
load_dotenv()  # MUST run before any code that reads env vars

from openenv.core.env_server.http_server import create_app

try:
    from ..models import CloudCostAction, CloudCostObservation
    from .cloud_cost_environment import CloudCostEnvironment
except ImportError:
    from models import CloudCostAction, CloudCostObservation
    from server.cloud_cost_environment import CloudCostEnvironment


app = create_app(
    CloudCostEnvironment,       # class (factory), NOT an instance
    CloudCostAction,
    CloudCostObservation,
    env_name="cloud_cost_optimizer",
    max_concurrent_envs=1,
)

from pathlib import Path
from fastapi.staticfiles import StaticFiles

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")

@app.get("/")
def health() -> dict:
    return {"status": "ok", "environment": "cloud_cost_optimizer"}


@app.get("/health")
def health_extended() -> dict:
    return {"status": "healthy", "name": "cloud_cost_optimizer"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "cloud_cost_optimizer",
        "description": "Cloud cost optimization simulation environment.",
    }


try:
    from .agent import decide as _agent_decide, DecideRequest, DecideResponse
except ImportError:
    from server.agent import decide as _agent_decide, DecideRequest, DecideResponse


@app.post("/agent/decide", response_model=DecideResponse)
async def agent_decide_endpoint(req: DecideRequest) -> DecideResponse:
    """LLM-powered action selection for the autonomous optimization cycle."""
    return await _agent_decide(req)


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
