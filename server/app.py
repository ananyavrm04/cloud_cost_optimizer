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


# ── per-IP rate limiting (defense-in-depth on the public POST endpoints) ──
try:
    from .ratelimit import rate_limit_middleware
except ImportError:
    from server.ratelimit import rate_limit_middleware
app.middleware("http")(rate_limit_middleware)


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


# ── training dashboard endpoints ──
try:
    from .trainer import (
        StartTrainingRequest, StartTrainingResponse, RunSummary,
        start_training, get_run_summary, list_runs, get_run_episodes,
    )
except ImportError:
    from server.trainer import (
        StartTrainingRequest, StartTrainingResponse, RunSummary,
        start_training, get_run_summary, list_runs, get_run_episodes,
    )


@app.post("/training/start", response_model=StartTrainingResponse)
async def training_start_endpoint(req: StartTrainingRequest) -> StartTrainingResponse:
    return await start_training(req)


@app.get("/training/runs")
def training_list_runs_endpoint() -> list[RunSummary]:
    return list_runs()


@app.get("/training/runs/{run_id}")
def training_run_summary_endpoint(run_id: str):
    summary = get_run_summary(run_id)
    if summary is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return summary


@app.get("/training/runs/{run_id}/episodes")
def training_run_episodes_endpoint(run_id: str) -> list[dict]:
    return get_run_episodes(run_id)


# ── static frontend ──
# Mounted LAST so this catch-all ("/") never shadows the API routes defined above.
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
