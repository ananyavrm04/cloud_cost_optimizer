"""
FastAPI application for the Cloud Cost Optimizer Environment.

Usage (dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

Usage (prod / Docker):
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

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


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
