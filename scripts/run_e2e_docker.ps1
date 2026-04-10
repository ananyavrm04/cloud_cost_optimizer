$ErrorActionPreference = "Stop"

Write-Host "[E2E] Building and running dockerized stack..."
docker compose -f docker-compose.e2e.yml up --build --abort-on-container-exit --exit-code-from inference-runner

Write-Host "[E2E] Stack finished. Cleaning up containers..."
docker compose -f docker-compose.e2e.yml down --remove-orphans
