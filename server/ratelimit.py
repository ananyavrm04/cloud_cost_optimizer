"""
Lightweight in-memory, per-client-IP rate limiting for the public API.

The Space runs as a single container/process, so in-memory counters are enough
(no Redis needed). This is defense-in-depth on top of the agent's *outbound*
provider rate limiter (which already caps LLM spend): it stops request floods
and abuse of the expensive state-changing POST endpoints, while leaving GET
reads (dashboard polling, static assets) unthrottled so the UI stays responsive.

Tunable via env:
  CCO_RL_WINDOW_SEC        sliding window in seconds        (default 60)
  CCO_RL_POST_PER_MIN      max POSTs per IP per window       (default 40)
  CCO_RL_TRAINING_PER_MIN  max /training/start per IP/window (default 3)
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

WINDOW_SEC   = float(os.getenv("CCO_RL_WINDOW_SEC", "60"))
POST_MAX     = int(os.getenv("CCO_RL_POST_PER_MIN", "40"))
TRAINING_MAX = int(os.getenv("CCO_RL_TRAINING_PER_MIN", "3"))

_buckets: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    """Real client IP — behind the Space proxy it's the first X-Forwarded-For entry."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _hit(key: str, limit: int, window: float) -> int:
    """Record a hit; return seconds-to-wait if the limit is exceeded, else 0."""
    now = time.monotonic()
    dq = _buckets[key]
    while dq and dq[0] < now - window:
        dq.popleft()
    if len(dq) >= limit:
        return int(window - (now - dq[0])) + 1
    dq.append(now)
    return 0


async def rate_limit_middleware(request: Request, call_next):
    # Only meter state-changing/expensive POSTs. GETs (polling, static files) pass through.
    if request.method == "POST":
        ip = client_ip(request)
        path = request.url.path.rstrip("/")
        retry = _hit(f"post:{ip}", POST_MAX, WINDOW_SEC)
        if not retry and path == "/training/start":
            retry = _hit(f"start:{ip}", TRAINING_MAX, WINDOW_SEC)
        if retry:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded — slow down. Retry in ~{retry}s."},
                headers={"Retry-After": str(retry)},
            )
    return await call_next(request)
