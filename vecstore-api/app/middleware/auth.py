"""API key authentication middleware.

Checks X-API-Key header. Skips auth for /health and /docs paths.
Returns 401 with a structured error body on failure.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.logging_config import get_logger

log = get_logger(__name__)

SKIP_PATHS = {"/health", "/status/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in SKIP_PATHS or request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get("x-api-key", "")
        if provided != self._api_key:
            log.warning("auth_failed", path=request.url.path)
            origin = request.headers.get("origin", "*")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key header"},
                headers={"Access-Control-Allow-Origin": origin},
            )

        return await call_next(request)
