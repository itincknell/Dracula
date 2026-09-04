"""Small ASGI safeguards specific to the public stateless route surface."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi.responses import JSONResponse

from dracula.api.stateless_contracts import StatelessApiErrorResponse

MAX_STATELESS_REQUEST_BYTES = 64 * 1024


class StatelessRequestBodyLimitMiddleware:
    """Reject oversized request bodies before JSON or Pydantic parsing."""

    def __init__(self, app: Any, max_bytes: int = MAX_STATELESS_REQUEST_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("request body limit must be a positive integer")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Buffer one mutating request up to the limit, then replay it downstream."""

        if scope.get("type") != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                await self.app(scope, receive, send)
                return
            if message.get("type") != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                error = StatelessApiErrorResponse(
                    code="request_too_large",
                    message="request body exceeds the stateless API limit",
                    retryable=False,
                )
                response = JSONResponse(
                    status_code=413,
                    content=error.model_dump(mode="json"),
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            # Downstream ASGI code receives the same body once, independent of
            # how many chunks arrived from API Gateway or the local server.
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay_receive, send)


__all__ = (
    "MAX_STATELESS_REQUEST_BYTES",
    "StatelessRequestBodyLimitMiddleware",
)
