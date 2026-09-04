"""FastAPI transport wiring for the stateless gameplay services."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dracula.api.narration import NarrationAdapter, NarrationService
from dracula.api.policy import PolicyDescriptor, PolicyExecutor, ServiceResponse
from dracula.api.stateless_contracts import (
    ApplyStatelessCommandRequest,
    NarrationRequest,
    NarrationResponse,
    ResumeStatelessGameRequest,
    StartStatelessGameRequest,
    StatelessApiErrorResponse,
    StatelessGameResponse,
    StatelessHealthResponse,
)
from dracula.api.stateless_http import StatelessRequestBodyLimitMiddleware
from dracula.api.stateless_service import ReplayCache, StatelessGameplayService


def _json(response: ServiceResponse) -> JSONResponse:
    return JSONResponse(status_code=response.status_code, content=response.body)


def configure_stateless_routes(
    application: FastAPI,
    *,
    policy_executor: PolicyExecutor | None,
    policy_descriptor: PolicyDescriptor | None,
    narration_enabled: bool,
    narration_adapter: NarrationAdapter | None,
    replay_cache_entries: int,
    game_seed_factory: Callable[[], str] | None,
) -> None:
    """Attach the production transport to already-defined domain services."""

    application.state.replay_cache = ReplayCache(replay_cache_entries)
    application.add_middleware(StatelessRequestBodyLimitMiddleware)
    gameplay = StatelessGameplayService(
        policy_executor=policy_executor,
        policy_descriptor=policy_descriptor,
        game_seed_factory=game_seed_factory,
        cache=application.state.replay_cache,
    )
    narration = NarrationService(
        gameplay,
        enabled=narration_enabled,
        adapter=narration_adapter,
    )
    application.state.stateless_gameplay_service = gameplay
    application.state.narration_service = narration

    @application.exception_handler(RequestValidationError)
    async def stateless_validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        body = StatelessApiErrorResponse(
            code="validation_error",
            message="request did not match the stateless API contract",
            retryable=False,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @application.get("/health", response_model=StatelessHealthResponse)
    def stateless_health() -> StatelessHealthResponse:
        """Return stateless dependency configuration without private details."""

        return StatelessHealthResponse(
            opponent_configured=gameplay.opponent_configured,
            narration_enabled=narration_enabled,
            narration_configured=narration.configured,
        )

    @application.post(
        "/games",
        response_model=StatelessGameResponse,
        status_code=201,
        responses={
            422: {"model": StatelessApiErrorResponse},
            503: {"model": StatelessApiErrorResponse},
        },
    )
    def start_stateless_game(request: StartStatelessGameRequest) -> JSONResponse:
        """Start a game and return its first authoritative recovery envelope."""

        return _json(gameplay.create_game(request))

    @application.post(
        "/games/command",
        response_model=StatelessGameResponse,
        responses={
            409: {"model": StatelessApiErrorResponse},
            422: {"model": StatelessApiErrorResponse},
            503: {"model": StatelessApiErrorResponse},
        },
    )
    def apply_stateless_command(
        request: ApplyStatelessCommandRequest,
    ) -> JSONResponse:
        """Validate and append one gameplay command."""

        return _json(gameplay.apply_command(request))

    @application.post(
        "/games/resume",
        response_model=StatelessGameResponse,
        responses={
            422: {"model": StatelessApiErrorResponse},
            503: {"model": StatelessApiErrorResponse},
        },
    )
    def resume_stateless_game(
        request: ResumeStatelessGameRequest,
    ) -> JSONResponse:
        """Reconstruct and return a game without changing its history."""

        return _json(gameplay.resume_game(request))

    @application.post(
        "/narration",
        response_model=NarrationResponse,
        responses={
            409: {"model": StatelessApiErrorResponse},
            422: {"model": StatelessApiErrorResponse},
        },
    )
    def narrate_stateless_game(request: NarrationRequest) -> JSONResponse:
        """Generate one optional cue independently of gameplay mutation."""

        return _json(narration.generate(request))


__all__ = ("configure_stateless_routes",)
