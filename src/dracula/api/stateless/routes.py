"""Connect stateless gameplay services to public HTTP endpoints.

FastAPI decorators such as ``application.post('/games')`` register the nested
function beneath them as the handler for that method and path. FastAPI parses
the request JSON into the handler's annotated Pydantic model before the handler
runs. It also uses ``response_model`` to document the successful response.

The handlers in this module only pass typed requests to service objects and
convert service results into HTTP responses. Reconstruction, game rules,
policy execution, and narration logic remain outside the HTTP layer.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dracula.api.contracts import ServiceResponse
from dracula.api.narration.service import NarrationAdapter, NarrationService
from dracula.policy.contracts import PolicyExecutor
from dracula.api.stateless.contracts import (
    ApplyStatelessCommandRequest,
    NarrationRequest,
    NarrationResponse,
    ResumeStatelessGameRequest,
    StartStatelessGameRequest,
    StatelessApiErrorResponse,
    StatelessGameResponse,
    StatelessHealthResponse,
)
from dracula.api.stateless.replay import ReplayCache
from dracula.api.stateless.service import StatelessGameplayService


def _json(response: ServiceResponse) -> JSONResponse:
    """Turn a framework-independent service result into an HTTP JSON response."""

    return JSONResponse(status_code=response.status_code, content=response.body)


def _register_validation_error(application: FastAPI) -> None:
    """Install the public response used when FastAPI rejects request data."""

    # ``exception_handler`` registers this nested async function for Pydantic
    # request-validation failures anywhere in this FastAPI application.
    @application.exception_handler(RequestValidationError)
    async def stateless_validation_error(
        _request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        # Pydantic details may echo caller data. The public API returns one
        # stable category rather than reflecting those details to the browser.
        body = StatelessApiErrorResponse(
            code="validation_error",
            message="request did not match the stateless API contract",
            retryable=False,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


def _register_health_route(
    application: FastAPI,
    gameplay: StatelessGameplayService,
    narration: NarrationService,
) -> None:
    """Register GET /health against the configured service instances."""

    # ``response_model`` is the documented success shape. FastAPI can serialize
    # the returned Pydantic model directly.
    @application.get("/health", response_model=StatelessHealthResponse)
    def stateless_health() -> StatelessHealthResponse:
        """Return dependency configuration without private runtime details."""

        return StatelessHealthResponse(
            opponent_configured=gameplay.opponent_configured,
            narration_enabled=narration.enabled,
        )


def _register_start_route(
    application: FastAPI,
    gameplay: StatelessGameplayService,
) -> None:
    """Register POST /games and capture the gameplay service in its handler."""

    # The nested function closes over ``gameplay``. The service is constructed
    # once at startup rather than rebuilt for every HTTP request.
    @application.post(
        "/games",
        response_model=StatelessGameResponse,
        status_code=201,
        # ``responses`` adds known error bodies to OpenAPI documentation; the
        # service still chooses the actual status code at runtime.
        responses={
            422: {"model": StatelessApiErrorResponse},
            503: {"model": StatelessApiErrorResponse},
        },
    )
    def start_stateless_game(request: StartStatelessGameRequest) -> JSONResponse:
        """Start a game and return its first authoritative recovery envelope."""

        return _json(gameplay.create_game(request))


def _register_command_route(
    application: FastAPI,
    gameplay: StatelessGameplayService,
) -> None:
    """Register the endpoint that proposes one next gameplay command."""

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


def _register_resume_route(
    application: FastAPI,
    gameplay: StatelessGameplayService,
) -> None:
    """Register reconstruction of an existing seed-and-history envelope."""

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


def _register_narration_route(
    application: FastAPI,
    narration: NarrationService,
) -> None:
    """Register the optional, non-mutating narration request endpoint."""

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


def configure_stateless_routes(
    application: FastAPI,
    *,
    policy_executor: PolicyExecutor | None,
    narration_adapter: NarrationAdapter | None,
    replay_cache_entries: int,
    game_seed_factory: Callable[[], str] | None,
) -> None:
    """Create one service graph and attach all stateless HTTP routes.

    ``application.state`` is FastAPI/Starlette's storage for process-lifetime
    objects. The retained gameplay-service reference lets privacy tests inspect
    reconstructed state; it does not make cached data authoritative.
    """

    cache = ReplayCache(replay_cache_entries)
    gameplay = StatelessGameplayService(
        policy_executor=policy_executor,
        game_seed_factory=game_seed_factory,
        cache=cache,
    )
    narration = NarrationService(
        gameplay,
        adapter=narration_adapter,
    )
    # Tests inspect replayed private state through this service to verify that
    # none of it crosses the public projection boundary.
    application.state.stateless_gameplay_service = gameplay

    _register_validation_error(application)
    _register_health_route(application, gameplay, narration)
    _register_start_route(application, gameplay)
    _register_command_route(application, gameplay)
    _register_resume_route(application, gameplay)
    _register_narration_route(application, narration)
