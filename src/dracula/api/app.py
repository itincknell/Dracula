"""Compatible FastAPI entry point for local and stateless gameplay."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, Path as ApiPath, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dracula.api.contracts import (
    API_VERSION,
    ApiErrorResponse,
    CreateGameRequest,
    EventsResponse,
    HealthResponse,
    HumanGameView,
    MoveRequest,
    VersionedMutationRequest,
)
from dracula.api.bedrock import BedrockRuntimeAdapter
from dracula.api.local_controllers import resolve_local_controller
from dracula.api.narration import NarrationAdapter
from dracula.api.policy import PolicyDescriptor, PolicyExecutor, ServiceResponse
from dracula.api.repository import (
    GameRepository,
    InMemoryGameRepository,
    SQLiteGameRepository,
)
from dracula.api.service import GameplayService
from dracula.api.stateless_app import create_stateless_app

LOCAL_GAME_SEED_ENV = "DRACULA_LOCAL_GAME_SEED"
GAMEPLAY_MODE_ENV = "DRACULA_GAMEPLAY_MODE"
REPLAY_CACHE_ENTRIES_ENV = "DRACULA_REPLAY_CACHE_ENTRIES"
DEFAULT_GAMEPLAY_MODE = "local"
DEFAULT_REPLAY_CACHE_ENTRIES = 256


def _environment_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _default_repository() -> GameRepository:
    database_path = os.getenv("DRACULA_DATABASE_PATH")
    if database_path is None:
        return InMemoryGameRepository()
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    return SQLiteGameRepository(database_path)


def _nonnegative_integer_environment(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _gameplay_mode(value: str | None) -> str:
    resolved = DEFAULT_GAMEPLAY_MODE if value is None else value.strip().lower()
    if resolved not in {"local", "stateless"}:
        raise ValueError(f"{GAMEPLAY_MODE_ENV} must be local or stateless")
    return resolved


def _json(response: ServiceResponse) -> JSONResponse:
    return JSONResponse(status_code=response.status_code, content=response.body)


def create_app(
    *,
    narration_enabled: bool | None = None,
    repository: GameRepository | None = None,
    policy_executor: PolicyExecutor | None = None,
    policy_descriptor: PolicyDescriptor | None = None,
    gameplay_mode: str | None = None,
    replay_cache_entries: int | None = None,
    narration_adapter: NarrationAdapter | None = None,
) -> FastAPI:
    """Assemble an explicitly selected local-stateful or stateless application."""

    resolved_narration = (
        _environment_flag("DRACULA_NARRATION_ENABLED", default=False)
        if narration_enabled is None
        else narration_enabled
    )
    resolved_gameplay_mode = _gameplay_mode(
        os.getenv(GAMEPLAY_MODE_ENV) if gameplay_mode is None else gameplay_mode
    )
    local_game_seed = os.getenv(LOCAL_GAME_SEED_ENV)
    if local_game_seed is not None and not local_game_seed:
        raise ValueError(f"{LOCAL_GAME_SEED_ENV} must be a nonempty value")
    resolved_executor, resolved_descriptor = resolve_local_controller(
        policy_executor, policy_descriptor
    )

    if resolved_gameplay_mode == "stateless":
        if repository is not None:
            raise ValueError("stateless gameplay cannot be configured with a repository")
        cache_entries = (
            _nonnegative_integer_environment(
                REPLAY_CACHE_ENTRIES_ENV, DEFAULT_REPLAY_CACHE_ENTRIES
            )
            if replay_cache_entries is None
            else replay_cache_entries
        )
        resolved_narration_adapter = narration_adapter
        if resolved_narration and resolved_narration_adapter is None:
            resolved_narration_adapter = BedrockRuntimeAdapter.from_environment()
        return create_stateless_app(
            policy_executor=resolved_executor,
            policy_descriptor=resolved_descriptor,
            narration_enabled=resolved_narration,
            narration_adapter=resolved_narration_adapter,
            replay_cache_entries=cache_entries,
            game_seed_factory=(None if local_game_seed is None else lambda: local_game_seed),
        )

    application = FastAPI(title="Dracula API", version=API_VERSION)
    application.state.narration_enabled = resolved_narration
    application.state.gameplay_mode = resolved_gameplay_mode
    application.state.game_repository = repository or _default_repository()
    application.state.gameplay_service = GameplayService(
        application.state.game_repository,
        policy_executor=resolved_executor,
        policy_descriptor=resolved_descriptor,
        narration_enabled=resolved_narration,
        game_seed_factory=(None if local_game_seed is None else lambda: local_game_seed),
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        body = ApiErrorResponse(
            code="validation_error",
            message="request did not match the API contract",
            retryable=False,
        )
        return JSONResponse(status_code=422, content=body.model_dump(mode="json"))

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(narration_enabled=resolved_narration)

    @application.post(
        "/games",
        response_model=HumanGameView,
        status_code=201,
        responses={409: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def start_game(request: CreateGameRequest) -> JSONResponse:
        return _json(application.state.gameplay_service.create_game(request))

    @application.get(
        "/games/{game_id}",
        response_model=HumanGameView,
        responses={404: {"model": ApiErrorResponse}},
    )
    def get_game(game_id: UUID) -> JSONResponse:
        return _json(application.state.gameplay_service.get_game(game_id))

    @application.post(
        "/games/{game_id}/moves",
        response_model=HumanGameView,
        responses={
            404: {"model": ApiErrorResponse},
            409: {"model": ApiErrorResponse},
            422: {"model": ApiErrorResponse},
        },
    )
    def submit_move(game_id: UUID, request: MoveRequest) -> JSONResponse:
        return _json(application.state.gameplay_service.human_move(game_id, request))

    @application.post(
        "/games/{game_id}/opponent-turn",
        response_model=HumanGameView,
        responses={
            202: {"model": HumanGameView},
            404: {"model": ApiErrorResponse},
            409: {"model": ApiErrorResponse},
            422: {"model": ApiErrorResponse},
            503: {"model": ApiErrorResponse},
        },
    )
    def opponent_turn(
        game_id: UUID, request: VersionedMutationRequest
    ) -> JSONResponse:
        return _json(application.state.gameplay_service.opponent_turn(game_id, request))

    @application.post(
        "/games/{game_id}/rounds/{round_number}/advance",
        response_model=HumanGameView,
        responses={
            404: {"model": ApiErrorResponse},
            409: {"model": ApiErrorResponse},
            422: {"model": ApiErrorResponse},
        },
    )
    def advance_round(
        game_id: UUID,
        request: VersionedMutationRequest,
        round_number: int = ApiPath(ge=1, le=6),
    ) -> JSONResponse:
        return _json(
            application.state.gameplay_service.advance_round(
                game_id, round_number, request
            )
        )

    @application.get(
        "/games/{game_id}/events",
        response_model=EventsResponse,
        responses={404: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def get_events(
        game_id: UUID, after_sequence: int = Query(default=0, ge=0)
    ) -> JSONResponse:
        return _json(
            application.state.gameplay_service.get_events(game_id, after_sequence)
        )

    return application


app = create_app()
