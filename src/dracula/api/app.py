"""FastAPI entry point for local and deployed browser gameplay."""

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
from dracula.api.repository import (
    GameRepository,
    InMemoryGameRepository,
    SQLiteGameRepository,
)
from dracula.api.service import (
    GameplayService,
    PolicyDescriptor,
    PolicyExecutor,
    ServiceResponse,
)

LOCAL_POLICY_ARCHIVE_ENV = "DRACULA_POLICY_ARCHIVE"
LOCAL_INFERENCE_PROFILE_ENV = "DRACULA_POLICY_INFERENCE_PROFILE"
DEFAULT_LOCAL_INFERENCE_PROFILE = "argmax-v1"
LOCAL_GAME_SEED_ENV = "DRACULA_LOCAL_GAME_SEED"


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


def _json(response: ServiceResponse) -> JSONResponse:
    return JSONResponse(status_code=response.status_code, content=response.body)


def create_app(
    *,
    narration_enabled: bool | None = None,
    repository: GameRepository | None = None,
    policy_executor: PolicyExecutor | None = None,
    policy_descriptor: PolicyDescriptor | None = None,
) -> FastAPI:
    resolved_narration = (
        _environment_flag("DRACULA_NARRATION_ENABLED", default=False)
        if narration_enabled is None
        else narration_enabled
    )
    resolved_executor = policy_executor
    resolved_descriptor = policy_descriptor
    archive_path = os.getenv(LOCAL_POLICY_ARCHIVE_ENV)
    local_game_seed = os.getenv(LOCAL_GAME_SEED_ENV)
    if local_game_seed is not None and not local_game_seed:
        raise ValueError(f"{LOCAL_GAME_SEED_ENV} must be a nonempty value")
    if resolved_executor is None and archive_path is not None:
        if not archive_path.strip():
            raise ValueError(f"{LOCAL_POLICY_ARCHIVE_ENV} must be a nonempty path")
        # PyTorch remains outside the ordinary API import path unless a local
        # candidate is explicitly selected.
        from dracula.local_policy import InlinePolicyExecutor

        resolved_executor = InlinePolicyExecutor.from_archive(
            archive_path,
            os.getenv(
                LOCAL_INFERENCE_PROFILE_ENV, DEFAULT_LOCAL_INFERENCE_PROFILE
            ),
        )
        resolved_descriptor = resolved_executor.descriptor

    application = FastAPI(title="Dracula API", version=API_VERSION)
    application.state.narration_enabled = resolved_narration
    application.state.game_repository = repository or _default_repository()
    application.state.gameplay_service = GameplayService(
        application.state.game_repository,
        policy_executor=resolved_executor,
        policy_descriptor=resolved_descriptor,
        narration_enabled=resolved_narration,
        game_seed_factory=(
            None if local_game_seed is None else lambda: local_game_seed
        ),
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
        return _json(
            application.state.gameplay_service.opponent_turn(game_id, request)
        )

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
