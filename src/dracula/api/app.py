"""FastAPI entry point for local and deployed browser gameplay."""

from __future__ import annotations

import math
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
LOCAL_OPPONENT_MODE_ENV = "DRACULA_OPPONENT_MODE"
SEARCH_SIMULATIONS_ENV = "DRACULA_SEARCH_SIMULATIONS"
SEARCH_EXPLORATION_ENV = "DRACULA_SEARCH_EXPLORATION"
SEARCH_RESPONSE_COMPLETIONS_ENV = "DRACULA_SEARCH_RESPONSE_COMPLETIONS"
SEARCH_RESPONSE_SIMULATIONS_ENV = "DRACULA_SEARCH_RESPONSE_SIMULATIONS"
BELIEF_COMPLETIONS_ENV = "DRACULA_BELIEF_COMPLETIONS"
RESPONSE_RANKER_ARTIFACT_ENV = "DRACULA_RESPONSE_RANKER_ARTIFACT"
GUIDED_ARTIFACT_ENV = "DRACULA_POLICY_VALUE_ARTIFACT"
GUIDED_SIMULATIONS_ENV = "DRACULA_GUIDED_SIMULATIONS"
SAM_POLICY_ARTIFACT_ENV = "DRACULA_SAM_POLICY_ARTIFACT"
BGC_PI0_ARTIFACT_ENV = "DRACULA_BGC_PI0_ARTIFACT"
DEFAULT_SEARCH_SIMULATIONS = 500
DEFAULT_SEARCH_EXPLORATION = math.sqrt(2.0)
DEFAULT_SEARCH_RESPONSE_COMPLETIONS = 1
DEFAULT_SEARCH_RESPONSE_SIMULATIONS = 32
DEFAULT_BELIEF_COMPLETIONS = 8
DEFAULT_GUIDED_SIMULATIONS = 100


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


def _positive_integer_environment(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_float_environment(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative finite number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return value


def _search_response_completions_environment() -> int:
    value = _positive_integer_environment(
        SEARCH_RESPONSE_COMPLETIONS_ENV,
        DEFAULT_SEARCH_RESPONSE_COMPLETIONS,
    )
    if value not in {1, 2, 4}:
        raise ValueError(f"{SEARCH_RESPONSE_COMPLETIONS_ENV} must be 1, 2, or 4")
    return value


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
    opponent_mode = os.getenv(LOCAL_OPPONENT_MODE_ENV)
    if opponent_mode is not None:
        opponent_mode = opponent_mode.strip().lower()
        if opponent_mode not in {
            "search",
            "search-v2",
            "search-v2-nested",
            "search-belief-greedy",
            "search-belief-greedy-pi0",
            "bgc-policy",
            "search-v2-student-direct",
            "search-v2-student-top-2",
            "search-v2-student-top-3",
            "guided",
            "archive",
            "sam-policy",
        }:
            raise ValueError(
                f"{LOCAL_OPPONENT_MODE_ENV} must select a documented opponent mode"
            )
    elif archive_path is not None:
        opponent_mode = "archive"
    local_game_seed = os.getenv(LOCAL_GAME_SEED_ENV)
    if local_game_seed is not None and not local_game_seed:
        raise ValueError(f"{LOCAL_GAME_SEED_ENV} must be a nonempty value")
    if resolved_executor is None:
        if opponent_mode == "search":
            from dracula.search_policy import InlineSearchExecutor

            resolved_executor = InlineSearchExecutor.from_values(
                _positive_integer_environment(
                    SEARCH_SIMULATIONS_ENV, DEFAULT_SEARCH_SIMULATIONS
                ),
                _nonnegative_float_environment(
                    SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
                ),
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode in {
            "search-v2",
            "search-v2-student-direct",
            "search-v2-student-top-2",
            "search-v2-student-top-3",
        }:
            from dracula.search_policy import InlineStrategicSearchExecutor
            from dracula.search import StrategicResponseMode

            response_mode = {
                "search-v2": StrategicResponseMode.PURE,
                "search-v2-student-direct": (
                    StrategicResponseMode.STUDENT_DIRECT
                ),
                "search-v2-student-top-2": (
                    StrategicResponseMode.STUDENT_TOP_2
                ),
                "search-v2-student-top-3": (
                    StrategicResponseMode.STUDENT_TOP_3
                ),
            }[opponent_mode]
            response_ranker_artifact = os.getenv(
                RESPONSE_RANKER_ARTIFACT_ENV
            )
            if (
                response_mode is not StrategicResponseMode.PURE
                and (
                    response_ranker_artifact is None
                    or not response_ranker_artifact.strip()
                )
            ):
                raise ValueError(
                    f"{RESPONSE_RANKER_ARTIFACT_ENV} must be a nonempty path"
                )

            resolved_executor = InlineStrategicSearchExecutor.from_values(
                _positive_integer_environment(
                    SEARCH_SIMULATIONS_ENV, DEFAULT_SEARCH_SIMULATIONS
                ),
                _search_response_completions_environment(),
                _nonnegative_float_environment(
                    SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
                ),
                response_mode=response_mode,
                response_ranker_artifact=response_ranker_artifact,
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "search-v2-nested":
            from dracula.search_policy import (
                InlineNestedStrategicSearchExecutor,
            )

            resolved_executor = (
                InlineNestedStrategicSearchExecutor.from_values(
                    _positive_integer_environment(
                        SEARCH_SIMULATIONS_ENV,
                        32,
                    ),
                    _positive_integer_environment(
                        SEARCH_RESPONSE_SIMULATIONS_ENV,
                        DEFAULT_SEARCH_RESPONSE_SIMULATIONS,
                    ),
                    _nonnegative_float_environment(
                        SEARCH_EXPLORATION_ENV,
                        DEFAULT_SEARCH_EXPLORATION,
                    ),
                    _nonnegative_float_environment(
                        SEARCH_EXPLORATION_ENV,
                        DEFAULT_SEARCH_EXPLORATION,
                    ),
                )
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "search-belief-greedy":
            from dracula.search_policy import (
                InlineBeliefGreedySearchExecutor,
            )

            resolved_executor = InlineBeliefGreedySearchExecutor.from_values(
                _positive_integer_environment(
                    SEARCH_SIMULATIONS_ENV,
                    32,
                ),
                _positive_integer_environment(
                    BELIEF_COMPLETIONS_ENV,
                    DEFAULT_BELIEF_COMPLETIONS,
                ),
                _nonnegative_float_environment(
                    SEARCH_EXPLORATION_ENV,
                    DEFAULT_SEARCH_EXPLORATION,
                ),
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "search-belief-greedy-pi0":
            artifact_path = os.getenv(BGC_PI0_ARTIFACT_ENV)
            if artifact_path is None or not artifact_path.strip():
                raise ValueError(
                    f"{BGC_PI0_ARTIFACT_ENV} must be a nonempty path"
                )
            from dracula.search_policy import (
                InlinePi0BeliefGreedySearchExecutor,
            )

            resolved_executor = InlinePi0BeliefGreedySearchExecutor.from_values(
                artifact_path,
                _positive_integer_environment(
                    SEARCH_SIMULATIONS_ENV,
                    128,
                ),
                _positive_integer_environment(
                    BELIEF_COMPLETIONS_ENV,
                    DEFAULT_BELIEF_COMPLETIONS,
                ),
                _nonnegative_float_environment(
                    SEARCH_EXPLORATION_ENV,
                    DEFAULT_SEARCH_EXPLORATION,
                ),
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "bgc-policy":
            artifact_path = os.getenv(BGC_PI0_ARTIFACT_ENV)
            if artifact_path is None or not artifact_path.strip():
                raise ValueError(
                    f"{BGC_PI0_ARTIFACT_ENV} must be a nonempty path"
                )
            from dracula.search_policy import InlineBGCPolicyExecutor

            resolved_executor = InlineBGCPolicyExecutor(artifact_path)
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "guided":
            artifact_path = os.getenv(GUIDED_ARTIFACT_ENV)
            if artifact_path is None or not artifact_path.strip():
                raise ValueError(f"{GUIDED_ARTIFACT_ENV} must be a nonempty path")
            from dracula.guided_policy import InlineGuidedSearchExecutor
            from dracula.search import GuidedSearchConfig

            resolved_executor = InlineGuidedSearchExecutor(
                artifact_path,
                GuidedSearchConfig(
                    simulation_budget=_positive_integer_environment(
                        GUIDED_SIMULATIONS_ENV, DEFAULT_GUIDED_SIMULATIONS
                    )
                ),
            )
            resolved_descriptor = resolved_executor.descriptor
        elif opponent_mode == "archive":
            if archive_path is None or not archive_path.strip():
                raise ValueError(
                    f"{LOCAL_POLICY_ARCHIVE_ENV} must be a nonempty path"
                )
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
        elif opponent_mode == "sam-policy":
            artifact_path = os.getenv(SAM_POLICY_ARTIFACT_ENV)
            if artifact_path is None or not artifact_path.strip():
                raise ValueError(
                    f"{SAM_POLICY_ARTIFACT_ENV} must be a nonempty path"
                )
            # PyTorch remains outside the ordinary API import path unless the
            # standalone classifier is selected explicitly.
            from dracula.standalone_policy import (
                StandaloneSamPolicyExecutor,
            )

            resolved_executor = StandaloneSamPolicyExecutor(artifact_path)
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
