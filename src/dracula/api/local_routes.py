"""Expose the retained SQLite gameplay service through local FastAPI routes.

These handlers translate validated request models to local transaction-service
calls and return its public responses. Production stateless routes are defined
separately in :mod:`dracula.api.stateless_routes`.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, Path as ApiPath, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dracula.api.contracts import (
    ApiErrorResponse,
    CreateGameRequest,
    EventsResponse,
    HealthResponse,
    HumanGameView,
    MoveRequest,
    VersionedMutationRequest,
)
from dracula.api.policy import ServiceResponse
from dracula.api.service import GameplayService


def _json(response: ServiceResponse) -> JSONResponse:
    return JSONResponse(status_code=response.status_code, content=response.body)


def configure_local_routes(
    application: FastAPI,
    *,
    gameplay: GameplayService,
    narration_enabled: bool,
) -> None:
    """Attach the local stateful HTTP surface to one configured service."""

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
        return HealthResponse(narration_enabled=narration_enabled)

    @application.post(
        "/games",
        response_model=HumanGameView,
        status_code=201,
        responses={409: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def start_game(request: CreateGameRequest) -> JSONResponse:
        return _json(gameplay.create_game(request))

    @application.get(
        "/games/{game_id}",
        response_model=HumanGameView,
        responses={404: {"model": ApiErrorResponse}},
    )
    def get_game(game_id: UUID) -> JSONResponse:
        return _json(gameplay.get_game(game_id))

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
        return _json(gameplay.human_move(game_id, request))

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
        return _json(gameplay.opponent_turn(game_id, request))

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
        return _json(gameplay.advance_round(game_id, round_number, request))

    @application.get(
        "/games/{game_id}/events",
        response_model=EventsResponse,
        responses={404: {"model": ApiErrorResponse}, 422: {"model": ApiErrorResponse}},
    )
    def get_events(
        game_id: UUID, after_sequence: int = Query(default=0, ge=0)
    ) -> JSONResponse:
        return _json(gameplay.get_events(game_id, after_sequence))


__all__ = ("configure_local_routes",)
