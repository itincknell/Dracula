"""Assemble explicitly selected local or stateless FastAPI applications.

Local mode wires the SQLite session service; stateless mode wires replay-based
production services. Configuration never silently switches between them.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from dracula.api.bedrock import BedrockRuntimeAdapter
from dracula.api.local_controllers import resolve_local_controller
from dracula.api.local_routes import configure_local_routes
from dracula.api.narration import NarrationAdapter
from dracula.api.policy import PolicyDescriptor, PolicyExecutor
from dracula.api.production_config import (
    DEFAULT_REPLAY_CACHE_ENTRIES,
    NARRATION_ENABLED_ENV,
    REPLAY_CACHE_ENTRIES_ENV,
    boolean_environment,
    nonnegative_integer_environment,
)
from dracula.api.repository import (
    GameRepository,
    InMemoryGameRepository,
    SQLiteGameRepository,
)
from dracula.api.service import GameplayService
from dracula.api.stateless_app import create_stateless_app

LOCAL_GAME_SEED_ENV = "DRACULA_LOCAL_GAME_SEED"
GAMEPLAY_MODE_ENV = "DRACULA_GAMEPLAY_MODE"
DEFAULT_GAMEPLAY_MODE = "local"


def _default_repository() -> GameRepository:
    database_path = os.getenv("DRACULA_DATABASE_PATH")
    if database_path is None:
        return InMemoryGameRepository()
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    return SQLiteGameRepository(database_path)


def _gameplay_mode(value: str | None) -> str:
    resolved = DEFAULT_GAMEPLAY_MODE if value is None else value.strip().lower()
    if resolved not in {"local", "stateless"}:
        raise ValueError(f"{GAMEPLAY_MODE_ENV} must be local or stateless")
    return resolved


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
        boolean_environment(NARRATION_ENABLED_ENV, False)
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
            nonnegative_integer_environment(
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

    application = FastAPI(title="Dracula API", version="1.0")
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

    configure_local_routes(
        application,
        gameplay=application.state.gameplay_service,
        narration_enabled=resolved_narration,
    )

    return application


app = create_app()
