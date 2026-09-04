"""Assembly of the production-neutral stateless FastAPI application."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from dracula.api.contracts import API_VERSION
from dracula.api.narration import NarrationAdapter
from dracula.api.policy import PolicyDescriptor, PolicyExecutor
from dracula.api.stateless_routes import configure_stateless_routes


def create_stateless_app(
    *,
    policy_executor: PolicyExecutor | None,
    policy_descriptor: PolicyDescriptor | None,
    narration_enabled: bool,
    narration_adapter: NarrationAdapter | None,
    replay_cache_entries: int,
    game_seed_factory: Callable[[], str] | None = None,
) -> FastAPI:
    """Create an API whose state is reconstructed solely from each request."""

    application = FastAPI(title="Dracula API", version=API_VERSION)
    application.state.narration_enabled = narration_enabled
    application.state.gameplay_mode = "stateless"
    configure_stateless_routes(
        application,
        policy_executor=policy_executor,
        policy_descriptor=policy_descriptor,
        narration_enabled=narration_enabled,
        narration_adapter=narration_adapter,
        replay_cache_entries=replay_cache_entries,
        game_seed_factory=game_seed_factory,
    )
    return application


__all__ = ("create_stateless_app",)
