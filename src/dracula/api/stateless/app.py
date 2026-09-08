"""Construct the FastAPI object used for stateless gameplay.

This is the composition boundary: callers provide the policy, narration
adapter, cache size, and optional seed factory, and this module connects them
to the route layer. It does not load model files, read production environment
variables, or choose fallback dependencies.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from dracula.api.narration.service import NarrationAdapter
from dracula.policy.contracts import PolicyExecutor
from dracula.api.stateless.routes import configure_stateless_routes


def create_stateless_app(
    *,
    policy_executor: PolicyExecutor | None,
    narration_adapter: NarrationAdapter | None,
    replay_cache_entries: int,
    game_seed_factory: Callable[[], str] | None = None,
) -> FastAPI:
    """Create an API whose game state is reconstructed from every request.

    The ``*`` in the signature makes all following arguments keyword-only,
    preventing similar Boolean and optional values from being passed in the
    wrong position.
    """

    # FastAPI owns routing, request parsing, response serialization, and the
    # generated OpenAPI documentation.
    application = FastAPI(title="Dracula API")
    configure_stateless_routes(
        application,
        policy_executor=policy_executor,
        narration_adapter=narration_adapter,
        replay_cache_entries=replay_cache_entries,
        game_seed_factory=game_seed_factory,
    )
    return application
