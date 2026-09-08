"""Construct the stateless FastAPI application used for local development.

Local development now exercises the same seed-and-command replay service as
production. Tests and previews may inject a policy, narration adapter, cache
size, or deterministic game seed; environment settings supply omitted values.
No local database or alternate gameplay transport is involved.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import FastAPI

from dracula.api.config import (
    LOCAL_GAME_SEED_ENV,
    ProductionSettings,
)
from dracula.api.narration.bedrock import BedrockRuntimeAdapter
from dracula.api.narration.service import NarrationAdapter
from dracula.api.stateless.app import create_stateless_app
from dracula.policy.contracts import PolicyExecutor


def _fixed_seed_factory(seed: str | None) -> Callable[[], str] | None:
    """Return a fixed seed source for repeatable local games.

    The stateless service normally creates a fresh random seed. Supplying this
    factory makes browser demonstrations reproducible without changing the
    request or recovery-envelope formats.
    """

    if seed is None:
        return None
    if not seed:
        raise ValueError(f"{LOCAL_GAME_SEED_ENV} must be a nonempty value")
    return lambda: seed


def _configured_policy(
    injected: PolicyExecutor | None,
    artifact_path: str | None,
) -> PolicyExecutor | None:
    """Use an injected policy or load the configured artifact once.

    Tests pass a lightweight executor directly. Ordinary development reads the
    artifact path from the same environment setting used by Lambda.
    """

    if injected is not None or artifact_path is None:
        return injected
    # Keep PyTorch out of policy-free API imports and tests.
    from dracula.policy.runtime import ActivePolicyExecutor

    return ActivePolicyExecutor(artifact_path)


def create_app(
    *,
    narration_enabled: bool | None = None,
    policy_executor: PolicyExecutor | None = None,
    replay_cache_entries: int | None = None,
    narration_adapter: NarrationAdapter | None = None,
) -> FastAPI:
    """Assemble a production-shaped stateless app with explicit local overrides.

    Each optional argument replaces one environment-derived dependency. This
    keeps tests and the dummy-dialogue preview on the real stateless routes
    without introducing another application mode.
    """

    settings = ProductionSettings.from_environment()
    enabled = (
        settings.narration_enabled
        if narration_enabled is None
        else narration_enabled
    )
    adapter = narration_adapter if enabled else None
    # Enabling real narration without an injected test adapter constructs the
    # Bedrock client once at application startup.
    if enabled and adapter is None:
        adapter = BedrockRuntimeAdapter.from_environment()
    return create_stateless_app(
        policy_executor=_configured_policy(policy_executor, settings.policy_artifact),
        narration_adapter=adapter,
        replay_cache_entries=(
            settings.replay_cache_entries
            if replay_cache_entries is None
            else replay_cache_entries
        ),
        game_seed_factory=_fixed_seed_factory(os.getenv(LOCAL_GAME_SEED_ENV)),
    )


# Local Uvicorn commands import this application directly.
app = create_app()
