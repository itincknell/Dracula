"""Explicit selection of the production policy for local gameplay."""

from __future__ import annotations

import os

from dracula.api.policy import PolicyDescriptor, PolicyExecutor
from dracula.api.production_config import POLICY_ARTIFACT_ENV

LOCAL_OPPONENT_MODE_ENV = "DRACULA_OPPONENT_MODE"
ACTIVE_OPPONENT_MODE = "bgc-policy"


def resolve_local_controller(
    executor: PolicyExecutor | None,
    descriptor: PolicyDescriptor | None,
) -> tuple[PolicyExecutor | None, PolicyDescriptor | None]:
    """Load pi1 only when the local environment explicitly selects it."""

    if executor is not None:
        return executor, descriptor
    raw_mode = os.getenv(LOCAL_OPPONENT_MODE_ENV)
    if raw_mode is None:
        return None, descriptor
    if raw_mode.strip().lower() != ACTIVE_OPPONENT_MODE:
        raise ValueError(
            f"{LOCAL_OPPONENT_MODE_ENV} must be {ACTIVE_OPPONENT_MODE}"
        )
    artifact = os.getenv(POLICY_ARTIFACT_ENV)
    if artifact is None or not artifact.strip():
        raise ValueError(f"{POLICY_ARTIFACT_ENV} must be a nonempty path")
    from dracula.active_policy import ActivePolicyExecutor

    resolved = ActivePolicyExecutor(artifact)
    return resolved, resolved.descriptor


__all__ = (
    "ACTIVE_OPPONENT_MODE",
    "LOCAL_OPPONENT_MODE_ENV",
    "resolve_local_controller",
)
