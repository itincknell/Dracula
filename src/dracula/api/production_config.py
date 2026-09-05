"""Parse environment configuration for the stateless production runtime.

This module owns policy-artifact, CORS, replay-cache, and narration settings.
It supplies no account-specific credentials and selects no silent fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

POLICY_ARTIFACT_ENV = "DRACULA_POLICY_ARTIFACT"
NARRATION_ENABLED_ENV = "DRACULA_NARRATION_ENABLED"
REPLAY_CACHE_ENTRIES_ENV = "DRACULA_REPLAY_CACHE_ENTRIES"
DEFAULT_REPLAY_CACHE_ENTRIES = 256


def boolean_environment(name: str, default: bool) -> bool:
    """Read one conventional boolean environment value."""

    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def nonnegative_integer_environment(name: str, default: int) -> int:
    """Read a non-negative integer environment value."""

    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProductionSettings:
    """Validated environment settings used by the Lambda composition root."""

    policy_artifact: str | None
    narration_enabled: bool
    replay_cache_entries: int

    @classmethod
    def from_environment(cls) -> ProductionSettings:
        """Read production-only policy, narration, and replay-cache settings."""

        raw_artifact = os.getenv(POLICY_ARTIFACT_ENV)
        artifact = None if raw_artifact is None else raw_artifact.strip()
        if raw_artifact is not None and not artifact:
            raise ValueError(f"{POLICY_ARTIFACT_ENV} must be a nonempty path")
        return cls(
            policy_artifact=artifact,
            narration_enabled=boolean_environment(
                NARRATION_ENABLED_ENV, False
            ),
            replay_cache_entries=nonnegative_integer_environment(
                REPLAY_CACHE_ENTRIES_ENV, DEFAULT_REPLAY_CACHE_ENTRIES
            ),
        )


__all__ = (
    "DEFAULT_REPLAY_CACHE_ENTRIES",
    "NARRATION_ENABLED_ENV",
    "POLICY_ARTIFACT_ENV",
    "ProductionSettings",
    "REPLAY_CACHE_ENTRIES_ENV",
    "boolean_environment",
    "nonnegative_integer_environment",
)
