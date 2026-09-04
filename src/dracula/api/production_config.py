"""Central configuration for the active stateless deployment runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass

POLICY_ARTIFACT_ENV = "DRACULA_POLICY_ARTIFACT"
NARRATION_ENABLED_ENV = "DRACULA_NARRATION_ENABLED"
REPLAY_CACHE_ENTRIES_ENV = "DRACULA_REPLAY_CACHE_ENTRIES"


def _boolean_environment(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _cache_entries() -> int:
    raw = os.getenv(REPLAY_CACHE_ENTRIES_ENV, "256")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{REPLAY_CACHE_ENTRIES_ENV} must be a non-negative integer"
        ) from error
    if value < 0:
        raise ValueError(
            f"{REPLAY_CACHE_ENTRIES_ENV} must be a non-negative integer"
        )
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
            narration_enabled=_boolean_environment(
                NARRATION_ENABLED_ENV, False
            ),
            replay_cache_entries=_cache_entries(),
        )


__all__ = (
    "NARRATION_ENABLED_ENV",
    "POLICY_ARTIFACT_ENV",
    "ProductionSettings",
    "REPLAY_CACHE_ENTRIES_ENV",
)
