"""Read process configuration from environment variables.

Lambda and local Uvicorn commands both configure the API through environment
variables. This module converts their string values into validated Python
values before application construction begins. It supplies no credentials and
does not silently choose a policy or narration provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dracula.api.stateless.replay import DEFAULT_REPLAY_CACHE_ENTRIES

POLICY_ARTIFACT_ENV = "DRACULA_POLICY_ARTIFACT"
NARRATION_ENABLED_ENV = "DRACULA_NARRATION_ENABLED"
REPLAY_CACHE_ENTRIES_ENV = "DRACULA_REPLAY_CACHE_ENTRIES"
LOCAL_GAME_SEED_ENV = "DRACULA_LOCAL_GAME_SEED"


def boolean_environment(name: str, default: bool) -> bool:
    """Read conventional deployment Boolean spellings from one variable.

    AWS templates and shell commands do not preserve a Boolean type: every
    environment value arrives as text. Accepting the common spellings here
    keeps that parsing at startup rather than spreading it through composition
    code. Unsupported values fail application construction.
    """

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
    """Parse a nonnegative startup value, including zero to disable the cache."""

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
    """Validated settings passed into the Lambda application constructor.

    A frozen dataclass gives startup code ordinary typed values after the
    environment-string boundary has been checked once.
    """

    policy_artifact: str | None
    """Container or workstation path to the selected policy artifact."""

    narration_enabled: bool
    """Whether application composition creates a narration adapter."""

    replay_cache_entries: int
    """Maximum reconstructed responses retained by each process."""

    @classmethod
    def from_environment(cls) -> ProductionSettings:
        """Read production-only policy, narration, and replay-cache settings."""

        # An omitted path means no policy was configured. A present but empty
        # path is rejected because it is usually a deployment mistake.
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
