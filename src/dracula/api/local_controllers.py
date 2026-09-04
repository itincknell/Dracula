"""Explicit environment-driven controller selection for local gameplay."""

from __future__ import annotations

import math
import os

from dracula.api.policy import PolicyDescriptor, PolicyExecutor
from dracula.api.production_config import POLICY_ARTIFACT_ENV

LOCAL_POLICY_ARCHIVE_ENV = "DRACULA_POLICY_ARCHIVE"
LOCAL_INFERENCE_PROFILE_ENV = "DRACULA_POLICY_INFERENCE_PROFILE"
DEFAULT_LOCAL_INFERENCE_PROFILE = "argmax-v1"
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

# Historical and diagnostic controllers remain available only by explicit
# local selection; production construction never consults this registry.
_OPPONENT_MODES = frozenset(
    {
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
    }
)


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


def _response_completions() -> int:
    value = _positive_integer_environment(
        SEARCH_RESPONSE_COMPLETIONS_ENV, DEFAULT_SEARCH_RESPONSE_COMPLETIONS
    )
    if value not in {1, 2, 4}:
        raise ValueError(f"{SEARCH_RESPONSE_COMPLETIONS_ENV} must be 1, 2, or 4")
    return value


def _required_path(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be a nonempty path")
    return value


def _opponent_mode() -> tuple[str | None, str | None]:
    archive_path = os.getenv(LOCAL_POLICY_ARCHIVE_ENV)
    raw = os.getenv(LOCAL_OPPONENT_MODE_ENV)
    if raw is None:
        return ("archive" if archive_path is not None else None), archive_path
    mode = raw.strip().lower()
    if mode not in _OPPONENT_MODES:
        raise ValueError(
            f"{LOCAL_OPPONENT_MODE_ENV} must select a documented opponent mode"
        )
    return mode, archive_path


def resolve_local_controller(
    executor: PolicyExecutor | None,
    descriptor: PolicyDescriptor | None,
) -> tuple[PolicyExecutor | None, PolicyDescriptor | None]:
    """Resolve one explicitly selected local controller without a fallback."""

    if executor is not None:
        return executor, descriptor
    mode, archive_path = _opponent_mode()
    if mode is None:
        return None, descriptor

    if mode == "search":
        from dracula.search_policy import InlineSearchExecutor

        resolved = InlineSearchExecutor.from_values(
            _positive_integer_environment(
                SEARCH_SIMULATIONS_ENV, DEFAULT_SEARCH_SIMULATIONS
            ),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
        )
    elif mode in {
        "search-v2",
        "search-v2-student-direct",
        "search-v2-student-top-2",
        "search-v2-student-top-3",
    }:
        from dracula.search.strategic import StrategicResponseMode
        from dracula.search_policy import InlineStrategicSearchExecutor

        response_mode = {
            "search-v2": StrategicResponseMode.PURE,
            "search-v2-student-direct": StrategicResponseMode.STUDENT_DIRECT,
            "search-v2-student-top-2": StrategicResponseMode.STUDENT_TOP_2,
            "search-v2-student-top-3": StrategicResponseMode.STUDENT_TOP_3,
        }[mode]
        response_ranker_artifact = os.getenv(RESPONSE_RANKER_ARTIFACT_ENV)
        if response_mode is not StrategicResponseMode.PURE and (
            response_ranker_artifact is None or not response_ranker_artifact.strip()
        ):
            raise ValueError(
                f"{RESPONSE_RANKER_ARTIFACT_ENV} must be a nonempty path"
            )
        resolved = InlineStrategicSearchExecutor.from_values(
            _positive_integer_environment(
                SEARCH_SIMULATIONS_ENV, DEFAULT_SEARCH_SIMULATIONS
            ),
            _response_completions(),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
            response_mode=response_mode,
            response_ranker_artifact=response_ranker_artifact,
        )
    elif mode == "search-v2-nested":
        from dracula.search_policy import InlineNestedStrategicSearchExecutor

        resolved = InlineNestedStrategicSearchExecutor.from_values(
            _positive_integer_environment(SEARCH_SIMULATIONS_ENV, 32),
            _positive_integer_environment(
                SEARCH_RESPONSE_SIMULATIONS_ENV,
                DEFAULT_SEARCH_RESPONSE_SIMULATIONS,
            ),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
        )
    elif mode == "search-belief-greedy":
        from dracula.search_policy import InlineBeliefGreedySearchExecutor

        resolved = InlineBeliefGreedySearchExecutor.from_values(
            _positive_integer_environment(SEARCH_SIMULATIONS_ENV, 32),
            _positive_integer_environment(
                BELIEF_COMPLETIONS_ENV, DEFAULT_BELIEF_COMPLETIONS
            ),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
        )
    elif mode == "search-belief-greedy-pi0":
        from dracula.search_policy import InlinePi0BeliefGreedySearchExecutor

        resolved = InlinePi0BeliefGreedySearchExecutor.from_values(
            _required_path(BGC_PI0_ARTIFACT_ENV),
            _positive_integer_environment(SEARCH_SIMULATIONS_ENV, 128),
            _positive_integer_environment(
                BELIEF_COMPLETIONS_ENV, DEFAULT_BELIEF_COMPLETIONS
            ),
            _nonnegative_float_environment(
                SEARCH_EXPLORATION_ENV, DEFAULT_SEARCH_EXPLORATION
            ),
        )
    elif mode == "bgc-policy":
        from dracula.active_policy import ActivePolicyExecutor

        resolved = ActivePolicyExecutor(_required_path(POLICY_ARTIFACT_ENV))
    elif mode == "guided":
        from dracula.guided_policy import InlineGuidedSearchExecutor
        from dracula.search.guided import GuidedSearchConfig

        resolved = InlineGuidedSearchExecutor(
            _required_path(GUIDED_ARTIFACT_ENV),
            GuidedSearchConfig(
                simulation_budget=_positive_integer_environment(
                    GUIDED_SIMULATIONS_ENV, DEFAULT_GUIDED_SIMULATIONS
                )
            ),
        )
    elif mode == "archive":
        from dracula.local_policy import InlinePolicyExecutor

        if archive_path is None or not archive_path.strip():
            raise ValueError(f"{LOCAL_POLICY_ARCHIVE_ENV} must be a nonempty path")
        resolved = InlinePolicyExecutor.from_archive(
            archive_path,
            os.getenv(LOCAL_INFERENCE_PROFILE_ENV, DEFAULT_LOCAL_INFERENCE_PROFILE),
        )
    else:
        from dracula.standalone_policy import StandaloneSamPolicyExecutor

        resolved = StandaloneSamPolicyExecutor(_required_path(SAM_POLICY_ARTIFACT_ENV))
    return resolved, resolved.descriptor
