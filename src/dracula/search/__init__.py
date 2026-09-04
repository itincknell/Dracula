"""Compatibility facade for information-safe search implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _exports(module: str, names: tuple[str, ...]) -> dict[str, str]:
    return {name: module for name in names}


# Lazy compatibility exports keep a narrow search import from loading every
# retained planner and model implementation.
_EXPORTS = {
    **_exports(
        "dracula.search.information",
        (
            "BELIEF_SAMPLE_NAMESPACE",
            "INFORMATION_STATE_SCHEMA_VERSION",
            "ROLLOUT_CHOICE_NAMESPACE",
            "SEARCH_REQUEST_NAMESPACE",
            "TREE_SELECTION_NAMESPACE",
            "InformationContractViolation",
            "SearchInformationState",
            "PublicGameHistory",
            "PublicPlayedMove",
            "PublicRoundRecord",
            "SampledDeterminization",
            "build_information_state",
            "canonical_information_json",
            "derive_belief_sample_seed",
            "derive_rollout_choice_seed",
            "derive_search_request_seed",
            "derive_tree_selection_seed",
            "information_state_fingerprint",
            "information_state_from_engine",
            "information_state_from_simulation",
            "policy_input_from_information_state",
            "public_history_from_engine",
            "sample_determinization",
            "sample_uniform_action_index",
        ),
    ),
    **_exports(
        "dracula.search.planner",
        (
            "ROUND_SCORE_NORMALIZER",
            "SEARCH_SCHEMA_VERSION",
            "ContinuationStep",
            "InformationSetSearch",
            "PrincipalContinuation",
            "SearchConfig",
            "SearchContractViolation",
            "SearchInterrupted",
            "SearchResult",
            "normalized_round_return",
        ),
    ),
    **_exports(
        "dracula.search.guided",
        (
            "GUIDED_SEARCH_SCHEMA_VERSION",
            "GuidedInformationSetSearch",
            "GuidedLeafValueMode",
            "GuidedPrincipalContinuation",
            "GuidedSearchConfig",
            "GuidedSearchResult",
            "PolicyValueEvaluation",
            "PolicyValueModelEvaluator",
            "SearchModelEvaluator",
            "UniformPolicyValueEvaluator",
            "player_relative_value_for_root",
        ),
    ),
    **_exports(
        "dracula.search.strategic",
        (
            "GREEDY_RESPONSE_CONTINUATION_PROFILE",
            "GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE",
            "GREEDY_RESPONSE_REQUEST_NAMESPACE",
            "GREEDY_RESPONSE_ROLLOUT_NAMESPACE",
            "GREEDY_RESPONSE_SCHEMA_VERSION",
            "GREEDY_RESPONSE_SELECTION_PROFILE",
            "HYBRID_RESPONSE_REQUEST_NAMESPACE",
            "HYBRID_RESPONSE_SCHEMA_VERSION",
            "STRATEGIC_DETERMINIZATION_NAMESPACE",
            "STRATEGIC_DESTINATION_CHOICE_NAMESPACE",
            "STRATEGIC_DESTINATION_CHOICE_PROFILE",
            "STRATEGIC_SEARCH_REQUEST_NAMESPACE",
            "STRATEGIC_SEARCH_SCHEMA_VERSION",
            "STRATEGIC_SELECTION_NAMESPACE",
            "STRATEGIC_SELECTION_PROFILE",
            "SUPPORTED_RESPONSE_COMPLETIONS",
            "GreedyResponseResult",
            "ShallowGreedyResponseEvaluator",
            "ShallowResponseConfig",
            "StrategicActionGroup",
            "StrategicActionGroupDiagnostic",
            "StrategicInformationSetSearch",
            "StrategicGroupRanker",
            "StrategicNodeKey",
            "StrategicResponseMode",
            "StrategicSearchConfig",
            "StrategicSearchResult",
            "actor_relative_value",
            "derive_greedy_response_determinization_seed",
            "derive_greedy_response_request_seed",
            "derive_greedy_response_rollout_seed",
            "derive_hybrid_response_request_seed",
            "derive_strategic_destination_choice_seed",
            "derive_strategic_determinization_seed",
            "derive_strategic_search_request_seed",
            "derive_strategic_selection_seed",
            "select_concrete_action_index",
            "strategic_action_groups",
        ),
    ),
    **_exports(
        "dracula.search.symmetry",
        (
            "DESTINATION_SYMMETRY_SCHEMA_VERSION",
            "DestinationSymmetryError",
            "DestinationSymmetryGroup",
            "destination_symmetry_groups",
        ),
    ),
    "ResponseRankerGroupEvaluator": "dracula.search.response_student",
    **_exports(
        "dracula.search.sam_teacher",
        (
            "SAM_TEACHER_DETERMINIZATION_NAMESPACE",
            "SAM_TEACHER_RECURSION_LIMIT",
            "SAM_TEACHER_REQUEST_NAMESPACE",
            "SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE",
            "SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE",
            "SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE",
            "SAM_TEACHER_RESPONSE_SCHEMA_VERSION",
            "SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE",
            "SAM_TEACHER_SEARCH_SCHEMA_VERSION",
            "SAM_TEACHER_SELECTION_NAMESPACE",
            "SAM_TEACHER_SELECTION_PROFILE",
            "SamTeacherInformationSetSearch",
            "SamTeacherNodeKey",
            "SamTeacherSearchConfig",
            "SamTeacherSearchResult",
            "derive_sam_teacher_destination_seed",
            "derive_sam_teacher_determinization_seed",
            "derive_sam_teacher_request_seed",
            "derive_sam_teacher_response_determinization_seed",
            "derive_sam_teacher_response_request_seed",
            "derive_sam_teacher_response_rollout_seed",
            "derive_sam_teacher_response_selection_seed",
            "derive_sam_teacher_selection_seed",
            "sam_teacher_action_groups",
        ),
    ),
    **_exports(
        "dracula.search.belief_greedy",
        (
            "BELIEF_GREEDY_RESPONSE_SCHEMA_VERSION",
            "BELIEF_GREEDY_SEARCH_SCHEMA_VERSION",
            "BeliefGreedyInformationSetSearch",
            "BeliefGreedyResponseEvaluator",
            "BeliefGreedyResponseResult",
            "BeliefGreedySearchConfig",
            "BeliefGreedySearchResult",
            "derive_belief_greedy_request_seed",
        ),
    ),
}

_EXPORTS.update(
    _exports(
        "dracula.strategic_actions",
        (
            "STRATEGIC_DESTINATION_CHOICE_NAMESPACE",
            "STRATEGIC_DESTINATION_CHOICE_PROFILE",
            "StrategicActionGroup",
            "derive_strategic_destination_choice_seed",
            "select_concrete_action_index",
            "strategic_action_groups",
        ),
    )
)

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
