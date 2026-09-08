"""Use a verified standalone policy for information-set UCT continuations.

The outer algorithm still samples hidden worlds, advances the exact engine,
and backs up exact round scores. Only its simulated continuation choices come
from the neural policy: the acting player supplies a 659-bit visible
observation to one masked policy inference.

Artifact verification occurs once at construction. Ordinary selections reuse
the frozen CPU model and never receive the sampled engine state.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dracula.policy.artifact import LoadedPolicyArtifact, load_policy_artifact
from dracula.policy.observation import select_policy_group
from dracula.search.information_set_uct import (
    InformationSetUCTConfig,
    InformationSetUCTSearch,
)
from dracula.search.contracts import (
    ContinuationDecision,
    SearchInterrupted,
)
from dracula.decision.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.decision.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
)


class PolicyContinuation:
    """Choose one simulated response with a verified 659-bit policy artifact."""

    def __init__(self, artifact: LoadedPolicyArtifact) -> None:
        # Loading has already verified the artifact and tensor contract. Search
        # keeps one inference-only CPU model for repeated response calls.
        self._artifact_digest = artifact.artifact_digest
        self._model = artifact.model.cpu().eval().requires_grad_(False)

    @classmethod
    def from_artifact(cls, path: str | Path) -> PolicyContinuation:
        """Load, verify, and freeze one policy artifact for continuation use."""

        return cls(load_policy_artifact(path))

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> ContinuationDecision:
        """Choose one legal group from the simulated actor's visible state."""

        if should_stop is not None and should_stop():
            raise SearchInterrupted("policy continuation interrupted")

        # The shared inference boundary encodes the visible state, masks illegal
        # and non-representative actions, and returns one strategic group.
        selected_group = select_policy_group(self._model, information)
        return ContinuationDecision(
            selected_group=selected_group,
            selected_action_index=self._concrete_action(
                information,
                selected_group,
            ),
            group_statistics=(),
            terminal_evaluation_count=0,
            model_inference_count=1,
        )

    def _concrete_action(
        self,
        information: SearchInformationState,
        selected_group: StrategicActionGroup,
    ) -> int:
        """Resolve a selected mirrored group without changing model choice."""

        # Visible-state identity and the exact artifact bytes make the coin
        # reproducible for retries while keeping it independent of hidden cards.
        return select_concrete_action_index(
            selected_group,
            information_state_fingerprint(information),
            self._artifact_digest,
            selected_group.representative_action_index,
        )


class PolicyContinuationInformationSetSearch(InformationSetUCTSearch):
    """Run information-set UCT with policy-selected continuation moves."""

    def __init__(
        self,
        continuation: PolicyContinuation,
        search_config: InformationSetUCTConfig = InformationSetUCTConfig(),
    ) -> None:
        super().__init__(continuation, search_config)

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        search_config: InformationSetUCTConfig = InformationSetUCTConfig(),
    ) -> PolicyContinuationInformationSetSearch:
        """Construct UCT search from one strictly verified policy artifact."""

        return cls(PolicyContinuation.from_artifact(path), search_config)


__all__ = (
    "PolicyContinuation",
    "PolicyContinuationInformationSetSearch",
)
