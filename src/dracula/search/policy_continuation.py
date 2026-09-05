"""Use a verified standalone policy for simulated BGC continuation moves.

This is the phase-two alternative to belief-greedy continuation. The outer BGC
algorithm still samples hidden worlds, performs 128 UCT simulations, advances
the exact engine, and backs up exact round scores. Only its simulated response
choice changes: the acting player supplies a 659-bit visible observation to one
masked policy inference.

Artifact verification occurs once at construction. Ordinary selections reuse
the frozen CPU model and never receive the sampled engine state.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from dracula.bgc_policy import LoadedPolicyArtifact, load_policy_artifact
from dracula.policy_observation import select_policy_group
from dracula.search.bgc import BGCInformationSetSearch, BGCSearchConfig
from dracula.search.contracts import (
    ContinuationDecision,
    SearchInterrupted,
)
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.strategic_actions import (
    select_concrete_action_index,
)


class PolicyContinuation:
    """Choose one simulated response with a verified 659-bit policy artifact."""

    def __init__(self, artifact: LoadedPolicyArtifact) -> None:
        # Loading has already verified the artifact and tensor contract. Search
        # keeps one inference-only CPU model for repeated response calls.
        self.artifact = artifact
        self.model = artifact.model.cpu().eval().requires_grad_(False)

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
        selected_group = select_policy_group(self.model, information)
        fingerprint = information_state_fingerprint(information)
        return ContinuationDecision(
            selected_group=selected_group,
            selected_action_index=select_concrete_action_index(
                selected_group,
                fingerprint,
                self.artifact.artifact_digest,
                selected_group.representative_action_index,
            ),
            group_statistics=(),
            terminal_evaluation_count=0,
            model_inference_count=1,
        )


class PolicyContinuationInformationSetSearch(BGCInformationSetSearch):
    """BGC-128 whose simulated responses use one policy inference."""

    def __init__(
        self,
        continuation: PolicyContinuation,
        search_config: BGCSearchConfig = BGCSearchConfig(),
    ) -> None:
        super().__init__(continuation, search_config)

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        search_config: BGCSearchConfig = BGCSearchConfig(),
    ) -> PolicyContinuationInformationSetSearch:
        """Construct BGC search from one strictly verified policy artifact."""

        return cls(PolicyContinuation.from_artifact(path), search_config)


__all__ = (
    "PolicyContinuation",
    "PolicyContinuationInformationSetSearch",
)
