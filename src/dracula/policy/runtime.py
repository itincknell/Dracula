"""Adapt the selected standalone policy to live gameplay decisions.

The runtime loads one verified artifact, performs actor-visible masked inference,
resolves paired destinations deterministically, and returns one legal action
through the shared gameplay-policy boundary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dracula.policy.contracts import (
    PolicyContractError,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.policy.artifact import load_policy_artifact
from dracula.policy.model import PolicyModel
from dracula.policy.observation import select_policy_group
from dracula.decision.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.decision.strategic_actions import (
    select_concrete_action_index,
)

LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class ActivePolicyDecision:
    """Selected proxy, resolved legal action, and complete decision latency."""

    representative_action_index: int
    concrete_action_index: int
    latency_seconds: float


class ActivePolicyRuntime:
    """Load and run one production policy without evaluation-code dependencies."""

    def __init__(
        self,
        model: PolicyModel,
        *,
        artifact_digest: str,
    ) -> None:
        self.model = model.cpu().eval().requires_grad_(False)
        self.artifact_digest = artifact_digest

    @classmethod
    def from_artifact(cls, path: str | Path) -> ActivePolicyRuntime:
        """Load the selected artifact through the strict shared verifier."""

        artifact_path = Path(path).expanduser().resolve()
        loaded = load_policy_artifact(artifact_path)
        return cls(
            loaded.model,
            artifact_digest=loaded.artifact_digest,
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
    ) -> ActivePolicyDecision:
        """Run one actor-visible inference and resolve its paired destination."""

        started = time.perf_counter()
        selected_group = select_policy_group(self.model, information)
        representative = selected_group.representative_action_index
        # The final coin depends only on this accepted decision's stable facts.
        concrete = select_concrete_action_index(
            selected_group,
            fixture_id,
            information_state_fingerprint(information),
            self.artifact_digest,
            decision_index,
            representative,
        )
        return ActivePolicyDecision(
            representative_action_index=representative,
            concrete_action_index=concrete,
            latency_seconds=time.perf_counter() - started,
        )


class ActivePolicyExecutor:
    """Adapt the selected standalone policy to the gameplay policy boundary."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.policy = ActivePolicyRuntime.from_artifact(artifact_path)

    @property
    def artifact_digest(self) -> str:
        return self.policy.artifact_digest

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        """Validate one gameplay request and return its concrete policy action."""

        information = self._information_for_request(request)
        decision = self.policy.decide(
            information,
            fixture_id=request.game_key,
            decision_index=request.turn_number,
        )
        self._confirm_service_legality(request, decision.concrete_action_index)
        self._log_decision(request, decision)
        return PolicyTurnResult(decision.concrete_action_index)

    def _information_for_request(
        self,
        request: PolicyTurnRequest,
    ) -> SearchInformationState:
        """Verify that policy-visible state describes the requested engine turn."""

        if not isinstance(request.information_state, SearchInformationState):
            raise PolicyContractError("policy requires a player information state")
        information = request.information_state
        if (
            information.player is not request.player
            or information.round_number != request.round_number
            or information.turn_number != request.turn_number
        ):
            raise PolicyContractError(
                "policy information does not match the requested turn"
            )
        return information

    @staticmethod
    def _confirm_service_legality(
        request: PolicyTurnRequest,
        selected: int,
    ) -> None:
        """Confirm that policy and gameplay action tables still agree."""

        # Inference derives legality from SearchInformationState. The separate
        # service table check protects the transaction boundary against drift.
        if (
            selected >= len(request.action_table)
            or request.action_table[selected] is None
        ):
            raise PolicyContractError(
                "policy selected an action outside service legality"
            )

    def _log_decision(
        self,
        request: PolicyTurnRequest,
        decision: ActivePolicyDecision,
    ) -> None:
        """Record operational policy identity, action, and latency."""

        # The game key contains the replay seed; it is not operational log data.
        LOGGER.info(
            (
                "standalone-policy turn artifact_sha256=%s "
                "round=%d turn=%d representative_action=%d concrete_action=%d "
                "decision_latency_seconds=%.6f"
            ),
            self.policy.artifact_digest,
            request.round_number,
            request.turn_number,
            decision.representative_action_index,
            decision.concrete_action_index,
            decision.latency_seconds,
        )


__all__ = (
    "ActivePolicyDecision",
    "ActivePolicyExecutor",
    "ActivePolicyRuntime",
)
