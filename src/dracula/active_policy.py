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

from dracula.api.policy import (
    PolicyContractError,
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.bgc_policy import load_policy_artifact
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.policy_observation import select_policy_group
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.strategic_actions import (
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
        model: BGCPolicyModel,
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
        self._descriptor = PolicyDescriptor(
            policy_id="pi1",
            artifact_digest=self.policy.artifact_digest,
        )

    @property
    def descriptor(self) -> PolicyDescriptor:
        return self._descriptor

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        if request.policy != self.descriptor:
            raise PolicyContractError("game opponent session does not match policy")
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
        decision = self.policy.decide(
            information,
            fixture_id=str(request.game_id),
            decision_index=request.turn_number,
        )
        selected = decision.concrete_action_index
        if (
            selected >= len(request.action_table)
            or request.action_table[selected] is None
        ):
            raise PolicyContractError(
                "policy selected an action outside service legality"
            )
        LOGGER.info(
            (
                "standalone-policy turn artifact_sha256=%s game_id=%s "
                "round=%d turn=%d representative_action=%d concrete_action=%d "
                "decision_latency_seconds=%.6f"
            ),
            self.policy.artifact_digest,
            request.game_id,
            request.round_number,
            request.turn_number,
            decision.representative_action_index,
            decision.concrete_action_index,
            decision.latency_seconds,
        )
        return PolicyTurnResult(selected)


__all__ = (
    "ActivePolicyDecision",
    "ActivePolicyExecutor",
    "ActivePolicyRuntime",
)
