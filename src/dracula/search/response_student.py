"""Response-ranker inference at the actor information-state boundary."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from dracula.policy_value import ACTION_COUNT, PolicyValueModel
from dracula.search.information import (
    SearchInformationState,
    policy_input_from_information_state,
)
from dracula.search.planner import SearchContractViolation
from dracula.search.strategic import StrategicActionGroup


class ResponseRankerGroupEvaluator:
    """Rank every legal strategic group without receiving an engine state."""

    def __init__(
        self,
        model: PolicyValueModel,
        artifact_digest: str,
    ) -> None:
        if not isinstance(model, PolicyValueModel):
            raise SearchContractViolation(
                "response ranker requires the documented policy/value model"
            )
        if (
            not isinstance(artifact_digest, str)
            or len(artifact_digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact_digest)
        ):
            raise SearchContractViolation(
                "response-ranker artifact digest is invalid"
            )
        self.model = model.cpu().eval()
        self.artifact_digest = artifact_digest

    @classmethod
    def from_artifact(
        cls, artifact_path: str | Path
    ) -> ResponseRankerGroupEvaluator:
        path = Path(artifact_path).expanduser().resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise SearchContractViolation(
                "response-ranker artifact could not be read"
            ) from error
        # Kept lazy so ordinary search imports do not load dataset/training code.
        from dracula.response_ranker import load_response_ranker_artifact

        try:
            loaded = load_response_ranker_artifact(path)
        except ValueError as error:
            raise SearchContractViolation(
                "response-ranker artifact is incompatible"
            ) from error
        return cls(loaded.model, digest)

    def rank(
        self,
        information: SearchInformationState,
        groups: tuple[StrategicActionGroup, ...],
    ) -> tuple[int, ...]:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "student response requires an actor information state"
            )
        if not isinstance(groups, tuple) or not groups:
            raise SearchContractViolation(
                "student response requires strategic action groups"
            )
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        if (
            representatives != tuple(sorted(representatives))
            or len(set(representatives)) != len(representatives)
        ):
            raise SearchContractViolation(
                "student response groups must have canonical representatives"
            )
        legal = {
            index
            for index, allowed in enumerate(
                value for row in information.legal_mask for value in row
            )
            if allowed
        }
        grouped = {
            index
            for group in groups
            for index in group.member_action_indices
        }
        if (
            grouped != legal
            or len(grouped)
            != sum(len(group.member_action_indices) for group in groups)
        ):
            raise SearchContractViolation(
                "student response groups must partition legal actions"
            )
        policy_input = policy_input_from_information_state(information)
        with torch.no_grad():
            logits, _ = self.model(policy_input.observation)
        if (
            logits.shape != (4, 8)
            or logits.dtype is not torch.float32
            or not torch.isfinite(logits).all()
        ):
            raise SearchContractViolation(
                "response ranker returned invalid logits"
            )
        flat = logits.reshape(ACTION_COUNT)
        scored: list[tuple[float, int]] = []
        for group in groups:
            score = float(
                flat[list(group.member_action_indices)].mean().detach().cpu()
            )
            if not math.isfinite(score):
                raise SearchContractViolation(
                    "response group score is non-finite"
                )
            scored.append((score, group.representative_action_index))
        return tuple(
            representative
            for _, representative in sorted(
                scored, key=lambda item: (-item[0], item[1])
            )
        )


__all__ = ("ResponseRankerGroupEvaluator",)
