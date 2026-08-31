"""Accepted pi0 continuation responses for the prospective D1 teacher."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from dracula.bgc_policy_evaluation import (
    AcceptedPi0Bundle,
    load_accepted_pi0_bundle,
)
from dracula.randomness import derive_seed
from dracula.sam_policy import (
    build_representative_action_projection,
    representative_policy_probabilities,
)
from dracula.bgc_policy_model import (
    compact_action_tensor_from_legacy,
    compact_observation_from_legacy,
    legacy_action_index_from_compact,
)
from dracula.search import (
    BeliefGreedyInformationSetSearch,
    BeliefGreedyResponseResult,
    BeliefGreedySearchConfig,
    SearchInformationState,
    SearchInterrupted,
    StrategicActionGroupDiagnostic,
    derive_strategic_destination_choice_seed,
    information_state_fingerprint,
    policy_input_from_information_state,
    select_concrete_action_index,
    strategic_action_groups,
)

ACCEPTED_PI0_RESPONSE_SCHEMA_VERSION = "dracula-accepted-pi0-response-v1"
ACCEPTED_PI0_RESPONSE_CACHE_SCHEMA_VERSION = (
    "dracula-accepted-pi0-response-cache-v1"
)
ACCEPTED_PI0_BGC_SEARCH_SCHEMA_VERSION = "dracula-pi0-bgc-search-v1"
ACCEPTED_PI0_RESPONSE_SEED_NAMESPACE = "dracula-accepted-pi0-response-seed-v1"
ACCEPTED_PI0_DESTINATION_SCOPE = "dracula-accepted-pi0-destination-v1"


class AcceptedPi0Error(ValueError):
    """An accepted artifact or continuation request violates its contract."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise SearchInterrupted("accepted pi0 continuation was interrupted")


@dataclass(frozen=True, slots=True)
class AcceptedPi0ResponseIdentity:
    accepted_checkpoint_digest: str
    acceptance_manifest_digest: str
    response_schema_version: str
    response_cache_schema_version: str

    @property
    def digest(self) -> str:
        return _digest(
            {
                "acceptance_manifest_digest": self.acceptance_manifest_digest,
                "accepted_checkpoint_digest": self.accepted_checkpoint_digest,
                "response_cache_schema_version": self.response_cache_schema_version,
                "response_schema_version": self.response_schema_version,
            }
        )


class AcceptedPi0ContinuationAdapter:
    """One model call over only the acting simulated player's information."""

    def __init__(self, bundle: AcceptedPi0Bundle) -> None:
        if not isinstance(bundle, AcceptedPi0Bundle):
            raise AcceptedPi0Error("continuation adapter requires an accepted bundle")
        self.bundle = bundle
        self.model = bundle.artifact.model.cpu().eval().requires_grad_(False)
        self.identity = AcceptedPi0ResponseIdentity(
            bundle.checkpoint_digest,
            bundle.acceptance_content_digest,
            ACCEPTED_PI0_RESPONSE_SCHEMA_VERSION,
            ACCEPTED_PI0_RESPONSE_CACHE_SCHEMA_VERSION,
        )

    @classmethod
    def from_directory(cls, path: str | Path) -> AcceptedPi0ContinuationAdapter:
        return cls(load_accepted_pi0_bundle(path))

    @property
    def digest(self) -> str:
        return self.identity.digest

    def evaluate(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> BeliefGreedyResponseResult:
        if not isinstance(information, SearchInformationState):
            raise AcceptedPi0Error(
                "accepted pi0 receives only SearchInformationState"
            )
        _check_stop(should_stop)
        fingerprint = information_state_fingerprint(information)
        policy_input = policy_input_from_information_state(information)
        groups = strategic_action_groups(information, True)
        projection = build_representative_action_projection(
            policy_input.legal_mask, groups
        )
        compact_observation = compact_observation_from_legacy(
            policy_input.observation
        )
        compact_mask = compact_action_tensor_from_legacy(
            policy_input.observation, projection.mask
        )
        with torch.inference_mode():
            logits = self.model(compact_observation)
            probabilities = representative_policy_probabilities(
                logits, compact_mask
            )
        _check_stop(should_stop)
        flattened_probabilities = probabilities.reshape(-1)
        flattened_mask = compact_mask.reshape(-1)
        if (
            not bool(torch.isfinite(flattened_probabilities).all().item())
            or not torch.equal(
                flattened_probabilities.masked_select(~flattened_mask),
                torch.zeros_like(
                    flattened_probabilities.masked_select(~flattened_mask)
                ),
            )
        ):
            raise AcceptedPi0Error("accepted pi0 masking produced invalid probability")
        maximum = torch.max(flattened_probabilities.masked_select(flattened_mask))
        compact_representative = int(
            torch.nonzero(
                flattened_mask & (flattened_probabilities == maximum),
                as_tuple=False,
            )[0].item()
        )
        representative = legacy_action_index_from_compact(
            policy_input.observation, compact_representative
        )
        response_seed = derive_seed(
            ACCEPTED_PI0_RESPONSE_SEED_NAMESPACE,
            fingerprint,
            self.bundle.checkpoint_digest,
            self.bundle.acceptance_content_digest,
            self.digest,
        )
        concrete = select_concrete_action_index(
            information,
            representative,
            derive_strategic_destination_choice_seed(
                response_seed,
                ACCEPTED_PI0_DESTINATION_SCOPE,
                information,
                representative,
                0,
            ),
            True,
        )
        diagnostics = tuple(
            StrategicActionGroupDiagnostic(group, 1, 0.0) for group in groups
        )
        return BeliefGreedyResponseResult(
            information_state_fingerprint=fingerprint,
            config_digest=self.digest,
            selected_action_index=concrete,
            selected_representative_action_index=representative,
            group_diagnostics=diagnostics,
            belief_completion_count=1,
            potential_evaluation_count=len(groups),
        )


class AcceptedPi0BGCSearch(BeliefGreedyInformationSetSearch):
    """BGC outer UCT with only its continuation response implementation changed."""

    def __init__(
        self,
        adapter: AcceptedPi0ContinuationAdapter,
        config: BeliefGreedySearchConfig | None = None,
    ) -> None:
        resolved = config or BeliefGreedySearchConfig(
            outer_simulation_budget=128,
            belief_completion_count=8,
        )
        if resolved.outer_simulation_budget != 128:
            raise AcceptedPi0Error("pi0-BGC requires exactly 128 outer simulations")
        super().__init__(resolved)
        self.adapter = adapter

    @property
    def controller_digest(self) -> str:
        return _digest(
            {
                "accepted_checkpoint_digest": (
                    self.adapter.bundle.checkpoint_digest
                ),
                "acceptance_manifest_digest": (
                    self.adapter.bundle.acceptance_content_digest
                ),
                "base_outer_config_digest": self.config.digest,
                "response_adapter_digest": self.adapter.digest,
                "search_schema_version": ACCEPTED_PI0_BGC_SEARCH_SCHEMA_VERSION,
            }
        )

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> BeliefGreedyResponseResult:
        response = self.adapter.evaluate(information, should_stop)
        # The base outer search validates this field against its configured
        # response digest. The accepted adapter identity is independently bound
        # into controller_digest, D1 rows, and every D1 manifest.
        return BeliefGreedyResponseResult(
            information_state_fingerprint=(
                response.information_state_fingerprint
            ),
            config_digest=self.config.response_digest,
            selected_action_index=response.selected_action_index,
            selected_representative_action_index=(
                response.selected_representative_action_index
            ),
            group_diagnostics=response.group_diagnostics,
            belief_completion_count=response.belief_completion_count,
            potential_evaluation_count=response.potential_evaluation_count,
        )


__all__ = (
    "ACCEPTED_PI0_BGC_SEARCH_SCHEMA_VERSION",
    "ACCEPTED_PI0_RESPONSE_CACHE_SCHEMA_VERSION",
    "ACCEPTED_PI0_RESPONSE_SCHEMA_VERSION",
    "AcceptedPi0BGCSearch",
    "AcceptedPi0ContinuationAdapter",
    "AcceptedPi0Error",
    "AcceptedPi0ResponseIdentity",
)
