"""Response-ranking dataset collection and sealing invariants."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import dracula.response_dataset as dataset
from dracula.policy_value import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from dracula.response_distillation import (
    RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
    RESPONSE_EXAMPLE_SCHEMA_VERSION,
    RESPONSE_RANKING_SCHEMA_VERSION,
    ResponseActionGroupTarget,
    ResponseCacheIdentity,
    ResponseDistillationExample,
)
from dracula.search import (
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    GREEDY_RESPONSE_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    information_state_fingerprint,
    policy_input_from_information_state,
    strategic_action_groups,
)
from dracula.teacher import FixtureSplit


class _FastPlanner:
    """Emit exact contract examples while tests avoid expensive Monte Carlo."""

    def __init__(self, config, *, response_observer=None):
        self.config = config
        self.observer = response_observer

    def search(self, information, request_seed, should_stop=None):
        del request_seed
        if should_stop is not None and should_stop():
            raise dataset.ResponseDatasetInterrupted("test interruption")
        policy_input = policy_input_from_information_state(information)
        search_groups = strategic_action_groups(information, True)
        targets = tuple(
            ResponseActionGroupTarget(
                hand_slot=group.hand_slot,
                representative_action_index=group.representative_action_index,
                member_action_indices=group.member_action_indices,
                mean_terminal_differential=(
                    1.0 - group.representative_action_index / 31.0
                ),
            )
            for group in search_groups
        )
        selected = min(
            targets,
            key=lambda group: (
                -group.mean_terminal_differential,
                group.representative_action_index,
            ),
        )
        fingerprint = information_state_fingerprint(information)
        example = ResponseDistillationExample(
            schema_version=RESPONSE_EXAMPLE_SCHEMA_VERSION,
            ranking_schema_version=RESPONSE_RANKING_SCHEMA_VERSION,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            action_schema_version=ACTION_SCHEMA_VERSION,
            observation=tuple(
                bool(value) for value in policy_input.observation.tolist()
            ),
            legal_mask=tuple(
                tuple(bool(value) for value in row)
                for row in policy_input.legal_mask.tolist()
            ),
            groups=targets,
            selected_representative_action_index=(
                selected.representative_action_index
            ),
            selected_concrete_action_index=(
                selected.representative_action_index
            ),
            placement_number=information.turn_number,
            actor_role=information.player.value,
            actor_is_dealer=information.player is information.dealer,
            completions_per_action=4,
            search_config_digest=self.config.digest,
            response_config_digest=self.config.response_config.digest,
            cache_identity=ResponseCacheIdentity(
                fingerprint, self.config.response_config.digest
            ),
        )
        if self.observer is not None:
            # A repeated notification simulates duplicate cache evidence. The
            # game shard must retain the identity only once.
            self.observer(example)
            self.observer(example)
        representatives = [
            group.representative_action_index for group in search_groups
        ]
        quotient, remainder = divmod(
            self.config.outer_simulation_budget, len(representatives)
        )
        visits = [0] * 32
        for ordinal, action_index in enumerate(representatives):
            visits[action_index] = quotient + int(ordinal < remainder)
        selected_root = min(
            representatives, key=lambda index: (-visits[index], index)
        )
        return SimpleNamespace(
            action_visits=tuple(visits),
            selected_action_index=selected_root,
            response_request_count=2,
        )


def _config(output: Path) -> dataset.ResponseDatasetConfig:
    return dataset.ResponseDatasetConfig(
        run_id="response-dataset-test",
        root_seed="response-dataset-test-root",
        output_directory=str(output),
        training_games=1,
        validation_games=1,
        workers=1,
    )


@pytest.fixture
def collected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset, "StrategicInformationSetSearch", _FastPlanner
    )
    config = _config(tmp_path)
    inspection = dataset.collect_response_dataset(config)
    return config, inspection


# Split derivation must make every game identity unique across data roles.
def test_training_and_validation_fixtures_are_disjoint(tmp_path) -> None:
    config = dataset.ResponseDatasetConfig(
        "split-test",
        "split-root",
        str(tmp_path),
        training_games=4,
        validation_games=2,
        workers=4,
    )
    fixtures = dataset.fixture_schedule(config)
    training = {
        fixture.fixture_id
        for fixture in fixtures
        if fixture.split is FixtureSplit.TRAINING
    }
    validation = {
        fixture.fixture_id
        for fixture in fixtures
        if fixture.split is FixtureSplit.VALIDATION
    }

    assert len(training) == 4
    assert len(validation) == 2
    assert training.isdisjoint(validation)


# A complete game includes both roles under both dealer assignments while
# forced eighth placements never reach response-ranking rows.
def test_complete_shards_have_role_dealer_coverage_without_forced_rows(
    collected,
) -> None:
    config, inspection = collected
    assert inspection.training_games == inspection.validation_games == 1
    for fixture in dataset.fixture_schedule(config):
        shard = dataset.load_response_shard(
            config.output_path
            / "response-distillation"
            / "games"
            / fixture.split.value
            / f"{fixture.fixture_id}.pt"
        )
        columns = shard["columns"]
        coverage = {
            (role, bool(columns["actor_is_dealer"][index].item()))
            for index, role in enumerate(columns["actor_roles"])
        }
        assert coverage == {
            ("queen", False),
            ("queen", True),
            ("king", False),
            ("king", True),
        }
        assert 8 not in columns["placement_numbers"].tolist()
        assert "search_visits" not in columns
        assert "round_returns" not in columns


# Repeated cache evidence is counted but sealed only once per identity.
def test_duplicate_cache_evidence_does_not_duplicate_rows(collected) -> None:
    _, inspection = collected
    assert inspection.observer_emissions == inspection.response_requests
    assert inspection.duplicates_removed == inspection.examples
    assert inspection.observer_emissions == inspection.examples * 2


# Variable-length groups must preserve evaluator means, canonical maxima, and
# an exact partition of the server-issued legal concrete actions.
def test_group_values_selection_and_legality_round_trip(collected) -> None:
    config, _ = collected
    fixture = dataset.fixture_schedule(config)[0]
    shard = dataset.load_response_shard(
        config.output_path
        / "response-distillation"
        / "games"
        / fixture.split.value
        / f"{fixture.fixture_id}.pt"
    )
    columns = shard["columns"]
    group_offsets = columns["group_offsets"]
    member_offsets = columns["member_offsets"]

    for row in range(columns["observations"].shape[0]):
        start = int(group_offsets[row].item())
        end = int(group_offsets[row + 1].item())
        values = columns["group_mean_values"][start:end]
        representatives = columns["group_representative_actions"][start:end]
        selected_offset = min(
            range(end - start),
            key=lambda offset: (
                -float(values[offset].item()),
                int(representatives[offset].item()),
            ),
        )
        assert (
            int(columns["selected_representative_actions"][row].item())
            == int(representatives[selected_offset].item())
        )
        members = {
            int(action)
            for group_index in range(start, end)
            for action in columns["group_member_actions"][
                int(member_offsets[group_index].item()) :
                int(member_offsets[group_index + 1].item())
            ].tolist()
        }
        legal = {
            index
            for index, allowed in enumerate(
                columns["legal_masks"][row].flatten().tolist()
            )
            if allowed
        }
        assert members == legal
        assert all(
            float(value)
            == pytest.approx(
                1.0 - int(representative.item()) / 31.0,
                abs=0.0,
            )
            for value, representative in zip(
                values, representatives, strict=True
            )
        )


# Shards and manifests bind every schema needed to reject incompatible data.
def test_artifacts_bind_the_complete_response_contract(collected) -> None:
    config, inspection = collected
    manifest = dataset.load_response_manifest(
        config.output_path, FixtureSplit.TRAINING
    )["manifest"]

    assert manifest["search_schema_version"] == STRATEGIC_SEARCH_SCHEMA_VERSION
    assert manifest["response_schema_version"] == GREEDY_RESPONSE_SCHEMA_VERSION
    assert (
        manifest["symmetry_schema_version"]
        == DESTINATION_SYMMETRY_SCHEMA_VERSION
    )
    assert manifest["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert manifest["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert (
        manifest["action_group_schema_version"]
        == RESPONSE_ACTION_GROUP_SCHEMA_VERSION
    )
    assert manifest["ranking_schema_version"] == RESPONSE_RANKING_SCHEMA_VERSION
    assert manifest["collection_config_digest"] == inspection.config_digest
    assert manifest["search_config_digest"] == config.search_config.digest


# A sealed row has only the model view, targets, and public contract identities.
def test_sealed_payload_has_no_hidden_or_authoritative_fields(collected) -> None:
    config, _ = collected
    fixture = dataset.fixture_schedule(config)[0]
    shard = dataset.load_response_shard(
        config.output_path
        / "response-distillation"
        / "games"
        / fixture.split.value
        / f"{fixture.fixture_id}.pt"
    )
    serialized_keys = set(shard["metadata"]) | set(shard["columns"])

    assert serialized_keys.isdisjoint(dataset._FORBIDDEN_KEYS)


# CPU collection is content deterministic even when the output directory differs.
def test_fixed_inputs_reproduce_dataset_digest(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dataset, "StrategicInformationSetSearch", _FastPlanner
    )
    first = dataset.collect_response_dataset(_config(tmp_path / "first"))
    second = dataset.collect_response_dataset(_config(tmp_path / "second"))

    assert first.dataset_digest == second.dataset_digest


# Interruption can leave complete games but cannot expose a partial game shard;
# resume skips the seal and reproduces the uninterrupted dataset digest.
def test_resume_starts_at_a_sealed_game_boundary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        dataset, "StrategicInformationSetSearch", _FastPlanner
    )
    baseline_config = _config(tmp_path / "baseline")
    baseline = dataset.collect_response_dataset(baseline_config)

    interrupted_config = _config(tmp_path / "interrupted")
    checks = 0

    def stop() -> bool:
        nonlocal checks
        checks += 1
        return checks > 100

    with pytest.raises(dataset.ResponseDatasetInterrupted):
        dataset.collect_response_dataset(
            interrupted_config, should_stop=stop
        )
    shard_paths = list(
        (interrupted_config.output_path / "response-distillation" / "games").rglob(
            "*.pt"
        )
    )
    assert len(shard_paths) == 1

    resumed = dataset.collect_response_dataset(
        interrupted_config, resume=True
    )
    assert resumed.dataset_digest == baseline.dataset_digest
