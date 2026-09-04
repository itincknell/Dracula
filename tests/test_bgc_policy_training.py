"""Current 659-bit corpus reader and visit-distillation trainer tests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
import torch

import dracula.bgc_policy_training as training
from dracula.action_contract import build_representative_action_projection
from dracula.bgc_policy import load_bgc_policy_artifact
from dracula.bgc_policy_migration import (
    CORPUS_SCHEMA_VERSION,
    GAME_SCHEMA_VERSION,
    LOADER_SCHEMA_VERSION,
    MIGRATION_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
)
from dracula.bgc_policy_model import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from dracula.engine import create_game
from dracula.policy_observation import candidate_action_tensor, encode_policy_observation
from dracula.search.information import information_state_from_engine
from dracula.strategic_actions import strategic_action_groups


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _pack(bits: torch.Tensor) -> str:
    values = bits.flatten().tolist()
    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return base64.b64encode(bytes(packed)).decode("ascii")


def _fixture_row(placement: int, *, fixture: str) -> dict[str, object]:
    information = information_state_from_engine(create_game(f"{fixture}-state"))
    observation = encode_policy_observation(information)
    engine_mask = torch.tensor(information.legal_mask, dtype=torch.bool)
    groups = strategic_action_groups(information)
    projection = build_representative_action_projection(engine_mask, groups)
    compact_engine = candidate_action_tensor(information, engine_mask)
    compact_groups = []
    representatives = []
    for group in groups:
        candidate_row = sum(
            card is not None for card in information.own_hand[: group.hand_slot]
        )
        members = [
            candidate_row * 8 + member % 8 for member in group.member_action_indices
        ]
        representative = candidate_row * 8 + group.representative_action_index % 8
        compact_groups.append(members)
        representatives.append(representative)
    visits = [0] * len(representatives)
    visits[0] = 128
    return {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "dealer": information.dealer.value,
        "fixture_id": fixture,
        "information_state_fingerprint": "1" * 64,
        "legal_mask_packed": _pack(compact_engine),
        "observation_packed": _pack(observation),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "placement_number": placement,
        "player": information.player.value,
        "round_number": 1,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "search_config_digest": "2" * 64,
        "selected_group_representative": representatives[0],
        "strategic_group_representatives": representatives,
        "strategic_group_visits": visits,
        "strategic_groups": compact_groups,
        "trajectory_profile": "teacher",
    }


@pytest.fixture()
def card_set_corpus(tmp_path: Path) -> Path:
    source_content = {"source_revision": "a" * 40, "source_tree_digest": "b" * 64}
    source = {"content": source_content, "content_digest": _digest(source_content)}
    source_path = tmp_path / "source-snapshot.json"
    _write(source_path, source)
    entries = []
    for ordinal, overlay in enumerate(("training", "validation")):
        fixture = f"fixture-{ordinal}"
        rows = [_fixture_row(index % 7 + 1, fixture=fixture) for index in range(42)]
        content = {
            "fixture_id": fixture,
            "game_schema_version": GAME_SCHEMA_VERSION,
            "ordinal": ordinal,
            "overlay": overlay,
            "rows": rows,
            "source_content_digest": "3" * 64,
            "trajectory_profile": "teacher",
        }
        document = {"content": content, "content_digest": _digest(content)}
        path = tmp_path / f"games/{ordinal:06d}.json"
        _write(path, document)
        entries.append(
            {
                "content_digest": document["content_digest"],
                "file_digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fixture_id": fixture,
                "ordinal": ordinal,
                "overlay": overlay,
                "path": f"games/{ordinal:06d}.json",
                "row_count": 42,
                "trajectory_profile": "teacher",
            }
        )
    manifest_content = {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "game_count": 2,
        "games": entries,
        "loader_schema_version": LOADER_SCHEMA_VERSION,
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "placement_counts": {
            overlay: {str(index): 6 for index in range(1, 8)}
            for overlay in ("training", "validation")
        },
        "row_count": 84,
        "source_snapshot_digest": source["content_digest"],
        "source_snapshot_path": str(source_path),
    }
    _write(
        tmp_path / "corpus-manifest.json",
        {"content": manifest_content, "content_digest": _digest(manifest_content)},
    )
    return tmp_path


def _config(dataset: Path, output: Path) -> training.BGCPolicyTrainingConfig:
    return training.BGCPolicyTrainingConfig(
        training.RunSection("bgc-smoke", "bgc-smoke-root", str(output)),
        training.DatasetSection(str(dataset)),
        training.ModelSection("pi1-smoke", 0),
        training.OptimizationSection("cpu"),
    )


def test_current_reader_loads_only_659_bit_rows(card_set_corpus: Path) -> None:
    bundle = training.load_bgc_card_policy_dataset(card_set_corpus)
    assert bundle.training.example_count == bundle.validation.example_count == 42
    observations, legal, representatives, targets, selected = (
        bundle.training.decoded_batch(torch.arange(42), device=torch.device("cpu"))
    )
    assert observations.shape == (42, 659)
    assert legal.shape == representatives.shape == (42, 4, 8)
    assert targets.shape == (42, 32)
    assert torch.equal(targets.sum(dim=1), torch.ones(42))
    assert representatives.flatten(1).gather(1, selected[:, None]).all()


def test_distributional_loss_masks_probability_and_gradient() -> None:
    logits = torch.randn(2, 4, 8, requires_grad=True)
    masks = torch.zeros(2, 4, 8, dtype=torch.bool)
    masks[:, 0, :3] = True
    targets = torch.zeros(2, 32)
    targets[:, :3] = torch.tensor([0.25, 0.5, 0.25])
    loss = training.distributional_policy_cross_entropy(logits, masks, targets)
    loss.backward()
    probabilities = torch.softmax(
        torch.where(masks, logits.detach(), -torch.inf).flatten(1), dim=1
    )
    assert torch.count_nonzero(probabilities.masked_select(~masks.flatten(1))) == 0
    assert torch.count_nonzero(logits.grad.masked_select(~masks)) == 0


def test_reader_rejects_private_fields(card_set_corpus: Path) -> None:
    path = card_set_corpus / "games/000000.json"
    document = json.loads(path.read_text())
    document["content"]["rows"][0]["opponent_hand"] = ["AC"]
    document["content_digest"] = _digest(document["content"])
    _write(path, document)
    manifest = json.loads((card_set_corpus / "corpus-manifest.json").read_text())
    manifest["content"]["games"][0]["content_digest"] = document["content_digest"]
    manifest["content"]["games"][0]["file_digest"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["content_digest"] = _digest(manifest["content"])
    _write(card_set_corpus / "corpus-manifest.json", manifest)
    with pytest.raises(training.BGCPolicyTrainingError, match="forbidden"):
        training.load_bgc_card_policy_dataset(card_set_corpus)


def test_smoke_training_resume_and_export(card_set_corpus: Path, tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    direct_result = training.train_bgc_policy(
        _config(card_set_corpus, direct), smoke_epochs=2
    )
    with pytest.raises(training.BGCPolicyTrainingInterrupted):
        training.train_bgc_policy(
            _config(card_set_corpus, resumed),
            smoke_epochs=2,
            interrupt_after_batches=1,
        )
    resumed_result = training.train_bgc_policy(
        _config(card_set_corpus, resumed), resume=True, smoke_epochs=2
    )
    direct_final = torch.load(direct_result.final_checkpoint, weights_only=True)
    resumed_final = torch.load(resumed_result.final_checkpoint, weights_only=True)
    assert direct_final["state_dict_digest"] == resumed_final["state_dict_digest"]
    artifact = load_bgc_policy_artifact(direct_result.exported_artifact)
    assert artifact.metadata.parameter_count == 754_601
    assert not list(direct.rglob("*.tmp-*"))
