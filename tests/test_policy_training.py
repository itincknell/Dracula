"""Exercise the current corpus reader and visit-distillation trainer.

Synthetic sealed data verifies target projection, masked distributional loss,
deterministic optimization and resume, checkpoint selection, and artifact export.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
import torch

import dracula.policy.training.runner as training
from dracula.policy.artifact import load_policy_artifact
from dracula.policy.training.dataset import DATASET_FORMAT, load_policy_dataset
from dracula.policy.training.metrics import distributional_policy_cross_entropy
from dracula.policy.training.contracts import (
    PolicyTrainingConfig,
    PolicyTrainingError,
    PolicyTrainingInterrupted,
    DatasetSection,
    OptimizationSection,
    RunSection,
)
from dracula.game.engine import create_game
from dracula.policy.observation import (
    encode_policy_observation,
    pack_action_rows_for_model,
)
from dracula.decision.information import information_state_from_engine
from dracula.decision.strategic_actions import strategic_action_groups


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
    compact_engine = pack_action_rows_for_model(information, engine_mask)
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
        "dealer": information.dealer.value,
        "fixture_id": fixture,
        "information_state_fingerprint": "1" * 64,
        "legal_mask_packed": _pack(compact_engine),
        "observation_packed": _pack(observation),
        "placement_number": placement,
        "player": information.player.value,
        "round_number": 1,
        "selected_group_representative": representatives[0],
        "strategic_group_representatives": representatives,
        "strategic_group_visits": visits,
        "strategic_groups": compact_groups,
        "trajectory_profile": "teacher",
    }


@pytest.fixture()
def card_set_corpus(tmp_path: Path) -> Path:
    entries = []
    for ordinal, overlay in enumerate(("training", "validation")):
        fixture = f"fixture-{ordinal}"
        rows = [_fixture_row(index % 7 + 1, fixture=fixture) for index in range(42)]
        content = {
            "fixture_id": fixture,
            "ordinal": ordinal,
            "overlay": overlay,
            "rows": rows,
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
        "corpus_schema_version": DATASET_FORMAT,
        "game_count": 2,
        "games": entries,
        "placement_counts": {
            overlay: {str(index): 6 for index in range(1, 8)}
            for overlay in ("training", "validation")
        },
        "row_count": 84,
    }
    _write(
        tmp_path / "corpus-manifest.json",
        {"content": manifest_content, "content_digest": _digest(manifest_content)},
    )
    return tmp_path


def _config(dataset: Path, output: Path) -> PolicyTrainingConfig:
    return PolicyTrainingConfig(
        RunSection("bgc-smoke", 606, str(output)),
        DatasetSection(str(dataset)),
        OptimizationSection("cpu"),
    )


def test_current_reader_loads_only_659_bit_rows(card_set_corpus: Path) -> None:
    bundle = load_policy_dataset(card_set_corpus)
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
    loss = distributional_policy_cross_entropy(logits, masks, targets)
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
    with pytest.raises(PolicyTrainingError, match="forbidden"):
        load_policy_dataset(card_set_corpus)


def test_reader_rejects_fixture_overlap_between_splits(card_set_corpus: Path) -> None:
    """A complete fixture may contribute to exactly one model-selection split."""

    manifest_path = card_set_corpus / "corpus-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training_entry, validation_entry = manifest["content"]["games"]
    validation_path = card_set_corpus / validation_entry["path"]
    validation = json.loads(validation_path.read_text())
    validation["content"]["fixture_id"] = training_entry["fixture_id"]
    for row in validation["content"]["rows"]:
        row["fixture_id"] = training_entry["fixture_id"]
    validation["content_digest"] = _digest(validation["content"])
    _write(validation_path, validation)
    validation_entry["fixture_id"] = training_entry["fixture_id"]
    validation_entry["content_digest"] = validation["content_digest"]
    validation_entry["file_digest"] = hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    manifest["content_digest"] = _digest(manifest["content"])
    _write(manifest_path, manifest)

    with pytest.raises(PolicyTrainingError, match="fixtures overlap"):
        load_policy_dataset(card_set_corpus)


def test_smoke_training_resume_and_export(card_set_corpus: Path, tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    direct_result = training.train_policy(
        _config(card_set_corpus, direct), smoke_epochs=2
    )
    with pytest.raises(PolicyTrainingInterrupted):
        training.train_policy(
            _config(card_set_corpus, resumed),
            smoke_epochs=2,
            interrupt_after_batches=1,
        )
    resumed_result = training.train_policy(
        _config(card_set_corpus, resumed), resume=True, smoke_epochs=2
    )
    direct_final = torch.load(direct_result.final_checkpoint, weights_only=True)
    resumed_final = torch.load(resumed_result.final_checkpoint, weights_only=True)
    assert all(
        torch.equal(direct_final["model_state_dict"][name], tensor)
        for name, tensor in resumed_final["model_state_dict"].items()
    )
    artifact = load_policy_artifact(direct_result.exported_artifact)
    assert artifact.artifact_digest == hashlib.sha256(
        Path(direct_result.exported_artifact).read_bytes()
    ).hexdigest()
    assert not list(direct.rglob("*.tmp-*"))


def test_validation_rejects_nonfinite_checkpoint_state(
    card_set_corpus: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "nonfinite-checkpoint"
    training.train_policy(_config(card_set_corpus, output), smoke_epochs=1)
    checkpoint = output / "checkpoints/best-validation-candidate.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["maximum_gradient_norm"] = float("nan")
    torch.save(payload, checkpoint)

    with pytest.raises(PolicyTrainingError, match="progress is invalid"):
        training.validate_policy_run(output)
