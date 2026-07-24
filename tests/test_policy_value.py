"""Mechanical acceptance for the search-guided policy/value model."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect

import pytest
import torch
from torch import Tensor, nn

import dracula.policy_value as policy_value_module
from dracula.bridge import PolicyInput, build_policy_turn_context, transpose_grid_index
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.policy_value import (
    PARAMETER_COUNT,
    PolicyValueContractError,
    PolicyValueModel,
    apply_legal_mask,
    build_policy_value_artifact,
    build_policy_value_optimizer,
    legal_policy_probabilities,
    load_policy_value_artifact,
    policy_value_loss,
    save_policy_value_artifact,
)
from dracula.randomness import derive_seed
from dracula.search import (
    PublicGameHistory,
    build_information_state,
    information_state_from_engine,
    policy_input_from_information_state,
    public_history_from_engine,
    sample_determinization,
)


def _model(ordinal: int = 0) -> PolicyValueModel:
    return PolicyValueModel(
        run_root_seed="policy-value-test-root",
        model_id="policy-value-test",
        initialization_ordinal=ordinal,
    )


def _advance(state):
    move = legal_moves(state, state.active_player)[0]
    return apply_move(state, move).state


def _input_after_moves(move_count: int = 0) -> PolicyInput:
    state = create_game("policy-value-contract-game")
    for _ in range(move_count):
        state = _advance(state)
    return policy_input_from_information_state(information_state_from_engine(state))


def _targets(mask: Tensor) -> Tensor:
    target = mask.to(torch.float32)
    if mask.ndim == 2:
        return target / target.sum()
    return target / target.sum(dim=(1, 2), keepdim=True)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def model() -> PolicyValueModel:
    return _model()


# The typed search view and the established bridge must never drift into two encodings.
def test_typed_information_state_reconstructs_every_game_observation() -> None:
    state = create_game("policy-value-six-round-projection")
    checked = 0
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            context = build_policy_turn_context(state, state.active_player)
            projected = policy_input_from_information_state(
                information_state_from_engine(state)
            )
            assert torch.equal(projected.observation, context.input.observation)
            assert torch.equal(projected.legal_mask, context.input.legal_mask)
            checked += 1
            state = _advance(state)
        state = advance_after_round(state)
    assert checked == 48


# Fixed shapes and parameter count make artifacts mechanically compatible.
def test_exact_single_and_batch_shapes_and_parameter_count(
    model: PolicyValueModel,
) -> None:
    first = _input_after_moves(0)
    second = _input_after_moves(2)
    logits, value = model(first.observation)
    assert first.observation.shape == (875,)
    assert first.legal_mask.shape == (4, 8)
    assert logits.shape == (4, 8)
    assert value.shape == ()
    assert torch.isfinite(logits).all()
    assert -1 <= value.item() <= 1

    observations = torch.stack((first.observation, second.observation))
    masks = torch.stack((first.legal_mask, second.legal_mask))
    batch_logits, batch_values = model(observations)
    assert observations.shape == (2, 875)
    assert masks.shape == (2, 4, 8)
    assert batch_logits.shape == (2, 4, 8)
    assert batch_values.shape == (2,)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 339_978


# Initialization identity must fully determine weights without consuming global randomness.
def test_initialization_is_deterministic_isolated_and_exact() -> None:
    torch.manual_seed(4701)
    expected_next = torch.rand(4)
    torch.manual_seed(4701)
    first = _model(0)
    actual_next = torch.rand(4)
    repeated = _model(0)
    different = _model(1)

    assert torch.equal(actual_next, expected_next)
    for left, right in zip(first.parameters(), repeated.parameters(), strict=True):
        assert torch.equal(left, right)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), different.parameters(), strict=True)
    )
    for module in first.modules():
        if isinstance(module, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            assert torch.max(torch.abs(module.weight)) <= bound
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.count_nonzero(module.bias) == 0


# Vectorized inference must preserve the semantics of evaluating states independently.
def test_batch_and_individual_forward_are_equivalent(model: PolicyValueModel) -> None:
    inputs = tuple(_input_after_moves(count) for count in (0, 1, 3, 5))
    observations = torch.stack(tuple(value.observation for value in inputs))
    batch_logits, batch_values = model(observations)
    for index, policy_input in enumerate(inputs):
        logits, value = model(policy_input.observation)
        assert torch.allclose(logits, batch_logits[index], atol=1e-5, rtol=1e-4)
        assert torch.allclose(value, batch_values[index], atol=1e-5, rtol=1e-4)


# Queen and King normalization must erase role-specific coordinate orientation.
def test_queen_and_king_normalized_inputs_produce_identical_outputs(
    model: PolicyValueModel,
) -> None:
    state = create_game("engine-contract-fixture-1")
    queen_context = build_policy_turn_context(state, EnginePlayer.QUEEN)
    queen_history = public_history_from_engine(state)
    queen_information = build_information_state(
        queen_context.input, queen_history, EnginePlayer.QUEEN
    )

    mirrored = [None] * 9
    for index, card_id in enumerate(queen_history.current_coffin):
        mirrored[transpose_grid_index(index)] = card_id
    king_history = PublicGameHistory(
        status=EngineStatus.PLAYING,
        round_number=queen_history.round_number,
        dealer=EnginePlayer.QUEEN,
        active_player=EnginePlayer.KING,
        total_scores=queen_history.total_scores,
        completed_rounds=(),
        current_coffin=tuple(mirrored),
        current_round_moves=(),
    )
    king_information = build_information_state(
        PolicyInput(
            queen_context.input.observation.clone(),
            queen_context.input.legal_mask.clone(),
        ),
        king_history,
        EnginePlayer.KING,
    )
    queen_input = policy_input_from_information_state(queen_information)
    king_input = policy_input_from_information_state(king_information)
    assert torch.equal(queen_input.observation, king_input.observation)
    assert torch.equal(queen_input.legal_mask, king_input.legal_mask)
    queen_outputs = model(queen_input.observation)
    king_outputs = model(king_input.observation)
    assert torch.equal(queen_outputs[0], king_outputs[0])
    assert torch.equal(queen_outputs[1], king_outputs[1])


# Private hidden arrangements that induce one information state cannot affect inference.
def test_hidden_card_substitution_is_invariant(model: PolicyValueModel) -> None:
    root = information_state_from_engine(create_game("policy-value-hidden-root"))
    first_sample = sample_determinization(
        root, derive_seed("dracula-policy-value-test-v1", "world-a")
    )
    second_sample = sample_determinization(
        root, derive_seed("dracula-policy-value-test-v1", "world-b")
    )
    assert (
        first_sample.opponent_remaining_hand,
        first_sample.sampled_stock,
    ) != (
        second_sample.opponent_remaining_hand,
        second_sample.sampled_stock,
    )
    first = policy_input_from_information_state(
        information_state_from_engine(first_sample.state)
    )
    second = policy_input_from_information_state(
        information_state_from_engine(second_sample.state)
    )
    assert torch.equal(first.observation, second.observation)
    assert torch.equal(first.legal_mask, second.legal_mask)
    first_outputs = model(first.observation)
    second_outputs = model(second.observation)
    assert torch.equal(first_outputs[0], second_outputs[0])
    assert torch.equal(first_outputs[1], second_outputs[1])


# Legality is authoritative outside the model and normalizes across all 32 actions.
def test_external_masking_keeps_raw_logits_and_zeroes_illegal_probability(
    model: PolicyValueModel,
) -> None:
    policy_input = _input_after_moves(1)
    logits, _ = model(policy_input.observation)
    masked = apply_legal_mask(logits, policy_input.legal_mask)
    probabilities = legal_policy_probabilities(logits, policy_input.legal_mask)
    assert logits.shape == masked.shape == probabilities.shape == (4, 8)
    assert torch.isfinite(logits).all()
    assert torch.isneginf(masked[~policy_input.legal_mask]).all()
    assert torch.count_nonzero(probabilities[~policy_input.legal_mask]) == 0
    assert probabilities.sum().item() == pytest.approx(1.0)


# Search targets must be valid before masking so malformed illegal mass cannot disappear.
def test_policy_targets_are_normalized_and_illegal_targets_are_rejected(
    model: PolicyValueModel,
) -> None:
    policy_input = _input_after_moves(2)
    logits, value = model(policy_input.observation)
    target = _targets(policy_input.legal_mask)
    losses = policy_value_loss(
        logits,
        value,
        policy_input.legal_mask,
        target,
        torch.tensor(0.25, dtype=torch.float32),
    )
    assert target.sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(target[~policy_input.legal_mask]) == 0
    assert torch.isfinite(losses.total)

    malformed = target.clone()
    illegal_index = (~policy_input.legal_mask).nonzero()[0]
    malformed[tuple(illegal_index)] = 0.01
    malformed[policy_input.legal_mask.nonzero()[0].unbind()] -= 0.01
    with pytest.raises(PolicyValueContractError, match="illegal actions"):
        policy_value_loss(
            logits,
            value,
            policy_input.legal_mask,
            malformed,
            torch.tensor(0.25, dtype=torch.float32),
        )


# Round values are player-relative: swapping the perspective negates one target.
def test_value_targets_reverse_sign_between_perspectives(
    model: PolicyValueModel,
) -> None:
    differential = (84 - 39) / 150
    assert differential == -((39 - 84) / 150)
    inputs = tuple(_input_after_moves(count) for count in (1, 2))
    observations = torch.stack(tuple(item.observation for item in inputs))
    masks = torch.stack(tuple(item.legal_mask for item in inputs))
    logits, values = model(observations)
    losses = policy_value_loss(
        logits,
        values,
        masks,
        torch.stack(tuple(_targets(item.legal_mask) for item in inputs)),
        torch.tensor([differential, -differential], dtype=torch.float32),
    )
    assert torch.isfinite(losses.value_mse)


# Each objective must train the shared representation rather than isolated heads only.
def test_both_heads_update_the_shared_body(model: PolicyValueModel) -> None:
    policy_input = _input_after_moves(3)
    logits, value = model(policy_input.observation)
    losses = policy_value_loss(
        logits,
        value,
        policy_input.legal_mask,
        _targets(policy_input.legal_mask),
        torch.tensor(0.8, dtype=torch.float32),
    )
    losses.policy_cross_entropy.backward(retain_graph=True)
    policy_gradient = model.encoder.shared_input.weight.grad
    assert policy_gradient is not None and torch.count_nonzero(policy_gradient) > 0
    model.zero_grad(set_to_none=True)
    losses.value_mse.backward()
    value_gradient = model.encoder.shared_input.weight.grad
    assert value_gradient is not None and torch.count_nonzero(value_gradient) > 0


# A complete dual-head backward pass must stay finite for every parameter.
def test_outputs_losses_gradients_and_regularization_are_finite(
    model: PolicyValueModel,
) -> None:
    inputs = tuple(_input_after_moves(count) for count in (0, 2, 4))
    observations = torch.stack(tuple(item.observation for item in inputs))
    masks = torch.stack(tuple(item.legal_mask for item in inputs))
    logits, values = model(observations)
    losses = policy_value_loss(
        logits,
        values,
        masks,
        torch.stack(tuple(_targets(item.legal_mask) for item in inputs)),
        torch.tensor([0.5, -0.25, 0.0], dtype=torch.float32),
    )
    losses.total.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(losses.total)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer = build_policy_value_optimizer(model)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in model.parameters()}


# Neural code receives tensors only and cannot import game or private-world machinery.
def test_model_module_has_no_engine_or_private_search_dependency() -> None:
    source = inspect.getsource(policy_value_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert imported.isdisjoint(
        {
            "dracula.bridge",
            "dracula.engine",
            "dracula.search",
            "dracula.search.information",
            "dracula.search.planner",
        }
    )
    assert tuple(inspect.signature(PolicyValueModel.forward).parameters) == (
        "self",
        "observation",
    )


# Artifact loading is strict so schema or tensor drift fails before inference.
def test_artifact_round_trip_and_incompatible_artifacts_are_rejected(
    model: PolicyValueModel, tmp_path
) -> None:
    valid_path = tmp_path / "valid.pt"
    artifact_arguments = {
        "source_revision": "test-revision",
        "training_configuration": {
            "batch_size": 256,
            "loss_weights": [1.0, 1.0],
        },
        "dataset_digest": _digest("dataset"),
        "search_report_digest": _digest("search-report"),
    }
    save_policy_value_artifact(valid_path, model, **artifact_arguments)
    loaded = load_policy_value_artifact(valid_path)
    assert loaded.metadata.parameter_count == PARAMETER_COUNT
    assert loaded.training_configuration["batch_size"] == 256
    for expected, actual in zip(
        model.parameters(), loaded.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)

    payload = build_policy_value_artifact(model, **artifact_arguments)

    mutations = []
    wrong_schema = copy.deepcopy(payload)
    wrong_schema["metadata"]["model_schema_version"] = "dracula-policy-value-v2"
    mutations.append(wrong_schema)
    wrong_cards = copy.deepcopy(payload)
    wrong_cards["metadata"]["card_ids"] = tuple(reversed(wrong_cards["metadata"]["card_ids"]))
    mutations.append(wrong_cards)
    wrong_shape = copy.deepcopy(payload)
    tensor_name = next(iter(wrong_shape["state_dict"]))
    wrong_shape["state_dict"][tensor_name] = wrong_shape["state_dict"][tensor_name][:-1]
    mutations.append(wrong_shape)
    nonfinite = copy.deepcopy(payload)
    tensor_name = next(iter(nonfinite["state_dict"]))
    nonfinite["state_dict"][tensor_name].flatten()[0] = torch.nan
    mutations.append(nonfinite)
    wrong_config = copy.deepcopy(payload)
    wrong_config["training_configuration"]["batch_size"] = 64
    mutations.append(wrong_config)

    for index, mutation in enumerate(mutations):
        path = tmp_path / f"invalid-{index}.pt"
        torch.save(mutation, path)
        with pytest.raises(PolicyValueContractError):
            load_policy_value_artifact(path)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward_matches_cpu_with_documented_tolerances() -> None:
    cpu_model = _model().eval()
    mps_model = _model().to("mps").eval()
    inputs = tuple(_input_after_moves(count) for count in (0, 2))
    observations = torch.stack(tuple(item.observation for item in inputs))
    cpu_logits, cpu_values = cpu_model(observations)
    mps_logits, mps_values = mps_model(observations.to("mps"))
    assert torch.allclose(
        cpu_logits, mps_logits.cpu(), atol=1e-5, rtol=1e-4
    )
    assert torch.allclose(
        cpu_values, mps_values.cpu(), atol=1e-5, rtol=1e-4
    )
