"""Mechanical acceptance for the standalone Sam-32 policy classifier."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import dracula.sam_policy as sam_policy_module
from dracula.bridge import (
    PolicyInput,
    build_policy_turn_context,
    global_grid_index,
    transpose_grid_index,
)
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.policy_value import (
    PARAMETER_COUNT as HISTORICAL_PARAMETER_COUNT,
    PolicyValueContractError,
    PolicyValueModel,
    build_policy_value_artifact,
    load_policy_value_artifact,
)
from dracula.sam_policy import (
    INFERENCE_DESTINATION_SCOPE,
    PARAMETER_COUNT,
    POLICY_GRID_INDICES,
    RepresentativeActionProjection,
    SamPolicyContractError,
    SamPolicyModel,
    apply_representative_mask,
    build_representative_action_projection,
    build_sam_policy_artifact,
    load_sam_policy_artifact,
    representative_policy_probabilities,
    resolve_representative_action,
    save_sam_policy_artifact,
    select_representative_action,
)
from dracula.search import (
    PublicGameHistory,
    SamTeacherSearchConfig,
    build_information_state,
    derive_sam_teacher_request_seed,
    derive_strategic_destination_choice_seed,
    information_state_from_engine,
    policy_input_from_information_state,
    public_history_from_engine,
    sam_teacher_action_groups,
    select_concrete_action_index,
)


AUTHORIZED_CASES = (
    ({5}, (2, 4)),
    ({2, 5}, (1, 4, 8)),
    ({5, 8}, (2, 4, 9)),
    ({4, 5}, (1, 2, 6)),
    ({5, 6}, (2, 3, 4)),
    ({4, 5, 6}, (1, 2, 3)),
    ({2, 5, 8}, (1, 4, 7)),
)


def _model(ordinal: int = 0) -> SamPolicyModel:
    return SamPolicyModel(
        run_root_seed="sam-policy-test-root",
        model_id="sam-policy-test",
        initialization_ordinal=ordinal,
    )


def _apply_relative_destination(state, perspective, position: int):
    actor = state.active_player
    assert actor is not None
    hand_slot = next(
        index
        for index, card_id in enumerate(state.hands[actor])
        if card_id is not None
    )
    move = EngineMove(
        actor,
        hand_slot,
        global_grid_index(perspective, position - 1),
    )
    assert move in legal_moves(state, actor)
    return apply_move(state, move).state


def _state_for_pattern(occupied: set[int], seed: str):
    state = create_game(seed)
    non_center = sorted(occupied - {5})
    if not non_center:
        return state
    if len(non_center) == 1:
        return _apply_relative_destination(
            state, state.dealer, non_center[0]
        )
    perspective = state.active_player
    for position in non_center:
        state = _apply_relative_destination(
            state, perspective, position
        )
    return state


def _projection_for_state(state) -> tuple:
    information = information_state_from_engine(state)
    policy_input = policy_input_from_information_state(information)
    projection = build_representative_action_projection(
        policy_input.legal_mask,
        sam_teacher_action_groups(information),
    )
    return information, policy_input, projection


def _proxy_destinations(
    projection: RepresentativeActionProjection,
    hand_slot: int,
) -> tuple[int, ...]:
    return tuple(
        POLICY_GRID_INDICES[position] + 1
        for position, enabled in enumerate(
            projection.mask[hand_slot].tolist()
        )
        if enabled
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def model() -> SamPolicyModel:
    return _model()


# Fixed tensor and parameter counts prevent silent architecture drift.
def test_exact_single_batch_shapes_and_parameter_count(
    model: SamPolicyModel,
) -> None:
    first = _projection_for_state(
        create_game("sam-policy-shape-a")
    )[1]
    second = _projection_for_state(
        _state_for_pattern({2, 5}, "sam-policy-shape-b")
    )[1]

    single_logits = model(first.observation)
    batch = torch.stack(
        (first.observation, second.observation)
    )
    batch_logits = model(batch)

    assert first.observation.shape == (875,)
    assert first.legal_mask.shape == (4, 8)
    assert single_logits.shape == (4, 8)
    assert batch.shape == (2, 875)
    assert batch_logits.shape == (2, 4, 8)
    assert single_logits.dtype is torch.float32
    assert batch_logits.dtype is torch.float32
    assert sum(
        parameter.numel() for parameter in model.parameters()
    ) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 738_569


# Initialization identity determines every parameter without consuming the
# process-wide PyTorch random stream.
def test_initialization_is_deterministic_and_rng_isolated() -> None:
    torch.manual_seed(7309)
    expected_next = torch.rand(5)
    torch.manual_seed(7309)
    first = _model(0)
    actual_next = torch.rand(5)
    repeated = _model(0)
    different = _model(1)

    assert torch.equal(actual_next, expected_next)
    for left, right in zip(
        first.parameters(), repeated.parameters(), strict=True
    ):
        assert torch.equal(left, right)
    assert any(
        not torch.equal(left, right)
        for left, right in zip(
            first.parameters(), different.parameters(), strict=True
        )
    )
    for module in first.modules():
        if isinstance(module, nn.Linear):
            fan_in, fan_out = (
                nn.init._calculate_fan_in_and_fan_out(module.weight)
            )
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            assert torch.max(torch.abs(module.weight)) <= bound
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.LayerNorm):
            assert module.eps == pytest.approx(1e-5)
            assert torch.equal(
                module.weight, torch.ones_like(module.weight)
            )
            assert torch.count_nonzero(module.bias) == 0


# Vectorized execution must equal independent execution of every state.
def test_batch_and_individual_inference_are_equivalent(
    model: SamPolicyModel,
) -> None:
    inputs = tuple(
        _projection_for_state(
            _state_for_pattern(
                occupied,
                f"sam-policy-batch-{index}",
            )
        )[1]
        for index, occupied in enumerate(
            ({5}, {2, 5}, {4, 5, 6}, {2, 3, 5})
        )
    )
    batch = torch.stack(
        tuple(value.observation for value in inputs)
    )
    batch_logits = model(batch)
    for index, policy_input in enumerate(inputs):
        assert torch.allclose(
            model(policy_input.observation),
            batch_logits[index],
            atol=1e-5,
            rtol=1e-4,
        )


# Queen/King coordinate normalization erases role-specific orientation before
# the standalone classifier sees a tensor.
def test_queen_and_king_normalized_inputs_are_equivalent(
    model: SamPolicyModel,
) -> None:
    state = create_game("engine-contract-fixture-1")
    queen_context = build_policy_turn_context(
        state, EnginePlayer.QUEEN
    )
    queen_history = public_history_from_engine(state)
    queen_information = build_information_state(
        queen_context.input,
        queen_history,
        EnginePlayer.QUEEN,
    )

    mirrored = [None] * 9
    for index, card_id in enumerate(
        queen_history.current_coffin
    ):
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
    queen_input = policy_input_from_information_state(
        queen_information
    )
    king_input = policy_input_from_information_state(
        king_information
    )
    assert torch.equal(
        queen_input.observation, king_input.observation
    )
    assert torch.equal(
        model(queen_input.observation),
        model(king_input.observation),
    )


# A complete standalone backward pass must keep every output and gradient
# finite while training only the shared body and pair head.
def test_outputs_and_gradients_are_finite(
    model: SamPolicyModel,
) -> None:
    values = tuple(
        _projection_for_state(
            _state_for_pattern(
                occupied,
                f"sam-policy-gradient-{index}",
            )
        )
        for index, occupied in enumerate(
            ({5}, {2, 5}, {4, 5, 6})
        )
    )
    observations = torch.stack(
        tuple(value[1].observation for value in values)
    )
    masks = torch.stack(
        tuple(value[2].mask for value in values)
    )
    logits = model(observations)
    masked = apply_representative_mask(logits, masks)
    targets = torch.tensor(
        [
            value[2].groups[0].representative_action_index
            for value in values
        ],
        dtype=torch.long,
    )
    loss = F.cross_entropy(
        masked.flatten(start_dim=1), targets
    )
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert torch.count_nonzero(
        model.encoder.shared_input.weight.grad
    )
    assert torch.count_nonzero(model.pair_hidden.weight.grad)


# Every listed board exposes only its documented destination proxies,
# independently for every remaining hand card.
@pytest.mark.parametrize(
    ("occupied", "expected_destinations"),
    AUTHORIZED_CASES,
)
def test_every_authoritative_pattern_uses_exact_proxies(
    occupied: set[int],
    expected_destinations: tuple[int, ...],
) -> None:
    information, policy_input, projection = _projection_for_state(
        _state_for_pattern(
            occupied,
            "engine-contract-fixture-1",
        )
    )
    legal_slots = tuple(
        index
        for index, card_id in enumerate(information.own_hand)
        if card_id is not None
    )
    for hand_slot in legal_slots:
        assert (
            _proxy_destinations(projection, hand_slot)
            == expected_destinations
        )
    assert int(projection.mask.sum().item()) == (
        len(legal_slots) * len(expected_destinations)
    )
    assert torch.all(
        projection.mask <= policy_input.legal_mask
    )


# The opening reduction is exactly two destinations per card: positions 2 and 4.
def test_center_only_exposes_destinations_two_and_four_per_card() -> None:
    information, _input, projection = _projection_for_state(
        create_game("sam-policy-center-only")
    )
    assert all(
        _proxy_destinations(projection, hand_slot) == (2, 4)
        for hand_slot, card_id in enumerate(information.own_hand)
        if card_id is not None
    )


# Any occupied pattern outside the exhaustive table keeps every concrete
# engine-legal action rather than inventing another equivalence.
def test_unlisted_pattern_retains_every_engine_legal_action() -> None:
    _information, policy_input, projection = (
        _projection_for_state(
            _state_for_pattern(
                {2, 3, 5}, "sam-policy-unlisted"
            )
        )
    )
    assert torch.equal(
        projection.mask, policy_input.legal_mask
    )
    assert all(
        len(group.member_action_indices) == 1
        for group in projection.groups
    )


# Card rank, suit, color, and identity cannot alter a spatial proxy table.
@pytest.mark.parametrize(
    ("occupied", "expected_destinations"),
    AUTHORIZED_CASES,
)
def test_card_substitutions_cannot_change_destination_grouping(
    occupied: set[int],
    expected_destinations: tuple[int, ...],
) -> None:
    first = _projection_for_state(
        _state_for_pattern(occupied, "sam-policy-cards-a")
    )
    second = _projection_for_state(
        _state_for_pattern(occupied, "sam-policy-cards-b")
    )
    assert first[0].own_hand != second[0].own_hand
    assert tuple(
        _proxy_destinations(first[2], slot)
        for slot, card in enumerate(first[0].own_hand)
        if card is not None
    ) == tuple(
        expected_destinations
        for _ in range(
            sum(card is not None for card in first[0].own_hand)
        )
    )
    assert tuple(
        _proxy_destinations(second[2], slot)
        for slot, card in enumerate(second[0].own_hand)
        if card is not None
    ) == tuple(
        expected_destinations
        for _ in range(
            sum(card is not None for card in second[0].own_hand)
        )
    )


# Symmetry reduces destinations only; separate hand slots remain separate
# classifier actions.
def test_distinct_hand_cards_remain_distinct_actions() -> None:
    information, _input, projection = _projection_for_state(
        create_game("sam-policy-distinct-cards")
    )
    slots = {
        slot
        for slot, card in enumerate(information.own_hand)
        if card is not None
    }
    assert slots == {0, 1, 2, 3}
    assert {group.hand_slot for group in projection.groups} == slots
    assert all(
        sum(group.hand_slot == slot for group in projection.groups)
        == 2
        for slot in slots
    )


# Each enabled proxy has one and only one complete legal strategic group.
def test_representative_maps_to_exactly_one_group() -> None:
    _information, policy_input, projection = (
        _projection_for_state(
            _state_for_pattern(
                {5, 8}, "sam-policy-representative-map"
            )
        )
    )
    all_members = []
    for representative in projection.mask.flatten().nonzero().flatten():
        group = projection.group_for(int(representative.item()))
        assert group.representative_action_index == representative
        assert all(
            bool(policy_input.legal_mask.flatten()[member].item())
            for member in group.member_action_indices
        )
        all_members.extend(group.member_action_indices)
    assert sorted(all_members) == sorted(
        policy_input.legal_mask.flatten().nonzero().flatten().tolist()
    )


# The standalone concrete resolver is exactly the established strategic fair
# coin and reaches both members of every authorized pair.
@pytest.mark.parametrize(
    "occupied",
    tuple(case[0] for case in AUTHORIZED_CASES),
)
def test_fair_coin_matches_existing_resolution_and_reaches_both_members(
    occupied: set[int],
) -> None:
    information, _input, projection = _projection_for_state(
        _state_for_pattern(
            occupied, "engine-contract-fixture-1"
        )
    )
    request_seed = derive_sam_teacher_request_seed(
        information, SamTeacherSearchConfig().digest
    )
    paired = tuple(
        group
        for group in projection.groups
        if len(group.member_action_indices) == 2
    )
    assert paired
    for group in paired:
        choices = set()
        for choice_index in range(64):
            choice_seed = derive_strategic_destination_choice_seed(
                request_seed,
                INFERENCE_DESTINATION_SCOPE,
                information,
                group.representative_action_index,
                choice_index,
            )
            resolved = resolve_representative_action(
                projection,
                group.representative_action_index,
                choice_seed,
            )
            assert resolved == select_concrete_action_index(
                information,
                group.representative_action_index,
                choice_seed,
            )
            choices.add(resolved)
        assert choices == set(group.member_action_indices)


# Concrete coin outcomes cannot feed back into neural logits or change the
# already selected strategic representative.
def test_fair_coin_cannot_change_logits_or_selected_group(
    model: SamPolicyModel,
) -> None:
    information, policy_input, projection = (
        _projection_for_state(
            create_game("sam-policy-coin-isolation")
        )
    )
    paired = next(
        group
        for group in projection.groups
        if len(group.member_action_indices) == 2
    )
    logits = model(policy_input.observation)
    before = logits.clone()
    request_seed = derive_sam_teacher_request_seed(
        information, SamTeacherSearchConfig().digest
    )
    for choice_index in range(16):
        choice_seed = derive_strategic_destination_choice_seed(
            request_seed,
            INFERENCE_DESTINATION_SCOPE,
            information,
            paired.representative_action_index,
            choice_index,
        )
        resolve_representative_action(
            projection,
            paired.representative_action_index,
            choice_seed,
        )
        assert (
            projection.group_for(
                paired.representative_action_index
            )
            == paired
        )
    assert torch.equal(logits, before)


# External masking gives exactly zero probability to engine-illegal actions and
# paired non-proxy members while preserving unit probability mass.
def test_illegal_and_non_proxy_logits_have_zero_probability(
    model: SamPolicyModel,
) -> None:
    _information, policy_input, projection = (
        _projection_for_state(
            create_game("sam-policy-mask")
        )
    )
    logits = model(policy_input.observation)
    masked = apply_representative_mask(logits, projection.mask)
    probabilities = representative_policy_probabilities(
        logits, projection.mask
    )
    non_proxy_legal = policy_input.legal_mask & ~projection.mask
    assert non_proxy_legal.any()
    assert torch.isneginf(masked[~projection.mask]).all()
    assert (
        torch.count_nonzero(probabilities[~projection.mask]) == 0
    )
    assert (
        torch.count_nonzero(probabilities[non_proxy_legal]) == 0
    )
    assert probabilities.sum().item() == pytest.approx(1.0)
    selected = select_representative_action(
        logits, projection.mask
    )
    assert isinstance(selected, int)
    assert bool(projection.mask.flatten()[selected].item())


# The neural module accepts tensors and structural action groups only; private
# engine and search machinery cannot enter its imports or forward signature.
def test_model_module_has_no_private_engine_or_search_dependency() -> None:
    source = inspect.getsource(sam_policy_module)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert imported.isdisjoint(
        {
            "dracula.bridge",
            "dracula.engine",
            "dracula.search",
            "dracula.search.information",
            "dracula.search.planner",
            "dracula.search.strategic",
            "dracula.search.symmetry",
        }
    )
    assert tuple(
        inspect.signature(SamPolicyModel.forward).parameters
    ) == ("self", "observation")
    assert all(
        token not in source
        for token in (
            "opponent_hand",
            "stock_order",
            "determinization",
            "engine_seed",
            "search_tree",
            "line_score",
            "round_score",
        )
    )


# Artifact loading binds every schema and rejects wrong shapes, non-finite
# tensors, metadata drift, and historical policy/value payloads.
def test_artifact_round_trip_and_strict_compatibility(
    model: SamPolicyModel, tmp_path
) -> None:
    arguments = {
        "source_revision": "a" * 40,
        "source_tree_digest": _digest("source-tree"),
        "training_configuration": {
            "batch_size": 256,
            "loss": "masked-cross-entropy",
        },
        "corpus_snapshot_digest": _digest("snapshot"),
    }
    valid = tmp_path / "sam-policy.pt"
    save_sam_policy_artifact(valid, model, **arguments)
    loaded = load_sam_policy_artifact(valid)
    assert loaded.metadata.parameter_count == PARAMETER_COUNT
    assert not any(
        name.startswith(("value", "critic"))
        for name in loaded.model.state_dict()
    )
    for expected, actual in zip(
        model.parameters(), loaded.model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)

    payload = build_sam_policy_artifact(model, **arguments)
    mutations = []
    wrong_schema = copy.deepcopy(payload)
    wrong_schema["metadata"]["model_schema_version"] = (
        "dracula-sam-policy-v2"
    )
    mutations.append(wrong_schema)
    wrong_mask = copy.deepcopy(payload)
    wrong_mask["metadata"][
        "representative_mask_schema_version"
    ] = "dracula-sam-representative-mask-v2"
    mutations.append(wrong_mask)
    wrong_shape = copy.deepcopy(payload)
    tensor_name = next(iter(wrong_shape["state_dict"]))
    wrong_shape["state_dict"][tensor_name] = (
        wrong_shape["state_dict"][tensor_name][:-1]
    )
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
        with pytest.raises(SamPolicyContractError):
            load_sam_policy_artifact(path)

    historical = PolicyValueModel(
        run_root_seed="historical-artifact",
        model_id="historical-artifact",
        initialization_ordinal=0,
    )
    historical_payload = build_policy_value_artifact(
        historical,
        source_revision="historical-test",
        training_configuration={"batch_size": 2},
        dataset_digest=_digest("historical-dataset"),
        search_report_digest=_digest("historical-search"),
    )
    historical_path = tmp_path / "historical.pt"
    torch.save(historical_payload, historical_path)
    with pytest.raises(SamPolicyContractError):
        load_sam_policy_artifact(historical_path)
    with pytest.raises(PolicyValueContractError):
        load_policy_value_artifact(valid)


# Adding and initializing the standalone model cannot alter the frozen
# historical PolicyValueModel construction or outputs.
def test_historical_policy_value_model_remains_unchanged() -> None:
    arguments = {
        "run_root_seed": "historical-regression",
        "model_id": "historical-regression",
        "initialization_ordinal": 2,
    }
    before = PolicyValueModel(**arguments)
    _model()
    after = PolicyValueModel(**arguments)
    observation = _projection_for_state(
        create_game("historical-regression-input")
    )[1].observation
    before_outputs = before(observation)
    after_outputs = after(observation)

    assert HISTORICAL_PARAMETER_COUNT == 339_978
    for left, right in zip(
        before.parameters(), after.parameters(), strict=True
    ):
        assert torch.equal(left, right)
    assert torch.equal(before_outputs[0], after_outputs[0])
    assert torch.equal(before_outputs[1], after_outputs[1])
