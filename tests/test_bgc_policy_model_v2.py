"""Verify the 659-bit observation and standalone policy model contract.

Tests protect candidate-card packing, permutation invariance, exact architecture,
role equivalence, row translation, deterministic initialization, and frozen π1
logits and actions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from dracula.action_contract import build_representative_action_mask
from dracula.active_policy import ActivePolicyRuntime
from dracula.bgc_policy_model import (
    BGCPolicyModel,
    OBSERVATION_SIZE,
    PARAMETER_COUNT,
    pack_hand_card_indices,
)
from dracula.bridge import move_for_action_index
from dracula.cards import CARD_COUNT, CARD_INDEX_BY_ID
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    PlayerValues,
    SimulationEngineState,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.engine_dealing import as_hand, deal_round, shuffled_deck, starting_coffin
from dracula.policy_observation import (
    PolicyObservationError,
    encode_policy_observation,
    engine_action_index_from_model,
    pack_action_rows_for_model,
)
from dracula.search.information import (
    canonical_information_data,
    information_state_from_engine,
)
from dracula.strategic_actions import strategic_action_groups


ROOT = Path(__file__).resolve().parents[1]
SELECTED_ARTIFACT = (
    ROOT / "runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt"
)


def _information():
    state = create_game("bgc-card-policy-v2-fixture")
    return information_state_from_engine(state)


def _simulation_state(
    simulation_deck: tuple[str, ...], dealer: EnginePlayer
) -> SimulationEngineState:
    deal = deal_round(simulation_deck, dealer)
    return SimulationEngineState(
        seed="policy-role-equivalence",
        status=EngineStatus.PLAYING,
        round_number=1,
        dealer=dealer,
        active_player=deal.non_dealer,
        stock=deal.remaining_stock,
        hands=PlayerValues(
            queen=as_hand(deal.queen_hand),
            king=as_hand(deal.king_hand),
        ),
        coffin=starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=(),
        total_scores=PlayerValues(queen=0, king=0),
        simulation_deck=simulation_deck,
    )


def test_direct_card_set_shapes_and_exact_parameter_count() -> None:
    observation = encode_policy_observation(_information())
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-test",
        model_id="pi1",
        initialization_ordinal=0,
    )
    candidates, present = pack_hand_card_indices(observation)
    assert observation.shape == (OBSERVATION_SIZE,) == (659,)
    assert candidates.shape == present.shape == (4,)
    assert present.all()
    assert model(observation).shape == (4, 8)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 754_601


def test_direct_observations_match_frozen_pre_cleanup_fixtures() -> None:
    expected = (
        "880dbad90758315bec5da197a6932f9ae949f06294c73f829283c6c5f668fe4a",
        "7f97a139f89a3fb85b1dda482f0f3f65b844c70222cf06c73162329effb2e639",
        "6df85b298e1d0b82205761dd1352b9b3615da510d6cdd71931a8b84f64ecc591",
        "118a287940067fa0845cb39692ebec4632e52a51c242088310b86b77e8c2421d",
        "6506b744e76c1e71a2aa7b77b94bbd550bec1accbcfa053e915ff8e6b6605a22",
        "4c3e073d0102ccaa1890fbb803238f1b5608dee2fc12d471fdfbba458c595b75",
        "ce37a0d1e5a8a76748d74eb22ef5c98b9d37a849bd40b8c91ec7255b3b76cdf2",
    )
    state = create_game("pi1-direct-queen")
    actual = []
    for _ in range(7):
        observation = encode_policy_observation(information_state_from_engine(state))
        actual.append(hashlib.sha256(observation.numpy().tobytes()).hexdigest())
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    assert tuple(actual) == expected


def test_model_initialization_is_deterministic_and_has_no_slot_embeddings() -> None:
    torch.manual_seed(17)
    global_state = torch.random.get_rng_state().clone()
    first = BGCPolicyModel(
        run_root_seed="bgc-card-model-test", model_id="pi1", initialization_ordinal=0
    )
    second = BGCPolicyModel(
        run_root_seed="bgc-card-model-test", model_id="pi1", initialization_ordinal=0
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not any("slot" in name for name, _ in first.named_parameters())


def test_batch_equivalence_gradients_and_action_round_trip() -> None:
    information = _information()
    observation = encode_policy_observation(information)
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-batch-test",
        model_id="pi1",
        initialization_ordinal=0,
    )
    individual = model(observation)
    batched = model(torch.stack((observation, observation)))
    assert torch.allclose(individual, batched[0], atol=1e-6, rtol=1e-6)
    assert torch.equal(batched[0], batched[1])
    batched.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    legal = torch.tensor(information.legal_mask, dtype=torch.bool)
    compact = pack_action_rows_for_model(information, legal)
    for compact_index in torch.nonzero(
        compact.flatten(), as_tuple=False
    ).flatten().tolist():
        engine_index = engine_action_index_from_model(information, compact_index)
        selected = torch.zeros_like(legal)
        selected.flatten()[engine_index] = True
        remapped = pack_action_rows_for_model(information, selected)
        assert torch.nonzero(remapped.flatten()).item() == compact_index


def test_engine_and_candidate_action_rows_round_trip_as_hands_shrink() -> None:
    state = create_game("candidate-action-shrinking-hands")
    slot_order = (1, 3, 0, 2)
    decisions = {player: 0 for player in EnginePlayer}

    for placement in range(8):
        information = information_state_from_engine(state)
        observation = encode_policy_observation(information)
        candidate_cards, candidate_is_present = pack_hand_card_indices(observation)
        engine_mask = torch.tensor(information.legal_mask, dtype=torch.bool)
        compact_mask = pack_action_rows_for_model(information, engine_mask)
        occupied_slots = tuple(
            slot
            for slot, card_id in enumerate(information.own_hand)
            if card_id is not None
        )
        expected_cards = tuple(
            CARD_INDEX_BY_ID[card_id]
            for card_id in information.own_hand
            if card_id is not None
        )
        assert (
            tuple(candidate_cards[candidate_is_present].tolist())
            == expected_cards
        )
        assert torch.all(candidate_cards[~candidate_is_present] == CARD_COUNT)
        assert not compact_mask[len(occupied_slots) :].any()
        for engine_index in torch.nonzero(
            engine_mask.flatten(), as_tuple=False
        ).flatten().tolist():
            engine_row, destination = divmod(engine_index, 8)
            candidate_index = occupied_slots.index(engine_row) * 8 + destination
            assert compact_mask.flatten()[candidate_index]
            assert (
                engine_action_index_from_model(information, candidate_index)
                == engine_index
            )
        if len(occupied_slots) < 4:
            with pytest.raises(PolicyObservationError, match="padding row"):
                engine_action_index_from_model(information, len(occupied_slots) * 8)

        if placement < 7:
            assert state.active_player is not None
            player = state.active_player
            selected_slot = slot_order[decisions[player]]
            decisions[player] += 1
            move = next(
                move
                for move in legal_moves(state, player)
                if move.hand_slot == selected_slot
            )
            state = apply_move(state, move).state


def test_queen_and_king_relative_views_have_identical_policy_semantics() -> None:
    deck = shuffled_deck("policy-role-equivalent-deck")
    queen_actor_state = _simulation_state(deck, EnginePlayer.KING)
    king_actor_state = _simulation_state(deck, EnginePlayer.QUEEN)

    for _ in range(2):
        queen_information = information_state_from_engine(queen_actor_state)
        king_information = information_state_from_engine(king_actor_state)
        assert queen_information.player is other_player(king_information.player)
        assert canonical_information_data(queen_information) == (
            canonical_information_data(king_information)
        )
        assert torch.equal(
            encode_policy_observation(queen_information),
            encode_policy_observation(king_information),
        )
        queen_groups = strategic_action_groups(queen_information)
        king_groups = strategic_action_groups(king_information)
        assert queen_groups == king_groups
        queen_mask = build_representative_action_mask(
            torch.tensor(queen_information.legal_mask, dtype=torch.bool),
            queen_groups,
        )
        king_mask = build_representative_action_mask(
            torch.tensor(king_information.legal_mask, dtype=torch.bool),
            king_groups,
        )
        assert torch.equal(
            pack_action_rows_for_model(queen_information, queen_mask),
            pack_action_rows_for_model(king_information, king_mask),
        )

        if queen_information.player is EnginePlayer.QUEEN:
            queen_actor_state = apply_simulation_move(
                queen_actor_state,
                move_for_action_index(queen_actor_state.active_player, 1),
            )
            king_actor_state = apply_simulation_move(
                king_actor_state,
                move_for_action_index(king_actor_state.active_player, 1),
            )


def test_policy_observation_rejects_values_at_its_external_boundary() -> None:
    information = _information()
    with pytest.raises(PolicyObservationError, match="SearchInformationState"):
        encode_policy_observation(object())  # type: ignore[arg-type]
    with pytest.raises(PolicyObservationError, match=r"shape \[4,8\]"):
        pack_action_rows_for_model(information, torch.zeros(3, 8, dtype=torch.bool))


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
def test_selected_artifact_matches_pre_cleanup_logits_and_actions() -> None:
    expected = (
        ("48724f8ae407964c2db1686e7c30d717cc21772042d9457eae4055a2ac38ab82", 17, 22),
        ("5f17692aaca65b3d9e3bd96e53e7caf04335bc78e58d997c37f1808d0d9f85e2", 22, 22),
        ("b0dbf34005820e0381fd489431a265971d285995f2cb681d9392b50b4219fe26", 29, 29),
        ("e908f1b3e1e257ac387ae773f22c9a650dc499201f5c8333dc7a464062c101ac", 22, 22),
        ("973304b77219ed992a627450641ce55cdb91049f03b6c5f8e5b3871cef3cc380", 18, 18),
        ("75fe4716c73cbaa03fa24feb0c01c4b0eb3b55f7312b25057e78d4fdf92b5784", 22, 22),
        ("c5cefaa98d891a134deae5e7ddd9357c66b4f0c2a7c75ae25acf743aa1f0dcd9", 31, 31),
    )
    runtime = ActivePolicyRuntime.from_artifact(SELECTED_ARTIFACT)
    state = create_game("pi1-direct-queen")
    for placement, (logits_digest, representative, concrete) in enumerate(
        expected, start=1
    ):
        information = information_state_from_engine(state)
        with torch.inference_mode():
            logits = runtime.model(encode_policy_observation(information))
        decision = runtime.decide(
            information,
            fixture_id="pi1-direct-queen",
            decision_index=placement,
        )
        assert hashlib.sha256(logits.numpy().tobytes()).hexdigest() == logits_digest
        assert decision.representative_action_index == representative
        assert decision.concrete_action_index == concrete
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
