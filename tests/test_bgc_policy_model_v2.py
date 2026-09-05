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
    ROOT / "runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt"
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
    model = BGCPolicyModel(101)
    candidates, present = pack_hand_card_indices(observation)
    assert observation.shape == (OBSERVATION_SIZE,) == (659,)
    assert candidates.shape == present.shape == (4,)
    assert present.all()
    assert model(observation).shape == (4, 8)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 754_601


def test_direct_observations_match_the_current_golden_fixture() -> None:
    expected = (
        "3cb9cb465c9566555aa71416ab064d79b05bab4be29c710b4088a86a765df3eb",
        "81b43fb93c622ac61a85e4d13282e516ee88b60fbac70c27f62debd5139a18c4",
        "ddfeb9fd55a5ec2a69c6ef829613e1e179b2140e93b0161c761e857755b3e367",
        "9f4313f07b35964080579cd9c6854e81888ca83109d72048d51b9a8be5d5dd0b",
        "e0ff19b76989a875e6ec6216b9b7c7de93b43247e1bbdb3404b8aebf149a1172",
        "d971cac655467084400af1a88bc2da4775ca69c3e73719997d688c93ed7a4a12",
        "d3fa24694a052af3e6be50385b516650544cc77dac403f2c09cb8cbaceec9e21",
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
    first = BGCPolicyModel(202)
    second = BGCPolicyModel(202)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not any("slot" in name for name, _ in first.named_parameters())


def test_batch_equivalence_gradients_and_action_round_trip() -> None:
    information = _information()
    observation = encode_policy_observation(information)
    model = BGCPolicyModel(303)
    individual = model(observation)
    batched = model(torch.stack((observation, observation)))
    assert torch.allclose(individual, batched[0], atol=1e-5, rtol=1e-5)
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
def test_selected_artifact_has_frozen_logits_and_strategic_actions() -> None:
    expected = (
        ("733fbd2a7d55fe157f2c7fbb38bad4b52b5cff7b2ed52a117774e628b75a0ab7", 17),
        ("183f49e3722ed0a5902cb8d35239cd59b282698f149493b75fc129fa94261bfb", 16),
        ("ac3a86029aef30fb3158b92b07312f8e5335238315efee712fb873812a1428b7", 22),
        ("32a9cefe536933db8d42f60a5db34d87da993a07fa0e6062a76da815263f97bb", 20),
        ("05aaca655ff9647f499734d7d217aef22bdb3cbdbda092af56443e120c8e0cdd", 29),
        ("72bdd69887d78a37bfb8c65fde536e2afbbe4efba51bc0a2095c61d34e13b2b5", 20),
        ("e77b91a4b62449ecf5a341f0cd01662eeb44699b70e185683489eb474a6f33f6", 31),
    )
    runtime = ActivePolicyRuntime.from_artifact(SELECTED_ARTIFACT)
    state = create_game("pi1-direct-queen")
    for placement, (logits_digest, representative) in enumerate(
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
        assert decision.concrete_action_index in next(
            group.member_action_indices
            for group in strategic_action_groups(information)
            if group.representative_action_index == representative
        )
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
