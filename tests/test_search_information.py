"""Information-boundary and hidden-card sampling contract tests."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from dracula.bridge import PolicyInput, build_policy_turn_context, transpose_grid_index
from dracula.cards import CARD_IDS, sort_card_ids
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    InvalidLifecycleTransition,
    SimulationEngineState,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
    validate_state,
)
from dracula.randomness import derive_seed
from dracula.search import (
    BELIEF_SAMPLE_NAMESPACE,
    ROLLOUT_CHOICE_NAMESPACE,
    SEARCH_REQUEST_NAMESPACE,
    TREE_SELECTION_NAMESPACE,
    PublicGameHistory,
    PublicPlayedMove,
    build_information_state,
    derive_belief_sample_seed,
    derive_rollout_choice_seed,
    derive_search_request_seed,
    derive_tree_selection_seed,
    information_state_fingerprint,
    information_state_from_engine,
    information_state_from_simulation,
    public_history_from_engine,
    sample_determinization,
    sample_uniform_action_index,
)


def _first_move(state):
    return legal_moves(state, state.active_player)[0]


def _advance_authoritative(state):
    return apply_move(state, _first_move(state)).state


def _sample_seed(label: str) -> bytes:
    return derive_seed("dracula-search-test-sample-v1", label)


def _owned_cards(state: SimulationEngineState) -> tuple[str, ...]:
    cards = list(state.stock)
    cards.extend(
        card
        for hand in (state.hands.queen, state.hands.king)
        for card in hand
        if card is not None
    )
    cards.extend(card for card in state.coffin if card is not None)
    cards.extend(card for result in state.completed_rounds for card in result.coffin)
    return tuple(cards)


# The root contract contains immutable observations, never mutable tensors or private locations.
def test_information_state_is_immutable_and_reconstructs_the_complete_visible_view() -> None:
    state = create_game("engine-contract-fixture-1")
    state = _advance_authoritative(state)
    information = information_state_from_engine(state)
    player = state.active_player
    assert player is not None

    assert information.player is player
    assert information.dealer is state.dealer
    assert information.round_number == state.round_number
    assert information.turn_number == len(state.current_round_moves) + 1
    assert information.own_hand == state.hands[player]
    assert set(information.played_card_ids) == {
        card for card in state.coffin if card is not None
    }
    assert set(information.unseen_card_ids) == {
        card for card in (*state.hands[other_player(player)], *state.stock) if card is not None
    }
    assert not hasattr(information, "opponent_hand")
    assert not hasattr(information, "stock")
    assert all("slot" not in field.name for field in fields(PublicPlayedMove))
    assert {field.name for field in fields(PublicGameHistory)}.isdisjoint(
        {"seed", "stock", "hands", "opponent_hand"}
    )
    with pytest.raises(FrozenInstanceError):
        information.round_number = 2  # type: ignore[misc]


# Every sampled state must remain a complete physical deck accepted by the rule engine.
@pytest.mark.parametrize("label", tuple(str(index) for index in range(12)))
def test_samples_conserve_every_card_and_preserve_visible_history(label: str) -> None:
    state = create_game("engine-contract-fixture-1")
    for _ in range(5):
        state = _advance_authoritative(state)
    information = information_state_from_engine(state)
    sampled = sample_determinization(information, _sample_seed(label))

    validate_state(sampled.state)
    owned = _owned_cards(sampled.state)
    assert len(owned) == len(set(owned)) == len(CARD_IDS)
    assert set(owned) == set(CARD_IDS)
    assert sampled.state.coffin == state.coffin
    assert tuple(
        (move.player, move.card_id, move.global_grid_index, move.turn_number)
        for move in sampled.state.current_round_moves
    ) == tuple(
        (move.player, move.card_id, move.global_grid_index, move.turn_number)
        for move in state.current_round_moves
    )
    assert information_state_from_simulation(sampled.state) == information


# A fixed information state and seed are a complete reproducibility boundary.
def test_equal_information_and_seed_reproduce_the_identical_sample() -> None:
    information = information_state_from_engine(create_game("engine-contract-fixture-1"))
    seed = _sample_seed("repeat")
    first = sample_determinization(information, seed)
    repeated = sample_determinization(information, seed)
    assert first == repeated
    assert first.state.simulation_deck == repeated.state.simulation_deck


# Rearranging cards between hidden locations cannot alter the view, its digest, or later samples.
def test_indistinguishable_hidden_worlds_have_identical_fingerprints_and_distributions() -> None:
    root = information_state_from_engine(create_game("engine-contract-fixture-1"))
    hidden_a = sample_determinization(root, _sample_seed("hidden-a")).state
    hidden_b = sample_determinization(root, _sample_seed("hidden-b")).state
    assert hidden_a.hands[other_player(root.player)] != hidden_b.hands[other_player(root.player)]
    assert hidden_a.stock != hidden_b.stock
    assert hidden_a.seed != hidden_b.seed

    view_a = information_state_from_engine(hidden_a)
    view_b = information_state_from_engine(hidden_b)
    assert view_a == view_b == root
    assert information_state_fingerprint(view_a) == information_state_fingerprint(view_b)

    resample_seed = _sample_seed("common-resample")
    assert sample_determinization(view_a, resample_seed) == sample_determinization(
        view_b, resample_seed
    )


# Queen and King use one normalized geometry and relative-role fingerprint.
def test_queen_and_king_transpose_equivalent_views_share_a_fingerprint() -> None:
    state = create_game("engine-contract-fixture-1")
    assert state.active_player is EnginePlayer.QUEEN
    queen_context = build_policy_turn_context(state, EnginePlayer.QUEEN)
    queen_history = public_history_from_engine(state)
    queen_information = build_information_state(
        queen_context.input, queen_history, EnginePlayer.QUEEN
    )

    mirrored_coffin = [None] * 9
    for index, card in enumerate(queen_history.current_coffin):
        mirrored_coffin[transpose_grid_index(index)] = card
    king_history = PublicGameHistory(
        status=EngineStatus.PLAYING,
        round_number=queen_history.round_number,
        dealer=EnginePlayer.QUEEN,
        active_player=EnginePlayer.KING,
        total_scores=queen_history.total_scores,
        completed_rounds=(),
        current_coffin=tuple(mirrored_coffin),
        current_round_moves=(),
    )
    king_input = PolicyInput(
        queen_context.input.observation.clone(), queen_context.input.legal_mask.clone()
    )
    king_information = build_information_state(
        king_input, king_history, EnginePlayer.KING
    )

    assert queen_information.coffin == king_information.coffin
    assert queen_information.own_hand == king_information.own_hand
    assert queen_information.legal_mask == king_information.legal_mask
    assert information_state_fingerprint(queen_information) == information_state_fingerprint(
        king_information
    )


# On each simulated turn, the new actor sees its own sampled cards and only a pooled remainder.
def test_simulated_opponent_receives_only_its_own_information() -> None:
    root_state = create_game("engine-contract-fixture-1")
    root_information = information_state_from_engine(root_state)
    sampled = sample_determinization(root_information, _sample_seed("actor-view"))
    after_root_move = apply_move(sampled.state, _first_move(sampled.state)).state
    opponent_information = information_state_from_simulation(after_root_move)
    opponent = after_root_move.active_player
    assert opponent is not None and opponent is other_player(root_information.player)

    assert opponent_information.own_hand == after_root_move.hands[opponent]
    assert set(opponent_information.unseen_card_ids) == {
        card
        for card in (*after_root_move.hands[root_information.player], *after_root_move.stock)
        if card is not None
    }
    assert set(opponent_information.unseen_card_ids).isdisjoint(
        card for card in opponent_information.own_hand if card is not None
    )
    assert not hasattr(opponent_information, "simulation_deck")


# Rollout choice takes only the actor view and its named seed, so private worlds cannot steer it.
def test_sampled_moves_do_not_depend_on_private_runtime_or_model_data() -> None:
    root = information_state_from_engine(create_game("engine-contract-fixture-1"))
    first_world = sample_determinization(root, _sample_seed("move-world-a")).state
    second_world = sample_determinization(root, _sample_seed("move-world-b")).state
    first_view = information_state_from_simulation(first_world)
    second_view = information_state_from_simulation(second_world)
    choice_seed = derive_seed("dracula-search-test-choice-v1", "same-choice")

    assert first_view == second_view
    action_index = sample_uniform_action_index(first_view, choice_seed)
    assert action_index == sample_uniform_action_index(second_view, choice_seed)
    assert first_view.legal_mask[action_index // 8][action_index % 8]
    source = inspect.getsource(inspect.getmodule(sample_determinization))
    for forbidden in ("policy_adapter", "local_policy", "model_path", "hidden_state"):
        assert forbidden not in source
    assert tuple(inspect.signature(sample_determinization).parameters) == (
        "information",
        "sample_seed",
    )


# Revealed cards fix private slots while only genuinely unseen remaining cards are sampled.
def test_late_round_samples_respect_revealed_cards_and_forced_final_placement() -> None:
    state = create_game("engine-contract-fixture-1")
    for _ in range(7):
        state = _advance_authoritative(state)
    information = information_state_from_engine(state)
    sampled = sample_determinization(information, _sample_seed("forced"))

    assert information.turn_number == 8
    assert information.player is state.dealer
    assert information.opponent_remaining_count == 0
    assert sampled.opponent_remaining_hand == ()
    assert sampled.state.hands[other_player(information.player)] == (None,) * 4
    assert sum(value for row in information.legal_mask for value in row) == 1
    for public, private in zip(
        information.current_round_moves,
        sampled.state.current_round_moves,
        strict=True,
    ):
        assert (private.player, private.card_id, private.global_grid_index) == (
            public.player,
            public.card_id,
            public.global_grid_index,
        )
    validate_state(apply_move(sampled.state, _first_move(sampled.state)).state)


# Every lifecycle position and dealer orientation must admit at least one valid belief sample.
@pytest.mark.parametrize("game_seed", ("search-role-0", "search-role-2"))
def test_sampling_is_valid_at_every_turn_round_and_dealer_role(game_seed: str) -> None:
    state = create_game(game_seed)
    seen_dealers: set[EnginePlayer] = set()
    samples = 0
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            seen_dealers.add(state.dealer)
            information = information_state_from_engine(state)
            request_seed = derive_search_request_seed(
                f"fixture-{game_seed}", information, "0" * 64
            )
            sampled = sample_determinization(
                information, derive_belief_sample_seed(request_seed, samples)
            )
            validate_state(sampled.state)
            assert information_state_from_simulation(sampled.state) == information
            samples += 1
            state = _advance_authoritative(state)
        state = advance_after_round(state)

    assert samples == 48
    assert seen_dealers == set(EnginePlayer)


# Named namespaces keep belief, tree, and rollout randomness reproducibly independent.
def test_search_seed_namespaces_are_versioned_reproducible_and_separated() -> None:
    information = information_state_from_engine(create_game("engine-contract-fixture-1"))
    request = derive_search_request_seed("fixture", information, "f" * 64)
    assert request == derive_search_request_seed("fixture", information, "f" * 64)
    belief = derive_belief_sample_seed(request, 3)
    tree = derive_tree_selection_seed(request, 3, "a" * 64)
    rollout = derive_rollout_choice_seed(request, 3, 2)
    assert len({request, belief, tree, rollout}) == 4
    assert SEARCH_REQUEST_NAMESPACE.endswith("-v1")
    assert BELIEF_SAMPLE_NAMESPACE.endswith("-v1")
    assert TREE_SELECTION_NAMESPACE.endswith("-v1")
    assert ROLLOUT_CHOICE_NAMESPACE.endswith("-v1")


# Sampled stock provenance cannot advance beyond the round-local horizon.
def test_simulation_state_is_typed_and_round_local() -> None:
    information = information_state_from_engine(create_game("engine-contract-fixture-1"))
    sampled = sample_determinization(information, _sample_seed("typed"))
    assert isinstance(sampled.state, SimulationEngineState)
    assert sampled.sampled_stock == sampled.state.stock
    assert sort_card_ids(sampled.opponent_remaining_hand) == sampled.opponent_remaining_hand
    with pytest.raises(
        InvalidLifecycleTransition, match="round-local simulation cannot advance rounds"
    ):
        advance_after_round(sampled.state)
