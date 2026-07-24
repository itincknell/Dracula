from __future__ import annotations

from dataclasses import asdict, fields, replace

import pytest
import torch

from dracula.engine import apply_move, create_game, legal_moves
from dracula.policy_value import PolicyValueModel
from dracula.response_distillation import (
    RESPONSE_EXAMPLE_SCHEMA_VERSION,
    RESPONSE_RANKING_SCHEMA_VERSION,
    ResponseActionGroupTarget,
    ResponseDistillationContractError,
    ResponseDistillationExample,
    response_group_logits,
    response_pairwise_ranking_loss,
    select_response_group,
)
from dracula.search import (
    SearchContractViolation,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_search_request_seed,
    information_state_from_engine,
)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(
            state, legal_moves(state, state.active_player)[0]
        ).state
    return state


def _search_with_examples():
    information = information_state_from_engine(
        _advance(create_game("response-distillation-fixture"), 5)
    )
    config = StrategicSearchConfig(
        outer_simulation_budget=8,
        response_completions_per_action=4,
    )
    request_seed = derive_strategic_search_request_seed(
        "response-distillation-test",
        information,
        config.digest,
    )
    examples: list[ResponseDistillationExample] = []
    result = StrategicInformationSetSearch(
        config,
        response_observer=examples.append,
    ).search(information, request_seed)
    return information, config, request_seed, result, tuple(examples)


def test_observer_is_behaviorally_inert_and_sees_each_cache_identity_once() -> None:
    information, config, request_seed, observed, examples = (
        _search_with_examples()
    )
    unobserved = StrategicInformationSetSearch(config).search(
        information, request_seed
    )

    # Instrumentation is outside search state: enabling it cannot alter the
    # visits, values, chosen move, diagnostics, or configuration identity.
    assert observed == unobserved
    assert observed.config_digest == config.digest
    assert len(examples) == observed.unique_response_evaluation_count
    assert len({example.cache_identity for example in examples}) == len(
        examples
    )
    assert all(
        example.search_config_digest == config.digest for example in examples
    )
    assert all(
        example.response_config_digest == config.response_config.digest
        for example in examples
    )


def test_captured_example_is_exact_model_visible_response_evidence() -> None:
    _, config, _, _, examples = _search_with_examples()
    example = examples[0]

    assert example.schema_version == RESPONSE_EXAMPLE_SCHEMA_VERSION
    assert example.ranking_schema_version == RESPONSE_RANKING_SCHEMA_VERSION
    assert len(example.observation) == 875
    assert len(example.legal_mask) == 4
    assert all(len(row) == 8 for row in example.legal_mask)
    assert example.completions_per_action == 4
    assert example.actor_role in {"queen", "king"}
    assert 1 <= example.placement_number <= 7
    assert (
        example.cache_identity.information_state_fingerprint
        and example.cache_identity.response_config_digest
        == config.response_config.digest
    )
    legal = {
        row * 8 + column
        for row, values in enumerate(example.legal_mask)
        for column, allowed in enumerate(values)
        if allowed
    }
    grouped = {
        index
        for group in example.groups
        for index in group.member_action_indices
    }
    assert grouped == legal
    assert all(
        -1.0 <= group.mean_terminal_differential <= 1.0
        for group in example.groups
    )
    assert example.selected_representative_action_index == min(
        example.groups,
        key=lambda group: (
            -group.mean_terminal_differential,
            group.representative_action_index,
        ),
    ).representative_action_index


def test_response_example_has_no_private_search_or_model_channel() -> None:
    _, _, _, _, examples = _search_with_examples()
    example = examples[0]
    field_names = {field.name for field in fields(example)}
    forbidden = {
        "determinization",
        "opponent_hand",
        "stock",
        "stock_order",
        "engine_seed",
        "outer_state",
        "search_tree",
        "policy_hidden_state",
        "model",
        "model_path",
        "logits",
    }

    # The observer receives this immutable record, not a simulation or
    # SearchInformationState from which private sampled-world fields could leak.
    assert field_names.isdisjoint(forbidden)
    assert set(asdict(example)) == field_names
    assert not hasattr(example, "__dict__")


def test_group_logits_use_arithmetic_mean_without_pair_size_advantage() -> None:
    groups = (
        ResponseActionGroupTarget(0, 0, (0, 1), 0.5),
        ResponseActionGroupTarget(0, 2, (2,), 0.25),
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    logits[0, 0] = 1.0
    logits[0, 1] = 3.0
    logits[0, 2] = 2.0

    scores = response_group_logits(logits, groups)

    assert scores.tolist() == [2.0, 2.0]
    assert select_response_group(logits, groups) == 0


def test_pairwise_loss_uses_value_gap_weights_and_omits_exact_ties() -> None:
    _, _, _, _, examples = _search_with_examples()
    base = examples[0]
    groups = (
        ResponseActionGroupTarget(0, 0, (0,), 0.75),
        ResponseActionGroupTarget(0, 1, (1,), 0.25),
        ResponseActionGroupTarget(0, 2, (2,), 0.25),
    )
    legal_mask = (
        (True, True, True, False, False, False, False, False),
        (False,) * 8,
        (False,) * 8,
        (False,) * 8,
    )
    example = replace(
        base,
        legal_mask=legal_mask,
        groups=groups,
        selected_representative_action_index=0,
        selected_concrete_action_index=0,
    )
    logits = torch.zeros((4, 8), dtype=torch.float32, requires_grad=True)

    loss = response_pairwise_ranking_loss(logits, example)
    loss.total.backward()

    # Two 0.5-weight comparisons contribute. The exact tie between groups 1
    # and 2 contributes neither loss nor weight.
    assert loss.contributing_pair_count == 2
    assert loss.total_pair_weight == pytest.approx(1.0)
    assert loss.total.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] > 0


def test_all_tied_groups_produce_zero_loss_and_zero_gradient() -> None:
    _, _, _, _, examples = _search_with_examples()
    base = examples[0]
    tied_groups = tuple(
        replace(group, mean_terminal_differential=0.0)
        for group in base.groups
    )
    example = replace(
        base,
        groups=tied_groups,
        selected_representative_action_index=min(
            group.representative_action_index for group in tied_groups
        ),
        selected_concrete_action_index=next(
            group
            for group in tied_groups
            if group.representative_action_index
            == min(item.representative_action_index for item in tied_groups)
        ).member_action_indices[0],
    )
    logits = torch.randn((4, 8), dtype=torch.float32, requires_grad=True)

    loss = response_pairwise_ranking_loss(logits, example)
    loss.total.backward()

    assert loss.total.item() == 0.0
    assert loss.contributing_pair_count == 0
    assert loss.total_pair_weight == 0.0
    assert torch.count_nonzero(logits.grad) == 0


def test_response_loss_uses_policy_and_shared_body_but_not_value_head() -> None:
    _, _, _, _, examples = _search_with_examples()
    example = examples[0]
    model = PolicyValueModel(
        run_root_seed="response-distillation",
        model_id="ranker",
        initialization_ordinal=0,
    )
    observation = torch.tensor(example.observation, dtype=torch.bool)

    logits, _unused_value = model(observation)
    loss = response_pairwise_ranking_loss(logits, example)
    loss.total.backward()

    assert model.policy_pair_output.weight.grad is not None
    assert model.encoder.shared_output.weight.grad is not None
    assert model.value_hidden.weight.grad is None
    assert model.value_output.weight.grad is None


def test_response_example_rejects_groups_that_do_not_match_legality() -> None:
    _, _, _, _, examples = _search_with_examples()
    example = examples[0]
    first = example.groups[0]
    illegal_index = next(
        index
        for index, allowed in enumerate(
            value for row in example.legal_mask for value in row
        )
        if not allowed and index // 8 == first.hand_slot
    )
    malformed = replace(
        first,
        member_action_indices=tuple(
            sorted((*first.member_action_indices, illegal_index))
        ),
    )

    with pytest.raises(ResponseDistillationContractError):
        replace(example, groups=(malformed, *example.groups[1:]))


def test_observer_requires_the_locked_four_completion_response() -> None:
    with pytest.raises(SearchContractViolation, match="four completions"):
        StrategicInformationSetSearch(
            StrategicSearchConfig(8, 2),
            response_observer=lambda _example: None,
        )
