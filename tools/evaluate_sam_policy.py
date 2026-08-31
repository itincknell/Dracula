#!/usr/bin/env python3
"""Deterministic standalone Sam-policy fixtures and absolute controls."""

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    derive_game_outcome,
    legal_moves,
    other_player,
)
from dracula.local_policy import LocalTorchPolicyAdapter
from dracula.policy_adapter import (
    POLICY_INFERENCE_CONTRACT_VERSION,
    PolicyInferenceRequest,
    resolve_inference_profile,
    select_masked_action,
)
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.search import (
    information_state_from_engine,
    sam_teacher_action_groups,
)
from dracula.search.nested_strategic import (
    NestedStrategicInformationSetSearch,
    NestedStrategicSearchConfig,
    derive_nested_strategic_search_request_seed,
)
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    replay_fixture,
)
from dracula.search.defensive_fixtures import (
    DEFENSIVE_FIXTURES,
    replay_defensive_fixture,
)
from dracula.standalone_policy import StandaloneSamPolicyExecutor
from dracula.sam_policy import POLICY_GRID_INDICES

EVALUATION_SCHEMA_VERSION = "dracula-sam-policy-evaluation-v1"
GAME_SEED_NAMESPACE = "dracula-sam-policy-evaluation-game-v1"
RANDOM_SELECTION_NAMESPACE = "dracula-sam-policy-evaluation-random-v1"
CONTROL_KINDS = ("random", "ppo", "nested-sam-32")
@dataclass(frozen=True, slots=True)
class DecisionRecord:
    controller: str
    round_number: int
    accepted_moves: int
    action_index: int
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class RoundRecord:
    round_number: int
    subject_dealer: bool
    subject_score: int
    opponent_score: int


@dataclass(frozen=True, slots=True)
class GameRecord:
    control: str
    seed_ordinal: int
    game_seed: str
    subject_role: str
    subject_won: bool
    tied: bool
    rounds: tuple[RoundRecord, ...]
    subject_decisions: tuple[DecisionRecord, ...]


@dataclass(frozen=True, slots=True)
class FixtureRecord:
    catalog: str
    fixture_id: str
    expected_action_indices: tuple[int, ...]
    expected_representative_actions: tuple[int, ...]
    expected_action_labels: tuple[str, ...]
    policy_representative_action: int
    policy_concrete_action: int
    policy_action_label: str
    nested_representative_action: int
    nested_concrete_action: int
    nested_action_label: str
    policy_passed: bool
    nested_passed: bool
    policy_concrete_expected: bool
    nested_concrete_expected: bool
    same_strategic_group: bool
    policy_latency_seconds: float
    nested_latency_seconds: float


class _StandaloneController:
    def __init__(
        self, executor: StandaloneSamPolicyExecutor, game_id: str
    ) -> None:
        self.executor = executor
        self.game_id = game_id

    def select(self, state: EngineState) -> tuple[EngineMove, DecisionRecord | None]:
        actor = state.active_player
        if actor is None:
            raise RuntimeError("standalone controller has no active player")
        moves = legal_moves(state, actor)
        if len(moves) == 1:
            return moves[0], None
        context = build_policy_turn_context(state, actor)
        information = information_state_from_engine(state, actor)
        decision = self.executor.decide(
            game_id=self.game_id,
            information=information,
            policy_input=context.input,
        )
        move = context.action_table[decision.concrete_action_index]
        if move is None or move not in moves:
            raise RuntimeError("standalone policy selected an illegal action")
        return move, DecisionRecord(
            "standalone-sam-policy",
            state.round_number,
            len(state.current_round_moves),
            decision.concrete_action_index,
            decision.latency_seconds,
        )


class _NestedController:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.planner = NestedStrategicInformationSetSearch(
            NestedStrategicSearchConfig(
                outer_simulation_budget=32,
                response_simulation_budget=32,
            )
        )

    def select(self, state: EngineState) -> tuple[EngineMove, None]:
        actor = state.active_player
        if actor is None:
            raise RuntimeError("nested controller has no active player")
        moves = legal_moves(state, actor)
        if len(moves) == 1:
            return moves[0], None
        context = build_policy_turn_context(state, actor)
        information = information_state_from_engine(state, actor)
        seed = derive_nested_strategic_search_request_seed(
            self.game_id, information, self.planner.config.digest
        )
        result = self.planner.search(information, seed)
        move = context.action_table[result.selected_action_index]
        if move is None or move not in moves:
            raise RuntimeError("nested Sam selected an illegal action")
        return move, None


class _RandomController:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id

    def select(self, state: EngineState) -> tuple[EngineMove, None]:
        actor = state.active_player
        if actor is None:
            raise RuntimeError("random controller has no active player")
        moves = legal_moves(state, actor)
        if len(moves) == 1:
            return moves[0], None
        seed = derive_seed(
            RANDOM_SELECTION_NAMESPACE,
            self.game_id,
            actor.value,
            str(state.round_number),
            str(len(state.current_round_moves)),
        )
        return moves[Sha256CounterStream(seed).randbelow(len(moves))], None


class _PpoController:
    def __init__(self, archive_path: Path, game_id: str) -> None:
        self.game_id = game_id
        self.adapter = LocalTorchPolicyAdapter(archive_path)
        self.profile = resolve_inference_profile("argmax-v1")
        self.hidden = bytes(512)

    def select(self, state: EngineState) -> tuple[EngineMove, None]:
        actor = state.active_player
        if actor is None:
            raise RuntimeError("PPO controller has no active player")
        moves = legal_moves(state, actor)
        context = build_policy_turn_context(state, actor)
        observation = tuple(
            bool(value)
            for value in context.input.observation.detach().cpu().tolist()
        )
        legal_mask = tuple(
            tuple(bool(value) for value in row)
            for row in context.input.legal_mask.detach().cpu().tolist()
        )
        response = self.adapter.invoke(
            PolicyInferenceRequest(
                POLICY_INFERENCE_CONTRACT_VERSION,
                self.adapter.metadata.artifact_id,
                observation,
                legal_mask,
                self.hidden,
            )
        )
        self.hidden = response.next_hidden_state
        if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            return moves[0], None
        index = select_masked_action(
            response.raw_logits,
            legal_mask,
            self.profile,
            game_id=self.game_id,
            round_number=state.round_number,
            turn_number=len(state.current_round_moves) + 1,
            artifact_id=self.adapter.metadata.artifact_id,
        )
        move = context.action_table[index]
        if move is None or move not in moves:
            raise RuntimeError("PPO control selected an illegal action")
        return move, None


def _controller(
    kind: str,
    *,
    executor: StandaloneSamPolicyExecutor,
    game_id: str,
    ppo_archive: Path,
) -> object:
    if kind == "standalone":
        return _StandaloneController(executor, game_id)
    if kind == "random":
        return _RandomController(game_id)
    if kind == "ppo":
        return _PpoController(ppo_archive, game_id)
    if kind == "nested-sam-32":
        return _NestedController(game_id)
    raise ValueError(f"unknown controller: {kind}")


def _play_game(
    *,
    control: str,
    seed_ordinal: int,
    game_seed: str,
    subject_role: EnginePlayer,
    executor: StandaloneSamPolicyExecutor,
    ppo_archive: Path,
) -> GameRecord:
    identity = (
        f"sam-policy-evaluation:{control}:{seed_ordinal}:"
        f"{subject_role.value}"
    )
    controllers = {
        subject_role: _controller(
            "standalone",
            executor=executor,
            game_id=f"{identity}:subject",
            ppo_archive=ppo_archive,
        ),
        other_player(subject_role): _controller(
            control,
            executor=executor,
            game_id=f"{identity}:control",
            ppo_archive=ppo_archive,
        ),
    }
    state = create_game(game_seed)
    rounds: list[RoundRecord] = []
    decisions: list[DecisionRecord] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            actor = state.active_player
            if actor is None:
                raise RuntimeError("playing game has no active player")
            move, timing = controllers[actor].select(state)  # type: ignore[attr-defined]
            if timing is not None and actor is subject_role:
                decisions.append(timing)
            state = apply_move(state, move).state
        result = state.pending_round_result
        if result is None:
            raise RuntimeError("completed round has no score")
        rounds.append(
            RoundRecord(
                state.round_number,
                state.dealer is subject_role,
                result.round_scores[subject_role],
                result.round_scores[other_player(subject_role)],
            )
        )
        state = advance_after_round(state)
    outcome = derive_game_outcome(state)
    return GameRecord(
        control,
        seed_ordinal,
        game_seed,
        subject_role.value,
        outcome.winner is subject_role,
        outcome.winner is None,
        tuple(rounds),
        tuple(decisions),
    )


def _play_game_task(
    task: tuple[str, int, str, str, str, str]
) -> GameRecord:
    (
        control,
        seed_ordinal,
        game_seed,
        subject_role,
        artifact_path,
        ppo_archive,
    ) = task
    return _play_game(
        control=control,
        seed_ordinal=seed_ordinal,
        game_seed=game_seed,
        subject_role=EnginePlayer(subject_role),
        executor=StandaloneSamPolicyExecutor(artifact_path),
        ppo_archive=Path(ppo_archive),
    )


def _group_representative_for_action(
    state: EngineState, action_index: int
) -> int:
    information = information_state_from_engine(state)
    return next(
        group.representative_action_index
        for group in sam_teacher_action_groups(information)
        if action_index in group.member_action_indices
    )


def _action_label(state: EngineState, action_index: int) -> str:
    actor = state.active_player
    if actor is None:
        raise RuntimeError("fixture action has no active player")
    hand_slot, policy_position = divmod(action_index, 8)
    card_id = state.hands[actor][hand_slot]
    if card_id is None:
        raise RuntimeError("fixture action names an empty hand slot")
    return f"{card_id}@{POLICY_GRID_INDICES[policy_position] + 1}"


def _fixture_records(
    executor: StandaloneSamPolicyExecutor,
) -> tuple[FixtureRecord, ...]:
    records: list[FixtureRecord] = []
    nested = _NestedController("sam-policy-fixture-nested")
    fixtures = (
        tuple(
            ("constructive", fixture, replay_fixture)
            for fixture in STRATEGIC_FIXTURES
        )
        + tuple(
            ("defensive", fixture, replay_defensive_fixture)
            for fixture in DEFENSIVE_FIXTURES
        )
    )
    for catalog, fixture, replay in fixtures:
        state = replay(fixture)
        context = build_policy_turn_context(state, state.active_player)
        information = information_state_from_engine(state)
        moves = legal_moves(state, state.active_player)
        if len(moves) == 1:
            only = next(
                index
                for index, move in enumerate(context.action_table)
                if move is not None
            )
            policy_rep = nested_rep = policy_action = nested_action = only
            policy_latency = nested_latency = 0.0
        else:
            policy = executor.decide(
                game_id=f"fixture:{fixture.fixture_id}:policy",
                information=information,
                policy_input=context.input,
            )
            policy_rep = policy.representative_action_index
            policy_action = policy.concrete_action_index
            policy_latency = policy.latency_seconds
            nested_seed = derive_nested_strategic_search_request_seed(
                f"fixture:{fixture.fixture_id}:nested",
                information,
                nested.planner.config.digest,
            )
            started = time.perf_counter()
            nested_result = nested.planner.search(
                information, nested_seed
            )
            nested_latency = time.perf_counter() - started
            nested_action = nested_result.selected_action_index
            nested_rep = _group_representative_for_action(
                state, nested_action
            )
        expected_representatives = tuple(
            sorted(
                {
                    _group_representative_for_action(state, index)
                    for index in fixture.expected_action_indices
                }
            )
        )
        records.append(
            FixtureRecord(
                catalog,
                fixture.fixture_id,
                fixture.expected_action_indices,
                expected_representatives,
                tuple(
                    _action_label(state, index)
                    for index in fixture.expected_action_indices
                ),
                policy_rep,
                policy_action,
                _action_label(state, policy_action),
                nested_rep,
                nested_action,
                _action_label(state, nested_action),
                policy_rep in expected_representatives,
                nested_rep in expected_representatives,
                policy_action in fixture.expected_action_indices,
                nested_action in fixture.expected_action_indices,
                policy_rep == nested_rep,
                policy_latency,
                nested_latency,
            )
        )
    return tuple(records)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1)
    )
    return ordered[index]


def _game_metrics(records: list[GameRecord]) -> dict[str, object]:
    games = len(records)
    rounds = [item for record in records for item in record.rounds]
    timings = [
        timing.latency_seconds
        for record in records
        for timing in record.subject_decisions
    ]

    def split(selected: list[GameRecord]) -> dict[str, object]:
        selected_rounds = [
            item for record in selected for item in record.rounds
        ]
        return {
            "games": len(selected),
            "wins": sum(record.subject_won for record in selected),
            "ties": sum(record.tied for record in selected),
            "rounds": len(selected_rounds),
            "round_wins": sum(
                item.subject_score > item.opponent_score
                for item in selected_rounds
            ),
            "round_ties": sum(
                item.subject_score == item.opponent_score
                for item in selected_rounds
            ),
            "mean_score_differential": (
                statistics.fmean(
                    item.subject_score - item.opponent_score
                    for item in selected_rounds
                )
                if selected_rounds
                else None
            ),
        }

    by_role = {
        role.value: split(
            [record for record in records if record.subject_role == role.value]
        )
        for role in EnginePlayer
    }
    by_dealer = {
        label: {
            "rounds": len(selected),
            "round_wins": sum(
                item.subject_score > item.opponent_score
                for item in selected
            ),
            "round_ties": sum(
                item.subject_score == item.opponent_score
                for item in selected
            ),
            "mean_score_differential": (
                statistics.fmean(
                    item.subject_score - item.opponent_score
                    for item in selected
                )
                if selected
                else None
            ),
        }
        for label, selected in (
            ("dealer", [item for item in rounds if item.subject_dealer]),
            (
                "non_dealer",
                [item for item in rounds if not item.subject_dealer],
            ),
        )
    }
    return {
        **split(records),
        "by_role": by_role,
        "by_dealer": by_dealer,
        "decision_latency_seconds": {
            "count": len(timings),
            "p50": _percentile(timings, 0.5),
            "p95": _percentile(timings, 0.95),
            "maximum": max(timings) if timings else None,
        },
        "game_count": games,
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ppo-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--game-pairs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--refresh-fixtures", action="store_true")
    arguments = parser.parse_args()
    if arguments.game_pairs < 1:
        parser.error("--game-pairs must be positive")
    if arguments.workers not in {1, 2, 3}:
        parser.error("--workers must be 1, 2, or 3")
    if not arguments.ppo_archive.is_file():
        parser.error("PPO archive is missing")
    arguments.output.mkdir(parents=True, exist_ok=True)
    game_directory = arguments.output / "games"
    game_directory.mkdir(exist_ok=True)
    executor = StandaloneSamPolicyExecutor(arguments.artifact)

    fixture_path = arguments.output / "fixtures.json"
    expected_fixture_count = len(STRATEGIC_FIXTURES) + len(
        DEFENSIVE_FIXTURES
    )
    if arguments.refresh_fixtures:
        fixture_payload = [
            asdict(record) for record in _fixture_records(executor)
        ]
        _atomic_json(fixture_path, fixture_payload)
    elif fixture_path.exists():
        fixture_payload = json.loads(
            fixture_path.read_text(encoding="utf-8")
        )
    elif arguments.fixtures is not None:
        fixture_payload = json.loads(
            arguments.fixtures.read_text(encoding="utf-8")
        )
        if (
            not isinstance(fixture_payload, list)
            or len(fixture_payload) != expected_fixture_count
        ):
            parser.error("--fixtures is not a complete fixture result")
        _atomic_json(fixture_path, fixture_payload)
    else:
        fixture_payload = [
            asdict(record) for record in _fixture_records(executor)
        ]
        _atomic_json(fixture_path, fixture_payload)
    print(f"fixtures={len(fixture_payload)}", flush=True)

    records: list[GameRecord] = []
    pending: list[tuple[Path, tuple[str, int, str, str, str, str]]] = []
    total = len(CONTROL_KINDS) * arguments.game_pairs * 2
    completed = 0
    for control in CONTROL_KINDS:
        for seed_ordinal in range(arguments.game_pairs):
            game_seed = derive_seed(
                GAME_SEED_NAMESPACE, f"pair-{seed_ordinal:03d}"
            ).hex()
            for role in EnginePlayer:
                path = (
                    game_directory
                    / f"{control}-{seed_ordinal:03d}-{role.value}.json"
                )
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    record = GameRecord(
                        control=raw["control"],
                        seed_ordinal=raw["seed_ordinal"],
                        game_seed=raw["game_seed"],
                        subject_role=raw["subject_role"],
                        subject_won=raw["subject_won"],
                        tied=raw["tied"],
                        rounds=tuple(
                            RoundRecord(**item) for item in raw["rounds"]
                        ),
                        subject_decisions=tuple(
                            DecisionRecord(**item)
                            for item in raw["subject_decisions"]
                        ),
                    )
                    records.append(record)
                    completed += 1
                    print(
                        f"games={completed}/{total} control={control} "
                        f"pair={seed_ordinal} role={role.value} cached=true",
                        flush=True,
                    )
                else:
                    pending.append(
                        (
                            path,
                            (
                                control,
                                seed_ordinal,
                                game_seed,
                                role.value,
                                str(arguments.artifact),
                                str(arguments.ppo_archive),
                            ),
                        )
                    )

    if arguments.workers == 1:
        for path, task in pending:
            record = _play_game_task(task)
            _atomic_json(path, asdict(record))
            records.append(record)
            completed += 1
            print(
                f"games={completed}/{total} control={record.control} "
                f"pair={record.seed_ordinal} role={record.subject_role}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=arguments.workers
        ) as pool:
            futures = {
                pool.submit(_play_game_task, task): path
                for path, task in pending
            }
            for future in as_completed(futures):
                record = future.result()
                _atomic_json(futures[future], asdict(record))
                records.append(record)
                completed += 1
                print(
                    f"games={completed}/{total} control={record.control} "
                    f"pair={record.seed_ordinal} "
                    f"role={record.subject_role}",
                    flush=True,
                )

    records.sort(
        key=lambda value: (
            CONTROL_KINDS.index(value.control),
            value.seed_ordinal,
            value.subject_role,
        )
    )
    metrics = {
        control: _game_metrics(
            [record for record in records if record.control == control]
        )
        for control in CONTROL_KINDS
    }
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact_path": str(arguments.artifact),
        "artifact_sha256": executor.artifact_sha256,
        "artifact_size_bytes": arguments.artifact.stat().st_size,
        "game_pairs": arguments.game_pairs,
        "fixture_records": fixture_payload,
        "game_records": [asdict(record) for record in records],
        "metrics": metrics,
        "peak_rss_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }
    _atomic_json(arguments.output / "results.json", payload)
    print(
        json.dumps(
            {
                "artifact_sha256": executor.artifact_sha256,
                "metrics": metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
