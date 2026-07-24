"""Absolute, role-balanced evaluation for one expert-iteration candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import (
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
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search import (
    GuidedInformationSetSearch,
    GuidedSearchConfig,
    InformationSetSearch,
    PolicyValueModelEvaluator,
    SearchConfig,
    derive_search_request_seed,
    information_state_from_engine,
)
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    FixtureEvidence,
    replay_fixture,
)

EVALUATION_FORMAT_VERSION = "dracula-expert-evaluation-v1"
EVALUATION_FIXTURE_NAMESPACE = "dracula-absolute-evaluation-v1"
EVALUATION_RANDOM_NAMESPACE = "dracula-expert-evaluation-random-v1"
EVALUATION_BOOTSTRAP_NAMESPACE = "dracula-expert-evaluation-bootstrap-v1"


class ExpertEvaluationError(RuntimeError):
    """Absolute evaluation cannot produce valid evidence."""


@dataclass(frozen=True, slots=True)
class ControllerSpec:
    name: str
    kind: str
    budget: int | None = None
    artifact: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRound:
    round_number: int
    candidate_dealer: bool
    candidate_score: int
    control_score: int


@dataclass(frozen=True, slots=True)
class EvaluationGame:
    comparison: str
    deck_index: int
    deck_id: str
    candidate_role: str
    candidate_won: bool
    tied: bool
    rounds: tuple[EvaluationRound, ...]
    candidate_decision_seconds: tuple[float, ...]
    control_decision_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FixtureEvaluation:
    fixture_id: str
    repetition: int
    selected_action_index: int
    passed: bool
    latency_seconds: float
    visits: tuple[int, ...]


class _Controller:
    def __init__(self, spec: ControllerSpec, identity: str) -> None:
        self.spec = spec
        self.identity = identity
        self.hidden = bytes(512)
        self.profile = resolve_inference_profile("argmax-v1")
        self.search = (
            InformationSetSearch(SearchConfig(spec.budget))
            if spec.kind == "search" and spec.budget is not None
            else None
        )
        self.guided = None
        if spec.kind == "guided":
            if spec.artifact is None or spec.budget is None:
                raise ExpertEvaluationError("guided controller is incomplete")
            evaluator = PolicyValueModelEvaluator.from_artifact(spec.artifact)
            self.guided = GuidedInformationSetSearch(
                evaluator, GuidedSearchConfig(simulation_budget=spec.budget)
            )
        self.ppo = (
            LocalTorchPolicyAdapter(spec.artifact)
            if spec.kind == "ppo" and spec.artifact is not None
            else None
        )

    def select(self, state: EngineState) -> tuple[object, float | None]:
        actor = state.active_player
        if actor is None:
            raise ExpertEvaluationError("controller received no active player")
        moves = legal_moves(state, actor)
        context = build_policy_turn_context(state, actor)
        if len(moves) == 1 and self.spec.kind != "ppo":
            return moves[0], None
        if self.spec.kind == "random":
            stream = Sha256CounterStream(
                derive_seed(
                    EVALUATION_RANDOM_NAMESPACE,
                    self.identity,
                    actor.value,
                    str(state.round_number),
                    str(len(state.current_round_moves)),
                )
            )
            return moves[stream.randbelow(len(moves))], None
        information = information_state_from_engine(state, actor)
        if self.search is not None:
            seed = derive_search_request_seed(self.identity, information, self.search.config.digest)
            started = time.perf_counter()
            result = self.search.search(information, seed)
            elapsed = time.perf_counter() - started
            move = context.action_table[result.selected_action_index]
            if move is None:
                raise ExpertEvaluationError("search control selected a masked action")
            return move, elapsed
        if self.guided is not None:
            seed = derive_search_request_seed(self.identity, information, self.guided.digest)
            started = time.perf_counter()
            result = self.guided.search(information, seed)
            elapsed = time.perf_counter() - started
            move = context.action_table[result.selected_action_index]
            if move is None:
                raise ExpertEvaluationError("guided controller selected a masked action")
            return move, elapsed
        if self.ppo is None:
            raise ExpertEvaluationError("evaluation controller is not configured")
        observation = tuple(bool(value) for value in context.input.observation.tolist())
        legal_mask = tuple(tuple(bool(value) for value in row) for row in context.input.legal_mask.tolist())
        response = self.ppo.invoke(
            PolicyInferenceRequest(
                POLICY_INFERENCE_CONTRACT_VERSION,
                self.ppo.metadata.artifact_id,
                observation,
                legal_mask,
                self.hidden,
            )
        )
        self.hidden = response.next_hidden_state
        if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            return moves[0], None
        action = select_masked_action(
            response.raw_logits,
            legal_mask,
            self.profile,
            game_id=self.identity,
            round_number=state.round_number,
            turn_number=len(state.current_round_moves) + 1,
            artifact_id=self.ppo.metadata.artifact_id,
        )
        move = context.action_table[action]
        if move is None:
            raise ExpertEvaluationError("PPO control selected a masked action")
        return move, None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, allow_nan=False, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _evaluation_seed(root_seed: str, deck_index: int) -> tuple[str, str]:
    digest = derive_seed(
        EVALUATION_FIXTURE_NAMESPACE,
        root_seed,
        "evaluation",
        "absolute",
        str(deck_index),
    )
    return seed_hex(digest), hashlib.sha256(digest).hexdigest()


def _play_game(
    candidate: ControllerSpec,
    control: ControllerSpec,
    *,
    root_seed: str,
    deck_index: int,
    candidate_role: EnginePlayer,
) -> EvaluationGame:
    game_seed, deck_id = _evaluation_seed(root_seed, deck_index)
    comparison = f"candidate-vs-{control.name}"
    controllers = {
        candidate_role: _Controller(candidate, f"{comparison}:{deck_id}:candidate:{candidate_role.value}"),
        other_player(candidate_role): _Controller(control, f"{comparison}:{deck_id}:control:{other_player(candidate_role).value}"),
    }
    state = create_game(game_seed)
    rounds: list[EvaluationRound] = []
    latencies: list[float] = []
    control_latencies: list[float] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            actor = state.active_player
            if actor is None:
                raise ExpertEvaluationError("evaluation game lost its active player")
            move, elapsed = controllers[actor].select(state)
            if actor is candidate_role and elapsed is not None:
                latencies.append(elapsed)
            elif actor is not candidate_role and elapsed is not None:
                control_latencies.append(elapsed)
            state = apply_move(state, move).state  # type: ignore[arg-type]
        result = state.pending_round_result
        if result is None:
            raise ExpertEvaluationError("evaluation round has no score")
        rounds.append(
            EvaluationRound(
                state.round_number,
                state.dealer is candidate_role,
                result.round_scores[candidate_role],
                result.round_scores[other_player(candidate_role)],
            )
        )
        state = advance_after_round(state)
    outcome = derive_game_outcome(state)
    return EvaluationGame(
        comparison,
        deck_index,
        deck_id,
        candidate_role.value,
        outcome.winner is candidate_role,
        outcome.winner is None,
        tuple(rounds),
        tuple(latencies),
        tuple(control_latencies),
    )


def _play_game_task(args) -> EvaluationGame:
    return _play_game(*args[:2], root_seed=args[2], deck_index=args[3], candidate_role=args[4])


def _fixture_candidate(
    candidate: ControllerSpec,
    fixture_id: str,
    repetition: int,
) -> FixtureEvaluation:
    fixture = next(value for value in STRATEGIC_FIXTURES if value.fixture_id == fixture_id)
    state = replay_fixture(fixture)
    information = information_state_from_engine(state, fixture.player)
    evaluator = PolicyValueModelEvaluator.from_artifact(str(candidate.artifact))
    planner = GuidedInformationSetSearch(
        evaluator, GuidedSearchConfig(simulation_budget=int(candidate.budget))
    )
    seed = derive_search_request_seed(
        f"absolute-fixture:{fixture_id}:{repetition}", information, planner.digest
    )
    started = time.perf_counter()
    result = planner.search(information, seed)
    return FixtureEvaluation(
        fixture_id,
        repetition,
        result.selected_action_index,
        result.selected_action_index in fixture.expected_action_indices,
        time.perf_counter() - started,
        result.action_visits,
    )


def _fixture_task(args) -> FixtureEvaluation:
    return _fixture_candidate(*args)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))]


def _bootstrap_interval(comparison: str, records: Sequence[EvaluationGame], samples: int) -> tuple[float, float]:
    blocks: dict[int, list[float]] = {}
    for row in records:
        score = 0.5 if row.tied else float(row.candidate_won)
        blocks.setdefault(row.deck_index, []).append(2 * score - 1)
    vectors = tuple(statistics.mean(values) for _, values in sorted(blocks.items()))
    if not vectors:
        raise ExpertEvaluationError("paired interval requires games")
    stream = Sha256CounterStream(
        derive_seed(
            EVALUATION_BOOTSTRAP_NAMESPACE,
            comparison,
            str(samples),
            *(f"{value:.17g}" for value in vectors),
        )
    )
    estimates = [statistics.mean(vectors[stream.randbelow(len(vectors))] for _ in vectors) for _ in range(samples)]
    return float(_percentile(estimates, 0.025)), float(_percentile(estimates, 0.975))


def _game_summary(name: str, records: Sequence[EvaluationGame], bootstrap_samples: int) -> dict[str, object]:
    games = [row for row in records if row.comparison == name]
    rounds = [value for row in games for value in row.rounds]
    latencies = [value for row in games for value in row.candidate_decision_seconds]
    control_latencies = [
        value for row in games for value in row.control_decision_seconds
    ]

    def split(rows: Sequence[EvaluationGame]) -> dict[str, object]:
        wins = sum(row.candidate_won for row in rows)
        ties = sum(row.tied for row in rows)
        return {
            "games": len(rows),
            "wins": wins,
            "ties": ties,
            "victory_percentage": (wins + 0.5 * ties) / len(rows) if rows else None,
        }

    return {
        "overall": split(games),
        "queen": split([row for row in games if row.candidate_role == "queen"]),
        "king": split([row for row in games if row.candidate_role == "king"]),
        "dealer_round_victory_percentage": statistics.mean(
            1.0 if row.candidate_score > row.control_score else 0.5 if row.candidate_score == row.control_score else 0.0
            for row in rounds if row.candidate_dealer
        ),
        "non_dealer_round_victory_percentage": statistics.mean(
            1.0 if row.candidate_score > row.control_score else 0.5 if row.candidate_score == row.control_score else 0.0
            for row in rounds if not row.candidate_dealer
        ),
        "round_victory_percentage": statistics.mean(
            1.0 if row.candidate_score > row.control_score else 0.5 if row.candidate_score == row.control_score else 0.0
            for row in rounds
        ),
        "mean_round_score_differential": statistics.mean(row.candidate_score - row.control_score for row in rounds),
        "paired_victory_difference_ci_95": list(_bootstrap_interval(name, games, bootstrap_samples)),
        "decision_latency_seconds": {
            "count": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "maximum": max(latencies) if latencies else None,
        },
        "control_decision_latency_seconds": {
            "count": len(control_latencies),
            "mean": statistics.mean(control_latencies) if control_latencies else None,
            "p50": _percentile(control_latencies, 0.5),
            "p95": _percentile(control_latencies, 0.95),
            "maximum": max(control_latencies) if control_latencies else None,
        },
    }


def _load_cache(
    path: Path, identity_digest: str
) -> tuple[list[EvaluationGame], list[FixtureEvaluation]]:
    if not path.is_file():
        return [], []
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("format_version") != EVALUATION_FORMAT_VERSION
        or value.get("identity_digest") != identity_digest
    ):
        return [], []
    games = [
        EvaluationGame(
            row["comparison"], int(row["deck_index"]), row["deck_id"], row["candidate_role"], bool(row["candidate_won"]), bool(row["tied"]),
            tuple(EvaluationRound(**item) for item in row["rounds"]),
            tuple(float(item) for item in row["candidate_decision_seconds"]),
            tuple(float(item) for item in row.get("control_decision_seconds", [])),
        )
        for row in value.get("games", [])
    ]
    fixtures = [
        FixtureEvaluation(
            row["fixture_id"], int(row["repetition"]), int(row["selected_action_index"]), bool(row["passed"]), float(row["latency_seconds"]), tuple(int(item) for item in row["visits"])
        )
        for row in value.get("fixtures", [])
    ]
    return games, fixtures


def _write_cache(
    path: Path,
    identity_digest: str,
    games: Sequence[EvaluationGame],
    fixtures: Sequence[FixtureEvaluation],
) -> None:
    _atomic_json(
        path,
        {
            "format_version": EVALUATION_FORMAT_VERSION,
            "identity_digest": identity_digest,
            "games": [
                asdict(value)
                for value in sorted(
                    games,
                    key=lambda row: (row.comparison, row.deck_index, row.candidate_role),
                )
            ],
            "fixtures": [
                asdict(value)
                for value in sorted(
                    fixtures, key=lambda row: (row.fixture_id, row.repetition)
                )
            ],
        },
    )


def _markdown(document: Mapping[str, object]) -> str:
    lines = [
        f"# Expert candidate evaluation {document['iteration']}",
        "",
        f"- Eligible for manual acceptance: **{'yes' if document['eligible'] else 'no'}**",
        f"- Strategic fixtures: {document['strategic_fixtures']['passes']}/{document['strategic_fixtures']['samples']}",  # type: ignore[index]
        "",
        "| Control | Games | Candidate VP | Mean round differential | Paired 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, value in document["comparisons"].items():  # type: ignore[union-attr]
        overall = value["overall"]
        interval = value["paired_victory_difference_ci_95"]
        lines.append(
            f"| {name} | {overall['games']} | {overall['victory_percentage']:.1%} | {value['mean_round_score_differential']:+.2f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] |"
        )
    lines.extend(("", "Candidate selection remains manual.", ""))
    return "\n".join(lines)


def evaluate_expert_candidate(
    config,
    iteration: int,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[str, int], bool] | None = None,
) -> dict[str, object]:
    iteration_dir = config.output_path / "iterations" / f"{iteration:06d}"
    training = json.loads((iteration_dir / "training-result.json").read_text(encoding="utf-8"))
    candidate = ControllerSpec("candidate", "guided", config.evaluation.candidate_budget, training["candidate_artifact"])
    pointer = json.loads((config.output_path / "accepted.json").read_text(encoding="utf-8"))
    accepted_history = pointer.get("history") or [{"version": pointer["accepted_version"], "artifact": pointer["artifact"]}]
    controls: list[tuple[ControllerSpec, int]] = [
        (ControllerSpec("random-legal", "random"), config.evaluation.random_pairs),
        (ControllerSpec("policy-2-v20", "ppo", artifact=config.controls.ppo_archive), config.evaluation.ppo_pairs),
        (ControllerSpec(f"search-{config.evaluation.search_control_budget}", "search", config.evaluation.search_control_budget), config.evaluation.search_pairs),
    ]
    for item in accepted_history[-config.replay.accepted_iterations :]:
        controls.append(
            (
                ControllerSpec(
                    f"accepted-v{item['version']}",
                    "guided",
                    config.evaluation.candidate_budget,
                    item["artifact"],
                ),
                config.evaluation.prior_pairs,
            )
        )
    output = iteration_dir / "evaluation"
    cache_path = output / "records.json"
    identity_digest = hashlib.sha256(
        json.dumps(
            {
                "candidate": hashlib.sha256(
                    Path(training["candidate_artifact"]).read_bytes()
                ).hexdigest(),
                "configuration": asdict(config.evaluation),
                "controls": [asdict(value[0]) | {"pairs": value[1]} for value in controls],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    games, fixtures = (
        _load_cache(cache_path, identity_digest) if resume else ([], [])
    )
    game_keys = {(row.comparison, row.deck_index, row.candidate_role) for row in games}
    fixture_keys = {(row.fixture_id, row.repetition) for row in fixtures}
    tasks = [
        (control, deck, role)
        for control, count in controls
        for deck in range(count)
        for role in EnginePlayer
        if (f"candidate-vs-{control.name}", deck, role.value) not in game_keys
    ]
    game_arguments = [
        (candidate, control, config.run.root_seed, deck, role)
        for control, deck, role in tasks
    ]
    if config.evaluation.workers == 1 or should_stop is not None:
        for index, arguments in enumerate(game_arguments):
            if should_stop is not None and should_stop("evaluation", len(games)):
                raise ExpertEvaluationError("expert evaluation interrupted")
            games.append(_play_game_task(arguments))
            _write_cache(cache_path, identity_digest, games, fixtures)
            if progress:
                progress(f"expert evaluation game={index + 1}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=config.evaluation.workers) as pool:
            futures = [pool.submit(_play_game_task, arguments) for arguments in game_arguments]
            for index, future in enumerate(as_completed(futures), start=1):
                games.append(future.result())
                _write_cache(cache_path, identity_digest, games, fixtures)
                if progress:
                    progress(f"expert evaluation game={index}/{len(tasks)}")
    fixture_tasks = [
        (fixture.fixture_id, repetition)
        for fixture in STRATEGIC_FIXTURES
        if fixture.evidence is not FixtureEvidence.FORCED
        for repetition in range(config.evaluation.strategic_repetitions)
        if (fixture.fixture_id, repetition) not in fixture_keys
    ]
    fixture_arguments = [(candidate, fixture_id, repetition) for fixture_id, repetition in fixture_tasks]
    if config.evaluation.workers == 1 or should_stop is not None:
        for arguments in fixture_arguments:
            if should_stop is not None and should_stop("evaluation", len(games) + len(fixtures)):
                raise ExpertEvaluationError("expert evaluation interrupted")
            fixtures.append(_fixture_task(arguments))
            _write_cache(cache_path, identity_digest, games, fixtures)
    else:
        with ProcessPoolExecutor(max_workers=config.evaluation.workers) as pool:
            futures = [pool.submit(_fixture_task, arguments) for arguments in fixture_arguments]
            for future in as_completed(futures):
                fixtures.append(future.result())
                _write_cache(cache_path, identity_digest, games, fixtures)
    games.sort(key=lambda row: (row.comparison, row.deck_index, row.candidate_role))
    fixtures.sort(key=lambda row: (row.fixture_id, row.repetition))
    _write_cache(cache_path, identity_digest, games, fixtures)
    comparisons = {
        name: _game_summary(name, games, config.evaluation.bootstrap_samples)
        for name in sorted({row.comparison for row in games})
    }
    fixture_pass = bool(fixtures) and all(row.passed for row in fixtures)
    sample_complete = all(
        sum(row.comparison == f"candidate-vs-{control.name}" for row in games) == pairs * 2
        for control, pairs in controls
    )
    permanent = ["candidate-vs-random-legal", "candidate-vs-policy-2-v20"]
    noninferiority = [name for name in comparisons if name not in permanent]
    accepted_comparisons = [
        name for name in comparisons if name.startswith("candidate-vs-accepted-v")
    ]
    search_comparisons = [
        name for name in comparisons if name.startswith("candidate-vs-search-")
    ]
    materially_faster_than_teacher = all(
        comparisons[name]["decision_latency_seconds"]["mean"] is not None
        and comparisons[name]["control_decision_latency_seconds"]["mean"] is not None
        and comparisons[name]["decision_latency_seconds"]["mean"]
        <= 0.8 * comparisons[name]["control_decision_latency_seconds"]["mean"]
        for name in search_comparisons
    )
    eligible = (
        fixture_pass
        and sample_complete
        and config.evaluation.random_pairs >= 12
        and config.evaluation.ppo_pairs >= 12
        and config.evaluation.search_pairs >= 60
        and config.evaluation.prior_pairs >= 60
        and all(comparisons[name]["paired_victory_difference_ci_95"][0] > 0 for name in permanent)
        and all(comparisons[name]["paired_victory_difference_ci_95"][0] >= -0.05 for name in noninferiority)
        and materially_faster_than_teacher
        and all(
            value[split]["victory_percentage"] is not None and value[split]["victory_percentage"] >= 0.4
            for value in comparisons.values()
            for split in ("queen", "king")
        )
        and all(
            comparisons[name][split] >= 0.4
            for name in accepted_comparisons
            for split in (
                "dealer_round_victory_percentage",
                "non_dealer_round_victory_percentage",
            )
        )
    )
    document: dict[str, object] = {
        "format_version": EVALUATION_FORMAT_VERSION,
        "iteration": iteration,
        "candidate_artifact_sha256": hashlib.sha256(Path(training["candidate_artifact"]).read_bytes()).hexdigest(),
        "candidate_budget": config.evaluation.candidate_budget,
        "eligible": eligible,
        "sample_complete": sample_complete,
        "strategic_fixtures": {
            "passes": sum(row.passed for row in fixtures),
            "samples": len(fixtures),
            "pass_rate": sum(row.passed for row in fixtures) / len(fixtures) if fixtures else None,
            "records": [asdict(row) for row in fixtures],
        },
        "comparisons": comparisons,
        "sample_design": {
            "role_assignments_per_deck": 2,
            "identical_deck_prefix_across_controls": True,
            "bootstrap_unit": "paired deck",
            "manual_acceptance": True,
            "teacher_latency_ratio_limit": 0.8,
        },
    }
    _atomic_json(output / "evaluation.json", document)
    _atomic_text(output / "evaluation.md", _markdown(document))
    return document


def benchmark_guided_candidate(
    artifact: str | Path,
    *,
    budgets: Sequence[int] = (20, 50, 100),
    repetitions: int = 1,
) -> dict[str, object]:
    """Measure tactical retention and latency without making an acceptance claim."""

    if not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ExpertEvaluationError("benchmark budgets must be positive integers")
    if type(repetitions) is not int or repetitions < 1:
        raise ExpertEvaluationError("benchmark repetitions must be positive")
    candidate_path = Path(artifact).expanduser().resolve()
    records: list[dict[str, object]] = []
    for budget in budgets:
        candidate = ControllerSpec("candidate", "guided", budget, str(candidate_path))
        for fixture in STRATEGIC_FIXTURES:
            if fixture.evidence is FixtureEvidence.FORCED:
                continue
            for repetition in range(repetitions):
                row = _fixture_candidate(candidate, fixture.fixture_id, repetition)
                records.append({"budget": budget, **asdict(row)})
    summary = {}
    for budget in budgets:
        rows = [row for row in records if row["budget"] == budget]
        latencies = [float(row["latency_seconds"]) for row in rows]
        summary[str(budget)] = {
            "passes": sum(bool(row["passed"]) for row in rows),
            "samples": len(rows),
            "pass_rate": sum(bool(row["passed"]) for row in rows) / len(rows),
            "latency_seconds": {
                "mean": statistics.mean(latencies),
                "p50": _percentile(latencies, 0.5),
                "p95": _percentile(latencies, 0.95),
                "maximum": max(latencies),
            },
        }
    return {
        "format_version": EVALUATION_FORMAT_VERSION,
        "artifact_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "repetitions": repetitions,
        "budgets": summary,
        "records": records,
    }


__all__ = (
    "ControllerSpec",
    "EVALUATION_FORMAT_VERSION",
    "EvaluationGame",
    "ExpertEvaluationError",
    "FixtureEvaluation",
    "benchmark_guided_candidate",
    "evaluate_expert_candidate",
)
