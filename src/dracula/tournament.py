"""Reproducible round-robin tournaments for manual policy selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch
from torch import Tensor, nn

from dracula.collection import PolicyVersion
from dracula.evaluation import (
    EvaluationFixtureResult,
    EvaluationSchedule,
    SplitMetrics,
    build_evaluation_schedule,
    evaluate_matchups,
)
from dracula.models import HIDDEN_SIZE
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.training import TrainingSuite, _load_checkpoint

TOURNAMENT_FORMAT_VERSION = "dracula-tournament-v2"
TOURNAMENT_BOOTSTRAP_NAMESPACE = "dracula-tournament-bootstrap-v1"
RANDOM_LEGAL_IDENTITY = PolicyVersion("random-legal", "uniform-v1")
DEFAULT_GENERATIONS = 256
DEFAULT_BOOTSTRAP_SAMPLES = 2_000


class TournamentContractViolation(ValueError):
    """A candidate tournament does not form a reproducible comparison."""


class UniformLegalPolicy(nn.Module):
    """Supply uniform legal-action probabilities without learned parameters."""

    def initial_hidden(self) -> Tensor:
        return torch.zeros(HIDDEN_SIZE, dtype=torch.float32)

    def forward(
        self, observation: Tensor, legal_mask: Tensor, hidden_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        if observation.shape != (875,) or observation.dtype is not torch.bool:
            raise TournamentContractViolation("random controller observation is invalid")
        if legal_mask.shape != (4, 8) or legal_mask.dtype is not torch.bool:
            raise TournamentContractViolation("random controller legal mask is invalid")
        if hidden_state.shape != (HIDDEN_SIZE,) or hidden_state.dtype is not torch.float32:
            raise TournamentContractViolation("random controller hidden state is invalid")
        return (
            torch.zeros(4, 8, dtype=torch.float32, device=observation.device),
            hidden_state.clone(),
        )


@dataclass(frozen=True, slots=True)
class TournamentArtifacts:
    json_path: Path
    markdown_path: Path


def build_candidate_tournament_schedule(
    candidates: Sequence[PolicyVersion],
    lane_roots: Sequence[str],
    *,
    generation_count: int = DEFAULT_GENERATIONS,
    start_game_counter: int = 0,
) -> EvaluationSchedule:
    ordered = tuple(sorted(candidates))
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise TournamentContractViolation("tournament candidates must be distinct")
    if len({candidate.policy_id for candidate in ordered}) != len(ordered):
        raise TournamentContractViolation("candidate policy IDs must be distinct")
    if RANDOM_LEGAL_IDENTITY in ordered:
        raise TournamentContractViolation("random legal is a control, not a candidate")

    schedule = build_evaluation_schedule(
        (*ordered, RANDOM_LEGAL_IDENTITY),
        lane_roots,
        generation_count=generation_count,
        start_game_counter=start_game_counter,
    )
    matchups = len(tuple(combinations(ordered, 2))) + len(ordered)
    expected = matchups * len(tuple(lane_roots)) * generation_count
    if len(schedule.fixtures) != expected:
        raise TournamentContractViolation("tournament fixture count is incorrect")
    return schedule


def run_candidate_tournament(
    *,
    run_directory: str | Path,
    checkpoint: str | Path,
    lane_roots: Sequence[str],
    generation_count: int = DEFAULT_GENERATIONS,
    start_game_counter: int = 0,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    output_stem: str | Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TournamentArtifacts:
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise TournamentContractViolation("bootstrap sample count must be positive")

    suite = TrainingSuite.open(run_directory)
    root = suite.run_directory
    checkpoint_path = _inside_run(root, checkpoint)
    manifest_hash = str(suite.state["manifest_hash"])
    snapshot = _load_checkpoint(checkpoint_path, suite.config, manifest_hash)
    if len(snapshot.policies) != 5:
        raise TournamentContractViolation("selection tournaments require five candidates")

    candidates = tuple(sorted(snapshot.policies))
    controllers: dict[PolicyVersion, nn.Module] = {
        **snapshot.policies,
        RANDOM_LEGAL_IDENTITY: UniformLegalPolicy(),
    }
    schedule = build_candidate_tournament_schedule(
        candidates,
        lane_roots,
        generation_count=generation_count,
        start_game_counter=start_game_counter,
    )
    results = evaluate_matchups(
        run_root_seed=f"{suite.config.run.root_seed}-candidate-tournament",
        schedule=schedule,
        policies=controllers,
        progress_callback=progress_callback,
    )
    document = _tournament_document(
        run_id=suite.config.run.run_id,
        checkpoint=checkpoint_path,
        lane_roots=tuple(lane_roots),
        schedule=schedule,
        results=results,
        candidates=candidates,
        bootstrap_samples=bootstrap_samples,
    )
    stem = _inside_run(root, output_stem)
    json_path = stem.with_suffix(".json")
    markdown_path = stem.with_suffix(".md")
    _atomic_write(
        json_path,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(markdown_path, _markdown_report(document).encode("utf-8"))
    return TournamentArtifacts(json_path=json_path, markdown_path=markdown_path)


def _tournament_document(
    *,
    run_id: str,
    checkpoint: Path,
    lane_roots: tuple[str, ...],
    schedule: EvaluationSchedule,
    results: Sequence[EvaluationFixtureResult],
    candidates: tuple[PolicyVersion, ...],
    bootstrap_samples: int,
) -> dict[str, object]:
    candidate_set = frozenset(candidates)
    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        opponent_rows: list[dict[str, object]] = []
        opponent_rates: list[float] = []
        for opponent in candidates:
            if opponent == candidate:
                continue
            metrics = _metrics_for(
                candidate, results, opponents=frozenset((opponent,))
            )
            rate = metrics["overall"]["victory_percentage"]  # type: ignore[index]
            if rate is None:
                raise TournamentContractViolation("candidate matchup has no games")
            opponent_rates.append(float(rate))
            opponent_rows.append(
                {
                    "opponent_policy_id": opponent.policy_id,
                    "opponent_version": opponent.version,
                    **metrics,
                }
            )
        opponent_rows.sort(key=lambda row: str(row["opponent_policy_id"]))
        versus_candidates = _metrics_for(
            candidate,
            results,
            opponents=candidate_set - frozenset((candidate,)),
        )
        versus_random = _metrics_for(
            candidate,
            results,
            opponents=frozenset((RANDOM_LEGAL_IDENTITY,)),
        )
        # The stored opponent rows are already small; selecting by rate avoids
        # making report order part of the ranking contract.
        worst_row = min(
            opponent_rows,
            key=lambda row: float(row["overall"]["victory_percentage"]),  # type: ignore[index]
        )
        candidate_rows.append(
            {
                "policy_id": candidate.policy_id,
                "version": candidate.version,
                "versus_candidates": versus_candidates,
                "versus_random_legal": versus_random,
                "macro_victory_percentage": statistics.mean(opponent_rates),
                "opponent_victory_standard_deviation": statistics.pstdev(
                    opponent_rates
                ),
                "worst_opponent": {
                    "policy_id": worst_row["opponent_policy_id"],
                    "version": worst_row["opponent_version"],
                    "victory_percentage": worst_row["overall"][  # type: ignore[index]
                        "victory_percentage"
                    ],
                },
                "opponents": opponent_rows,
            }
        )

    candidate_rows.sort(key=lambda row: str(row["policy_id"]))
    candidate_rows.sort(
        key=lambda row: (
            float(row["macro_victory_percentage"]),
            float(row["worst_opponent"]["victory_percentage"]),  # type: ignore[index]
            float(
                row["versus_random_legal"]["overall"][  # type: ignore[index]
                    "victory_percentage"
                ]
            ),
        ),
        reverse=True,
    )

    result_digest = _fixture_result_digest(results)
    bootstrap = _bootstrap_ranking(
        candidates=candidates,
        schedule=schedule,
        results=results,
        samples=bootstrap_samples,
        seed=derive_seed(
            TOURNAMENT_BOOTSTRAP_NAMESPACE,
            run_id,
            _sha256(checkpoint),
            result_digest,
            str(bootstrap_samples),
        ),
    )
    tier = 1
    for rank, row in enumerate(candidate_rows, start=1):
        identity = str(row["policy_id"])
        analysis = bootstrap["candidates"][identity]  # type: ignore[index]
        if rank > 1:
            previous = str(candidate_rows[rank - 2]["policy_id"])
            probability = bootstrap["superiority"][previous][identity]  # type: ignore[index]
            if float(probability) >= 0.95:
                tier += 1
        row.update(
            {
                "rank": rank,
                "tier": tier,
                "macro_victory_percentage_ci_95": analysis["macro_ci_95"],
                "rank_probabilities": analysis["rank_probabilities"],
                "first_place_probability": analysis["rank_probabilities"][0],
            }
        )

    games_per_matchup = schedule.lane_count * schedule.generation_count
    candidate_matchups = len(tuple(combinations(candidates, 2)))
    return {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "run_id": run_id,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "fixtures": {
            "lane_roots": list(lane_roots),
            "generation_count": schedule.generation_count,
            "start_game_counter": schedule.start_game_counter,
            "games_per_matchup": games_per_matchup,
            "total_games": len(schedule.fixtures),
            "candidate_matchups": candidate_matchups,
            "random_control_matchups": len(candidates),
            "queen_games_per_matchup": games_per_matchup // 2,
            "king_games_per_matchup": games_per_matchup // 2,
        },
        "ranking": {
            "primary_metric": "macro victory percentage against candidate opponents",
            "random_control_included": False,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "shared lane-generation deck fixture",
            "resolved": len({int(row["tier"]) for row in candidate_rows})
            == len(candidate_rows),
            "superiority": bootstrap["superiority"],
        },
        "random_legal_controller": {
            "policy_id": RANDOM_LEGAL_IDENTITY.policy_id,
            "version": RANDOM_LEGAL_IDENTITY.version,
            "implementation": "stateless uniform sampling after legal masking",
            "learned_parameters": 0,
        },
        "fixture_result_sha256": result_digest,
        "candidates": candidate_rows,
    }


def _bootstrap_ranking(
    *,
    candidates: tuple[PolicyVersion, ...],
    schedule: EvaluationSchedule,
    results: Sequence[EvaluationFixtureResult],
    samples: int,
    seed: bytes,
) -> dict[str, object]:
    fixture_by_id = {fixture.fixture_id: fixture for fixture in schedule.fixtures}
    if len(fixture_by_id) != len(schedule.fixtures):
        raise TournamentContractViolation("tournament fixture IDs are not unique")
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    block_scores: dict[str, list[float]] = {}
    block_counts: dict[str, list[int]] = {}
    for result in results:
        fixture = fixture_by_id.get(result.fixture_id)
        if fixture is None:
            raise TournamentContractViolation("result does not belong to the schedule")
        if RANDOM_LEGAL_IDENTITY in (result.queen, result.king):
            continue
        scores = block_scores.setdefault(fixture.game_seed, [0.0] * len(candidates))
        counts = block_counts.setdefault(fixture.game_seed, [0] * len(candidates))
        for identity in (result.queen, result.king):
            index = candidate_index[identity]
            scores[index] += _game_score(result, identity)
            counts[index] += 1

    expected_opponents = len(candidates) - 1
    vectors: list[tuple[float, ...]] = []
    for game_seed in sorted(block_scores):
        if block_counts[game_seed] != [expected_opponents] * len(candidates):
            raise TournamentContractViolation("bootstrap block is missing a candidate matchup")
        vectors.append(
            tuple(score / expected_opponents for score in block_scores[game_seed])
        )
    expected_blocks = schedule.lane_count * schedule.generation_count
    if len(vectors) != expected_blocks:
        raise TournamentContractViolation("bootstrap block count is incorrect")

    stream = Sha256CounterStream(seed)
    estimates: list[list[float]] = [[] for _ in candidates]
    rank_counts = [[0.0] * len(candidates) for _ in candidates]
    for _ in range(samples):
        totals = [0.0] * len(candidates)
        for _ in range(len(vectors)):
            vector = vectors[stream.randbelow(len(vectors))]
            for index, value in enumerate(vector):
                totals[index] += value
        means = [total / len(vectors) for total in totals]
        for index, value in enumerate(means):
            estimates[index].append(value)
        ordered_indices = sorted(
            range(len(candidates)), key=means.__getitem__, reverse=True
        )
        position = 0
        while position < len(ordered_indices):
            end = position + 1
            while (
                end < len(ordered_indices)
                and means[ordered_indices[end]] == means[ordered_indices[position]]
            ):
                end += 1
            share = 1.0 / (end - position)
            for candidate_index_in_tie in ordered_indices[position:end]:
                for rank_index in range(position, end):
                    rank_counts[candidate_index_in_tie][rank_index] += share
            position = end

    candidate_analysis: dict[str, object] = {}
    for index, candidate in enumerate(candidates):
        ordered = sorted(estimates[index])
        candidate_analysis[candidate.policy_id] = {
            "macro_ci_95": [
                _percentile(ordered, 0.025),
                _percentile(ordered, 0.975),
            ],
            "rank_probabilities": [count / samples for count in rank_counts[index]],
        }
    superiority: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(candidates):
        row: dict[str, float] = {}
        for right_index, right in enumerate(candidates):
            if left == right:
                row[right.policy_id] = 0.5
                continue
            wins = 0.0
            for left_value, right_value in zip(
                estimates[left_index], estimates[right_index], strict=True
            ):
                wins += 1.0 if left_value > right_value else 0.5 if left_value == right_value else 0.0
            row[right.policy_id] = wins / samples
        superiority[left.policy_id] = row
    return {"candidates": candidate_analysis, "superiority": superiority}


def _metrics_for(
    identity: PolicyVersion,
    results: Sequence[EvaluationFixtureResult],
    *,
    opponents: frozenset[PolicyVersion],
) -> dict[str, object]:
    selected = _select_results(identity, results, opponents)
    queen = [result for result in selected if result.queen == identity]
    king = [result for result in selected if result.king == identity]
    return {
        "overall": _split_dict(_split_metrics(identity, selected)),
        "queen": _split_dict(_split_metrics(identity, queen)),
        "king": _split_dict(_split_metrics(identity, king)),
    }


def _select_results(
    identity: PolicyVersion,
    results: Sequence[EvaluationFixtureResult],
    opponents: frozenset[PolicyVersion],
) -> list[EvaluationFixtureResult]:
    return [
        result
        for result in results
        if identity in (result.queen, result.king)
        and (result.king if result.queen == identity else result.queen) in opponents
    ]


def _split_metrics(
    identity: PolicyVersion, selected: Sequence[EvaluationFixtureResult]
) -> SplitMetrics:
    wins = sum(result.winner == identity for result in selected)
    ties = sum(result.winner is None for result in selected)
    returns: list[float] = []
    for result in selected:
        returns.extend(
            result.queen_round_returns
            if result.queen == identity
            else result.king_round_returns
        )
    games = len(selected)
    return SplitMetrics(
        games=games,
        wins=wins,
        ties=ties,
        victory_percentage=(wins + 0.5 * ties) / games if games else None,
        mean_round_return=sum(returns) / len(returns) if returns else None,
    )


def _split_dict(metrics: SplitMetrics) -> dict[str, object]:
    successes = metrics.wins + 0.5 * metrics.ties
    return {
        "games": metrics.games,
        "wins": metrics.wins,
        "ties": metrics.ties,
        "victory_percentage": metrics.victory_percentage,
        "victory_percentage_ci_95": (
            list(_wilson_interval(successes, metrics.games)) if metrics.games else None
        ),
        "mean_round_return": metrics.mean_round_return,
    }


def _wilson_interval(successes: float, games: int) -> tuple[float, float]:
    if games <= 0 or not 0 <= successes <= games:
        raise TournamentContractViolation("Wilson interval inputs are invalid")
    z = 1.959963984540054
    proportion = successes / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _game_score(result: EvaluationFixtureResult, identity: PolicyVersion) -> float:
    if identity not in (result.queen, result.king):
        raise TournamentContractViolation("score identity did not play the fixture")
    return 0.5 if result.winner is None else 1.0 if result.winner == identity else 0.0


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered or not 0 <= quantile <= 1:
        raise TournamentContractViolation("percentile inputs are invalid")
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _markdown_report(document: Mapping[str, object]) -> str:
    fixtures = document["fixtures"]
    ranking = document["ranking"]
    candidates = document["candidates"]
    lines = [
        "# Candidate tournament",
        "",
        f"Run: `{document['run_id']}`  ",
        f"Checkpoint: `{document['checkpoint']['path']}`  ",  # type: ignore[index]
        f"Games: `{fixtures['total_games']}`  ",  # type: ignore[index]
        (
            f"Games per matchup: `{fixtures['games_per_matchup']}` "  # type: ignore[index]
            f"({fixtures['queen_games_per_matchup']} Queen, "  # type: ignore[index]
            f"{fixtures['king_games_per_matchup']} King)"  # type: ignore[index]
        ),
        "",
        "Every candidate plays every other candidate and the stateless random-legal",
        "control on the same deck fixtures. Random-control games do not affect rank.",
        "Bootstrap intervals resample shared lane-generation deck fixtures.",
        "",
        "## Ranking",
        "",
        f"Ordering resolved at the 95% adjacent-comparison threshold: `{'yes' if ranking['resolved'] else 'no'}`",  # type: ignore[index]
        "",
        "| Rank | Tier | Candidate | Candidate win rate (95% bootstrap CI) | First-place probability | Worst opponent | Random legal | Mean round return |",
        "| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for candidate in candidates:  # type: ignore[assignment]
        overall = candidate["versus_candidates"]["overall"]
        interval = candidate["macro_victory_percentage_ci_95"]
        worst = candidate["worst_opponent"]
        random = candidate["versus_random_legal"]["overall"]
        lines.append(
            f"| {candidate['rank']} | {candidate['tier']} | {candidate['version']} "
            f"| {_percentage(candidate['macro_victory_percentage'])} "
            f"({_percentage(interval[0])}–{_percentage(interval[1])}) "
            f"| {_percentage(candidate['first_place_probability'])} "
            f"| {worst['version']} ({_percentage(worst['victory_percentage'])}) "
            f"| {_percentage(random['victory_percentage'])} "
            f"| {_number(overall['mean_round_return'])} |"
        )

    ordered_candidates = sorted(candidates, key=lambda row: str(row["policy_id"]))  # type: ignore[arg-type]
    lines.extend(["", "## Head-to-head victory matrix", ""])
    headings = " | ".join(str(row["version"]) for row in ordered_candidates)
    lines.append(f"| Candidate | {headings} |")
    lines.append(f"| --- | {' | '.join('---:' for _ in ordered_candidates)} |")
    for candidate in ordered_candidates:
        opponent_rates = {
            row["opponent_policy_id"]: row["overall"]["victory_percentage"]
            for row in candidate["opponents"]
        }
        cells = [
            "—"
            if opponent["policy_id"] == candidate["policy_id"]
            else _percentage(opponent_rates[opponent["policy_id"]])
            for opponent in ordered_candidates
        ]
        lines.append(f"| {candidate['version']} | {' | '.join(cells)} |")

    lines.extend(
        [
            "",
            "## Opponent detail",
            "",
            "| Candidate | Opponent | Games | Overall (95% Wilson CI) | Queen | King | Mean round return |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in ordered_candidates:
        for opponent in candidate["opponents"]:
            overall = opponent["overall"]
            interval = overall["victory_percentage_ci_95"]
            lines.append(
                f"| {candidate['version']} | {opponent['opponent_version']} "
                f"| {overall['games']} "
                f"| {_percentage(overall['victory_percentage'])} "
                f"({_percentage(interval[0])}–{_percentage(interval[1])}) "
                f"| {_percentage(opponent['queen']['victory_percentage'])} "
                f"| {_percentage(opponent['king']['victory_percentage'])} "
                f"| {_number(overall['mean_round_return'])} |"
            )

    lines.extend(
        [
            "",
            "## Random-legal control",
            "",
            "| Candidate | Games | Overall | Queen | King | Mean round return |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in ordered_candidates:
        random = candidate["versus_random_legal"]
        overall = random["overall"]
        lines.append(
            f"| {candidate['version']} | {overall['games']} "
            f"| {_percentage(overall['victory_percentage'])} "
            f"| {_percentage(random['queen']['victory_percentage'])} "
            f"| {_percentage(random['king']['victory_percentage'])} "
            f"| {_number(overall['mean_round_return'])} |"
        )
    return "\n".join(lines) + "\n"


def _fixture_result_digest(results: Sequence[EvaluationFixtureResult]) -> str:
    rows = [
        {
            "fixture_id": result.fixture_id,
            "queen": [result.queen.policy_id, result.queen.version],
            "king": [result.king.policy_id, result.king.version],
            "winner": (
                None
                if result.winner is None
                else [result.winner.policy_id, result.winner.version]
            ),
            "queen_round_returns": list(result.queen_round_returns),
            "king_round_returns": list(result.king_round_returns),
        }
        for result in results
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inside_run(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise TournamentContractViolation("tournament path must remain inside the run")
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentage(value: object) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.6g}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dracula-tournament")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--start-counter", type=int, default=0)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument("--output-stem", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    lane_roots = tuple(f"candidate-tournament-lane-{index:02d}" for index in range(12))

    def progress(completed: int, total: int) -> None:
        interval = max(1, total // 20)
        if completed == total or completed % interval == 0:
            print(f"tournament games={completed}/{total}", flush=True)

    artifacts = run_candidate_tournament(
        run_directory=arguments.run_directory,
        checkpoint=arguments.checkpoint,
        lane_roots=lane_roots,
        generation_count=arguments.generations,
        start_game_counter=arguments.start_counter,
        bootstrap_samples=arguments.bootstrap_samples,
        output_stem=arguments.output_stem,
        progress_callback=progress,
    )
    print(artifacts.markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
