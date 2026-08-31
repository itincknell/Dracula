"""Render the controlled prior-versus-run-1 response-ranker comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from run_teacher_v2_hybrid_experiment import _paired_analysis


TIMING_FIELDS = {
    "latency_seconds",
    "model_inference_seconds",
    "peak_rss_bytes",
}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _mib(value: int) -> str:
    return f"{value / (1024 * 1024):.1f}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime(item)
            for key, item in value.items()
            if key not in TIMING_FIELDS
        }
    if isinstance(value, list):
        return [_strip_runtime(item) for item in value]
    return value


def _record_index(document: dict[str, Any], mode: str) -> dict[tuple[str, str], Any]:
    return {
        (record["kind"], record["key"]): _strip_runtime(record)
        for record in document["records"][mode]
    }


def _assert_replay(
    left: dict[str, Any],
    right: dict[str, Any],
    mode: str,
    label: str,
) -> None:
    if _record_index(left, mode) != _record_index(right, mode):
        raise ValueError(f"deterministic replay mismatch: {label}")


def _fixture_index(
    document: dict[str, Any],
    mode: str,
) -> dict[str, dict[str, Any]]:
    return {
        record["key"]: record
        for record in document["records"][mode]
        if record["kind"] == "fixture"
    }


def _work_metrics(document: dict[str, Any], mode: str) -> dict[str, int | float]:
    records = [
        record
        for record in document["records"][mode]
        if record["kind"] != "game"
    ]
    requests = sum(int(record["response_requests"]) for record in records)
    hits = sum(int(record["response_cache_hits"]) for record in records)
    return {
        "requests": requests,
        "unique": sum(
            int(record["unique_response_evaluations"]) for record in records
        ),
        "hits": hits,
        "hit_rate": hits / requests if requests else 0.0,
        "candidate_actions": sum(
            int(record["response_candidate_actions"]) for record in records
        ),
    }


def _wilson(wins: int, total: int) -> tuple[float, float]:
    if not total:
        return (math.nan, math.nan)
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def _offline_top_two(report: Path) -> dict[str, dict[str, float | int]]:
    text = report.read_text(encoding="utf-8")
    recall = re.search(
        r"\| Top-two recall \| [^|]+ \| (?P<prior>[^|]+)% \| "
        r"\*\*(?P<new>[^|]+)%\*\* \|",
        text,
    )
    missing = re.search(
        r"\| Teacher group missing from top two \| [^|]+ \| "
        r"(?P<prior>[\d,]+) \| \*\*(?P<new>[\d,]+)\*\* \|",
        text,
    )
    examples = re.search(r"Validation examples \| (?P<count>[\d,]+)", text)
    if recall is None or missing is None or examples is None:
        raise ValueError("response-ranker report no longer matches its contract")
    total = int(examples.group("count").replace(",", ""))
    return {
        "prior": {
            "recall": float(recall.group("prior")) / 100.0,
            "missing": int(missing.group("prior").replace(",", "")),
            "total": total,
        },
        "run1": {
            "recall": float(recall.group("new")) / 100.0,
            "missing": int(missing.group("new").replace(",", "")),
            "total": total,
        },
    }


def _fixture_table(
    prior: dict[str, Any],
    run1: dict[str, Any],
) -> list[str]:
    pure = _fixture_index(run1, "pure")
    prior_top2 = _fixture_index(prior, "student-top-2")
    new_top2 = _fixture_index(run1, "student-top-2")
    direct = _fixture_index(run1, "student-direct")
    rows = [
        "| Suite | Fixture | P | Expected | Pure | Prior top-2 | "
        "Run-1 top-2 | Run-1 direct |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]

    def cell(record: dict[str, Any]) -> str:
        status = "pass" if record["strategic_group_pass"] else "fail"
        return f"{status}; g{record['selected_representative_action_index']}"

    for key, reference in sorted(
        pure.items(),
        key=lambda item: (item[1]["suite"], item[1]["fixture_id"]),
    ):
        expected = ",".join(
            str(value) for value in reference["expected_action_indices"]
        )
        rows.append(
            f"| {reference['suite']} | `{reference['fixture_id']}` | "
            f"{reference['placement_number']} | `{expected}` | "
            f"{cell(reference)} | {cell(prior_top2[key])} | "
            f"{cell(new_top2[key])} | {cell(direct[key])} |"
        )
    return rows


def _fixture_difference_rows(
    paired: dict[str, dict[str, Any]],
) -> list[str]:
    rows = [
        "| Controller | Fixture | P | Pure group | Hybrid group | "
        "Pure Q(pure) | Pure Q(hybrid) | Hybrid Q(pure) | "
        "Hybrid Q(hybrid) | Regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for label, analysis in paired.items():
        disagreements = [
            row
            for row in analysis["disagreements"]
            if row["kind"] == "fixture"
        ]
        for row in sorted(disagreements, key=lambda value: value["key"]):
            rows.append(
                f"| {label} | `{row['suite']}:{row['fixture_id']}` | "
                f"{row['placement_number']} | "
                f"{row['teacher_selected_representative']} | "
                f"{row['hybrid_selected_representative']} | "
                f"{_number(row['teacher_value_of_teacher_action'], 6)} | "
                f"{_number(row['teacher_value_of_hybrid_action'], 6)} | "
                f"{_number(row['hybrid_value_of_teacher_action'], 6)} | "
                f"{_number(row['hybrid_value_of_hybrid_action'], 6)} | "
                f"{_number(row['teacher_value_regret'], 6)} |"
            )
    return rows


def render(
    *,
    prior_path: Path,
    run1_path: Path,
    historical_path: Path,
    ranker_report: Path,
) -> str:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    run1 = json.loads(run1_path.read_text(encoding="utf-8"))
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    expected_search = {
        "outer_simulations": 32,
        "response_completions": 4,
        "placement_request_seeds": 5,
        "game_seed_count": 12,
        "game_roles_per_seed": 2,
        "workers": 4,
    }
    for document in (prior, run1):
        for key, value in expected_search.items():
            if document["configuration"].get(key) != value:
                raise ValueError(f"comparison configuration drift: {key}")

    _assert_replay(prior, run1, "pure", "independent pure reruns")
    _assert_replay(prior, historical, "student-top-2", "prior ranker rerun")
    _assert_replay(run1, historical, "pure", "historical pure control")

    prior_analysis = _paired_analysis(prior["records"])["student-top-2"]
    paired = {
        "prior top-2": prior_analysis,
        "run-1 top-2": run1["paired_decisions"]["student-top-2"],
        "run-1 direct": run1["paired_decisions"]["student-direct"],
    }
    controller_documents = {
        "prior top-2": (prior, "student-top-2"),
        "run-1 top-2": (run1, "student-top-2"),
        "run-1 direct": (run1, "student-direct"),
    }
    offline = _offline_top_two(ranker_report)
    pure_prior = prior["metrics"]["pure"]
    pure_run1 = run1["metrics"]["pure"]

    lines = [
        "# Teacher v2 hybrid experiment 002",
        "",
        "## Result",
        "",
        "The larger run-1 response ranker did not produce a clear hybrid "
        "improvement over the prior smoke ranker. Run-1 top-2 improved the "
        "fixed defensive fixtures to 13/14, versus 8/14 for prior top-2 and "
        "9/14 for pure Teacher v2, but it finished 8–16 against pure over "
        "the 24 paired games. Prior top-2 repeated its historical 11–13 "
        "record exactly. Run-1 direct finished 5–19.",
        "",
        "The computational result is stable. Both top-2 controllers removed "
        "about 72% of pure Teacher v2's inner terminal evaluations and cut "
        "isolated median decision latency by roughly 3.6×. Run-1 direct "
        "removed all inner response completions and was about 19× faster, "
        "but its game result and score differential were substantially worse.",
        "",
        "The evidence is mixed rather than promotive: run 1 learned a slightly "
        "different ranking that helped the curated defensive fixtures but "
        "regressed on this small absolute game sample and on fixed-state "
        "agreement/regret. Selection for manual testing remains the user's "
        "decision.",
        "",
        "## Method",
        "",
        "- Fixed decisions: 12 constructive fixtures, 14 defensive fixtures, "
        "and 24 role/dealer-balanced placement states repeated over five "
        "request seeds, for 146 decisions per controller.",
        "- Games: 12 deterministic deck seeds, each hybrid versus pure "
        "Teacher v2 in both roles, for 24 six-round games and 144 rounds per "
        "hybrid. Dealer assignment alternated across rounds.",
        "- Every controller retained 32 outer simulations and exact engine "
        "terminal scoring. Top-2 retained four-completion shallow evaluation "
        "for the model's two selected response groups.",
        "- The independently rerun pure matrices, and the rerun prior-ranker "
        "matrix versus its historical artifact, are tensor/search-result exact "
        "after excluding elapsed-time and RSS fields.",
        f"- Prior ranker: `{prior['configuration']['response_ranker_artifact_digest']}`.",
        f"- Run-1 ranker: `{run1['configuration']['response_ranker_artifact_digest']}`.",
        f"- Prior raw artifact: `{prior_path}` (`sha256:{_sha256(prior_path)}`).",
        f"- Run-1 raw artifact: `{run1_path}` (`sha256:{_sha256(run1_path)}`).",
        "",
        "## Strategic fixtures",
        "",
        *_fixture_table(prior, run1),
        "",
        "Totals:",
        "",
        "| Controller | Constructive | Defensive |",
        "| --- | ---: | ---: |",
    ]
    fixture_sources = {
        "pure": (run1, "pure"),
        **controller_documents,
    }
    for label, (document, mode) in fixture_sources.items():
        fixture = document["metrics"][mode]["fixtures"]
        lines.append(
            f"| {label} | "
            f"{fixture['constructive']['strategic_group_passes']}/"
            f"{fixture['constructive']['count']} | "
            f"{fixture['defensive']['strategic_group_passes']}/"
            f"{fixture['defensive']['count']} |"
        )

    lines.extend(
        [
            "",
            "## Paired game results",
            "",
            "| Hybrid | Record vs pure | Game VP (95% Wilson) | Round VP | "
            "Mean round differential |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, (document, mode) in controller_documents.items():
        metrics = document["metrics"][mode]
        games = metrics["games"]
        losses = games["count"] - games["wins"] - games["ties"]
        low, high = _wilson(games["wins"], games["count"])
        lines.append(
            f"| {label} | {games['wins']}-{losses} | "
            f"{_pct(games['victory_percentage'])} "
            f"({_pct(low)}–{_pct(high)}) | "
            f"{_pct(metrics['rounds']['victory_percentage'])} | "
            f"{metrics['rounds']['mean_score_differential']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "The intervals describe binomial sampling uncertainty only. With "
            "12 decks, the game results distinguish obvious weakness from "
            "parity more reliably than they distinguish the two top-2 "
            "rankers from each other.",
            "",
            "### Role and dealer splits",
            "",
            "| Hybrid | Queen games | King games | Queen round VP / diff | "
            "King round VP / diff |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, (document, mode) in controller_documents.items():
        metrics = document["metrics"][mode]
        games = metrics["games"]
        rounds = metrics["rounds"]
        lines.append(
            f"| {label} | {games['queen']['wins']}-"
            f"{games['queen']['games'] - games['queen']['wins']} | "
            f"{games['king']['wins']}-"
            f"{games['king']['games'] - games['king']['wins']} | "
            f"{_pct(rounds['queen']['victory_percentage'])} / "
            f"{rounds['queen']['mean_score_differential']:+.2f} | "
            f"{_pct(rounds['king']['victory_percentage'])} / "
            f"{rounds['king']['mean_score_differential']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "| Hybrid | Dealer round VP / diff | Non-dealer round VP / diff |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, (document, mode) in controller_documents.items():
        rounds = document["metrics"][mode]["rounds"]
        lines.append(
            f"| {label} | "
            f"{_pct(rounds['dealer']['victory_percentage'])} / "
            f"{rounds['dealer']['mean_score_differential']:+.2f} | "
            f"{_pct(rounds['non_dealer']['victory_percentage'])} / "
            f"{rounds['non_dealer']['mean_score_differential']:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Fixed-state agreement and Teacher-value regret",
            "",
            "| Placement | Prior top-2 | Run-1 top-2 | Run-1 direct |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for placement in range(1, 8):
        cells = []
        for analysis in paired.values():
            row = analysis["by_placement"][str(placement)]
            cells.append(
                f"{_pct(row['agreement'])} / "
                f"{row['mean_teacher_value_regret']:.3f}"
            )
        lines.append(
            f"| {placement} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    overall = [
        f"{_pct(analysis['strategic_group_agreement'])} / "
        f"{analysis['mean_teacher_value_regret']:.3f}"
        for analysis in paired.values()
    ]
    lines.extend(
        [
            f"| Overall | {overall[0]} | {overall[1]} | {overall[2]} |",
            "",
            "Each cell is strategic-group agreement with pure Teacher v2 / "
            "mean pure-Teacher value regret. Run-1 top-2 agreed less often "
            "and had higher regret than prior top-2 on this matrix.",
            "",
            "### Teacher group absent from neural top two",
            "",
            "| Ranker | Held-out response states | Missing | Omission rate |",
            "| --- | ---: | ---: | ---: |",
            f"| Prior smoke | {offline['prior']['total']:,} | "
            f"{offline['prior']['missing']:,} | "
            f"{_pct(1.0 - float(offline['prior']['recall']))} |",
            f"| Run 1 | {offline['run1']['total']:,} | "
            f"{offline['run1']['missing']:,} | "
            f"{_pct(1.0 - float(offline['run1']['recall']))} |",
            "",
            "This is measured at the actual shallow-response boundary over "
            "the shared 38,652-row held-out response set. It is not inferred "
            "from root actions. Run 1 reduced omissions by 66 rows (0.17 "
            "percentage points), a small offline change.",
            "",
            "## Efficiency",
            "",
            "The fixed matrix contains the same 146 decisions for every "
            "controller. Each hybrid is compared with the pure timing from "
            "its contemporaneous run; pure behavior and work counts were "
            "identical across the two runs.",
            "",
            "| Controller | p50 | p95 | Max | Inner terminals | Reduction | "
            "Model calls / time | ms/call | Peak RSS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| pure (prior run) | {pure_prior['latency_seconds']['p50']:.3f}s | "
            f"{pure_prior['latency_seconds']['p95']:.3f}s | "
            f"{pure_prior['latency_seconds']['maximum']:.3f}s | "
            f"{pure_prior['response_terminal_evaluations']:,} | 0.0% | "
            f"0 / 0.000s | — | {_mib(pure_prior['peak_rss_bytes'])} MiB |",
            f"| pure (run-1 run) | {pure_run1['latency_seconds']['p50']:.3f}s | "
            f"{pure_run1['latency_seconds']['p95']:.3f}s | "
            f"{pure_run1['latency_seconds']['maximum']:.3f}s | "
            f"{pure_run1['response_terminal_evaluations']:,} | 0.0% | "
            f"0 / 0.000s | — | {_mib(pure_run1['peak_rss_bytes'])} MiB |",
        ]
    )
    for label, (document, mode) in controller_documents.items():
        metrics = document["metrics"][mode]
        inference = metrics["model_inference"]
        pure_metrics = document["metrics"]["pure"]
        reduction = 1.0 - (
            metrics["response_terminal_evaluations"]
            / pure_metrics["response_terminal_evaluations"]
        )
        lines.append(
            f"| {label} | {metrics['latency_seconds']['p50']:.3f}s | "
            f"{metrics['latency_seconds']['p95']:.3f}s | "
            f"{metrics['latency_seconds']['maximum']:.3f}s | "
            f"{metrics['response_terminal_evaluations']:,} | "
            f"{_pct(reduction)} | {inference['calls']:,} / "
            f"{inference['seconds']:.3f}s | "
            f"{inference['per_call_milliseconds']:.3f} | "
            f"{_mib(metrics['peak_rss_bytes'])} MiB |"
        )

    lines.extend(
        [
            "",
            "### Response-cache work on the fixed matrix",
            "",
            "| Controller | Requests | Unique evaluations | Cache hits | "
            "Hit rate | Candidate actions evaluated |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    work_sources = {
        "pure": (run1, "pure"),
        **controller_documents,
    }
    for label, (document, mode) in work_sources.items():
        work = _work_metrics(document, mode)
        lines.append(
            f"| {label} | {work['requests']:,} | {work['unique']:,} | "
            f"{work['hits']:,} | {_pct(float(work['hit_rate']))} | "
            f"{work['candidate_actions']:,} |"
        )

    lines.extend(
        [
            "",
            "### Concurrent gameplay latency",
            "",
            "| Hybrid | Decisions | p50 | p95 | Max | Inner terminals | "
            "Model calls / time |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, (document, mode) in controller_documents.items():
        metrics = document["metrics"][mode]
        latency = metrics["gameplay_latency_seconds"]
        inference = metrics["gameplay_model_inference"]
        lines.append(
            f"| {label} | {metrics['gameplay_decision_count']} | "
            f"{latency['p50']:.3f}s | {latency['p95']:.3f}s | "
            f"{latency['maximum']:.3f}s | "
            f"{metrics['gameplay_response_terminal_evaluations']:,} | "
            f"{inference['calls']:,} / {inference['seconds']:.2f}s |"
        )

    lines.extend(
        [
            "",
            "## Every meaningful strategic-fixture disagreement",
            "",
            "All 26 fixtures are shown above, including pass/fail changes. "
            "The table below exhaustively lists every fixture where a hybrid "
            "selected a different strategic group from pure Teacher v2. "
            "`Q` values are the exact normalized four-completion/outer-search "
            "means stored in the raw artifacts; the report prints six decimal "
            "places. All 266 fixed-state disagreements, including repeated "
            "synthetic placement states, remain in the raw artifacts with "
            "full-precision group visits and values.",
            "",
            *_fixture_difference_rows(paired),
            "",
            "## Verification",
            "",
            "- Deterministic replay: 146/146 pure decisions matched across "
            "both fresh runs and the historical control after excluding only "
            "timing/RSS. The prior top-2 rerun matched all 146 decisions and "
            "all 24 complete games from the historical experiment.",
            "- Pure Teacher v2 invariance: selected actions, strategic groups, "
            "visits, action means, response-work counters, and exact terminal "
            "values were unchanged.",
            "- Privacy and legality: the focused search/model suite passed. "
            "Student inference accepts only actor-relative information-state "
            "tensors and legal masks; no authoritative hand, stock, seed, "
            "tree, or private model state enters it.",
            "- Exact scoring: every outer terminal value came from the "
            "deterministic engine. No model value head or scoring "
            "approximation was enabled.",
            "- Legal actions: no fixed decision or game produced an illegal "
            "action, and forced placements bypassed response inference.",
            "",
            "## Interpretation",
            "",
            "Run 1 is not a demonstrated upgrade to the hybrid. Its curated "
            "defensive fixture result is stronger, but its 8–16 game record, "
            "41.1% fixed-state agreement, and 0.100 mean Teacher-value regret "
            "are worse than the prior top-2 ranker's 11–13, 45.2%, and 0.084. "
            "The top-two omission rate barely changed. The game sample is "
            "small enough that the two top-2 records are not a precise "
            "strength ranking, but it provides no positive evidence that the "
            "larger response dataset improved gameplay.",
            "",
            "Both top-2 hybrids retain the useful efficiency result: about "
            "72% fewer inner terminal evaluations and roughly 3.6× lower "
            "isolated median latency. Whether the run-1 controller's defensive "
            "fixture behavior is worth manual browser comparison, or whether "
            "the prior top-2 controller remains the better experimental "
            "candidate, is deferred to the user.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--run1", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--ranker-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = render(
        prior_path=args.prior,
        run1_path=args.run1,
        historical_path=args.historical,
        ranker_report=args.ranker_report,
    )
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} sha256={_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
