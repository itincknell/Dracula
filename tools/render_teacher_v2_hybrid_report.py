"""Render the controlled Teacher v2 hybrid experiment report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

MODES = ("pure", "student-direct", "student-top-2", "student-top-3")
HYBRIDS = MODES[1:]


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _number(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _mib(value: int) -> str:
    return f"{value / (1024 * 1024):.1f}"


def _fixture_cell(record: dict[str, object]) -> str:
    label = "pass" if record["strategic_group_pass"] else "fail"
    return (
        f"{label}; group "
        f"{record['selected_representative_action_index']}"
    )


def _fixture_table(document: dict[str, object]) -> list[str]:
    indexed = {
        mode: {
            record["key"]: record
            for record in document["records"][mode]
            if record["kind"] == "fixture"
        }
        for mode in MODES
    }
    rows = [
        "| Suite | Fixture | Placement | Expected actions | Pure | Direct | "
        "Top-2 | Top-3 |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for key, pure in sorted(
        indexed["pure"].items(),
        key=lambda item: (item[1]["suite"], item[1]["fixture_id"]),
    ):
        rows.append(
            "| {suite} | `{fixture}` | {placement} | `{expected}` | "
            "{pure} | {direct} | {top2} | {top3} |".format(
                suite=pure["suite"],
                fixture=pure["fixture_id"],
                placement=pure["placement_number"],
                expected=",".join(
                    str(value) for value in pure["expected_action_indices"]
                ),
                pure=_fixture_cell(pure),
                direct=_fixture_cell(indexed["student-direct"][key]),
                top2=_fixture_cell(indexed["student-top-2"][key]),
                top3=_fixture_cell(indexed["student-top-3"][key]),
            )
        )
    return rows


def _performance_table(document: dict[str, object]) -> list[str]:
    pure_terminals = document["metrics"]["pure"][
        "response_terminal_evaluations"
    ]
    rows = [
        "| Controller | p50 | p95 | Max | Inner terminals | Reduction | "
        "Model calls | Model time | ms/call | Peak RSS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for mode in MODES:
        metrics = document["metrics"][mode]
        latency = metrics["latency_seconds"]
        inference = metrics["model_inference"]
        terminals = metrics["response_terminal_evaluations"]
        reduction = 1.0 - terminals / pure_terminals
        rows.append(
            "| {mode} | {p50}s | {p95}s | {maximum}s | {terminals:,} | "
            "{reduction} | {calls:,} | {model_seconds:.3f}s | {per_call} | "
            "{rss} MiB |".format(
                mode=mode,
                p50=_seconds(latency["p50"]),
                p95=_seconds(latency["p95"]),
                maximum=_seconds(latency["maximum"]),
                terminals=terminals,
                reduction=_percentage(reduction),
                calls=inference["calls"],
                model_seconds=inference["seconds"],
                per_call=(
                    "—"
                    if inference["per_call_milliseconds"] is None
                    else f"{inference['per_call_milliseconds']:.3f}"
                ),
                rss=_mib(metrics["peak_rss_bytes"]),
            )
        )
    return rows


def _game_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Hybrid subject | Record vs pure | Game VP | Round VP | "
        "Mean round differential |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in HYBRIDS:
        metrics = document["metrics"][mode]
        games = metrics["games"]
        rounds = metrics["rounds"]
        losses = games["count"] - games["wins"] - games["ties"]
        rows.append(
            f"| {mode} | {games['wins']}-{losses}"
            f"{f'-{games['ties']}' if games['ties'] else ''} | "
            f"{_percentage(games['victory_percentage'])} | "
            f"{_percentage(rounds['victory_percentage'])} | "
            f"{rounds['mean_score_differential']:+.2f} |"
        )
    return rows


def _role_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Hybrid | Queen games | King games | Queen round VP / diff | "
        "King round VP / diff |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in HYBRIDS:
        metrics = document["metrics"][mode]
        games = metrics["games"]
        rounds = metrics["rounds"]
        rows.append(
            "| {mode} | {qw}-{ql} | {kw}-{kl} | {qvp} / {qdiff:+.2f} | "
            "{kvp} / {kdiff:+.2f} |".format(
                mode=mode,
                qw=games["queen"]["wins"],
                ql=games["queen"]["games"] - games["queen"]["wins"],
                kw=games["king"]["wins"],
                kl=games["king"]["games"] - games["king"]["wins"],
                qvp=_percentage(rounds["queen"]["victory_percentage"]),
                qdiff=rounds["queen"]["mean_score_differential"],
                kvp=_percentage(rounds["king"]["victory_percentage"]),
                kdiff=rounds["king"]["mean_score_differential"],
            )
        )
    return rows


def _dealer_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Hybrid | Dealer round VP / diff | Non-dealer round VP / diff |",
        "| --- | ---: | ---: |",
    ]
    for mode in HYBRIDS:
        rounds = document["metrics"][mode]["rounds"]
        rows.append(
            "| {mode} | {dvp} / {ddiff:+.2f} | {nvp} / {ndiff:+.2f} |".format(
                mode=mode,
                dvp=_percentage(rounds["dealer"]["victory_percentage"]),
                ddiff=rounds["dealer"]["mean_score_differential"],
                nvp=_percentage(
                    rounds["non_dealer"]["victory_percentage"]
                ),
                ndiff=rounds["non_dealer"]["mean_score_differential"],
            )
        )
    return rows


def _agreement_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Placement | Direct agreement / regret | Top-2 agreement / regret | "
        "Top-3 agreement / regret |",
        "| ---: | ---: | ---: | ---: |",
    ]
    paired = document["paired_decisions"]
    for placement in range(1, 8):
        cells = []
        for mode in HYBRIDS:
            values = paired[mode]["by_placement"][str(placement)]
            cells.append(
                f"{_percentage(values['agreement'])} / "
                f"{values['mean_teacher_value_regret']:.3f}"
            )
        rows.append(
            f"| {placement} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    overall = []
    for mode in HYBRIDS:
        values = paired[mode]
        overall.append(
            f"{_percentage(values['strategic_group_agreement'])} / "
            f"{values['mean_teacher_value_regret']:.3f}"
        )
    rows.append(
        f"| Overall | {overall[0]} | {overall[1]} | {overall[2]} |"
    )
    return rows


def _gameplay_latency_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Hybrid | Decisions | p50 | p95 | Max | Inner terminals | "
        "Model calls / time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in HYBRIDS:
        metrics = document["metrics"][mode]
        latency = metrics["gameplay_latency_seconds"]
        inference = metrics["gameplay_model_inference"]
        rows.append(
            "| {mode} | {count} | {p50}s | {p95}s | {maximum}s | "
            "{terminals:,} | {calls:,} / {seconds:.2f}s |".format(
                mode=mode,
                count=metrics["gameplay_decision_count"],
                p50=_seconds(latency["p50"]),
                p95=_seconds(latency["p95"]),
                maximum=_seconds(latency["maximum"]),
                terminals=metrics[
                    "gameplay_response_terminal_evaluations"
                ],
                calls=inference["calls"],
                seconds=inference["seconds"],
            )
        )
    return rows


def _disagreement_table(document: dict[str, object]) -> list[str]:
    rows = [
        "| Mode | Source | P | Actor | Dealer | Teacher group | Hybrid group | "
        "TQ(T) | TQ(H) | HQ(T) | HQ(H) | Regret |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for mode in HYBRIDS:
        disagreements = document["paired_decisions"][mode]["disagreements"]
        for row in sorted(
            disagreements,
            key=lambda value: (
                value["kind"],
                value["key"],
            ),
        ):
            source = (
                f"{row['suite']}:{row['fixture_id']}"
                if row["kind"] == "fixture"
                else f"{row['fixture_id']}#{row['seed_index']}"
            )
            rows.append(
                "| {mode} | `{source}` | {placement} | {actor} | {dealer} | "
                "{teacher} | {hybrid} | {tt} | {th} | {ht} | {hh} | "
                "{regret} |".format(
                    mode=mode,
                    source=source,
                    placement=row["placement_number"],
                    actor=row["actor"],
                    dealer=row["dealer"],
                    teacher=row["teacher_selected_representative"],
                    hybrid=row["hybrid_selected_representative"],
                    tt=_number(row["teacher_value_of_teacher_action"]),
                    th=_number(row["teacher_value_of_hybrid_action"]),
                    ht=_number(row["hybrid_value_of_teacher_action"]),
                    hh=_number(row["hybrid_value_of_hybrid_action"]),
                    regret=_number(row["teacher_value_regret"]),
                )
            )
    return rows


def render(document: dict[str, object], raw_path: Path, raw_digest: str) -> str:
    artifact_digest = document["configuration"][
        "response_ranker_artifact_digest"
    ]
    lines = [
        "# Teacher v2 hybrid experiment",
        "",
        "## Conclusion",
        "",
        "Student-top-2 is the only hybrid that preserved recognizable Teacher "
        "v2 competence in this controlled comparison while materially reducing "
        "computation. It finished 11–13 against pure Teacher v2 across 12 "
        "identical decks in both roles, split 144 rounds exactly 50/50, and "
        "trailed by 1.19 points per round. On the fixed decision matrix it "
        "removed 71.7% of inner terminal evaluations and reduced p50 latency "
        "from 3.396 seconds to 0.969 seconds.",
        "",
        "This is evidence for using top-2 as an experimental gameplay controller, "
        "not an automatic promotion. It selected the same strategic group as "
        "pure Teacher v2 on 45.2% of fixed decisions and lost two defensive "
        "fixture passes that pure retained. The hybrid is therefore preserving "
        "outcomes more successfully than it is imitating individual choices.",
        "",
        "Student-direct was faster but clearly weaker. Student-top-3 improved "
        "fixture coverage but lost 5–19 in paired games; evaluating an extra "
        "noisy candidate did not improve the complete policy. No larger "
        "shortlist is justified by this experiment.",
        "",
        "## Method",
        "",
        "- Controllers: pure symmetry-aware 32×4 Teacher v2, student-direct, "
        "student-top-2, and the conditionally added student-top-3.",
        "- Fixed decisions: all 12 constructive fixtures, all 14 defensive "
        "fixtures, and 24 role/dealer-balanced placement states repeated over "
        "five deterministic request seeds.",
        "- Games: 12 fixed decks, each played with the hybrid as Queen and King "
        "against pure Teacher v2. Dealer alternated across all six rounds.",
        "- Every paired fixed decision used the same information state and raw "
        "request seed. All modes retained 32 outer simulations and exact engine "
        "terminal scoring.",
        "- Teacher-value regret is the best pure-search group mean minus the "
        "pure-search mean for the hybrid-selected group. It is diagnostic, not "
        "a promotion threshold.",
        f"- Response-ranker artifact: `{artifact_digest}`.",
        f"- Raw result: `{raw_path}` (`sha256:{raw_digest}`).",
        "",
        "## Strategic fixtures",
        "",
        "The table reports strategic-group equivalence, so a fair-coin-selected "
        "mirrored destination is not counted as a different strategic choice.",
        "",
        *_fixture_table(document),
        "",
        "Totals:",
        "",
        "| Controller | Constructive | Defensive |",
        "| --- | ---: | ---: |",
    ]
    for mode in MODES:
        fixtures = document["metrics"][mode]["fixtures"]
        lines.append(
            f"| {mode} | "
            f"{fixtures['constructive']['strategic_group_passes']}/"
            f"{fixtures['constructive']['count']} | "
            f"{fixtures['defensive']['strategic_group_passes']}/"
            f"{fixtures['defensive']['count']} |"
        )
    lines.extend(
        [
            "",
            "Top-3 was run because top-2 lost the "
            "`king-early-nondealer-safe-intersection` and "
            "`queen-middle-avoid-vampire-destruction` defensive actions that "
            "pure Teacher v2 passed. Top-3 recovered the safe-intersection "
            "fixture and added the `king-middle-avoid-suit` pass, but it still "
            "missed the Vampire-destruction fixture.",
            "",
            "## Paired game results",
            "",
            *_game_table(document),
            "",
            "### Role splits",
            "",
            *_role_table(document),
            "",
            "### Dealer splits",
            "",
            *_dealer_table(document),
            "",
            "Each role contains 12 games and 72 rounds. Each dealer split "
            "contains 72 rounds.",
            "",
            "## Efficiency",
            "",
            "The primary latency comparison uses the identical 146 fixed "
            "decisions for each controller.",
            "",
            *_performance_table(document),
            "",
            "Gameplay timings below cover the hybrid-controlled decisions in "
            "the 24 paired games. Concurrent full-game evaluation increases "
            "tail latency relative to isolated fixed decisions.",
            "",
            *_gameplay_latency_table(document),
            "",
            "Neural inference itself remained small: roughly 0.52–0.56 ms per "
            "call on the fixed matrix. Search continuations, not the network, "
            "dominate hybrid latency.",
            "",
            "## Agreement and teacher-value regret",
            "",
            *_agreement_table(document),
            "",
            "Agreement becomes exact at placements 6 and 7 for top-2 and "
            "top-3 because the remaining legal response space is small. Early "
            "placement disagreement is substantial, consistent with the "
            "ranker's approximately random offline pairwise accuracy.",
            "",
            "## Every strategic-group disagreement",
            "",
            "`TQ(T)` is pure Teacher v2's value for its selected group; `TQ(H)` "
            "is pure Teacher v2's value for the hybrid-selected group. `HQ(T)` "
            "and `HQ(H)` are the hybrid search's corresponding values. Full "
            "group visits and values for both controllers remain in the raw "
            "artifact.",
            "",
            *_disagreement_table(document),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_path = args.input.expanduser().resolve()
    content = raw_path.read_bytes()
    document = json.loads(content)
    if document.get("format_version") != (
        "dracula-teacher-v2-hybrid-experiment-v1"
    ):
        raise ValueError("hybrid experiment result schema is incompatible")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render(
            document,
            raw_path.relative_to(Path.cwd().resolve()),
            hashlib.sha256(content).hexdigest(),
        ),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
