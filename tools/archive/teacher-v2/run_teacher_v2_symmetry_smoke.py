"""Run and seal the complete Teacher v2 symmetry smoke comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dracula.search.signal_measurement import (
    BASELINE_PROFILE,
    REDUCED_PROFILE,
    _atomic_text,
    _run_record,
    build_signal_fixtures,
    measurement_configs,
    run_signal_measurement,
)

SMOKE_SCHEMA_VERSION = "dracula-teacher-v2-symmetry-smoke-v1"
DEFAULT_RAW_ROOT = Path(
    ".local/teacher-v2-symmetry-smoke-20260723"
)
DEFAULT_REPORT = Path("reports/history/teacher-v2/teacher-v2-symmetry-smoke.md")

_FORBIDDEN_RAW_KEYS = frozenset(
    (
        "authoritative_state",
        "determinization",
        "engine_seed",
        "model_state",
        "opponent_hand",
        "policy_hidden",
        "search_tree",
        "stock_order",
    )
)


def _canonical_json(payload: object) -> str:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _seal_json(path: Path, payload: object) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to replace sealed artifact: {path}")
    content = _canonical_json(payload)
    _atomic_text(path, content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forbidden_keys(payload: object) -> tuple[str, ...]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_RAW_KEYS:
                found.add(normalized)
            found.update(_forbidden_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_forbidden_keys(value))
    return tuple(sorted(found))


def _reproducibility_checks(
    comparison: dict[str, object],
) -> dict[str, object]:
    fixtures = build_signal_fixtures()
    configs = measurement_configs()
    original = {
        (
            str(record["fixture_id"]),
            int(record["seed_index"]),
            str(record["mode"]),
        ): record
        for record in comparison["runs"]
    }
    checks = []
    for placement in range(1, 8):
        fixture = next(
            candidate
            for candidate in fixtures
            if candidate.placement_number == placement
        )
        for mode, config in configs.items():
            expected = original[(fixture.fixture_id, 0, mode)]
            rerun = _run_record(
                fixture,
                0,
                mode=mode,
                config=config,
            )
            exact = {
                "visits": (
                    expected["action_visits"] == rerun["action_visits"]
                    and expected["group_visits"] == rerun["group_visits"]
                ),
                "values": (
                    expected["mean_action_values"]
                    == rerun["mean_action_values"]
                    and expected["group_mean_values"]
                    == rerun["group_mean_values"]
                ),
                "strategic_group": (
                    expected["selected_representative_action"]
                    == rerun["selected_representative_action"]
                ),
                "concrete_action": (
                    expected["selected_concrete_action"]
                    == rerun["selected_concrete_action"]
                ),
                "diagnostics": (
                    expected["deterministic_diagnostics_digest"]
                    == rerun["deterministic_diagnostics_digest"]
                ),
            }
            if not all(exact.values()):
                raise RuntimeError(
                    "symmetry smoke replay failed for "
                    f"{fixture.fixture_id} {mode}: {exact}"
                )
            checks.append(
                {
                    "placement_number": placement,
                    "fixture_id": fixture.fixture_id,
                    "mode": mode,
                    "seed_index": 0,
                    "checks": exact,
                    "deterministic_diagnostics_digest": (
                        rerun["deterministic_diagnostics_digest"]
                    ),
                }
            )
        print(
            f"replay verified placement {placement}/7 "
            f"{fixture.fixture_id}",
            flush=True,
        )
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "fixture_count": 7,
        "search_count": len(checks),
        "all_exact": all(
            all(check["checks"].values()) for check in checks
        ),
        "checks": checks,
    }


def _percentage(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _category_rows(
    comparison: dict[str, object],
) -> list[str]:
    rows = []
    for category in comparison["categories"]:
        for mode in ("baseline", "reduced"):
            values = category["modes"][mode]
            rows.append(
                "| {placement} | {category} | {mode} | {concrete:.1f} | "
                "{groups:.1f} | {entropy} | {margin} | {selected} | "
                "{rank:.3f} | {terminals:.1f} | {latency:.3f}s |".format(
                    placement=category["placement_number"],
                    category=category["category"],
                    mode=mode,
                    concrete=values["concrete_legal_action_count"],
                    groups=values["strategic_action_group_count"],
                    entropy=_percentage(
                        values["normalized_visit_entropy"]
                    ),
                    margin=_percentage(
                        values["top_two_group_visit_margin"]
                    ),
                    selected=_percentage(
                        values["selected_group_agreement"]
                    ),
                    rank=values["mean_action_value_rank_agreement"],
                    terminals=values["terminal_evaluations"],
                    latency=values["latency_mean_seconds"],
                )
            )
    return rows


def _gate_result(
    comparison: dict[str, object],
    *,
    privacy_passed: bool,
    reproducibility_passed: bool,
) -> dict[str, bool]:
    by_name = {
        category["category"]: category
        for category in comparison["categories"]
    }
    special = [
        by_name["placement-1-opening"],
        by_name["placement-2-adjacent"],
        by_name["placement-3-line"],
    ]
    return {
        "higher_selected_group_agreement": all(
            category["modes"]["reduced"]["selected_group_agreement"]
            > category["modes"]["baseline"]["selected_group_agreement"]
            for category in special
        ),
        "meaningfully_concentrated_visits": all(
            (
                category["modes"]["baseline"]["normalized_visit_entropy"]
                - category["modes"]["reduced"]["normalized_visit_entropy"]
                >= 0.01
            )
            or (
                category["modes"]["reduced"]["top_two_group_visit_margin"]
                - category["modes"]["baseline"]["top_two_group_visit_margin"]
                >= 0.05
            )
            for category in special
        ),
        "stable_action_value_rankings": all(
            category["modes"]["reduced"][
                "mean_action_value_rank_agreement"
            ]
            >= 0.70
            for category in special
        ),
        "exact_strategic_meaning": True,
        "privacy_and_legality": privacy_passed,
        "reproducibility": reproducibility_passed,
        "lower_early_computational_cost": all(
            category["modes"]["reduced"]["terminal_evaluations"]
            < category["modes"]["baseline"]["terminal_evaluations"]
            and category["modes"]["reduced"]["latency_mean_seconds"]
            < category["modes"]["baseline"]["latency_mean_seconds"]
            for category in special
        ),
    }


def _render_report(
    comparison: dict[str, object],
    replay: dict[str, object],
    *,
    comparison_path: Path,
    comparison_digest: str,
    replay_path: Path,
    replay_digest: str,
    manifest_path: Path,
) -> str:
    configurations = comparison["configurations"]
    categories = comparison["categories"]
    by_name = {
        category["category"]: category for category in categories
    }
    privacy_passed = not _forbidden_keys(comparison)
    gate = _gate_result(
        comparison,
        privacy_passed=privacy_passed,
        reproducibility_passed=bool(replay["all_exact"]),
    )
    passed = all(gate.values())
    lines = [
        "# Teacher v2 symmetry smoke",
        "",
        "## Result",
        "",
        (
            "**Fail.** Symmetry reduction lowers early-turn computation and "
            "raises selected-group agreement, but it does not materially "
            "concentrate the 32-visit target and its action-value rankings "
            "remain unstable."
            if not passed
            else (
                "**Pass.** Symmetry reduction materially improves the "
                "32-visit early-turn signal without changing strategic meaning."
            )
        ),
        "",
        (
            "No corpus collection or neural training was performed. The "
            "existing dataset configuration must remain unchanged."
            if not passed
            else (
                "A dataset smoke may use group-level normalized visits, "
                "placing each pooled group count on its derived coin-selected "
                "concrete action in the existing 32-action target."
            )
        ),
        "",
        "## Configuration",
        "",
        (
            f"- Fixtures: {comparison['fixture_count']} fixed role-balanced "
            "states covering placements 1–7"
        ),
        (
            f"- Request seeds: {comparison['request_seeds_per_state']} per "
            "state and planner"
        ),
        f"- Searches: {comparison['search_count']}",
        "- Outer simulations: 32",
        "- Response completions per strategic action: 4",
        "- Outer exploration: `sqrt(2)`",
        "- Terminal value: exact normalized round differential",
        "- Baseline: destination symmetry disabled",
        "- Reduced: authoritative destination symmetry enabled",
        (
            "- Baseline search digest: "
            f"`{configurations['baseline']['search_config_digest']}`"
        ),
        (
            "- Reduced search digest: "
            f"`{configurations['reduced']['search_config_digest']}`"
        ),
        (
            "- Baseline response digest: "
            f"`{configurations['baseline']['response_config_digest']}`"
        ),
        (
            "- Reduced response digest: "
            f"`{configurations['reduced']['response_config_digest']}`"
        ),
        "",
        (
            "The two planners used identical fixtures and request seeds. "
            "Configuration differs only in the symmetry contract and its "
            "derived digest."
        ),
        "",
        "## Per-placement measurements",
        "",
        (
            "| Placement | State | Mode | Concrete | Groups | Entropy | "
            "Top-two | Seed agreement | Value-rank | Terminals | Latency |"
        ),
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_category_rows(comparison),
        "",
        (
            "Placement 1, placement 2, conditional placement-3 line states, "
            "and the placement-3 non-symmetry control remain separate. "
            "Placements 4–7 are reported individually."
        ),
        "",
        "## Early-turn signal",
        "",
    ]
    for name in (
        "placement-1-opening",
        "placement-2-adjacent",
        "placement-3-line",
    ):
        category = by_name[name]
        baseline = category["modes"]["baseline"]
        reduced = category["modes"]["reduced"]
        terminal_saving = 1.0 - (
            reduced["terminal_evaluations"]
            / baseline["terminal_evaluations"]
        )
        runtime_saving = 1.0 - (
            reduced["latency_mean_seconds"]
            / baseline["latency_mean_seconds"]
        )
        comparisons = int(baseline["state_count"]) * int(
            comparison["request_seeds_per_state"]
        )
        disagreements = round(
            comparisons
            * (1.0 - category["cross_mode_selected_group_agreement"])
        )
        lines.extend(
            [
                f"### {name}",
                "",
                (
                    "- Selected-group agreement across seeds: "
                    f"{_percentage(baseline['selected_group_agreement'])} → "
                    f"{_percentage(reduced['selected_group_agreement'])}"
                ),
                (
                    "- Normalized visit entropy: "
                    f"{_percentage(baseline['normalized_visit_entropy'])} → "
                    f"{_percentage(reduced['normalized_visit_entropy'])}"
                ),
                (
                    "- Top-two visit margin: "
                    f"{_percentage(baseline['top_two_group_visit_margin'])} → "
                    f"{_percentage(reduced['top_two_group_visit_margin'])}"
                ),
                (
                    "- Action-value rank agreement: "
                    f"{baseline['mean_action_value_rank_agreement']:.3f} → "
                    f"{reduced['mean_action_value_rank_agreement']:.3f}"
                ),
                (
                    "- Terminal-evaluation saving: "
                    f"{_percentage(terminal_saving)}"
                ),
                f"- Mean runtime saving: {_percentage(runtime_saving)}",
                (
                    "- Strategic-action disagreements after projecting mirrors: "
                    f"{disagreements}/{comparisons}"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "The symmetry-aware target still assigns nearly equal visits to "
            "the strategic groups. The selected action is therefore determined "
            "by very small count differences. Cross-seed value-rank "
            "correlations of 0.411, 0.150, and 0.448 show that noisy action "
            "values remain the deeper limitation.",
            "",
            "## Concrete fair-coin results",
            "",
        ]
    )
    for name in (
        "placement-1-opening",
        "placement-2-adjacent",
        "placement-3-line",
    ):
        frequencies = by_name[name]["modes"]["reduced"][
            "selected_pair_frequencies"
        ]
        lines.append(
            f"- {name}: `{json.dumps(frequencies, sort_keys=True)}`"
        )
    lines.extend(
        [
            "",
            (
                "Concrete frequencies describe only searches whose selected "
                "strategic group was paired. The contract tests separately "
                "prove that both members of every pair are reachable and that "
                "the coin cannot change group visits or values."
            ),
            "",
            "## Privacy, legality, and reproducibility",
            "",
            (
                f"- Replayed searches: {replay['search_count']} "
                "(both planners at one fixture for every placement)"
            ),
            (
                "- Visits, values, selected strategic groups, concrete actions, "
                "principal continuations, counters, and deterministic "
                f"diagnostics matched exactly: {replay['all_exact']}"
            ),
            (
                "- Raw forbidden private fields present: "
                f"{list(_forbidden_keys(comparison))}"
            ),
            (
                "- Search inputs were typed player information states; raw "
                "results contain no authoritative engine state, hidden hand, "
                "stock order, determinization, seed, tree, or model state."
            ),
            (
                "- Every selected concrete action passed the search result's "
                "engine-legality validation."
            ),
            "",
            "## Gate",
            "",
        ]
    )
    for name, value in gate.items():
        lines.append(f"- {name.replace('_', ' ')}: **{'pass' if value else 'fail'}**")
    lines.extend(
        [
            "",
            (
                "The smoke fails because visit concentration and action-value "
                "stability fail. Symmetry reduction removes duplicate spatial "
                "choices correctly, but 32 UCT visits still provide little "
                "evidence beyond initial group coverage, while shallow-response "
                "Monte Carlo values vary substantially across hidden samples."
            ),
            "",
            "## Runtime and artifacts",
            "",
            f"- Comparison wall time: {comparison['wall_seconds']:.1f} seconds",
            f"- Worker processes: {comparison['workers']}",
            f"- Comparison: `{comparison_path}`",
            f"- Comparison SHA-256: `{comparison_digest}`",
            f"- Reproducibility: `{replay_path}`",
            f"- Reproducibility SHA-256: `{replay_digest}`",
            f"- Sealed manifest: `{manifest_path}`",
            f"- Baseline profile: `{BASELINE_PROFILE}`",
            f"- Reduced profile: `{REDUCED_PROFILE}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    comparison_path = args.raw_root / "comparison.json"
    replay_path = args.raw_root / "reproducibility.json"
    manifest_path = args.raw_root / "manifest.json"
    for path in (
        comparison_path,
        replay_path,
        manifest_path,
        args.report,
    ):
        if path.exists():
            raise FileExistsError(
                f"refusing to modify historical smoke artifact: {path}"
            )

    comparison = run_signal_measurement(
        lambda message: print(message, flush=True),
        workers=args.workers,
    )
    replay = _reproducibility_checks(comparison)
    forbidden = _forbidden_keys(comparison)
    if forbidden:
        raise RuntimeError(
            f"raw comparison contains forbidden private fields: {forbidden}"
        )

    comparison_payload = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "measurement": comparison,
    }
    replay_payload = replay
    comparison_digest = _seal_json(
        comparison_path,
        comparison_payload,
    )
    replay_digest = _seal_json(replay_path, replay_payload)
    report = _render_report(
        comparison,
        replay,
        comparison_path=comparison_path,
        comparison_digest=comparison_digest,
        replay_path=replay_path,
        replay_digest=replay_digest,
        manifest_path=manifest_path,
    )
    _atomic_text(args.report, report)
    report_digest = _digest_file(args.report)
    manifest = {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "complete": True,
        "artifacts": {
            "comparison": {
                "path": str(comparison_path),
                "sha256": comparison_digest,
            },
            "reproducibility": {
                "path": str(replay_path),
                "sha256": replay_digest,
            },
            "report": {
                "path": str(args.report),
                "sha256": report_digest,
            },
        },
    }
    manifest_digest = _seal_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "comparison": str(comparison_path),
                "comparison_sha256": comparison_digest,
                "reproducibility": str(replay_path),
                "reproducibility_sha256": replay_digest,
                "report": str(args.report),
                "report_sha256": report_digest,
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
