"""Choose a move under hidden information with outer UCT search.

The player whose move is being chosen is the root player. Search receives only
that player's hand and the public game state. Each simulation samples one
plausible assignment of the unseen cards, plays the rest of the current round,
and records the exact root-relative round score.

The tree contains only positions where the root player makes a non-forced
decision. A node is identified by what the root player can see, so different
sampled hidden worlds share a node when their visible position is identical.
Its outgoing edges are the legal strategic actions: one current card paired
with one distinct destination group. Visits count completed simulations using
an edge, while accumulated values record their root-relative results.

UCT gives every edge initial coverage and then balances its mean result against
how little it has been explored. The first unvisited edge chosen during a
simulation is the only new tree edge added on that simulation. A configurable
continuation policy supplies opponent decisions and every later decision after
that expansion. Once the round ends, its result updates every searched edge the
root player followed along the sampled line.

After the simulation budget is exhausted, the starting edge with the most
visits becomes the real strategic move. Mean result and canonical action order
break ties. Mirrored destinations share one edge; the deterministic fair coin
chooses a concrete member only after the strategic group has been selected.

This module exposes the search configuration, result, and request-level
orchestrator. ``uct_tree`` owns UCT statistics and backup, while
``uct_simulation`` plays each sampled round.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from dracula.decision.bridge import ACTION_COUNT
from dracula.randomness import stable_seed
from dracula.search.uct_simulation import OuterSimulationRunner
from dracula.search.uct_tree import NodeKey, SearchWork, root_rank
from dracula.search.contracts import (
    ContinuationPolicy,
    PrincipalContinuation,
    SearchContractViolation,
    StrategicGroupStatistics,
)
from dracula.decision.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.decision.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
    strategic_action_groups,
)


@dataclass(frozen=True, slots=True)
class InformationSetUCTConfig:
    """Configure the number of outer simulations and UCT exploration strength."""

    outer_simulation_budget: int = 128
    outer_exploration_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        if self.outer_simulation_budget < 1:
            raise SearchContractViolation("outer simulation budget must be positive")
        if (
            not math.isfinite(self.outer_exploration_constant)
            or self.outer_exploration_constant < 0
        ):
            raise SearchContractViolation(
                "outer exploration constant must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class InformationSetUCTResult:
    """Return the selected move, root evidence, and private diagnostics.

    The two 32-entry arrays use representative action indexes;
    non-representative positions remain zero or ``None``. The principal
    continuation contains a sampled private line and must not cross a public
    API boundary. The remaining counters describe work performed by outer
    search and its continuation policy.
    """

    information_state_fingerprint: str
    selected_group: StrategicActionGroup
    selected_action_index: int
    representative_action_visits: tuple[int, ...]
    representative_mean_values: tuple[float | None, ...]
    group_statistics: tuple[StrategicGroupStatistics, ...]
    simulation_count: int
    information_set_count: int
    principal_continuation: PrincipalContinuation | None
    continuation_request_count: int
    continuation_cache_hit_count: int
    continuation_terminal_evaluation_count: int
    continuation_model_inference_count: int
    elapsed_seconds: float
    peak_resident_memory_bytes: int


def derive_uct_request_seed(
    fixture_id: str,
    information: SearchInformationState,
) -> int:
    """Return the reproducible local seed for one visible search request."""

    return stable_seed(
        fixture_id,
        information_state_fingerprint(information),
        information.player.value,
        information.round_number,
        information.turn_number,
    )


def _peak_resident_memory_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class InformationSetUCTSearch:
    """Run outer UCT with one interchangeable actor-local continuation policy."""

    def __init__(
        self,
        continuation: ContinuationPolicy,
        config: InformationSetUCTConfig = InformationSetUCTConfig(),
    ) -> None:
        self.continuation = continuation
        self.config = config

    def search(
        self,
        information: SearchInformationState,
        request_seed: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> InformationSetUCTResult:
        """Evaluate one actor-visible position and return one concrete action."""

        started = time.perf_counter()
        root_fingerprint = information_state_fingerprint(information)
        root_groups = strategic_action_groups(information)
        if not root_groups:
            raise SearchContractViolation("UCT root has no strategic actions")

        if len(root_groups) == 1:
            # A forced decision needs neither a sampled world nor continuation work.
            group = root_groups[0]
            selected = select_concrete_action_index(
                group,
                request_seed,
                group.representative_action_index,
            )
            return self._forced_result(root_fingerprint, group, selected, started)
        if self.config.outer_simulation_budget < len(root_groups):
            raise SearchContractViolation(
                "outer budget cannot cover every strategic root action"
            )

        work = SearchWork({}, {}, [])
        runner = OuterSimulationRunner(
            root_information=information,
            request_seed=request_seed,
            continuation=self.continuation,
            exploration_constant=float(self.config.outer_exploration_constant),
            work=work,
            should_stop=should_stop,
        )
        for simulation_index in range(self.config.outer_simulation_budget):
            runner.run(simulation_index)

        return self._completed_result(
            information,
            request_seed,
            root_fingerprint,
            work,
            started,
        )

    def _completed_result(
        self,
        information: SearchInformationState,
        request_seed: int,
        root_fingerprint: str,
        work: SearchWork,
        started: float,
    ) -> InformationSetUCTResult:
        """Choose the root group and assemble aggregate search diagnostics."""

        root_key = NodeKey(information.player, root_fingerprint)
        root_node = work.nodes.get(root_key)
        if root_node is None or root_node.visits != self.config.outer_simulation_budget:
            raise SearchContractViolation("root backup count differs from its budget")

        selected_group = min(
            root_node.groups,
            key=lambda group: root_rank(root_node, group),
        )
        # Destination resolution happens after root selection, so its fair coin
        # cannot affect the strategic group's visits or mean result.
        selected_action = select_concrete_action_index(
            selected_group,
            request_seed,
            root_fingerprint,
            selected_group.representative_action_index,
        )
        # This line is diagnostic only; root selection used aggregate evidence.
        principal = min(
            (
                item
                for item in work.observed_continuations
                if item.root_group == selected_group
            ),
            key=lambda item: (-item.terminal_value, item.simulation_index),
        )

        statistics = tuple(
            StrategicGroupStatistics(
                group,
                root_node.action_visits[group.representative_action_index],
                root_node.action_value_sums[group.representative_action_index]
                / root_node.action_visits[group.representative_action_index],
            )
            for group in root_node.groups
        )
        means: list[float | None] = [None] * ACTION_COUNT
        for item in statistics:
            means[item.group.representative_action_index] = item.mean_value

        return InformationSetUCTResult(
            information_state_fingerprint=root_fingerprint,
            selected_group=selected_group,
            selected_action_index=selected_action,
            representative_action_visits=tuple(root_node.action_visits),
            representative_mean_values=tuple(means),
            group_statistics=statistics,
            simulation_count=root_node.visits,
            information_set_count=len(work.nodes),
            principal_continuation=PrincipalContinuation(
                principal.root_action_index,
                principal.terminal_value,
                principal.simulation_index,
                principal.steps,
            ),
            continuation_request_count=work.continuation_requests,
            continuation_cache_hit_count=work.continuation_cache_hits,
            continuation_terminal_evaluation_count=work.terminal_evaluations,
            continuation_model_inference_count=work.model_inferences,
            elapsed_seconds=time.perf_counter() - started,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )

    @staticmethod
    def _forced_result(
        fingerprint: str,
        group: StrategicActionGroup,
        action_index: int,
        started: float,
    ) -> InformationSetUCTResult:
        """Return a legal forced move with explicit zero search work."""

        return InformationSetUCTResult(
            information_state_fingerprint=fingerprint,
            selected_group=group,
            selected_action_index=action_index,
            representative_action_visits=(0,) * ACTION_COUNT,
            representative_mean_values=(None,) * ACTION_COUNT,
            group_statistics=(),
            simulation_count=0,
            information_set_count=0,
            principal_continuation=None,
            continuation_request_count=0,
            continuation_cache_hit_count=0,
            continuation_terminal_evaluation_count=0,
            continuation_model_inference_count=0,
            elapsed_seconds=time.perf_counter() - started,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
        )


__all__ = (
    "InformationSetUCTConfig",
    "InformationSetUCTResult",
    "InformationSetUCTSearch",
    "derive_uct_request_seed",
)
