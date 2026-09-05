"""Choose a move under hidden information by simulating complete round endings.

The player whose current move is being chosen is the *root player*. Search
receives that player's hand and the public game state, but not the opponent's
remaining cards or the stock order. Before each simulation, it samples one
possible assignment of those unseen cards that is consistent with everything
the root player can see. Repeating this process tests candidate moves across
many possible hidden deals.

The search tree records only decision positions reached by the root player:

- A node represents one player-visible position. It includes the root player's
  remaining hand and the public round state. Sampled worlds with different
  hidden cards share a node whenever they look identical to the root player.
- An edge leaving a node represents one legal strategic action: playing one
  particular card into one strategically distinct destination group.
- An edge's visit count is the number of completed simulations that selected
  that action from that visible position.
- An edge's value records how well those simulations ended for the root player,
  expressed from the root player's score relative to the opponent's.

Each simulation starts at the current decision and plays until the engine
completes and scores the round. Whenever the root player acts while the
simulation is still following the tree, UCT selects an outgoing edge. It first
gathers evidence for actions not yet sampled from that node; after that, it
balances actions with strong average results against actions with comparatively
little evidence. Opponent moves are never tree nodes and are chosen by the
configured continuation policy.

The first time a simulation selects a previously unexplored root-player action,
that edge becomes the simulation's one possible tree expansion. The continuation
policy then chooses every remaining move for both players so the round can be
completed and scored.

Once the round is scored, the same root-relative result is recorded on every
tree edge followed in that simulation. Recording the result increments the
edge's visit count and updates its accumulated evidence. Over many simulations,
earlier actions therefore reflect the outcomes of all later continuations that
followed them.

After the configured simulation budget is exhausted, the outgoing action at the
starting node with the most visits is selected as the real move. Its mean result
and then canonical action order break exact visit ties.

Destination symmetry changes only the action representation. Mirrored
destinations belong to one strategic edge and share its visits and values.
After search selects that strategic action, the deterministic fair coin chooses
which concrete mirrored destination to play.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

from dracula.bridge import ACTION_COUNT, action_index_for_move, move_for_action_index
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
)
from dracula.randomness import stable_seed
from dracula.search.contracts import (
    ContinuationDecision,
    ContinuationPolicy,
    ContinuationStep,
    PrincipalContinuation,
    SearchContractViolation,
    SearchInterrupted,
    StrategicGroupStatistics,
    normalized_round_return,
)
from dracula.search.determinization import sample_determinization
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
    information_state_from_simulation,
)
from dracula.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
    strategic_action_groups,
)


@dataclass(frozen=True, slots=True)
class BGCSearchConfig:
    """The retained BGC-128 outer-search configuration."""

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
class BGCSearchResult:
    """The selected move, root evidence, and private search diagnostics.

    The two 32-entry arrays use representative action indexes; non-representative
    positions remain zero or ``None``. ``principal_continuation`` contains one
    sampled private line and therefore must not cross a public API boundary.
    Remaining counts measure the work performed by outer search and its
    continuation policy.
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


@dataclass(frozen=True, slots=True)
class _NodeKey:
    """Identify one actor-visible information set independently of hidden cards."""

    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(slots=True)
class _Node:
    """Mutable UCT statistics indexed by strategic-group representative action."""

    key: _NodeKey
    groups: tuple[StrategicActionGroup, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @classmethod
    def create(
        cls,
        key: _NodeKey,
        groups: tuple[StrategicActionGroup, ...],
    ) -> _Node:
        return cls(key, groups, 0, [0] * ACTION_COUNT, [0.0] * ACTION_COUNT)


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    """Retain one completed simulation for principal-line diagnostics."""

    root_group: StrategicActionGroup
    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(slots=True)
class _SearchWork:
    """Mutable tree, cache, diagnostics, and counters for one root request."""

    nodes: dict[_NodeKey, _Node]
    continuation_cache: dict[str, ContinuationDecision]
    observed_continuations: list[_ObservedContinuation]
    continuation_requests: int = 0
    continuation_cache_hits: int = 0
    terminal_evaluations: int = 0
    model_inferences: int = 0


@dataclass(slots=True)
class _SimulationTrace:
    """Tree path and private moves collected during one outer simulation."""

    path: list[tuple[_Node, StrategicActionGroup]]
    steps: list[ContinuationStep]
    root_group: StrategicActionGroup | None = None
    root_action_index: int | None = None


def derive_bgc_request_seed(
    fixture_id: str,
    information: SearchInformationState,
) -> int:
    """Return the local random seed for one visible search request."""

    return stable_seed(
        fixture_id,
        information_state_fingerprint(information),
        information.player.value,
        information.round_number,
        information.turn_number,
    )


def _simulation_seed(request_seed: int, simulation_index: int) -> int:
    """Give each outer simulation an independent reproducible hidden world."""

    return stable_seed(request_seed, simulation_index)


def _select_group(
    node: _Node,
    exploration_constant: float,
) -> StrategicActionGroup:
    """Apply UCT selection to one information-set node.

    Unvisited groups receive deterministic initial coverage. Later selections
    combine their actor-relative mean return with the usual exploration bonus;
    exact score ties use representative order.
    """

    # Canonical initial coverage gives every strategic action one observation.
    for group in node.groups:
        if node.action_visits[group.representative_action_index] == 0:
            return group

    log_parent = math.log(node.visits)
    best_score = -math.inf
    tied: list[StrategicActionGroup] = []
    for group in node.groups:
        representative = group.representative_action_index
        visits = node.action_visits[representative]
        mean = node.action_value_sums[representative] / visits
        score = mean + exploration_constant * math.sqrt(log_parent / visits)
        if score > best_score:
            best_score = score
            tied = [group]
        elif score == best_score:
            tied.append(group)
    return min(tied, key=lambda group: group.representative_action_index)


def _move_for_action(
    state: SimulationEngineState,
    action_index: int,
) -> EngineMove:
    """Translate a concrete action index and reject any search/engine mismatch."""

    actor = state.active_player
    if actor is None:
        raise SearchContractViolation("playing simulation has no active player")
    move = move_for_action_index(actor, action_index)
    if move not in legal_simulation_moves(state, actor):
        raise SearchContractViolation("search selected an illegal concrete action")
    return move


def _continuation_step(
    root_player: EnginePlayer,
    state: SimulationEngineState,
    move: EngineMove,
    action_index: int,
    forced: bool,
) -> ContinuationStep:
    """Describe a simulated move relative to the original decision maker."""

    card_id = state.hands[move.player][move.hand_slot]
    if card_id is None:
        raise SearchContractViolation("continuation refers to an empty hand slot")
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


def _peak_resident_memory_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class BGCInformationSetSearch:
    """Run outer UCT with one interchangeable actor-local continuation policy."""

    def __init__(
        self,
        continuation: ContinuationPolicy,
        config: BGCSearchConfig = BGCSearchConfig(),
    ) -> None:
        self.continuation = continuation
        self.config = config

    def search(
        self,
        information: SearchInformationState,
        request_seed: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> BGCSearchResult:
        """Evaluate one actor-visible position and return one concrete legal action."""

        started = time.perf_counter()
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_groups = strategic_action_groups(information)
        if not root_groups:
            raise SearchContractViolation("BGC root has no strategic actions")

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

        work = _SearchWork({}, {}, [])
        for simulation_index in range(self.config.outer_simulation_budget):
            self._run_simulation(
                information,
                request_seed,
                simulation_index,
                work,
                should_stop,
            )
        return self._completed_result(
            information,
            request_seed,
            root_fingerprint,
            work,
            started,
        )

    def _run_simulation(
        self,
        information: SearchInformationState,
        request_seed: int,
        simulation_index: int,
        work: _SearchWork,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        """Sample, play, and back up one complete round simulation."""

        self._check_interrupted(should_stop)
        # Hidden cards are resampled for every simulation; only visible-state
        # fingerprints are allowed to identify and merge tree nodes.
        state = sample_determinization(
            information,
            _simulation_seed(request_seed, simulation_index),
        ).state
        root_player = information.player
        trace = _SimulationTrace([], [])
        expanded = False

        while state.status is EngineStatus.PLAYING:
            self._check_interrupted(should_stop)
            actor = state.active_player
            if actor is None:
                raise SearchContractViolation(
                    "playing simulation lost its active player"
                )
            legal = legal_simulation_moves(state, actor)
            forced = len(legal) == 1

            if forced:
                # Forced engine moves are recorded but do not create tree nodes.
                move = legal[0]
                action_index = action_index_for_move(move, actor)
                selected_group = None
            elif actor is root_player and not expanded:
                move, selected_group, action_index, node, expanded = (
                    self._select_tree_move(
                        state,
                        request_seed,
                        simulation_index,
                        work,
                    )
                )
                trace.path.append((node, selected_group))
            else:
                # Opponent turns, and all turns after expansion, use the
                # configured actor-local response policy.
                move, selected_group, action_index = self._select_continuation_move(
                    state,
                    work,
                    should_stop,
                )

            if trace.root_group is None:
                # The first non-forced move is the candidate root decision whose
                # group receives this simulation's eventual terminal return.
                if actor is not root_player or selected_group is None:
                    raise SearchContractViolation(
                        "learned root simulation began without a strategic group"
                    )
                trace.root_group = selected_group
                trace.root_action_index = action_index
            trace.steps.append(
                _continuation_step(
                    root_player,
                    state,
                    move,
                    action_index,
                    forced,
                )
            )
            state = apply_simulation_move(state, move)

        self._back_up_simulation(
            state,
            root_player,
            trace,
            simulation_index,
            work,
        )

    @staticmethod
    def _back_up_simulation(
        state: SimulationEngineState,
        root_player: EnginePlayer,
        trace: _SimulationTrace,
        simulation_index: int,
        work: _SearchWork,
    ) -> None:
        """Validate a completed simulation and back its result through the path."""

        if (
            state.status is not EngineStatus.ROUND_COMPLETE
            or state.pending_round_result is None
            or trace.root_group is None
            or trace.root_action_index is None
            or not trace.path
        ):
            raise SearchContractViolation(
                "simulation did not reach a backed round result"
            )
        terminal_value = normalized_round_return(
            state.pending_round_result,
            root_player,
        )
        # Every node is backed up from the original decision maker's
        # actor-relative terminal score, even when opponents acted between.
        for node, selected_group in trace.path:
            representative = selected_group.representative_action_index
            node.visits += 1
            node.action_visits[representative] += 1
            node.action_value_sums[representative] += terminal_value
        work.observed_continuations.append(
            _ObservedContinuation(
                trace.root_group,
                trace.root_action_index,
                terminal_value,
                simulation_index,
                tuple(trace.steps),
            )
        )

    def _select_tree_move(
        self,
        state: SimulationEngineState,
        request_seed: int,
        simulation_index: int,
        work: _SearchWork,
    ) -> tuple[EngineMove, StrategicActionGroup, int, _Node, bool]:
        """Select one root-player move and report whether it expands the tree."""

        actor = state.active_player
        if actor is None:
            raise SearchContractViolation("tree selection has no active player")
        actor_information = information_state_from_simulation(state)
        key = _NodeKey(actor, information_state_fingerprint(actor_information))
        groups = strategic_action_groups(actor_information)
        node = work.nodes.get(key)
        if node is None:
            node = _Node.create(key, groups)
            work.nodes[key] = node
        elif node.groups != groups:
            raise SearchContractViolation(
                "one information set produced inconsistent groups"
            )

        selected_group = _select_group(
            node,
            float(self.config.outer_exploration_constant),
        )
        representative = selected_group.representative_action_index
        newly_expanded = node.action_visits[representative] == 0
        action_index = select_concrete_action_index(
            selected_group,
            request_seed,
            simulation_index,
            key.information_state_fingerprint,
            representative,
        )
        return (
            _move_for_action(state, action_index),
            selected_group,
            action_index,
            node,
            newly_expanded,
        )

    def _select_continuation_move(
        self,
        state: SimulationEngineState,
        work: _SearchWork,
        should_stop: Callable[[], bool] | None,
    ) -> tuple[EngineMove, StrategicActionGroup, int]:
        """Select or reuse one response from an actor-visible information state."""

        actor_information = information_state_from_simulation(state)
        fingerprint = information_state_fingerprint(actor_information)
        # Cache identity is actor-visible. Hidden determinizations cannot select
        # or distinguish a continuation response.
        cache_key = fingerprint
        work.continuation_requests += 1
        decision = work.continuation_cache.get(cache_key)
        if decision is None:
            decision = self.continuation.select(actor_information, should_stop)
            work.continuation_cache[cache_key] = decision
            work.terminal_evaluations += decision.terminal_evaluation_count
            work.model_inferences += decision.model_inference_count
        else:
            work.continuation_cache_hits += 1
        return (
            _move_for_action(state, decision.selected_action_index),
            decision.selected_group,
            decision.selected_action_index,
        )

    def _completed_result(
        self,
        information: SearchInformationState,
        request_seed: int,
        root_fingerprint: str,
        work: _SearchWork,
        started: float,
    ) -> BGCSearchResult:
        """Select the root group and assemble aggregate search diagnostics."""

        root_key = _NodeKey(information.player, root_fingerprint)
        root_node = work.nodes.get(root_key)
        if root_node is None or root_node.visits != self.config.outer_simulation_budget:
            raise SearchContractViolation("root backup count differs from its budget")
        selected_group = min(
            root_node.groups,
            key=lambda group: self._root_rank(root_node, group),
        )
        # Visits choose the strategic decision. Destination resolution happens
        # afterward so a paired group's coin cannot affect its search evidence.
        selected_action = select_concrete_action_index(
            selected_group,
            request_seed,
            root_fingerprint,
            selected_group.representative_action_index,
        )
        # The best sampled line is diagnostic context for the selected group;
        # root selection still depends on aggregate visits and mean value.
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
        return BGCSearchResult(
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

    def _forced_result(
        self,
        fingerprint: str,
        group: StrategicActionGroup,
        action_index: int,
        started: float,
    ) -> BGCSearchResult:
        """Return a legal forced move with explicit zero search work."""

        return BGCSearchResult(
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

    @staticmethod
    def _root_rank(
        node: _Node,
        group: StrategicActionGroup,
    ) -> tuple[int, float, int]:
        representative = group.representative_action_index
        visits = node.action_visits[representative]
        # Root choice follows visit count, then mean value, then canonical action.
        return (
            -visits,
            -node.action_value_sums[representative] / visits,
            representative,
        )

    @staticmethod
    def _check_interrupted(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted("BGC search interrupted before result commit")


__all__ = (
    "BGCInformationSetSearch",
    "BGCSearchConfig",
    "BGCSearchResult",
    "derive_bgc_request_seed",
)
