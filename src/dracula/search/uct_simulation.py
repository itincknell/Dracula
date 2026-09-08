"""Execute complete sampled rounds for information-set UCT.

One outer simulation samples a hidden world consistent with the root player's
information, then plays that world until the round is scored. Root-player turns
follow the information-set tree until one new edge is expanded. Opponent turns,
and every turn after that expansion, use the configured continuation policy.

The runner deliberately owns no final move-selection policy. It only adds one
completed simulation's evidence to the shared tree state supplied by
``information_set_uct``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dracula.decision.bridge import action_index_for_move, move_for_action_index
from dracula.game.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
)
from dracula.randomness import stable_seed
from dracula.search.uct_tree import (
    Node,
    NodeKey,
    SearchWork,
    SimulationTrace,
    back_up_simulation,
    select_group,
)
from dracula.search.contracts import (
    ContinuationDecision,
    ContinuationPolicy,
    ContinuationStep,
    SearchContractViolation,
    SearchInterrupted,
)
from dracula.search.determinization import sample_determinization
from dracula.decision.information import (
    SearchInformationState,
    information_state_fingerprint,
    information_state_from_simulation,
)
from dracula.decision.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
    strategic_action_groups,
)


@dataclass(frozen=True, slots=True)
class _SelectedMove:
    """Carry one simulated choice and its effect on the outer search tree."""

    move: EngineMove
    group: StrategicActionGroup | None
    action_index: int
    tree_node: Node | None
    expands_tree: bool
    forced: bool


def simulation_seed(request_seed: int, simulation_index: int) -> int:
    """Derive an independent reproducible hidden-world seed for one simulation."""

    return stable_seed(request_seed, simulation_index)


def _move_for_action(
    state: SimulationEngineState,
    action_index: int,
) -> EngineMove:
    """Translate an action index while enforcing the search/engine boundary."""

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
    """Describe a private simulated move relative to the root player."""

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


class OuterSimulationRunner:
    """Play outer simulations into one request's shared UCT tree and counters."""

    def __init__(
        self,
        root_information: SearchInformationState,
        request_seed: int,
        continuation: ContinuationPolicy,
        exploration_constant: float,
        work: SearchWork,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        self.root_information = root_information
        self.request_seed = request_seed
        self.continuation = continuation
        self.exploration_constant = exploration_constant
        self.work = work
        self.should_stop = should_stop

    def run(self, simulation_index: int) -> None:
        """Sample, play, and back up one complete round simulation."""

        self._check_interrupted()
        state = self._sample_world(simulation_index)
        root_player = self.root_information.player
        trace = SimulationTrace([], [])
        expanded = False

        while state.status is EngineStatus.PLAYING:
            self._check_interrupted()
            selected = self._select_move(
                state,
                root_player,
                expanded,
                simulation_index,
            )
            self._record_selection(trace, root_player, state, selected)
            expanded = expanded or selected.expands_tree
            trace.steps.append(
                _continuation_step(
                    root_player,
                    state,
                    selected.move,
                    selected.action_index,
                    selected.forced,
                )
            )
            state = apply_simulation_move(state, selected.move)

        back_up_simulation(
            state,
            root_player,
            trace,
            simulation_index,
            self.work,
        )

    def _sample_world(self, simulation_index: int) -> SimulationEngineState:
        """Sample hidden cards without letting them influence tree identity."""

        # Every simulation receives a fresh plausible world. Later tree lookup
        # deliberately uses actor-visible fingerprints rather than this state.
        return sample_determinization(
            self.root_information,
            simulation_seed(self.request_seed, simulation_index),
        ).state

    def _select_move(
        self,
        state: SimulationEngineState,
        root_player: EnginePlayer,
        tree_already_expanded: bool,
        simulation_index: int,
    ) -> _SelectedMove:
        """Route one turn to forced play, tree search, or continuation play."""

        actor = state.active_player
        if actor is None:
            raise SearchContractViolation("playing simulation lost its active player")
        legal = legal_simulation_moves(state, actor)

        if len(legal) == 1:
            # Forced moves change the sampled game but add no searchable edge.
            move = legal[0]
            return _SelectedMove(
                move=move,
                group=None,
                action_index=action_index_for_move(move, actor),
                tree_node=None,
                expands_tree=False,
                forced=True,
            )

        if actor is root_player and not tree_already_expanded:
            return self._select_tree_move(state, simulation_index)
        return self._select_continuation_move(state)

    @staticmethod
    def _record_selection(
        trace: SimulationTrace,
        root_player: EnginePlayer,
        state: SimulationEngineState,
        selected: _SelectedMove,
    ) -> None:
        """Add the selected edge and initial root choice to the simulation trace."""

        if selected.tree_node is not None:
            if selected.group is None:
                raise SearchContractViolation("tree move has no strategic group")
            trace.path.append((selected.tree_node, selected.group))

        if trace.root_group is not None:
            return
        # The caller handles forced root requests before starting simulations.
        # Therefore the first simulated move must be the searched root choice.
        if state.active_player is not root_player or selected.group is None:
            raise SearchContractViolation(
                "learned root simulation began without a strategic group"
            )
        trace.root_group = selected.group
        trace.root_action_index = selected.action_index

    def _select_tree_move(
        self,
        state: SimulationEngineState,
        simulation_index: int,
    ) -> _SelectedMove:
        """Select and concretize one UCT edge for a root-player turn."""

        node = self._tree_node(state)
        selected_group = select_group(node, self.exploration_constant)
        representative = selected_group.representative_action_index
        newly_expanded = node.action_visits[representative] == 0
        action_index = select_concrete_action_index(
            selected_group,
            self.request_seed,
            simulation_index,
            node.key.information_state_fingerprint,
            representative,
        )
        return _SelectedMove(
            move=_move_for_action(state, action_index),
            group=selected_group,
            action_index=action_index,
            tree_node=node,
            expands_tree=newly_expanded,
            forced=False,
        )

    def _tree_node(self, state: SimulationEngineState) -> Node:
        """Find or create the node shared by equivalent sampled worlds."""

        actor = state.active_player
        if actor is None:
            raise SearchContractViolation("tree selection has no active player")
        actor_information = information_state_from_simulation(state)
        key = NodeKey(actor, information_state_fingerprint(actor_information))
        groups = strategic_action_groups(actor_information)
        node = self.work.nodes.get(key)
        if node is None:
            node = Node.create(key, groups)
            self.work.nodes[key] = node
        elif node.groups != groups:
            # A fingerprint identifies the complete visible decision state, so
            # changing legal groups under the same key indicates contract drift.
            raise SearchContractViolation(
                "one information set produced inconsistent groups"
            )
        return node

    def _select_continuation_move(
        self,
        state: SimulationEngineState,
    ) -> _SelectedMove:
        """Concretize one cached or newly computed continuation decision."""

        actor_information = information_state_from_simulation(state)
        decision = self._continuation_decision(actor_information)
        return _SelectedMove(
            move=_move_for_action(state, decision.selected_action_index),
            group=decision.selected_group,
            action_index=decision.selected_action_index,
            tree_node=None,
            expands_tree=False,
            forced=False,
        )

    def _continuation_decision(
        self,
        actor_information: SearchInformationState,
    ) -> ContinuationDecision:
        """Reuse responses only when the simulated actor sees the same state."""

        fingerprint = information_state_fingerprint(actor_information)
        # A hidden determinization cannot distinguish otherwise identical
        # continuation requests or create a separate cached response.
        self.work.continuation_requests += 1
        decision = self.work.continuation_cache.get(fingerprint)
        if decision is None:
            decision = self.continuation.select(
                actor_information,
                self.should_stop,
            )
            self.work.continuation_cache[fingerprint] = decision
            self.work.terminal_evaluations += decision.terminal_evaluation_count
            self.work.model_inferences += decision.model_inference_count
        else:
            self.work.continuation_cache_hits += 1
        return decision

    def _check_interrupted(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise SearchInterrupted("UCT search interrupted before result commit")


__all__ = ("OuterSimulationRunner", "simulation_seed")
