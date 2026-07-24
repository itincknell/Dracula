"""Neural-guided information-set search with exact round scoring by default."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from dracula.bridge import ACTION_COUNT, action_index_for_move, build_policy_turn_context
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_moves,
)
from dracula.policy_value import (
    PolicyValueModel,
    legal_policy_probabilities,
    load_policy_value_artifact,
)
from dracula.randomness import Sha256CounterStream, seed_hex
from dracula.search.information import (
    SearchInformationState,
    derive_belief_sample_seed,
    derive_rollout_choice_seed,
    derive_tree_selection_seed,
    information_state_fingerprint,
    project_simulation_information_state,
    policy_input_from_information_state,
    sample_determinization,
    sample_uniform_action_index,
)
from dracula.search.planner import (
    ContinuationStep,
    SearchContractViolation,
    SearchInterrupted,
    normalized_round_return,
)

GUIDED_SEARCH_SCHEMA_VERSION = "dracula-guided-information-search-v1"
UNIFORM_EVALUATOR_DIGEST = hashlib.sha256(
    b"dracula-uniform-policy-value-evaluator-v1"
).hexdigest()


class GuidedLeafValueMode(StrEnum):
    TERMINAL = "terminal"
    EXPERIMENTAL_MODEL = "experimental-model"


@dataclass(frozen=True, slots=True)
class GuidedSearchConfig:
    simulation_budget: int = 100
    puct_constant: float = 1.5
    model_prior_weight: float = 0.95
    uniform_prior_weight: float = 0.05
    opponent_temperature: float = 1.0
    leaf_value_mode: GuidedLeafValueMode = GuidedLeafValueMode.TERMINAL

    def __post_init__(self) -> None:
        if type(self.simulation_budget) is not int or self.simulation_budget < 1:
            raise SearchContractViolation("simulation budget must be a positive integer")
        numeric = (
            self.puct_constant,
            self.model_prior_weight,
            self.uniform_prior_weight,
            self.opponent_temperature,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in numeric):
            raise SearchContractViolation("guided search values must be finite")
        if self.puct_constant < 0:
            raise SearchContractViolation("PUCT constant cannot be negative")
        if self.model_prior_weight < 0 or self.uniform_prior_weight < 0:
            raise SearchContractViolation("prior weights cannot be negative")
        if not math.isclose(
            self.model_prior_weight + self.uniform_prior_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SearchContractViolation("prior weights must sum to one")
        if self.opponent_temperature != 1.0:
            raise SearchContractViolation("version 1 opponent temperature is fixed at one")
        if not isinstance(self.leaf_value_mode, GuidedLeafValueMode):
            raise SearchContractViolation("guided leaf value mode is invalid")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "guided_search_schema_version": GUIDED_SEARCH_SCHEMA_VERSION,
                "simulation_budget": self.simulation_budget,
                "puct_constant": float(self.puct_constant),
                "model_prior_weight": float(self.model_prior_weight),
                "uniform_prior_weight": float(self.uniform_prior_weight),
                "opponent_temperature": float(self.opponent_temperature),
                "leaf_value_mode": self.leaf_value_mode.value,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyValueEvaluation:
    information_state_fingerprint: str
    legal_priors: tuple[float, ...]
    player_relative_value: float

    def __post_init__(self) -> None:
        if len(self.legal_priors) != ACTION_COUNT:
            raise SearchContractViolation("model priors must contain 32 entries")
        if any(not math.isfinite(value) or value < 0 for value in self.legal_priors):
            raise SearchContractViolation("model priors must be finite and non-negative")
        if (
            not math.isfinite(self.player_relative_value)
            or not -1 <= self.player_relative_value <= 1
        ):
            raise SearchContractViolation("model value must be finite and in [-1,1]")


@runtime_checkable
class SearchModelEvaluator(Protocol):
    @property
    def digest(self) -> str: ...

    def evaluate_batch(
        self, information_states: Sequence[SearchInformationState]
    ) -> tuple[PolicyValueEvaluation, ...]: ...


def _model_state_digest(model: PolicyValueModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class PolicyValueModelEvaluator:
    """Information-state-only boundary around one frozen policy/value model."""

    def __init__(
        self,
        model: PolicyValueModel,
        *,
        device: str | torch.device = "cpu",
        artifact_digest: str | None = None,
    ) -> None:
        if not isinstance(model, PolicyValueModel):
            raise SearchContractViolation("guided evaluator requires a policy/value model")
        selected = torch.device(device)
        if selected.type not in {"cpu", "mps"}:
            raise SearchContractViolation("guided evaluator device must be CPU or MPS")
        if selected.type == "mps" and not torch.backends.mps.is_available():
            raise SearchContractViolation("MPS evaluator requested when MPS is unavailable")
        self._model = model.to(selected).eval()
        self._device = selected
        state_digest = artifact_digest or _model_state_digest(model)
        try:
            bytes.fromhex(state_digest)
        except (TypeError, ValueError) as error:
            raise SearchContractViolation("evaluator artifact digest is invalid") from error
        if len(state_digest) != 64:
            raise SearchContractViolation("evaluator artifact digest is invalid")
        self._digest = hashlib.sha256(
            (GUIDED_SEARCH_SCHEMA_VERSION + "\0" + state_digest).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_artifact(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> PolicyValueModelEvaluator:
        loaded = load_policy_value_artifact(path)
        return cls(
            loaded.model,
            device=device,
            artifact_digest=loaded.metadata.state_dict_digest,
        )

    @property
    def digest(self) -> str:
        return self._digest

    @torch.no_grad()
    def evaluate_batch(
        self, information_states: Sequence[SearchInformationState]
    ) -> tuple[PolicyValueEvaluation, ...]:
        if not isinstance(information_states, Sequence) or not information_states:
            raise SearchContractViolation("model evaluation batch cannot be empty")
        if any(not isinstance(value, SearchInformationState) for value in information_states):
            raise SearchContractViolation("model evaluator accepts information states only")
        inputs = tuple(
            policy_input_from_information_state(value) for value in information_states
        )
        observations = torch.stack(tuple(value.observation for value in inputs)).to(
            self._device
        )
        masks = torch.stack(tuple(value.legal_mask for value in inputs)).to(self._device)
        logits, values = self._model(observations)
        priors = legal_policy_probabilities(logits, masks).flatten(start_dim=1)
        if not torch.isfinite(priors).all() or not torch.isfinite(values).all():
            raise SearchContractViolation("model evaluator produced non-finite output")
        return tuple(
            PolicyValueEvaluation(
                information_state_fingerprint(information),
                tuple(float(item) for item in prior.detach().cpu().tolist()),
                float(value.detach().cpu()),
            )
            for information, prior, value in zip(
                information_states, priors, values, strict=True
            )
        )


class UniformPolicyValueEvaluator:
    """Domain-neutral evaluator used only for search-contract controls."""

    @property
    def digest(self) -> str:
        return UNIFORM_EVALUATOR_DIGEST

    def evaluate_batch(
        self, information_states: Sequence[SearchInformationState]
    ) -> tuple[PolicyValueEvaluation, ...]:
        if not information_states:
            raise SearchContractViolation("model evaluation batch cannot be empty")
        evaluations: list[PolicyValueEvaluation] = []
        for information in information_states:
            if not isinstance(information, SearchInformationState):
                raise SearchContractViolation("model evaluator accepts information states only")
            legal = _legal_action_indexes(information)
            prior = 1.0 / len(legal)
            evaluations.append(
                PolicyValueEvaluation(
                    information_state_fingerprint(information),
                    tuple(prior if index in legal else 0.0 for index in range(ACTION_COUNT)),
                    0.0,
                )
            )
        return tuple(evaluations)


@dataclass(frozen=True, slots=True)
class GuidedPrincipalContinuation:
    root_action_index: int
    backed_value: float
    value_source: str
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(frozen=True, slots=True)
class GuidedSearchResult:
    information_state_fingerprint: str
    config_digest: str
    evaluator_digest: str
    selected_action_index: int
    action_visits: tuple[int, ...]
    mean_action_values: tuple[float | None, ...]
    simulation_count: int
    information_set_count: int
    model_evaluation_count: int
    terminal_evaluation_count: int
    cutoff_evaluation_count: int
    elapsed_seconds: float
    principal_continuation: GuidedPrincipalContinuation | None

    def __post_init__(self) -> None:
        if len(self.action_visits) != ACTION_COUNT:
            raise SearchContractViolation("action visits must contain 32 entries")
        if len(self.mean_action_values) != ACTION_COUNT:
            raise SearchContractViolation("action values must contain 32 entries")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise SearchContractViolation("search latency must be finite and non-negative")


@dataclass(slots=True)
class _GuidedNode:
    legal_actions: tuple[int, ...]
    priors: tuple[float, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @classmethod
    def create(
        cls, legal_actions: tuple[int, ...], priors: tuple[float, ...]
    ) -> _GuidedNode:
        return cls(
            legal_actions,
            priors,
            0,
            [0] * ACTION_COUNT,
            [0.0] * ACTION_COUNT,
        )


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    root_action_index: int
    backed_value: float
    value_source: str
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


def _legal_action_indexes(information: SearchInformationState) -> tuple[int, ...]:
    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


def _validate_evaluation(
    information: SearchInformationState,
    evaluation: PolicyValueEvaluation,
) -> PolicyValueEvaluation:
    fingerprint = information_state_fingerprint(information)
    if evaluation.information_state_fingerprint != fingerprint:
        raise SearchContractViolation("model evaluation belongs to another information state")
    legal = set(_legal_action_indexes(information))
    if any(
        evaluation.legal_priors[index] != 0
        for index in range(ACTION_COUNT)
        if index not in legal
    ):
        raise SearchContractViolation("model evaluation assigns prior mass to an illegal action")
    mass = sum(evaluation.legal_priors)
    if not math.isclose(mass, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SearchContractViolation("model legal priors must sum to one")
    return evaluation


def _mixed_priors(
    information: SearchInformationState,
    evaluation: PolicyValueEvaluation,
    config: GuidedSearchConfig,
) -> tuple[float, ...]:
    legal = _legal_action_indexes(information)
    floor = config.uniform_prior_weight / len(legal)
    mixed = tuple(
        config.model_prior_weight * evaluation.legal_priors[index] + floor
        if index in legal
        else 0.0
        for index in range(ACTION_COUNT)
    )
    if not math.isclose(sum(mixed), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SearchContractViolation("mixed legal priors must sum to one")
    return mixed


def _select_puct_action(
    node: _GuidedNode,
    config: GuidedSearchConfig,
    selection_seed: bytes,
) -> int:
    # Canonical initial coverage preserves evidence for every legal root action.
    for action_index in node.legal_actions:
        if node.action_visits[action_index] == 0:
            return action_index
    parent_scale = math.sqrt(node.visits)
    best_score = -math.inf
    tied: list[int] = []
    for action_index in node.legal_actions:
        visits = node.action_visits[action_index]
        mean = node.action_value_sums[action_index] / visits
        score = mean + (
            config.puct_constant
            * node.priors[action_index]
            * parent_scale
            / (1 + visits)
        )
        if score > best_score:
            best_score = score
            tied = [action_index]
        elif score == best_score:
            tied.append(action_index)
    return tied[Sha256CounterStream(selection_seed).randbelow(len(tied))]


def _sample_policy_action(
    information: SearchInformationState,
    priors: tuple[float, ...],
    choice_seed: bytes,
) -> int:
    legal = _legal_action_indexes(information)
    legal_values = tuple(priors[index] for index in legal)
    if all(value == legal_values[0] for value in legal_values[1:]):
        return sample_uniform_action_index(information, choice_seed)
    sample = int.from_bytes(Sha256CounterStream(choice_seed).next_block(), "big") / (1 << 256)
    cumulative = 0.0
    for action_index in legal:
        cumulative += priors[action_index]
        if sample < cumulative:
            return action_index
    return legal[-1]


def player_relative_value_for_root(
    value: float,
    evaluated_player: EnginePlayer,
    root_player: EnginePlayer,
) -> float:
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise SearchContractViolation("model value must be finite and in [-1,1]")
    return value if EnginePlayer(evaluated_player) is EnginePlayer(root_player) else -value


def _move_for_action(state: SimulationEngineState, action_index: int) -> EngineMove:
    actor = state.active_player
    if actor is None:
        raise SearchContractViolation("a playing simulation must have an active player")
    move = build_policy_turn_context(state, actor).action_table[action_index]
    if move is None:
        raise SearchContractViolation("guided search selected a masked action")
    return move


def _continuation_step(
    root_player: EnginePlayer,
    state: SimulationEngineState,
    move: EngineMove,
    action_index: int,
    forced: bool,
) -> ContinuationStep:
    card_id = state.hands[move.player][move.hand_slot]
    if card_id is None:
        raise SearchContractViolation("continuation move refers to an empty hand slot")
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


class GuidedInformationSetSearch:
    """PUCT planner whose evaluator receives only per-actor information states."""

    def __init__(
        self,
        evaluator: SearchModelEvaluator,
        config: GuidedSearchConfig = GuidedSearchConfig(),
    ) -> None:
        if not isinstance(evaluator, SearchModelEvaluator):
            raise SearchContractViolation("guided search evaluator is invalid")
        if not isinstance(config, GuidedSearchConfig):
            raise SearchContractViolation("guided search configuration is invalid")
        try:
            evaluator_digest = evaluator.digest
            bytes.fromhex(evaluator_digest)
        except (AttributeError, TypeError, ValueError) as error:
            raise SearchContractViolation("guided evaluator digest is invalid") from error
        if len(evaluator_digest) != 64:
            raise SearchContractViolation("guided evaluator digest is invalid")
        self.evaluator = evaluator
        self.config = config
        self._digest = hashlib.sha256(
            (config.digest + "\0" + evaluator_digest).encode("ascii")
        ).hexdigest()

    @property
    def digest(self) -> str:
        return self._digest

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes,
        should_stop: Callable[[], bool] | None = None,
    ) -> GuidedSearchResult:
        started = time.perf_counter()
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation("guided search requires an information state")
        seed_hex(request_seed)
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_legal = _legal_action_indexes(information)
        if not root_legal:
            raise SearchContractViolation("guided search root has no legal actions")
        if len(root_legal) == 1:
            return GuidedSearchResult(
                root_fingerprint,
                self.config.digest,
                self.evaluator.digest,
                root_legal[0],
                tuple(0 for _ in range(ACTION_COUNT)),
                tuple(None for _ in range(ACTION_COUNT)),
                0,
                0,
                0,
                0,
                0,
                time.perf_counter() - started,
                None,
            )
        if self.config.simulation_budget < len(root_legal):
            raise SearchContractViolation(
                "simulation budget must visit every legal root action"
            )

        nodes: dict[str, _GuidedNode] = {}
        evaluation_cache: dict[str, PolicyValueEvaluation] = {}
        model_evaluations = 0
        terminal_evaluations = 0
        cutoff_evaluations = 0
        continuations: list[_ObservedContinuation] = []

        def evaluate(actor_information: SearchInformationState) -> PolicyValueEvaluation:
            nonlocal model_evaluations
            fingerprint = information_state_fingerprint(actor_information)
            cached = evaluation_cache.get(fingerprint)
            if cached is not None:
                return cached
            values = self.evaluator.evaluate_batch((actor_information,))
            if len(values) != 1:
                raise SearchContractViolation("model evaluator returned another batch size")
            checked = _validate_evaluation(actor_information, values[0])
            evaluation_cache[fingerprint] = checked
            model_evaluations += 1
            return checked

        for simulation_index in range(self.config.simulation_budget):
            self._check_interrupted(should_stop)
            sampled = sample_determinization(
                information,
                derive_belief_sample_seed(request_seed, simulation_index),
            )
            state = sampled.state
            path: list[tuple[_GuidedNode, int]] = []
            steps: list[ContinuationStep] = []
            expanded = False
            root_action_index: int | None = None
            rollout_ply = 0
            backed_value: float | None = None
            value_source = "exact-terminal"

            while state.status is EngineStatus.PLAYING:
                self._check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation("playing simulation lost its active player")
                moves = legal_moves(state, actor)
                forced = len(moves) == 1
                expanded_now = False
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(state)
                    node_key = information_state_fingerprint(actor_information)
                    legal_actions = _legal_action_indexes(actor_information)
                    node = nodes.get(node_key)
                    if node is None:
                        model_output = evaluate(actor_information)
                        node = _GuidedNode.create(
                            legal_actions,
                            _mixed_priors(actor_information, model_output, self.config),
                        )
                        nodes[node_key] = node
                    elif node.legal_actions != legal_actions:
                        raise SearchContractViolation(
                            "one guided information node produced inconsistent legality"
                        )
                    action_index = _select_puct_action(
                        node,
                        self.config,
                        derive_tree_selection_seed(
                            request_seed, simulation_index, node_key
                        ),
                    )
                    path.append((node, action_index))
                    expanded_now = node.action_visits[action_index] == 0
                    if expanded_now:
                        expanded = True
                    move = _move_for_action(state, action_index)
                else:
                    actor_information = project_simulation_information_state(state)
                    model_output = evaluate(actor_information)
                    action_index = _sample_policy_action(
                        actor_information,
                        model_output.legal_priors,
                        derive_rollout_choice_seed(
                            request_seed, simulation_index, rollout_ply
                        ),
                    )
                    move = _move_for_action(state, action_index)

                if root_action_index is None:
                    if actor is not root_player:
                        raise SearchContractViolation("root simulation began on another actor")
                    root_action_index = action_index
                steps.append(
                    _continuation_step(root_player, state, move, action_index, forced)
                )
                state = apply_simulation_move(state, move)
                if not isinstance(state, SimulationEngineState):
                    raise SearchContractViolation("engine discarded simulation provenance")
                rollout_ply += 1

                if (
                    expanded_now
                    and self.config.leaf_value_mode
                    is GuidedLeafValueMode.EXPERIMENTAL_MODEL
                ):
                    # Forced placements remain engine transitions and never
                    # become neural leaves or consume inference work.
                    while state.status is EngineStatus.PLAYING:
                        next_actor = state.active_player
                        if next_actor is None:
                            raise SearchContractViolation(
                                "playing simulation lost its active player"
                            )
                        next_moves = legal_moves(state, next_actor)
                        if len(next_moves) != 1:
                            break
                        forced_move = next_moves[0]
                        forced_index = action_index_for_move(forced_move, next_actor)
                        steps.append(
                            _continuation_step(
                                root_player,
                                state,
                                forced_move,
                                forced_index,
                                True,
                            )
                        )
                        state = apply_simulation_move(state, forced_move)
                        if not isinstance(state, SimulationEngineState):
                            raise SearchContractViolation(
                                "engine discarded simulation provenance"
                            )
                    if state.status is EngineStatus.PLAYING:
                        leaf_information = project_simulation_information_state(state)
                        leaf_value = evaluate(leaf_information)
                        backed_value = player_relative_value_for_root(
                            leaf_value.player_relative_value,
                            leaf_information.player,
                            root_player,
                        )
                        cutoff_evaluations += 1
                        value_source = "experimental-model"
                        break

            if backed_value is None:
                if (
                    state.status is not EngineStatus.ROUND_COMPLETE
                    or state.pending_round_result is None
                ):
                    raise SearchContractViolation(
                        "guided round search did not reach an engine terminal"
                    )
                backed_value = normalized_round_return(
                    state.pending_round_result, root_player
                )
                terminal_evaluations += 1
            if root_action_index is None or not path:
                raise SearchContractViolation(
                    "a learned guided simulation has no tree path"
                )
            for node, action_index in path:
                node.visits += 1
                node.action_visits[action_index] += 1
                node.action_value_sums[action_index] += backed_value
            continuations.append(
                _ObservedContinuation(
                    root_action_index,
                    backed_value,
                    value_source,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_node = nodes.get(root_fingerprint)
        if root_node is None or root_node.visits != self.config.simulation_budget:
            raise SearchContractViolation("guided root backup count differs")
        selected = min(
            root_legal,
            key=lambda action: (
                -root_node.action_visits[action],
                -root_node.action_value_sums[action] / root_node.action_visits[action],
                action,
            ),
        )
        representative = min(
            (item for item in continuations if item.root_action_index == selected),
            key=lambda item: (-item.backed_value, item.simulation_index),
        )
        means = tuple(
            (
                root_node.action_value_sums[index] / root_node.action_visits[index]
                if root_node.action_visits[index]
                else None
            )
            for index in range(ACTION_COUNT)
        )
        return GuidedSearchResult(
            root_fingerprint,
            self.config.digest,
            self.evaluator.digest,
            selected,
            tuple(root_node.action_visits),
            means,
            root_node.visits,
            len(nodes),
            model_evaluations,
            terminal_evaluations,
            cutoff_evaluations,
            time.perf_counter() - started,
            GuidedPrincipalContinuation(
                representative.root_action_index,
                representative.backed_value,
                representative.value_source,
                representative.simulation_index,
                representative.steps,
            ),
        )

    @staticmethod
    def _check_interrupted(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted("guided search interrupted before result commit")


__all__ = (
    "GUIDED_SEARCH_SCHEMA_VERSION",
    "GuidedInformationSetSearch",
    "GuidedLeafValueMode",
    "GuidedPrincipalContinuation",
    "GuidedSearchConfig",
    "GuidedSearchResult",
    "PolicyValueEvaluation",
    "PolicyValueModelEvaluator",
    "SearchModelEvaluator",
    "UniformPolicyValueEvaluator",
    "player_relative_value_for_root",
)
