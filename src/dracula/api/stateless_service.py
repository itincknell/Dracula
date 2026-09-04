"""Deterministic production gameplay reconstructed from a browser envelope."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable
from uuid import NAMESPACE_URL, UUID, uuid5

from dracula.api.policy import (
    PolicyDescriptor,
    PolicyExecutionError,
    PolicyExecutor,
    PolicyTurnRequest,
    ServiceResponse,
    UnavailablePolicyExecutor,
    zero_hidden_state,
)
from dracula.api.stateless_contracts import (
    AdvanceRoundCommand,
    ApplyStatelessCommandRequest,
    MAX_ACCEPTED_COMMANDS,
    PlaceCardCommand,
    RecoveryEnvelope,
    ResumeStatelessGameRequest,
    SelectRoleCommand,
    StartStatelessGameRequest,
    StatelessApiErrorResponse,
    StatelessGameResponse,
)
from dracula.api.stateless_projection import project_stateless_game
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    InvalidLifecycleTransition,
    RuleViolation,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
)

DEFAULT_REPLAY_CACHE_ENTRIES = 256


class StatelessReplayError(Exception):
    """Publicly mappable rejection of an invalid history, command, or phase."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class StatelessPolicyError(Exception):
    """Opponent inference failed before a reconstructed move was accepted."""


@dataclass(frozen=True, slots=True)
class ReplayedGame:
    """Trusted private state reconstructed from one complete browser envelope."""

    state: EngineState
    human_role: EnginePlayer
    game_id: UUID


@dataclass(frozen=True, slots=True)
class ReplayCacheStatistics:
    """Process-local diagnostics with no effect on gameplay correctness."""

    hits: int
    misses: int
    evictions: int
    entries: int


class ReplayCache:
    """Bounded process-local cache whose eviction cannot affect correctness."""

    def __init__(self, max_entries: int = DEFAULT_REPLAY_CACHE_ENTRIES) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("replay cache size must be a non-negative integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, ReplayedGame] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> ReplayedGame | None:
        """Return and refresh one cached reconstruction, recording hit or miss."""

        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: ReplayedGame) -> None:
        """Store one reconstruction and evict least-recently-used entries."""

        if self.max_entries == 0:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """Discard cached reconstructions without changing accepted game data."""

        with self._lock:
            self._entries.clear()

    @property
    def statistics(self) -> ReplayCacheStatistics:
        """Return a locked snapshot of cache counters and current size."""

        with self._lock:
            return ReplayCacheStatistics(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._entries),
            )


def canonical_envelope_digest(envelope: RecoveryEnvelope) -> str:
    """Hash canonical seed/history content for optional replay caching."""

    encoded = json.dumps(
        envelope.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _game_id(envelope: RecoveryEnvelope) -> UUID:
    """Derive a stable public game identifier from seed and selected role."""

    role = envelope.history[0]
    assert isinstance(role, SelectRoleCommand)
    return uuid5(
        NAMESPACE_URL,
        f"dracula-stateless-game\0{envelope.seed}\0{role.human_role}",
    )


class StatelessGameplayService:
    """Reconstruct, validate, and advance games without persistent server state."""

    def __init__(
        self,
        *,
        policy_executor: PolicyExecutor | None = None,
        policy_descriptor: PolicyDescriptor | None = None,
        game_seed_factory: Callable[[], str] | None = None,
        cache: ReplayCache | None = None,
    ) -> None:
        self.policy_executor = policy_executor or UnavailablePolicyExecutor()
        self.policy_descriptor = policy_descriptor or PolicyDescriptor()
        self.game_seed_factory = game_seed_factory or (
            lambda: secrets.token_hex(32)
        )
        self.cache = cache or ReplayCache()

    @property
    def opponent_configured(self) -> bool:
        """Report whether opponent turns can be executed in this service."""

        return not isinstance(self.policy_executor, UnavailablePolicyExecutor)

    @staticmethod
    def _response(
        envelope: RecoveryEnvelope,
        game: ReplayedGame,
        *,
        status_code: int = 200,
    ) -> ServiceResponse:
        body = StatelessGameResponse(
            envelope=envelope,
            game=project_stateless_game(game),
        )
        return ServiceResponse(status_code, body.model_dump(mode="json"))

    @staticmethod
    def _error(error: StatelessReplayError) -> ServiceResponse:
        body = StatelessApiErrorResponse(
            code=error.code,  # type: ignore[arg-type]
            message=error.message,
            retryable=False,
        )
        return ServiceResponse(error.status_code, body.model_dump(mode="json"))

    @staticmethod
    def _dependency_error() -> ServiceResponse:
        body = StatelessApiErrorResponse(
            code="dependency_unavailable",
            message="opponent policy is temporarily unavailable",
            retryable=True,
        )
        return ServiceResponse(503, body.model_dump(mode="json"))

    def _opponent_move(self, game: ReplayedGame) -> ReplayedGame:
        """Apply one deterministic opponent action to trusted reconstructed state."""

        state = game.state
        opponent = other_player(game.human_role)
        if (
            state.status is not EngineStatus.PLAYING
            or state.active_player is not opponent
        ):
            return game
        from dracula.bridge import apply_policy_action, build_policy_turn_context
        from dracula.search import information_state_from_engine

        context = build_policy_turn_context(state, opponent)
        if context.forced_move is not None:
            # The unique legal move is an engine fact and requires no model call.
            transition = apply_policy_action(state, context, None)
        else:
            # Only the opponent's actor-relative information state crosses the
            # controller boundary; the reconstructed private state stays here.
            information = information_state_from_engine(state, opponent)
            try:
                result = self.policy_executor.invoke(
                    PolicyTurnRequest(
                        game_id=game.game_id,
                        policy=self.policy_descriptor,
                        turn_number=len(state.current_round_moves) + 1,
                        player=opponent,
                        round_number=state.round_number,
                        turn_kind=context.kind,
                        policy_input=context.input,
                        action_table=context.action_table,
                        information_state=information,
                        hidden_state=zero_hidden_state(),
                    )
                )
                if result.hidden_state is not None:
                    raise PolicyExecutionError(
                        "stateless standalone policy returned recurrent state"
                    )
                transition = apply_policy_action(
                    state,
                    context,
                    result.action_index,
                )
            except Exception as error:
                raise StatelessPolicyError from error
        return ReplayedGame(transition.state, game.human_role, game.game_id)

    def _settle_opponent(self, game: ReplayedGame) -> ReplayedGame:
        """Advance any opponent turn required before returning control to the human."""

        while (
            game.state.status is EngineStatus.PLAYING
            and game.state.active_player is other_player(game.human_role)
        ):
            game = self._opponent_move(game)
        return game

    def _apply_history_command(
        self,
        game: ReplayedGame,
        command: PlaceCardCommand | AdvanceRoundCommand,
        *,
        history_replay: bool,
    ) -> ReplayedGame:
        """Apply one validated command, distinguishing replay from live errors."""

        state = game.state
        if isinstance(command, PlaceCardCommand):
            if state.status is not EngineStatus.PLAYING:
                raise StatelessReplayError(
                    "invalid_history" if history_replay else "wrong_phase",
                    "game is not accepting a placement",
                    status_code=422 if history_replay else 409,
                )
            if state.active_player is not game.human_role:
                raise StatelessReplayError(
                    "invalid_history" if history_replay else "wrong_turn",
                    "it is not the human turn",
                    status_code=422 if history_replay else 409,
                )
            move = EngineMove(
                game.human_role,
                command.hand_slot,
                command.position,
            )
            if move not in legal_moves(state, game.human_role):
                raise StatelessReplayError(
                    "invalid_history" if history_replay else "invalid_command",
                    "placement is not legal in the reconstructed game",
                )
            game = ReplayedGame(
                apply_move(state, move).state,
                game.human_role,
                game.game_id,
            )
            return self._settle_opponent(game)

        if state.status is not EngineStatus.ROUND_COMPLETE:
            raise StatelessReplayError(
                "invalid_history" if history_replay else "wrong_phase",
                "round is not ready to advance",
                status_code=422 if history_replay else 409,
            )
        try:
            advanced = advance_after_round(state)
        except InvalidLifecycleTransition as error:
            raise StatelessReplayError(
                "invalid_history" if history_replay else "wrong_phase",
                "round cannot be advanced",
                status_code=422 if history_replay else 409,
            ) from error
        return self._settle_opponent(
            ReplayedGame(advanced, game.human_role, game.game_id)
        )

    def replay(self, envelope: RecoveryEnvelope) -> ReplayedGame:
        """Reconstruct the exact private game represented by an accepted history."""

        key = canonical_envelope_digest(envelope)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        role_command = envelope.history[0]
        if not isinstance(role_command, SelectRoleCommand):
            raise StatelessReplayError(
                "invalid_history", "history must begin with role selection"
            )
        human_role = EnginePlayer(role_command.human_role)
        # A cache miss always starts from the seed and replays every accepted
        # command; no process-local value is required for correctness.
        game = ReplayedGame(create_game(envelope.seed), human_role, _game_id(envelope))
        try:
            game = self._settle_opponent(game)
            for command in envelope.history[1:]:
                if isinstance(command, SelectRoleCommand):
                    raise StatelessReplayError(
                        "invalid_history", "role selection may appear only once"
                    )
                game = self._apply_history_command(
                    game,
                    command,
                    history_replay=True,
                )
        except RuleViolation as error:
            raise StatelessReplayError(
                "invalid_history", "history contains an invalid engine transition"
            ) from error
        self.cache.put(key, game)
        return game

    def create_game(self, request: StartStatelessGameRequest) -> ServiceResponse:
        """Create the leading role command and return the first settled view."""

        seed = request.seed if request.seed is not None else self.game_seed_factory()
        envelope = RecoveryEnvelope(
            seed=seed,
            history=(SelectRoleCommand(human_role=request.human_role),),
        )
        try:
            game = self.replay(envelope)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return self._error(error)
        return self._response(envelope, game, status_code=201)

    def resume_game(self, request: ResumeStatelessGameRequest) -> ServiceResponse:
        """Replay an existing envelope without changing its command history."""

        try:
            game = self.replay(request.envelope)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return self._error(error)
        return self._response(request.envelope, game)

    def apply_command(
        self,
        request: ApplyStatelessCommandRequest,
    ) -> ServiceResponse:
        """Replay, apply one new command, and return the extended envelope."""

        if len(request.envelope.history) >= MAX_ACCEPTED_COMMANDS:
            return self._error(
                StatelessReplayError(
                    "invalid_command", "game command history is already complete"
                )
            )
        try:
            game = self.replay(request.envelope)
            next_game = self._apply_history_command(
                game,
                request.command,
                history_replay=False,
            )
            next_envelope = RecoveryEnvelope(
                seed=request.envelope.seed,
                history=request.envelope.history + (request.command,),
            )
            self.cache.put(canonical_envelope_digest(next_envelope), next_game)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return self._error(error)
        return self._response(next_envelope, next_game)


__all__ = (
    "DEFAULT_REPLAY_CACHE_ENTRIES",
    "ReplayCache",
    "ReplayCacheStatistics",
    "ReplayedGame",
    "StatelessGameplayService",
    "StatelessPolicyError",
    "StatelessReplayError",
    "canonical_envelope_digest",
    "project_stateless_game",
)
