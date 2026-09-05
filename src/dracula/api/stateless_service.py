"""Execute stateless gameplay by replaying browser-held command history.

The service reconstructs authoritative engine state, validates and applies one
command, invokes the opponent when required, and returns a public projection.
Its bounded replay cache is optional and cannot affect correctness.
"""

from __future__ import annotations

import secrets
from typing import Callable, cast
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
from dracula.api.stateless_replay import (
    DEFAULT_REPLAY_CACHE_ENTRIES,
    ReplayCache,
    ReplayCacheStatistics,
    ReplayedGame,
    canonical_envelope_digest,
)
from dracula.bridge import apply_policy_action, build_policy_turn_context
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    RuleViolation,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.search import information_state_from_engine


class StatelessReplayError(Exception):
    """Publicly mappable rejection of an invalid history, command, or phase."""

    def __init__(self, code: str, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def stateless_replay_error_response(error: StatelessReplayError) -> ServiceResponse:
    """Convert a replay rejection into the shared public error contract."""

    body = StatelessApiErrorResponse(
        code=error.code,  # type: ignore[arg-type]
        message=error.message,
        retryable=False,
    )
    return ServiceResponse(error.status_code, body.model_dump(mode="json"))


class StatelessPolicyError(Exception):
    """Opponent inference failed before a reconstructed move was accepted."""


def _game_id(seed: str, human_role: EnginePlayer) -> UUID:
    """Derive a stable public game identifier from seed and selected role."""

    return uuid5(
        NAMESPACE_URL,
        f"dracula-stateless-game\0{seed}\0{human_role.value}",
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
    def _dependency_error() -> ServiceResponse:
        body = StatelessApiErrorResponse(
            code="dependency_unavailable",
            message="opponent policy is temporarily unavailable",
            retryable=True,
        )
        return ServiceResponse(503, body.model_dump(mode="json"))

    def _opponent_move(self, game: ReplayedGame) -> ReplayedGame:
        """Apply one opponent action after the settle loop establishes ownership."""

        state = game.state
        opponent = other_player(game.human_role)
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
                        policy_input=None,
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

    def _apply_place_command(
        self,
        game: ReplayedGame,
        command: PlaceCardCommand,
        *,
        history_replay: bool,
    ) -> ReplayedGame:
        """Validate one human placement against reconstructed engine state."""

        state = game.state
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
        move = EngineMove(game.human_role, command.hand_slot, command.position)
        if move not in legal_moves(state, game.human_role):
            raise StatelessReplayError(
                "invalid_history" if history_replay else "invalid_command",
                "placement is not legal in the reconstructed game",
            )
        placed = ReplayedGame(
            apply_move(state, move).state,
            game.human_role,
            game.game_id,
        )
        return self._settle_opponent(placed)

    def _apply_advance_command(
        self,
        game: ReplayedGame,
        *,
        history_replay: bool,
    ) -> ReplayedGame:
        """Advance exactly one completed round and settle any opponent opener."""

        if game.state.status is not EngineStatus.ROUND_COMPLETE:
            raise StatelessReplayError(
                "invalid_history" if history_replay else "wrong_phase",
                "round is not ready to advance",
                status_code=422 if history_replay else 409,
            )
        advanced = advance_after_round(game.state)
        return self._settle_opponent(
            ReplayedGame(advanced, game.human_role, game.game_id)
        )

    def _apply_history_command(
        self,
        game: ReplayedGame,
        command: PlaceCardCommand | AdvanceRoundCommand,
        *,
        history_replay: bool,
    ) -> ReplayedGame:
        """Dispatch one command while preserving replay-specific error codes."""

        if isinstance(command, PlaceCardCommand):
            return self._apply_place_command(
                game, command, history_replay=history_replay
            )
        return self._apply_advance_command(game, history_replay=history_replay)

    def replay(self, envelope: RecoveryEnvelope) -> ReplayedGame:
        """Reconstruct the exact private game represented by an accepted history."""

        key = canonical_envelope_digest(envelope)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        # RecoveryEnvelope guarantees that the first command is the sole role
        # selection, so replay consumes that established invariant directly.
        role_command = cast(SelectRoleCommand, envelope.history[0])
        human_role = EnginePlayer(role_command.human_role)
        # A cache miss always starts from the seed and replays every accepted
        # command; no process-local value is required for correctness.
        game = ReplayedGame(
            create_game(envelope.seed),
            human_role,
            _game_id(envelope.seed, human_role),
        )
        try:
            game = self._settle_opponent(game)
            for command in envelope.history[1:]:
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
            return stateless_replay_error_response(error)
        return self._response(envelope, game, status_code=201)

    def resume_game(self, request: ResumeStatelessGameRequest) -> ServiceResponse:
        """Replay an existing envelope without changing its command history."""

        try:
            game = self.replay(request.envelope)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return stateless_replay_error_response(error)
        return self._response(request.envelope, game)

    def apply_command(
        self,
        request: ApplyStatelessCommandRequest,
    ) -> ServiceResponse:
        """Replay, apply one new command, and return the extended envelope."""

        if len(request.envelope.history) >= MAX_ACCEPTED_COMMANDS:
            return stateless_replay_error_response(
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
            return stateless_replay_error_response(error)
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
    "stateless_replay_error_response",
)
