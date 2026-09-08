"""Execute stateless gameplay by replaying browser-held command history.

The service reconstructs authoritative engine state, validates and applies one
command, invokes the opponent when required, and returns a public projection.
Its bounded replay cache is optional and cannot affect correctness.
"""

from __future__ import annotations

import secrets
from typing import Callable, cast

from dracula.api.contracts import ServiceResponse
from dracula.policy.contracts import (
    PolicyExecutor,
    PolicyTurnRequest,
    UnavailablePolicyExecutor,
)
from dracula.api.stateless.contracts import (
    AdvanceRoundCommand,
    ApplyStatelessCommandRequest,
    MAX_ACCEPTED_COMMANDS,
    PlaceCardCommand,
    RecoveryEnvelope,
    ResumeStatelessGameRequest,
    SelectRoleCommand,
    StartStatelessGameRequest,
    StatelessApiErrorResponse,
    StatelessErrorCode,
    StatelessGameResponse,
)
from dracula.api.stateless.projection import project_stateless_game
from dracula.api.stateless.replay import ReplayCache, ReplayedGame
from dracula.decision.bridge import (
    PolicyTurnContext,
    apply_policy_action,
    build_policy_turn_context,
)
from dracula.game.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    EngineTransition,
    RuleViolation,
    advance_after_round,
    apply_move,
    create_game,
    other_player,
)
from dracula.decision.information import information_state_from_engine


class StatelessReplayError(Exception):
    """Publicly mappable rejection of an invalid history, command, or phase."""

    def __init__(
        self,
        code: StatelessErrorCode,
        message: str,
        *,
        status_code: int = 422,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def stateless_replay_error_response(error: StatelessReplayError) -> ServiceResponse:
    """Convert a replay rejection into the shared public error contract."""

    body = StatelessApiErrorResponse(
        code=error.code,
        message=error.message,
        retryable=False,
    )
    return ServiceResponse(error.status_code, body.model_dump(mode="json"))


class StatelessPolicyError(Exception):
    """Opponent inference failed before a reconstructed move was accepted."""


class StatelessGameplayService:
    """Reconstruct, validate, and advance games without persistent server state."""

    def __init__(
        self,
        *,
        policy_executor: PolicyExecutor | None = None,
        game_seed_factory: Callable[[], str] | None = None,
        cache: ReplayCache | None = None,
    ) -> None:
        self.policy_executor = (
            UnavailablePolicyExecutor() if policy_executor is None else policy_executor
        )
        self.game_seed_factory = (
            (lambda: secrets.token_hex(32))
            if game_seed_factory is None
            else game_seed_factory
        )
        self.cache = ReplayCache() if cache is None else cache

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
        # This projector is the only transition from reconstructed private
        # engine state to the browser-visible game representation.
        body = StatelessGameResponse(
            envelope=envelope,
            game=project_stateless_game(game.state, game.human_role),
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

    def _policy_request(
        self,
        game: ReplayedGame,
        context: PolicyTurnContext,
    ) -> PolicyTurnRequest:
        """Expose one actor-relative turn while retaining private state here."""

        state = game.state
        opponent = other_player(game.human_role)
        information = information_state_from_engine(state, opponent)
        return PolicyTurnRequest(
            game_key=game.policy_game_key,
            turn_number=len(state.current_round_moves) + 1,
            player=opponent,
            round_number=state.round_number,
            action_table=context.action_table,
            information_state=information,
        )

    def _opponent_transition(
        self,
        game: ReplayedGame,
        context: PolicyTurnContext,
    ) -> EngineTransition:
        """Resolve a forced turn or invoke the configured visible-state policy."""

        state = game.state
        if context.forced_move is not None:
            # The unique legal move is an engine fact and requires no model call.
            return apply_policy_action(state, context, None)
        try:
            result = self.policy_executor.invoke(self._policy_request(game, context))
            return apply_policy_action(state, context, result.action_index)
        except Exception as error:
            raise StatelessPolicyError from error

    def _opponent_move(self, game: ReplayedGame) -> ReplayedGame:
        """Apply one opponent action after the settle loop establishes ownership."""

        opponent = other_player(game.human_role)
        context = build_policy_turn_context(game.state, opponent)
        transition = self._opponent_transition(game, context)
        return ReplayedGame(
            transition.state,
            game.human_role,
            game.policy_game_key,
        )

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
        try:
            transition = apply_move(state, move)
        except RuleViolation as error:
            raise StatelessReplayError(
                "invalid_history" if history_replay else "invalid_command",
                "placement is not legal in the reconstructed game",
            ) from error
        placed = ReplayedGame(
            transition.state,
            game.human_role,
            game.policy_game_key,
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
            ReplayedGame(advanced, game.human_role, game.policy_game_key)
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

        cached = self.cache.get(envelope)
        if cached is not None:
            return cached
        # RecoveryEnvelope guarantees that the first command is the sole role
        # selection, so replay consumes that established invariant directly.
        role_command = cast(SelectRoleCommand, envelope.history[0])
        human_role = EnginePlayer(role_command.human_role)
        # A cache miss always starts from the seed and replays every accepted
        # command; no process-local value is required for correctness.
        # Role prefixes keep identical user-supplied seeds distinct when the
        # human swaps sides. The key feeds only deterministic policy coin tosses.
        policy_game_key = f"{human_role.value}:{envelope.seed}"
        game = ReplayedGame(
            create_game(envelope.seed),
            human_role,
            policy_game_key,
        )
        game = self._settle_opponent(game)
        for command in envelope.history[1:]:
            game = self._apply_history_command(
                game,
                command,
                history_replay=True,
            )
        self.cache.put(envelope, game)
        return game

    def _replay_response(
        self,
        envelope: RecoveryEnvelope,
        *,
        status_code: int = 200,
    ) -> ServiceResponse:
        """Map replay failures while keeping successful projection in one place."""

        try:
            game = self.replay(envelope)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return stateless_replay_error_response(error)
        return self._response(envelope, game, status_code=status_code)

    def create_game(self, request: StartStatelessGameRequest) -> ServiceResponse:
        """Create the leading role command and return the first settled view."""

        seed = request.seed if request.seed is not None else self.game_seed_factory()
        envelope = RecoveryEnvelope(
            seed=seed,
            history=(SelectRoleCommand(human_role=request.human_role),),
        )
        return self._replay_response(envelope, status_code=201)

    def resume_game(self, request: ResumeStatelessGameRequest) -> ServiceResponse:
        """Replay an existing envelope without changing its command history."""

        return self._replay_response(request.envelope)

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
            # Commit the accelerated result only after the command and every
            # automatic opponent move succeed. Repeating the original request
            # therefore cannot observe or apply a partial transition.
            self.cache.put(next_envelope, next_game)
        except StatelessPolicyError:
            return self._dependency_error()
        except StatelessReplayError as error:
            return stateless_replay_error_response(error)
        return self._response(next_envelope, next_game)
