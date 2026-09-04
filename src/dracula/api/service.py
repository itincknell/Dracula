"""Application orchestration for persisted human-versus-policy games."""

from __future__ import annotations

import secrets
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import UUID, uuid4

from dracula.api.contracts import (
    ApiErrorResponse,
    CreateGameRequest,
    EventsResponse,
    HumanGameView,
    MoveRequest,
    OpponentTurnPhase,
    VersionedMutationRequest,
)
from dracula.api.repository import (
    ConcurrentSessionUpdate,
    GameRepository,
    RequestIdConflict,
    SessionNotFound,
)
from dracula.api.policy import (
    PolicyDescriptor,
    PolicyExecutionError,
    PolicyExecutor,
    PolicyTurnRequest,
    PolicyTurnResult,
    ServiceResponse,
    UnavailablePolicyExecutor,
    zero_hidden_state,
)
from dracula.api.presentation import phase_for_state
from dracula.api.session import (
    GameSession,
    PolicySession,
    PolicyTurnClaim,
    make_event,
    project_human_game,
    request_fingerprint,
    resolve_human_move_id,
    response_body,
    response_record,
    validate_hidden_state,
)
from dracula.cards import CARD_SCHEMA_VERSION
from dracula.engine import (
    ENGINE_VERSION,
    RULES_VERSION,
    EngineMove,
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    other_player,
)
from dracula.randomness import RANDOMNESS_SCHEMA_VERSION


class GameplayService:
    """Repository-backed transaction service retained for explicit local gameplay."""

    def __init__(
        self,
        repository: GameRepository,
        *,
        policy_executor: PolicyExecutor | None = None,
        policy_descriptor: PolicyDescriptor | None = None,
        narration_enabled: bool = False,
        clock: Callable[[], datetime] | None = None,
        game_id_factory: Callable[[], UUID] | None = None,
        game_seed_factory: Callable[[], str] | None = None,
    ) -> None:
        self.repository = repository
        self.policy_executor = policy_executor or UnavailablePolicyExecutor()
        self.policy_descriptor = policy_descriptor or PolicyDescriptor()
        self.narration_enabled = narration_enabled
        self.clock = clock or (lambda: datetime.now(UTC))
        self.game_id_factory = game_id_factory or uuid4
        self.game_seed_factory = game_seed_factory or (lambda: secrets.token_hex(32))

    def _view(self, session: GameSession) -> HumanGameView:
        return project_human_game(
            session, narration_enabled=self.narration_enabled
        )

    def _response(self, view: HumanGameView, status_code: int = 200) -> ServiceResponse:
        return ServiceResponse(status_code, view.model_dump(mode="json"))

    def _error(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        session: GameSession | None = None,
    ) -> ServiceResponse:
        view = None if session is None else self._view(session)
        error = ApiErrorResponse(
            code=code,  # type: ignore[arg-type]
            message=message,
            retryable=retryable,
            current_version=None if session is None else session.version,
            current_game=view,
        )
        return ServiceResponse(status_code, error.model_dump(mode="json"))

    def _load(self, game_id: UUID) -> GameSession | ServiceResponse:
        try:
            return self.repository.load(game_id)
        except SessionNotFound:
            return self._error(404, "not_found", "game was not found")

    @staticmethod
    def _request_data(request: Any) -> dict[str, Any]:
        return request.model_dump(mode="json", exclude={"request_id"})

    def _replay_or_validate(
        self, session: GameSession, request: MoveRequest | VersionedMutationRequest
    ) -> tuple[str, str, ServiceResponse | None]:
        request_id = str(request.request_id)
        request_hash = request_fingerprint(self._request_data(request))
        existing = next(
            (
                record
                for record in session.idempotency_records
                if record.request_id == request_id
            ),
            None,
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                return (
                    request_id,
                    request_hash,
                    self._error(
                        409,
                        "request_id_conflict",
                        "request ID was already used with another body",
                        session=session,
                    ),
                )
            return (
                request_id,
                request_hash,
                ServiceResponse(existing.status_code, response_body(existing)),
            )
        if (
            session.policy_turn_claim is not None
            and session.policy_turn_claim.request_id == request_id
            and session.policy_turn_claim.request_hash != request_hash
        ):
            return (
                request_id,
                request_hash,
                self._error(
                    409,
                    "request_id_conflict",
                    "request ID was already used with another body",
                    session=session,
                ),
            )
        if request.expected_version != session.version:
            return (
                request_id,
                request_hash,
                self._error(
                    409,
                    "stale_version",
                    "expected version does not match the current game",
                    session=session,
                ),
            )
        return request_id, request_hash, None

    def _commit_or_conflict(
        self,
        session: GameSession,
        expected_revision: int,
        *,
        request_id: str | None = None,
        request_hash: str | None = None,
    ) -> ServiceResponse | None:
        try:
            self.repository.commit(session, expected_revision=expected_revision)
            return None
        except ConcurrentSessionUpdate:
            current = self.repository.load(session.game_id)
            if request_id is not None and request_hash is not None:
                record = next(
                    (
                        item
                        for item in current.idempotency_records
                        if item.request_id == request_id
                    ),
                    None,
                )
                if record is not None and record.request_hash == request_hash:
                    return ServiceResponse(record.status_code, response_body(record))
            return self._error(
                409,
                "stale_version",
                "game changed while the request was being processed",
                session=current,
            )

    def _completed_claim_response(
        self, session: GameSession, claim: PolicyTurnClaim
    ) -> ServiceResponse | None:
        record = next(
            (
                item
                for item in session.idempotency_records
                if item.request_id == claim.request_id
            ),
            None,
        )
        if record is None or record.request_hash != claim.request_hash:
            return None
        return ServiceResponse(record.status_code, response_body(record))

    def create_game(self, request: CreateGameRequest) -> ServiceResponse:
        """Create one idempotent persisted local game and its opening events."""

        request_id = str(request.request_id)
        request_hash = request_fingerprint(self._request_data(request))
        existing = self.repository.find_create_request(request_id)
        if existing is not None:
            if existing.request_hash != request_hash:
                return self._error(
                    409,
                    "request_id_conflict",
                    "request ID was already used with another body",
                )
            return ServiceResponse(existing.status_code, response_body(existing))

        human_role = EnginePlayer(request.human_role)
        seed = request.seed if request.seed is not None else self.game_seed_factory()
        state = create_game(seed)
        game_id = self.game_id_factory()
        descriptor = self.policy_descriptor
        now = self.clock()
        events = (
            make_event(
                game_id,
                1,
                "game_created",
                now,
                0,
                0,
                request_id,
                {
                    "human_role": human_role.value,
                    "opponent_role": other_player(human_role).value,
                    "policy_id": descriptor.policy_id,
                    "policy_version": descriptor.policy_version,
                    "artifact_id": descriptor.artifact_id,
                    "engine_version": ENGINE_VERSION,
                    "rules_version": RULES_VERSION,
                    "card_schema_version": CARD_SCHEMA_VERSION,
                    "randomness_schema_version": RANDOMNESS_SCHEMA_VERSION,
                    "observation_schema_version": descriptor.observation_schema_version,
                    "action_schema_version": descriptor.action_schema_version,
                    "hidden_state_schema_version": descriptor.hidden_state_schema_version,
                    "inference_profile": descriptor.inference_profile,
                    "narration_enabled": self.narration_enabled,
                },
            ),
            make_event(
                game_id,
                2,
                "round_started",
                now,
                0,
                0,
                request_id,
                {
                    "round_number": 1,
                    "center_card": state.coffin[4],
                    "dealer": state.dealer.value,
                    "first_player": state.active_player.value,
                },
            ),
        )
        session = GameSession(
            game_id=game_id,
            version=0,
            revision=0,
            human_role=human_role,
            engine_state=state,
            policy_session=PolicySession(
                policy_id=descriptor.policy_id,
                policy_version=descriptor.policy_version,
                artifact_id=descriptor.artifact_id,
                artifact_sha256=descriptor.artifact_sha256,
                observation_schema_version=descriptor.observation_schema_version,
                action_schema_version=descriptor.action_schema_version,
                hidden_state_schema_version=descriptor.hidden_state_schema_version,
                inference_profile=descriptor.inference_profile,
                hidden_state=zero_hidden_state(),
            ),
            move_secret=secrets.token_bytes(32),
            phase=phase_for_state(state, human_role),
            events=events,
            idempotency_records=(),
        )
        view = self._view(session)
        record = response_record(
            request_id, request_hash, 201, view.model_dump(mode="json")
        )
        session = replace(session, idempotency_records=(record,))
        try:
            prior = self.repository.create(session, record)
        except RequestIdConflict:
            return self._error(
                409,
                "request_id_conflict",
                "request ID was already used with another body",
            )
        if prior is not None:
            return ServiceResponse(prior.status_code, response_body(prior))
        return ServiceResponse(201, view.model_dump(mode="json"))

    def get_game(self, game_id: UUID) -> ServiceResponse:
        """Return the latest public view of a persisted local game."""

        session = self._load(game_id)
        if isinstance(session, ServiceResponse):
            return session
        return self._response(self._view(session))

    def get_events(self, game_id: UUID, after_sequence: int = 0) -> ServiceResponse:
        """Return local public events strictly after a sequence number."""

        session = self._load(game_id)
        if isinstance(session, ServiceResponse):
            return session
        events = tuple(
            event for event in session.events if event.sequence > after_sequence
        )
        body = EventsResponse(
            events=events, latest_sequence=len(session.events)
        ).model_dump(mode="json")
        return ServiceResponse(200, body)

    def _move_events(
        self,
        session: GameSession,
        transition: Any,
        request_id: str,
        resulting_version: int,
    ) -> tuple[Any, ...]:
        sequence = len(session.events) + 1
        now = self.clock()
        events = [
            make_event(
                session.game_id,
                sequence,
                "move_accepted",
                now,
                session.version,
                resulting_version,
                request_id,
                {
                    "player": transition.played_move.player.value,
                    "card_id": transition.played_move.card_id,
                    "hand_slot": transition.played_move.hand_slot,
                    "position": transition.played_move.global_grid_index,
                    "turn_number": transition.played_move.turn_number,
                    "round_number": transition.state.round_number,
                },
            )
        ]
        result = transition.round_result
        if result is not None:
            for player in (result.dealer, other_player(result.dealer)):
                sequence += 1
                event_type = (
                    "row_score_ready"
                    if player is EnginePlayer.QUEEN
                    else "column_score_ready"
                )
                events.append(
                    make_event(
                        session.game_id,
                        sequence,
                        event_type,
                        now,
                        session.version,
                        resulting_version,
                        request_id,
                        {
                            "round_number": result.round_number,
                            "player": player.value,
                            "round_score": result.round_scores[player],
                        },
                    )
                )
            sequence += 1
            events.append(
                make_event(
                    session.game_id,
                    sequence,
                    "round_completed",
                    now,
                    session.version,
                    resulting_version,
                    request_id,
                    {
                        "round_number": result.round_number,
                        "queen_score": result.round_scores.queen,
                        "king_score": result.round_scores.king,
                    },
                )
            )
        return tuple(events)

    def _finish_move(
        self,
        session: GameSession,
        transition: Any,
        request_id: str,
        request_hash: str,
        *,
        next_hidden_state: bytes | None = None,
    ) -> ServiceResponse:
        version = session.version + 1
        events = self._move_events(session, transition, request_id, version)
        policy_session = session.policy_session
        if next_hidden_state is not None:
            policy_session = replace(policy_session, hidden_state=next_hidden_state)
        next_session = replace(
            session,
            version=version,
            revision=session.revision + 1,
            engine_state=transition.state,
            policy_session=policy_session,
            phase=phase_for_state(transition.state, session.human_role),
            events=session.events + events,
            policy_turn_claim=None,
        )
        view = self._view(next_session)
        record = response_record(
            request_id, request_hash, 200, view.model_dump(mode="json")
        )
        next_session = replace(
            next_session,
            idempotency_records=next_session.idempotency_records + (record,),
        )
        conflict = self._commit_or_conflict(
            next_session,
            session.revision,
            request_id=request_id,
            request_hash=request_hash,
        )
        return conflict or self._response(view)

    def human_move(self, game_id: UUID, request: MoveRequest) -> ServiceResponse:
        """Idempotently apply one opaque human move token."""

        session = self._load(game_id)
        if isinstance(session, ServiceResponse):
            return session
        request_id, request_hash, replay = self._replay_or_validate(session, request)
        if replay is not None:
            return replay
        state = session.engine_state
        if state.status is not EngineStatus.PLAYING:
            return self._error(
                409, "wrong_phase", "game is not accepting moves", session=session
            )
        if state.active_player is not session.human_role:
            return self._error(
                409, "wrong_turn", "it is not the human turn", session=session
            )
        resolved = resolve_human_move_id(session, request.move_id)
        if resolved is None:
            return self._error(
                422,
                "invalid_move",
                "move ID is invalid or expired",
                session=session,
            )
        hand_slot, position = resolved
        transition = apply_move(
            state, EngineMove(session.human_role, hand_slot, position)
        )
        return self._finish_move(session, transition, request_id, request_hash)

    def opponent_turn(
        self, game_id: UUID, request: VersionedMutationRequest
    ) -> ServiceResponse:
        """Claim or resume one persisted opponent inference transaction."""

        session = self._load(game_id)
        if isinstance(session, ServiceResponse):
            return session
        request_id, request_hash, replay = self._replay_or_validate(session, request)
        if replay is not None:
            return replay
        state = session.engine_state
        opponent = other_player(session.human_role)
        if state.status is not EngineStatus.PLAYING:
            return self._error(
                409, "wrong_phase", "game is not awaiting an opponent move", session=session
            )
        if state.active_player is not opponent:
            return self._error(
                409, "wrong_turn", "it is not the opponent turn", session=session
            )
        if session.policy_turn_claim is not None:
            if (
                session.policy_turn_claim.request_id == request_id
                and session.policy_turn_claim.request_hash != request_hash
            ):
                return self._error(
                    409,
                    "request_id_conflict",
                    "request ID was already used with another body",
                    session=session,
                )
            if (
                session.policy_turn_claim.request_id == request_id
                and session.policy_turn_claim.request_hash == request_hash
            ):
                return self._complete_policy_turn(session, session.policy_turn_claim)
            return self._error(
                409,
                "wrong_phase",
                "an opponent turn is already pending",
                retryable=True,
                session=session,
            )

        # The public job identifier is the idempotency key for this claimed turn.
        # A reloaded browser can therefore resume the same claim without retaining
        # presentation-only request state.
        job_id = str(request_id)
        claim = PolicyTurnClaim(job_id, request_id, request_hash, session.version)
        claimed = replace(
            session,
            revision=session.revision + 1,
            phase=OpponentTurnPhase(status="pending", job_id=job_id),
            policy_turn_claim=claim,
        )
        conflict = self._commit_or_conflict(claimed, session.revision)
        if conflict is not None:
            return conflict

        return self._response(self._view(claimed), status_code=202)

    def _complete_policy_turn(
        self, claimed: GameSession, claim: PolicyTurnClaim
    ) -> ServiceResponse:
        state = claimed.engine_state
        opponent = other_player(claimed.human_role)
        persisted_policy = claimed.policy_session
        policy_descriptor = PolicyDescriptor(
            policy_id=persisted_policy.policy_id,
            policy_version=persisted_policy.policy_version,
            artifact_id=persisted_policy.artifact_id,
            artifact_sha256=persisted_policy.artifact_sha256,
            observation_schema_version=persisted_policy.observation_schema_version,
            action_schema_version=persisted_policy.action_schema_version,
            hidden_state_schema_version=persisted_policy.hidden_state_schema_version,
            inference_profile=persisted_policy.inference_profile,
        )

        try:
            # The bridge is loaded only when inference is requested, preserving a
            # lightweight FastAPI import path for health and public state reads.
            from dracula.bridge import apply_policy_action, build_policy_turn_context
            from dracula.search import information_state_from_engine

            context = build_policy_turn_context(state, opponent)
            information_state = information_state_from_engine(state, opponent)
            policy_result = self.policy_executor.invoke(
                PolicyTurnRequest(
                    game_id=claimed.game_id,
                    policy=policy_descriptor,
                    turn_number=len(state.current_round_moves) + 1,
                    player=opponent,
                    round_number=state.round_number,
                    turn_kind=context.kind,
                    policy_input=None,
                    action_table=context.action_table,
                    information_state=information_state,
                    hidden_state=claimed.policy_session.hidden_state,
                )
            )
            if policy_result.hidden_state is not None:
                validate_hidden_state(policy_result.hidden_state)
            transition = apply_policy_action(
                state, context, policy_result.action_index
            )
        except Exception:
            failed_base = self.repository.load(claimed.game_id)
            if failed_base.policy_turn_claim != claim:
                completed = self._completed_claim_response(failed_base, claim)
                if completed is not None:
                    return completed
                return self._error(
                    409,
                    "stale_version",
                    "opponent turn changed while inference was running",
                    session=failed_base,
                )
            failed = replace(
                failed_base,
                revision=failed_base.revision + 1,
                phase=OpponentTurnPhase(
                    status="failed", job_id=claim.job_id, retryable=True
                ),
                policy_turn_claim=None,
            )
            error_response = self._error(
                503,
                "dependency_unavailable",
                "opponent policy is temporarily unavailable",
                retryable=True,
                session=failed,
            )
            record = response_record(
                claim.request_id, claim.request_hash, 503, error_response.body
            )
            failed = replace(
                failed,
                idempotency_records=failed.idempotency_records + (record,),
            )
            conflict = self._commit_or_conflict(
                failed,
                failed_base.revision,
                request_id=claim.request_id,
                request_hash=claim.request_hash,
            )
            return conflict or error_response

        claimed_current = self.repository.load(claimed.game_id)
        if claimed_current.policy_turn_claim != claim:
            completed = self._completed_claim_response(claimed_current, claim)
            if completed is not None:
                return completed
            return self._error(
                409,
                "stale_version",
                "opponent turn changed while inference was running",
                session=claimed_current,
            )
        return self._finish_move(
            claimed_current,
            transition,
            claim.request_id,
            claim.request_hash,
            next_hidden_state=policy_result.hidden_state,
        )

    def advance_round(
        self,
        game_id: UUID,
        round_number: int,
        request: VersionedMutationRequest,
    ) -> ServiceResponse:
        """Idempotently archive a scored round and deal or finish the game."""

        session = self._load(game_id)
        if isinstance(session, ServiceResponse):
            return session
        request_id, request_hash, replay = self._replay_or_validate(session, request)
        if replay is not None:
            return replay
        state = session.engine_state
        if round_number < state.round_number or state.status is EngineStatus.GAME_COMPLETE:
            return self._error(
                409,
                "already_advanced",
                "round has already been advanced",
                session=session,
            )
        if round_number != state.round_number or state.status is not EngineStatus.ROUND_COMPLETE:
            return self._error(
                409,
                "wrong_phase",
                "round is not ready to advance",
                session=session,
            )
        next_state = advance_after_round(state)
        version = session.version + 1
        now = self.clock()
        if next_state.status is EngineStatus.GAME_COMPLETE:
            outcome = phase_for_state(next_state, session.human_role)
            event = make_event(
                game_id,
                len(session.events) + 1,
                "game_completed",
                now,
                session.version,
                version,
                request_id,
                {
                    "outcome": outcome.outcome,  # type: ignore[union-attr]
                    "queen_total": next_state.total_scores.queen,
                    "king_total": next_state.total_scores.king,
                },
            )
        else:
            event = make_event(
                game_id,
                len(session.events) + 1,
                "round_started",
                now,
                session.version,
                version,
                request_id,
                {
                    "round_number": next_state.round_number,
                    "center_card": next_state.coffin[4],
                    "dealer": next_state.dealer.value,
                    "first_player": next_state.active_player.value,
                },
            )
        next_session = replace(
            session,
            version=version,
            revision=session.revision + 1,
            engine_state=next_state,
            phase=phase_for_state(next_state, session.human_role),
            events=session.events + (event,),
        )
        view = self._view(next_session)
        record = response_record(
            request_id, request_hash, 200, view.model_dump(mode="json")
        )
        next_session = replace(
            next_session,
            idempotency_records=next_session.idempotency_records + (record,),
        )
        conflict = self._commit_or_conflict(
            next_session,
            session.revision,
            request_id=request_id,
            request_hash=request_hash,
        )
        return conflict or self._response(view)
