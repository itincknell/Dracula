"""Transactional persistence adapters for local gameplay sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol
from uuid import UUID

from dracula.api.contracts import PublicEvent
from dracula.api.session import (
    GameSession,
    IdempotencyRecord,
    session_core_data,
    session_from_core_data,
    validate_session,
)


class RepositoryError(Exception):
    pass


class SessionNotFound(RepositoryError):
    pass


class ConcurrentSessionUpdate(RepositoryError):
    pass


class RequestIdConflict(RepositoryError):
    pass


class GameRepository(Protocol):
    def create(
        self, session: GameSession, create_record: IdempotencyRecord
    ) -> IdempotencyRecord | None: ...

    def load(self, game_id: UUID) -> GameSession: ...

    def find_create_request(self, request_id: str) -> IdempotencyRecord | None: ...

    def commit(self, session: GameSession, *, expected_revision: int) -> None: ...


class InMemoryGameRepository:
    def __init__(self) -> None:
        self._games: dict[UUID, GameSession] = {}
        self._create_requests: dict[str, tuple[UUID, IdempotencyRecord]] = {}
        self._lock = threading.RLock()

    def create(
        self, session: GameSession, create_record: IdempotencyRecord
    ) -> IdempotencyRecord | None:
        validate_session(session)
        with self._lock:
            existing = self._create_requests.get(create_record.request_id)
            if existing is not None:
                _, record = existing
                if record.request_hash != create_record.request_hash:
                    raise RequestIdConflict("request ID was already used with another body")
                return record
            if session.game_id in self._games:
                raise ConcurrentSessionUpdate("game ID already exists")
            self._games[session.game_id] = session
            self._create_requests[create_record.request_id] = (session.game_id, create_record)
            return None

    def load(self, game_id: UUID) -> GameSession:
        with self._lock:
            try:
                return self._games[game_id]
            except KeyError as error:
                raise SessionNotFound(str(game_id)) from error

    def find_create_request(self, request_id: str) -> IdempotencyRecord | None:
        with self._lock:
            found = self._create_requests.get(request_id)
            return None if found is None else found[1]

    def commit(self, session: GameSession, *, expected_revision: int) -> None:
        validate_session(session)
        with self._lock:
            try:
                current = self._games[session.game_id]
            except KeyError as error:
                raise SessionNotFound(str(session.game_id)) from error
            if current.revision != expected_revision:
                raise ConcurrentSessionUpdate("game session changed during the operation")
            if session.revision != expected_revision + 1:
                raise ValueError("a committed session must advance its storage revision once")
            if session.events[: len(current.events)] != current.events:
                raise ValueError("commits may only append public events")
            if session.idempotency_records[: len(current.idempotency_records)] != current.idempotency_records:
                raise ValueError("commits may only append idempotency records")
            self._games[session.game_id] = session


class SQLiteGameRepository:
    """SQLite repository with a single explicit transaction per mutation."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL CHECK (version >= 0),
                revision INTEGER NOT NULL CHECK (revision >= 0),
                session_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS public_events (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK (sequence > 0),
                event_id TEXT NOT NULL UNIQUE,
                event_json TEXT NOT NULL,
                PRIMARY KEY (game_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS idempotency_records (
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                request_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                response_json TEXT NOT NULL,
                PRIMARY KEY (game_id, request_id)
            );
            CREATE TABLE IF NOT EXISTS create_requests (
                request_id TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
                status_code INTEGER NOT NULL,
                response_json TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteGameRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")

    def _load_unlocked(self, game_id: UUID) -> GameSession:
        row = self._connection.execute(
            "SELECT session_json FROM games WHERE game_id = ?", (str(game_id),)
        ).fetchone()
        if row is None:
            raise SessionNotFound(str(game_id))
        event_rows = self._connection.execute(
            "SELECT event_json FROM public_events WHERE game_id = ? ORDER BY sequence",
            (str(game_id),),
        ).fetchall()
        record_rows = self._connection.execute(
            """
            SELECT request_id, request_hash, status_code, response_json
            FROM idempotency_records WHERE game_id = ? ORDER BY rowid
            """,
            (str(game_id),),
        ).fetchall()
        events = tuple(
            PublicEvent.model_validate_json(event_row["event_json"])
            for event_row in event_rows
        )
        records = tuple(
            IdempotencyRecord(
                request_id=record_row["request_id"],
                request_hash=record_row["request_hash"],
                status_code=record_row["status_code"],
                response_json=record_row["response_json"],
            )
            for record_row in record_rows
        )
        return session_from_core_data(json.loads(row["session_json"]), events, records)

    def create(
        self, session: GameSession, create_record: IdempotencyRecord
    ) -> IdempotencyRecord | None:
        validate_session(session)
        with self._lock:
            try:
                self._begin()
                existing = self._connection.execute(
                    """
                    SELECT request_hash, status_code, response_json
                    FROM create_requests WHERE request_id = ?
                    """,
                    (create_record.request_id,),
                ).fetchone()
                if existing is not None:
                    if existing["request_hash"] != create_record.request_hash:
                        raise RequestIdConflict(
                            "request ID was already used with another body"
                        )
                    self._connection.execute("COMMIT")
                    return IdempotencyRecord(
                        request_id=create_record.request_id,
                        request_hash=existing["request_hash"],
                        status_code=existing["status_code"],
                        response_json=existing["response_json"],
                    )
                core = session_core_data(session)
                self._connection.execute(
                    "INSERT INTO games(game_id, version, revision, session_json) VALUES (?, ?, ?, ?)",
                    (
                        str(session.game_id),
                        session.version,
                        session.revision,
                        json.dumps(core, sort_keys=True, separators=(",", ":")),
                    ),
                )
                self._insert_events(session.events)
                self._insert_records(session.game_id, session.idempotency_records)
                self._connection.execute(
                    """
                    INSERT INTO create_requests
                    (request_id, request_hash, game_id, status_code, response_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        create_record.request_id,
                        create_record.request_hash,
                        str(session.game_id),
                        create_record.status_code,
                        create_record.response_json,
                    ),
                )
                self._connection.execute("COMMIT")
                return None
            except sqlite3.IntegrityError as error:
                self._rollback()
                raise ConcurrentSessionUpdate(str(error)) from error
            except Exception:
                self._rollback()
                raise

    def load(self, game_id: UUID) -> GameSession:
        with self._lock:
            return self._load_unlocked(game_id)

    def find_create_request(self, request_id: str) -> IdempotencyRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT request_hash, status_code, response_json
                FROM create_requests WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            return IdempotencyRecord(
                request_id=request_id,
                request_hash=row["request_hash"],
                status_code=row["status_code"],
                response_json=row["response_json"],
            )

    def _insert_events(self, events: tuple[PublicEvent, ...]) -> None:
        self._connection.executemany(
            """
            INSERT INTO public_events(game_id, sequence, event_id, event_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                (
                    str(event.game_id),
                    event.sequence,
                    event.event_id,
                    event.model_dump_json(),
                )
                for event in events
            ),
        )

    def _insert_records(
        self, game_id: UUID, records: tuple[IdempotencyRecord, ...]
    ) -> None:
        self._connection.executemany(
            """
            INSERT INTO idempotency_records
            (game_id, request_id, request_hash, status_code, response_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    str(game_id),
                    record.request_id,
                    record.request_hash,
                    record.status_code,
                    record.response_json,
                )
                for record in records
            ),
        )

    def commit(self, session: GameSession, *, expected_revision: int) -> None:
        validate_session(session)
        if session.revision != expected_revision + 1:
            raise ValueError("a committed session must advance its storage revision once")
        with self._lock:
            try:
                self._begin()
                current = self._load_unlocked(session.game_id)
                if current.revision != expected_revision:
                    raise ConcurrentSessionUpdate(
                        "game session changed during the operation"
                    )
                if session.events[: len(current.events)] != current.events:
                    raise ValueError("commits may only append public events")
                if session.idempotency_records[: len(current.idempotency_records)] != current.idempotency_records:
                    raise ValueError("commits may only append idempotency records")
                updated = self._connection.execute(
                    """
                    UPDATE games SET version = ?, revision = ?, session_json = ?
                    WHERE game_id = ? AND revision = ?
                    """,
                    (
                        session.version,
                        session.revision,
                        json.dumps(
                            session_core_data(session),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        str(session.game_id),
                        expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise ConcurrentSessionUpdate(
                        "game session changed during the operation"
                    )
                self._insert_events(session.events[len(current.events) :])
                self._insert_records(
                    session.game_id,
                    session.idempotency_records[len(current.idempotency_records) :],
                )
                self._connection.execute("COMMIT")
            except Exception:
                self._rollback()
                raise
