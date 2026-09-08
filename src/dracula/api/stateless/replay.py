"""Hold process-local acceleration state for stateless game reconstruction.

The browser's seed and accepted command history remain authoritative. This
module uses each immutable recovery envelope as the cache key and stores the
complete reconstructed game only as an optimization. Eviction or process
replacement cannot affect gameplay. Cache entries contain private engine state
and never cross the public API boundary.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from dracula.api.stateless.contracts import RecoveryEnvelope
from dracula.game.engine import EnginePlayer, EngineState

DEFAULT_REPLAY_CACHE_ENTRIES = 256


@dataclass(frozen=True, slots=True)
class ReplayedGame:
    """Trusted private state reconstructed from one complete browser envelope."""

    state: EngineState
    human_role: EnginePlayer
    policy_game_key: str


class ReplayCache:
    """Bounded least-recently-used cache of private reconstructed games.

    ``OrderedDict`` preserves access order so the oldest unused entry can be
    removed first. ``RLock`` protects that order when several FastAPI worker
    threads use the same Lambda process concurrently.
    """

    def __init__(self, max_entries: int = DEFAULT_REPLAY_CACHE_ENTRIES) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("replay cache size must be a non-negative integer")
        self.max_entries = max_entries
        self._entries: OrderedDict[RecoveryEnvelope, ReplayedGame] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, envelope: RecoveryEnvelope) -> ReplayedGame | None:
        """Return a cached reconstruction and mark it as recently used."""

        with self._lock:
            value = self._entries.get(envelope)
            if value is None:
                return None
            # Moving a hit to the end marks it as most recently used.
            self._entries.move_to_end(envelope)
            return value

    def put(self, envelope: RecoveryEnvelope, value: ReplayedGame) -> None:
        """Store one reconstruction and evict least-recently-used entries."""

        if self.max_entries == 0:
            return
        with self._lock:
            self._entries[envelope] = value
            self._entries.move_to_end(envelope)
            while len(self._entries) > self.max_entries:
                # ``last=False`` removes the least recently used first entry.
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Discard cached reconstructions without changing accepted game data."""

        with self._lock:
            self._entries.clear()
