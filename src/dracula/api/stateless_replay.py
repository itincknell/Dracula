"""Hold process-local acceleration state for stateless game reconstruction.

The browser's seed and accepted command history remain authoritative. This
module derives stable cache keys and stores complete reconstructed games only as
an optimization; eviction or process replacement cannot affect gameplay. Cache
entries contain private engine state and never cross the public API boundary.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from uuid import UUID

from dracula.api.stateless_contracts import RecoveryEnvelope
from dracula.engine import EnginePlayer, EngineState

DEFAULT_REPLAY_CACHE_ENTRIES = 256


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
    """Bounded least-recently-used cache of private reconstructed games."""

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


__all__ = (
    "DEFAULT_REPLAY_CACHE_ENTRIES",
    "ReplayCache",
    "ReplayCacheStatistics",
    "ReplayedGame",
    "canonical_envelope_digest",
)
