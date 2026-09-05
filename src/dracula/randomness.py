"""Create small, deterministic random generators from ordinary values.

Gameplay and search use local ``random.Random`` instances so unrelated work and
parallel workers never share mutable random state. SHA-256 is used only to turn
structured values into a stable integer; it is not a schema or security layer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import TypeVar

_T = TypeVar("_T")


def stable_seed(*parts: str | int) -> int:
    """Return a process-independent integer seed for the supplied values."""

    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def deterministic_random(*parts: str | int) -> random.Random:
    """Create an independent local generator reproducible from ``parts``."""

    return random.Random(stable_seed(*parts))


def shuffled(values: Sequence[_T], *seed_parts: str | int) -> tuple[_T, ...]:
    """Return a deterministic permutation without mutating ``values``."""

    result = list(values)
    deterministic_random(*seed_parts).shuffle(result)
    return tuple(result)


__all__ = ("deterministic_random", "shuffled", "stable_seed")
