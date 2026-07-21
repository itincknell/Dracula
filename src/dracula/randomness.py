"""Versioned deterministic seed derivation and random streams."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import TypeVar

RANDOMNESS_SCHEMA_VERSION = "dracula-randomness-v1"
COUNTER_NAMESPACE = "dracula-sha256-counter-v1"
SEED_BYTES = hashlib.sha256().digest_size

_NAMESPACE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*$")
_T = TypeVar("_T")


def derive_seed(namespace: str, *components: str) -> bytes:
    """Return the full SHA-256 digest for canonical NUL-separated fields."""

    _validate_namespace(namespace)
    fields = (namespace, *components)
    encoded: list[bytes] = []
    for position, field in enumerate(fields):
        if not isinstance(field, str):
            raise TypeError(f"seed field {position} must be a string")
        if "\0" in field:
            raise ValueError("seed fields cannot contain NUL")
        encoded.append(field.encode("utf-8"))
    return hashlib.sha256(b"\0".join(encoded)).digest()


def derive_pytorch_seed(namespace: str, *components: str) -> int:
    """Derive the documented unsigned 64-bit seed for a PyTorch generator."""

    return int.from_bytes(derive_seed(namespace, *components)[:8], "big", signed=False)


def seed_hex(seed: bytes) -> str:
    _validate_seed(seed)
    return seed.hex()


def seed_integer(seed: bytes) -> int:
    _validate_seed(seed)
    return int.from_bytes(seed, "big", signed=False)


class Sha256CounterStream:
    """Counter-mode SHA-256 stream used by engine randomness."""

    __slots__ = ("_counter", "_seed")

    def __init__(self, seed: bytes, *, counter: int = 0) -> None:
        _validate_seed(seed)
        if type(counter) is not int or counter < 0:
            raise ValueError("counter must be a non-negative integer")
        self._seed = seed
        self._counter = counter

    @property
    def counter(self) -> int:
        return self._counter

    def next_block(self) -> bytes:
        block = derive_seed(COUNTER_NAMESPACE, self._seed.hex(), str(self._counter))
        self._counter += 1
        return block

    def randbelow(self, upper_bound: int) -> int:
        if type(upper_bound) is not int or upper_bound <= 0:
            raise ValueError("upper_bound must be a positive integer")

        # Rejection preserves equal-size residue classes instead of introducing
        # modulo bias at the upper end of the 256-bit sample space.
        sample_space = 1 << (8 * SEED_BYTES)
        if upper_bound > sample_space:
            raise ValueError("upper_bound cannot exceed the 256-bit sample space")
        limit = sample_space - sample_space % upper_bound
        while True:
            candidate = int.from_bytes(self.next_block(), "big", signed=False)
            if candidate < limit:
                return candidate % upper_bound


def shuffled(values: Sequence[_T], seed: bytes) -> tuple[_T, ...]:
    """Return a Fisher-Yates permutation without mutating the input."""

    result = list(values)
    stream = Sha256CounterStream(seed)
    for index in range(len(result) - 1, 0, -1):
        selected = stream.randbelow(index + 1)
        result[index], result[selected] = result[selected], result[index]
    return tuple(result)


def _validate_namespace(namespace: str) -> None:
    if not isinstance(namespace, str):
        raise TypeError("seed namespace must be a string")
    if "\0" in namespace:
        raise ValueError("seed fields cannot contain NUL")
    if not _NAMESPACE.fullmatch(namespace):
        raise ValueError("seed namespace must end in a positive version such as '-v1'")


def _validate_seed(seed: bytes) -> None:
    if type(seed) is not bytes or len(seed) != SEED_BYTES:
        raise ValueError(f"seed must be exactly {SEED_BYTES} bytes")
