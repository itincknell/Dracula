"""Provide strict serialization and privacy checks for policy-training data.

Corpus migration and training both consume sealed JSON and packed Boolean
fields. This module owns that shared byte-level contract so neither layer must
import the other. Atomic checkpoint persistence remains with the trainer because
it handles mutable optimization state rather than corpus input.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from dracula.bgc_policy_training_contracts import BGCPolicyTrainingError

FORBIDDEN_DATA_KEYS = frozenset(
    {
        "action_values",
        "authoritative_state",
        "critic",
        "determinization",
        "determinizations",
        "engine_seed",
        "game_seed",
        "hidden_state",
        "model_data",
        "opponent_hand",
        "outer_sampled_state",
        "policy_hidden_state",
        "ppo",
        "return",
        "returns",
        "round_return",
        "score_estimate",
        "search_tree",
        "stock",
        "stock_order",
        "tree",
        "value",
        "values",
    }
)


def canonical_json(value: object) -> bytes:
    """Serialize one finite JSON value with stable keys and separators."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BGCPolicyTrainingError("artifact is not canonical JSON") from error


def json_digest(value: object) -> str:
    """Return the SHA-256 identity of canonical JSON content."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Hash one artifact without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    """Read one JSON artifact and normalize decoding failures."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BGCPolicyTrainingError(
            f"JSON artifact could not be read: {path}"
        ) from error


def assert_no_forbidden_keys(value: object) -> None:
    """Reject private engine, search, and disallowed training fields recursively."""

    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_DATA_KEYS.intersection(value)
        if forbidden:
            raise BGCPolicyTrainingError(
                "training data contains forbidden fields: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            assert_no_forbidden_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_no_forbidden_keys(nested)


def decode_packed(
    value: object,
    *,
    byte_count: int,
    bit_count: int,
    label: str,
) -> bytes:
    """Decode canonical base64 and verify its length and unused padding bits."""

    if not isinstance(value, str):
        raise BGCPolicyTrainingError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise BGCPolicyTrainingError(f"{label} is not canonical base64") from error
    if len(decoded) != byte_count:
        raise BGCPolicyTrainingError(f"{label} has the wrong byte count")
    unused = byte_count * 8 - bit_count
    if unused and decoded[-1] & ((1 << unused) - 1):
        raise BGCPolicyTrainingError(f"{label} has nonzero padding bits")
    return decoded


def unpack_bytes(value: bytes, bit_count: int) -> tuple[bool, ...]:
    """Expand most-significant-bit-first packed bytes into Boolean values."""

    return tuple(
        bool(value[index // 8] & (1 << (7 - index % 8)))
        for index in range(bit_count)
    )


__all__ = (
    "FORBIDDEN_DATA_KEYS",
    "assert_no_forbidden_keys",
    "canonical_json",
    "decode_packed",
    "file_digest",
    "json_digest",
    "load_json",
    "unpack_bytes",
)
