"""Git and authored-source identity used by reproducible training artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SOURCE_TREE_SCHEMA_VERSION = "dracula-source-tree-v1"


class SourceIdentityError(RuntimeError):
    """The current authored source cannot be identified reproducibly."""


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    revision: str
    tree_digest: str


def resolve_source_identity() -> SourceIdentity:
    """Hash the Git revision and authored Python, contract, and project files."""

    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceIdentityError("source identity requires a Git revision") from error
    if len(revision) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise SourceIdentityError("Git returned an invalid revision")
    paths = [repository / "NORTHSTARS", repository / "pyproject.toml"]
    paths.extend(sorted((repository / "src").rglob("*.py")))
    paths.extend(sorted((repository / "docs").rglob("*.md")))
    files = [
        {
            "path": path.relative_to(repository).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    encoded = json.dumps(
        {"schema_version": SOURCE_TREE_SCHEMA_VERSION, "files": files},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SourceIdentity(revision, hashlib.sha256(encoded).hexdigest())


__all__ = (
    "SOURCE_TREE_SCHEMA_VERSION",
    "SourceIdentity",
    "SourceIdentityError",
    "resolve_source_identity",
)
