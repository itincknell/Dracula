"""Assemble the deterministic identity of a production release candidate.

The manifest records the exact policy, authored source, frontend distribution,
Lambda build context, local container image, infrastructure, dependency locks,
and validation reports that would be promoted together. It is evidence for a
later release decision; running this tool does not publish or deploy anything.

Generated caches are excluded from the source inventory. The intentionally
ignored policy and build products are inventoried through their own sections so
they cannot silently enter or disappear from a release candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from build_lambda_context import (
    SELECTED_POLICY_RELATIVE_PATH,
    SELECTED_POLICY_SHA256,
)


RELEASE_EVIDENCE = (
    "docs/stateless-api.md",
    "docs/narrator.md",
    "docs/frontend-experience.md",
    "docs/deployment.md",
)
USER_DECISIONS = (
    "Lambda memory, timeout, reserved concurrency, and budget alarms.",
    "Final production go-live authorization.",
)
AWS_REGION = "us-east-1"

_GENERATED_SOURCE_PARTS = {"__pycache__", ".pytest_cache"}


def _is_release_source(path: Path) -> bool:
    """Return whether *path* is authored release input rather than build residue."""

    return (
        path.is_file()
        and not any(part in _GENERATED_SOURCE_PARTS for part in path.parts)
        and not any(part.endswith((".egg-info", ".dist-info")) for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    )


class ReleaseCandidateError(ValueError):
    """The release candidate cannot be identified reproducibly."""


def _sha256_bytes(payload: bytes) -> str:
    """Hash an in-memory release identity such as a Git diff."""

    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash one release file without reading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    """Hash structured inventory data with stable key and separator choices."""

    return _sha256_bytes(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .encode("utf-8")
    )


def _run(root: Path, *arguments: str) -> str:
    """Run one required local release-inspection command."""

    try:
        return subprocess.check_output(
            arguments,
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ReleaseCandidateError(
            f"command failed: {' '.join(arguments)}: {detail.strip()}"
        ) from error


def _release_paths(root: Path) -> list[Path]:
    """Enumerate authored runtime, frontend, deployment, and release-tool inputs."""

    # Directory roots capture additions automatically; individual files cover
    # authored release inputs that sit beside generated or unrelated content.
    roots = (
        root / "src",
        root / "deployment",
        root / "infrastructure",
        root / "frontend" / "src",
        root / "frontend" / "public",
        root / "frontend" / "e2e",
        root / "frontend" / "scripts",
        root / ".github" / "workflows",
    )
    individual = (
        root / "Makefile",
        root / "pyproject.toml",
        root / "frontend" / "index.html",
        root / "frontend" / "package.json",
        root / "frontend" / "package-lock.json",
        root / "frontend" / "playwright.config.ts",
        root / "frontend" / "tsconfig.json",
        root / "frontend" / "tsconfig.app.json",
        root / "frontend" / "tsconfig.node.json",
        root / "frontend" / "vite.config.ts",
        root / "tools" / "build_lambda_context.py",
        root / "tools" / "build_release_candidate.py",
        root / "tools" / "lambda_validation_gameplay.py",
        root / "tools" / "validate_lambda_container.py",
    )
    paths: set[Path] = set()
    for directory in roots:
        if not directory.is_dir():
            raise ReleaseCandidateError(f"required release directory is absent: {directory}")
        paths.update(path for path in directory.rglob("*") if _is_release_source(path))
    for path in individual:
        if path.is_file():
            paths.add(path)
        elif path.name not in {"tsconfig.app.json", "tsconfig.node.json"}:
            raise ReleaseCandidateError(f"required release file is absent: {path}")
    return sorted(paths)


def _file_inventory(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    """Record repository-relative path, bytes, and digest for each input."""

    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in paths
    ]


def _tree_inventory(root: Path) -> dict[str, object]:
    """Describe a required generated directory and hash its ordered inventory."""

    if not root.is_dir():
        raise ReleaseCandidateError(f"required generated directory is absent: {root}")
    files = _file_inventory(root, sorted(path for path in root.rglob("*") if path.is_file()))
    return {
        "file_count": len(files),
        "files": files,
        "tree_sha256": _canonical_digest(files),
    }


def _docker_image(root: Path, image: str) -> dict[str, object]:
    """Record the ARM64 image and its registry digest when already published."""

    # A local image ID identifies this workstation build. The registry digest
    # remains unset until the same image is pushed to ECR.
    raw = _run(
        root,
        "docker",
        "image",
        "inspect",
        image,
        "--format",
        "{{json .}}",
    )
    value = json.loads(raw)
    architecture = value.get("Architecture")
    if architecture != "arm64":
        raise ReleaseCandidateError(
            f"release image must be arm64, received {architecture!r}"
        )
    published = value.get("RepoDigests") or []
    registry_digest = published[0].split("@", 1)[1] if published else None
    return {
        "architecture": architecture,
        "local_image_id": value["Id"],
        "registry_manifest_digest": registry_digest,
        "registry_manifest_status": "published" if published else "pending immutable ECR publication",
        "size": value["Size"],
        "tag": image,
    }


def _verified_model(root: Path) -> tuple[Path, str]:
    """Verify the ignored selected policy before naming it in a release."""

    model = root / SELECTED_POLICY_RELATIVE_PATH
    if not model.is_file():
        raise ReleaseCandidateError(f"selected pi1 artifact is absent: {model}")
    model_digest = _sha256_file(model)
    if model_digest != SELECTED_POLICY_SHA256:
        raise ReleaseCandidateError(
            "selected policy artifact differs: "
            f"expected {SELECTED_POLICY_SHA256}, got {model_digest}"
        )
    return model, model_digest


def _git_identity(root: Path, release_names: set[str]) -> dict[str, object]:
    """Record the base commit and any tracked or release-relevant untracked work."""

    # HEAD alone cannot identify an uncommitted release candidate, so retain a
    # digest of the tracked binary diff and list release-relevant new files.
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=root
    )
    untracked = _run(
        root, "git", "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    return {
        "dirty": bool(_run(root, "git", "status", "--porcelain")),
        "head": _run(root, "git", "rev-parse", "HEAD"),
        "tracked_diff_sha256": _sha256_bytes(diff),
        "untracked_release_files": sorted(
            name for name in untracked if name in release_names
        ),
    }


def _release_inventories(root: Path) -> dict[str, list[dict[str, object]]]:
    """Build separately reviewable source, dependency, infrastructure, and evidence lists."""

    # Keep these inventories separate so reviewers can verify dependency or
    # infrastructure changes without diffing the full source inventory.
    source = _file_inventory(root, _release_paths(root))
    locks = _file_inventory(
        root,
        (
            root / "frontend" / "package-lock.json",
            root / "deployment" / "container" / "requirements-constraints.txt",
            root / "deployment" / "container" / "requirements-lambda.txt",
            root / "deployment" / "container" / "requirements-torch.txt",
            root / "pyproject.toml",
        ),
    )
    infrastructure = _file_inventory(
        root,
        sorted(path for path in (root / "infrastructure").rglob("*") if path.is_file()),
    )
    evidence = _file_inventory(root, [root / path for path in RELEASE_EVIDENCE])
    return {
        "source": source,
        "locks": locks,
        "infrastructure": infrastructure,
        "evidence": evidence,
    }


def _selected_bedrock_model(root: Path) -> str:
    """Read the one selected model identity from both environment inputs."""

    selected: set[str] = set()
    for environment in ("staging", "production"):
        path = root / "infrastructure" / "parameters" / f"{environment}.json"
        values = {
            item["ParameterKey"]: item["ParameterValue"]
            for item in json.loads(path.read_text())
        }
        selected.add(values.get("BedrockModelId", ""))
    if len(selected) != 1 or not next(iter(selected)):
        raise ReleaseCandidateError(
            "staging and production must select one Bedrock model"
        )
    return selected.pop()


def build_manifest(root: Path, image: str) -> dict[str, object]:
    """Construct the complete release identity without writing or deploying it."""

    root = root.resolve()
    _model, model_digest = _verified_model(root)
    files = _release_inventories(root)
    source_files = files["source"]

    release_names = {item["path"] for item in source_files}
    # Each major deployable is identified independently before the entire
    # manifest receives one final digest below.
    manifest: dict[str, object] = {
        "bedrock": {
            "model_id": _selected_bedrock_model(root),
            "region": AWS_REGION,
            "status": "selected; account-bound model ARN is rendered at deployment",
        },
        "dependencies": {
            "files": files["locks"],
            "tree_sha256": _canonical_digest(files["locks"]),
        },
        "frontend_build": _tree_inventory(root / "frontend" / "dist"),
        "git": _git_identity(root, release_names),
        "infrastructure": {
            "files": files["infrastructure"],
            "tree_sha256": _canonical_digest(files["infrastructure"]),
        },
        "lambda_build_context": _tree_inventory(root / "build" / "lambda-context"),
        "lambda_container": _docker_image(root, image),
        "pi1": {
            "path": SELECTED_POLICY_RELATIVE_PATH.as_posix(),
            "sha256": model_digest,
        },
        "source": {
            "file_count": len(source_files),
            "files": source_files,
            "tree_sha256": _canonical_digest(source_files),
        },
        "release_evidence": {
            "files": files["evidence"],
            "tree_sha256": _canonical_digest(files["evidence"]),
        },
        "user_decisions": list(USER_DECISIONS),
    }
    manifest["candidate_sha256"] = _canonical_digest(manifest)
    return manifest


def write_manifest(output: Path, manifest: dict[str, object]) -> None:
    """Atomically replace the ignored release-candidate manifest."""

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    """Define the local image and generated manifest command arguments."""

    parser = argparse.ArgumentParser(
        description="Seal the locally validated Dracula production release candidate."
    )
    parser.add_argument("--image", default="dracula-api:release-candidate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/release-candidate/manifest.json"),
    )
    return parser


def _display_path(path: Path, root: Path) -> str:
    """Prefer concise repository-relative output while supporting other paths."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    """Build, atomically write, and summarize one local release candidate."""

    arguments = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    manifest = build_manifest(root, arguments.image)
    write_manifest(output, manifest)
    print(
        json.dumps(
            {
                "candidate_sha256": manifest["candidate_sha256"],
                "output": _display_path(output, root),
                "source_tree_sha256": manifest["source"]["tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
