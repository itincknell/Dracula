"""Seal a deterministic manifest for the locally validated production candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from build_lambda_context import PI1_RELATIVE_PATH, PI1_SHA256


RELEASE_SCHEMA = "dracula-production-release-candidate-v1"
STAGE_REPORTS = (
    "reports/active/stateless-gameplay-api.md",
    "reports/active/bedrock-narration-implementation.md",
    "reports/active/lambda-packaging-and-infrastructure.md",
    "reports/active/github-pages-frontend-integration.md",
    "reports/active/staging-deployment-validation.md",
)
USER_DECISIONS = (
    "Lambda memory, timeout, reserved concurrency, and budget alarms.",
    "Final production go-live authorization.",
)

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
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .encode("utf-8")
    )


def _run(root: Path, *arguments: str) -> str:
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
    roots = (
        root / "src",
        root / "deployment",
        root / "infrastructure",
        root / "frontend" / "src",
        root / "frontend" / "public",
        root / "frontend" / "e2e",
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
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in paths
    ]


def _tree_inventory(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise ReleaseCandidateError(f"required generated directory is absent: {root}")
    files = _file_inventory(root, sorted(path for path in root.rglob("*") if path.is_file()))
    return {
        "file_count": len(files),
        "files": files,
        "tree_sha256": _canonical_digest(files),
    }


def _docker_image(root: Path, image: str) -> dict[str, object]:
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
    return {
        "architecture": architecture,
        "local_image_id": value["Id"],
        "registry_manifest_digest": None,
        "registry_manifest_status": "pending immutable ECR publication",
        "size": value["Size"],
        "tag": image,
    }


def build_manifest(root: Path, image: str) -> dict[str, object]:
    root = root.resolve()
    model = root / PI1_RELATIVE_PATH
    if not model.is_file():
        raise ReleaseCandidateError(f"selected pi1 artifact is absent: {model}")
    model_digest = _sha256_file(model)
    if model_digest != PI1_SHA256:
        raise ReleaseCandidateError(
            f"selected pi1 artifact differs: expected {PI1_SHA256}, got {model_digest}"
        )

    source_files = _file_inventory(root, _release_paths(root))
    report_files = _file_inventory(root, [root / path for path in STAGE_REPORTS])
    lock_paths = (
        root / "frontend" / "package-lock.json",
        root / "deployment" / "container" / "requirements-constraints.txt",
        root / "deployment" / "container" / "requirements-lambda.txt",
        root / "deployment" / "container" / "requirements-torch.txt",
        root / "pyproject.toml",
    )
    lock_files = _file_inventory(root, lock_paths)
    infrastructure_paths = sorted(
        path for path in (root / "infrastructure").rglob("*") if path.is_file()
    )
    infrastructure_files = _file_inventory(root, infrastructure_paths)

    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=root
    )
    untracked = _run(
        root, "git", "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    release_names = {item["path"] for item in source_files}
    untracked_release = sorted(name for name in untracked if name in release_names)

    context = _tree_inventory(root / "build" / "lambda-context")
    frontend = _tree_inventory(root / "frontend" / "dist")
    manifest: dict[str, object] = {
        "bedrock": {
            "model_id": None,
            "status": "unresolved; narration remains disabled",
        },
        "dependencies": {
            "files": lock_files,
            "tree_sha256": _canonical_digest(lock_files),
        },
        "frontend_build": frontend,
        "git": {
            "dirty": bool(_run(root, "git", "status", "--porcelain")),
            "head": _run(root, "git", "rev-parse", "HEAD"),
            "tracked_diff_sha256": _sha256_bytes(diff),
            "untracked_release_files": untracked_release,
        },
        "infrastructure": {
            "files": infrastructure_files,
            "tree_sha256": _canonical_digest(infrastructure_files),
        },
        "lambda_build_context": context,
        "lambda_container": _docker_image(root, image),
        "pi1": {
            "path": PI1_RELATIVE_PATH.as_posix(),
            "sha256": model_digest,
        },
        "release_schema": RELEASE_SCHEMA,
        "source": {
            "file_count": len(source_files),
            "files": source_files,
            "tree_sha256": _canonical_digest(source_files),
        },
        "staging_evidence": {
            "hosted": False,
            "limitation": "AWS credentials expired; validation is local staging-equivalent",
            "reports": report_files,
            "tree_sha256": _canonical_digest(report_files),
        },
        "user_decisions": list(USER_DECISIONS),
    }
    manifest["candidate_sha256"] = _canonical_digest(manifest)
    return manifest


def write_manifest(output: Path, manifest: dict[str, object]) -> None:
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
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
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
