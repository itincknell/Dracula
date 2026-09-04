"""Build an isolated Lambda image context containing the exact selected pi1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PI1_RELATIVE_PATH = Path(
    "runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt"
)
PI1_SHA256 = "70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c"
CONTEXT_SCHEMA = "dracula-lambda-build-context-v1"


class ReleaseBuildError(ValueError):
    """The release context cannot be constructed without ambiguity."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseBuildError("release build requires a Git checkout") from error


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ReleaseBuildError(f"required source directory is absent: {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            ".pytest_cache",
            "*.egg-info",
            "*.dist-info",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
        ),
    )


def _manifest_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "release-manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _digest(path),
                "size": path.stat().st_size,
            }
        )
    return files


def build_context(root: Path, artifact: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    artifact = artifact.resolve()
    output = output.resolve()
    if not artifact.is_file():
        raise ReleaseBuildError(f"selected pi1 artifact is absent: {artifact}")
    actual_digest = _digest(artifact)
    if actual_digest != PI1_SHA256:
        raise ReleaseBuildError(
            "selected pi1 artifact digest differs: "
            f"expected {PI1_SHA256}, received {actual_digest}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    previous = output.with_name(f".{output.name}.previous-{os.getpid()}")
    try:
        container = root / "deployment" / "container"
        for name in (
            "Dockerfile",
            "logging.json",
            "requirements-constraints.txt",
            "requirements-lambda.txt",
            "requirements-torch.txt",
        ):
            source = container / name
            if not source.is_file():
                raise ReleaseBuildError(f"required container file is absent: {source}")
            shutil.copy2(source, temporary / name)
        shutil.copy2(root / "pyproject.toml", temporary / "pyproject.toml")
        _copy_tree(root / "src", temporary / "src")
        (temporary / "artifacts").mkdir()
        shutil.copy2(artifact, temporary / "artifacts" / "pi1.pt")

        manifest: dict[str, object] = {
            "artifact_sha256": actual_digest,
            "context_schema": CONTEXT_SCHEMA,
            "files": _manifest_files(temporary),
            "git_revision": _git_revision(root),
        }
        encoded = json.dumps(
            manifest,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"
        (temporary / "release-manifest.json").write_text(encoded)

        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink(missing_ok=True)
        if output.exists():
            output.rename(previous)
        temporary.rename(output)
        shutil.rmtree(previous, ignore_errors=True)
        return manifest
    except Exception:
        if previous.exists() and not output.exists():
            previous.rename(output)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the ignored, digest-bound Lambda Docker build context."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=PI1_RELATIVE_PATH,
        help=f"selected pi1 artifact (default: {PI1_RELATIVE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/lambda-context"),
        help="generated Docker context (default: build/lambda-context)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    artifact = arguments.artifact
    output = arguments.output
    if not artifact.is_absolute():
        artifact = root / artifact
    if not output.is_absolute():
        output = root / output
    manifest = build_context(root, artifact, output)
    print(
        json.dumps(
            {
                "artifact_sha256": manifest["artifact_sha256"],
                "file_count": len(manifest["files"]),
                "git_revision": manifest["git_revision"],
                "output": str(output.relative_to(root)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
