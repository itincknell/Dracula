"""Build an isolated Lambda container context for the selected π1 release.

The tool verifies the pinned local artifact, computes the production import
closure, and copies only required source and runtime files into generated output.
"""

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
    "runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt"
)
PI1_SHA256 = "d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203"
RUNTIME_MODULES = (
    "__init__.py",
    "action_contract.py",
    "active_policy.py",
    "bgc_policy.py",
    "bgc_policy_model.py",
    "bridge.py",
    "cards.py",
    "engine.py",
    "engine_dealing.py",
    "engine_serialization.py",
    "engine_types.py",
    "engine_validation.py",
    "policy_observation.py",
    "production_logging.py",
    "randomness.py",
    "scoring.py",
    "strategic_actions.py",
)
RUNTIME_API_MODULES = (
    "__init__.py",
    "bedrock.py",
    "contracts.py",
    "narration.py",
    "policy.py",
    "presentation.py",
    "production.py",
    "production_config.py",
    "stateless_app.py",
    "stateless_contracts.py",
    "stateless_http.py",
    "stateless_projection.py",
    "stateless_routes.py",
    "stateless_service.py",
)
RUNTIME_SEARCH_MODULES = ("__init__.py", "information.py", "symmetry.py")


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


def _copy_runtime_source(source_root: Path, destination_root: Path) -> None:
    """Copy the production import closure without training or local API code."""

    source_package = source_root / "dracula"
    destination_package = destination_root / "dracula"
    for relative_directory, modules in (
        (Path(), RUNTIME_MODULES),
        (Path("api"), RUNTIME_API_MODULES),
        (Path("search"), RUNTIME_SEARCH_MODULES),
    ):
        destination = destination_package / relative_directory
        destination.mkdir(parents=True, exist_ok=True)
        for module in modules:
            source = source_package / relative_directory / module
            if not source.is_file():
                raise ReleaseBuildError(f"required runtime module is absent: {source}")
            shutil.copy2(source, destination / module)


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
        _copy_runtime_source(root / "src", temporary / "src")
        (temporary / "artifacts").mkdir()
        shutil.copy2(artifact, temporary / "artifacts" / "pi1.pt")

        manifest: dict[str, object] = {
            "artifact_sha256": actual_digest,
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
    try:
        displayed_output = output.relative_to(root).as_posix()
    except ValueError:
        displayed_output = output.as_posix()
    print(
        json.dumps(
            {
                "artifact_sha256": manifest["artifact_sha256"],
                "file_count": len(manifest["files"]),
                "git_revision": manifest["git_revision"],
                "output": displayed_output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
