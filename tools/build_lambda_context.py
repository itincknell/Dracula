"""Build an isolated Lambda container context for the selected policy.

The policy artifact lives in an ignored training run and must not be committed.
This tool verifies that file against the release digest, discovers the local
Python modules imported by the production entry point, and copies those inputs
into an ignored Docker build directory. Training code, tests, local preview
code, and unrelated repository files are therefore excluded from the image.

The context is disposable build output. A failed build removes its incomplete
output, and the calling Make target will not proceed to Docker.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
from pathlib import Path

# Local source and exact file identity of the model selected for deployment.
SELECTED_POLICY_RELATIVE_PATH = Path(
    "runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt"
)
SELECTED_POLICY_SHA256 = (
    "d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203"
)
RUNTIME_ENTRY_MODULES = (
    "dracula.api.production",
    # This module is referenced by logging.json rather than a Python import.
    "dracula.api.logging",
)
CONTAINER_FILES = (
    "Dockerfile",
    "logging.json",
    "requirements-constraints.txt",
    "requirements-lambda.txt",
    "requirements-torch.txt",
)


class ReleaseBuildError(ValueError):
    """The release context cannot be constructed without ambiguity."""


def _digest(path: Path) -> str:
    """Return the SHA-256 identity of one release input file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_source(source_root: Path, module: str) -> Path | None:
    """Resolve a local dotted module name to its module or package file."""

    relative = Path(*module.split("."))
    module_file = source_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = source_root / relative / "__init__.py"
    return package_file if package_file.is_file() else None


def _local_imports(source_root: Path, source: Path) -> set[str]:
    """Find direct imports that belong to the local ``dracula`` package."""

    imports: set[str] = set()
    # ast.parse turns source text into a tree of Python statements without
    # executing it. ast.walk visits that tree, including imports inside functions.
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            # "import dracula.game.engine" stores module names in node.names.
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            # "from dracula.game.engine import ..." names the module separately
            # from the functions or classes imported from it.
            names = (node.module,)
        else:
            continue
        imports.update(
            name for name in names if name == "dracula" or name.startswith("dracula.")
        )
    return {name for name in imports if _module_source(source_root, name) is not None}


def _package_initializers(source_root: Path, source: Path) -> set[Path]:
    """Include package initializers required to import one copied module."""

    paths: set[Path] = set()
    parent = source.parent
    # Importing a leaf module also executes every parent package initializer.
    while parent != source_root:
        initializer = parent / "__init__.py"
        if initializer.is_file():
            paths.add(initializer)
        parent = parent.parent
    return paths


def _runtime_source_files(source_root: Path) -> list[Path]:
    """Resolve every local module reachable from the production entry points.

    The pending list starts with the FastAPI application and JSON log
    formatter. Parsing each module's imports adds its local dependencies until
    no unseen ``dracula`` imports remain.
    """

    pending = list(RUNTIME_ENTRY_MODULES)
    visited: set[str] = set()
    sources: set[Path] = set()
    # ``visited`` makes circular imports safe and prevents duplicate work.
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        source = _module_source(source_root, module)
        if source is None:
            raise ReleaseBuildError(f"required runtime module is absent: {module}")
        sources.add(source)
        sources.update(_package_initializers(source_root, source))
        pending.extend(sorted(_local_imports(source_root, source) - visited))
    return sorted(sources)


def _copy_runtime_source(source_root: Path, destination_root: Path) -> None:
    """Copy the production import closure without training or local API code."""

    for source in _runtime_source_files(source_root):
        # Preserve the package path expected by the container's PYTHONPATH.
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # copy2 creates an independent file and preserves its metadata, including
        # permissions and modification time; it does not link to mutable source.
        shutil.copy2(source, destination)


def _verified_artifact_digest(artifact: Path) -> str:
    """Reject a missing or substituted policy before copying any release input."""

    if not artifact.is_file():
        raise ReleaseBuildError(f"selected policy artifact is absent: {artifact}")
    actual_digest = _digest(artifact)
    # The hash, rather than the filename, identifies the selected weights.
    if actual_digest != SELECTED_POLICY_SHA256:
        raise ReleaseBuildError(
            "selected policy artifact digest differs: "
            f"expected {SELECTED_POLICY_SHA256}, received {actual_digest}"
        )
    return actual_digest


def _populate_context(root: Path, artifact: Path, output: Path) -> None:
    """Copy the container inputs, runtime source, and verified policy."""

    container = root / "deployment" / "container"
    for name in CONTAINER_FILES:
        source = container / name
        if not source.is_file():
            raise ReleaseBuildError(f"required container file is absent: {source}")
        shutil.copy2(source, output / name)
    _copy_runtime_source(root / "src", output / "src")
    frontend = root / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        raise ReleaseBuildError("frontend distribution is absent; run make web-build")
    # Copy only the built site, never frontend source, dependencies, or local files.
    shutil.copytree(frontend, output / "frontend")
    (output / "artifacts").mkdir()
    shutil.copy2(artifact, output / "artifacts" / "policy.pt")


def build_context(root: Path, output: Path) -> str:
    """Recreate the Docker context and return the verified policy digest."""

    root = root.resolve()
    output = output.resolve()
    artifact = root / SELECTED_POLICY_RELATIVE_PATH

    # A wrong model fails before an existing build context is removed.
    actual_digest = _verified_artifact_digest(artifact)

    # Rebuild from empty output so deleted source files cannot linger.
    if output.is_dir():
        # rmtree deletes the directory and all files and subdirectories inside it.
        shutil.rmtree(output)
    else:
        output.unlink(missing_ok=True)
    output.mkdir(parents=True)
    try:
        _populate_context(root, artifact, output)
    except Exception:
        # Never leave an incomplete directory that could be passed to Docker.
        # ignore_errors keeps cleanup failures from hiding the original build error.
        shutil.rmtree(output, ignore_errors=True)
        raise
    return actual_digest


def _parser() -> argparse.ArgumentParser:
    """Define the ignored output path accepted by the build tool."""

    parser = argparse.ArgumentParser(
        description="Create the ignored, digest-bound Lambda Docker build context."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/lambda-context"),
        help="generated Docker context (default: build/lambda-context)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Resolve repository-relative arguments, build the context, and summarize it."""

    arguments = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    artifact_digest = build_context(root, output)
    try:
        displayed_output = output.relative_to(root).as_posix()
    except ValueError:
        displayed_output = output.as_posix()
    print(f"policy_sha256={artifact_digest} output={displayed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
