"""Release-context, logging, and infrastructure contract checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys

import pytest

from dracula.production_logging import JsonLogFormatter
from dracula.api.production_config import ProductionSettings


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt"
EXPECTED_ARTIFACT_SHA256 = (
    "70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c"
)


def _load_builder():
    path = ROOT / "tools/build_lambda_context.py"
    spec = importlib.util.spec_from_file_location("build_lambda_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_release_builder():
    path = ROOT / "tools/build_release_candidate.py"
    spec = importlib.util.spec_from_file_location("build_release_candidate", path)
    assert spec is not None and spec.loader is not None
    tools_path = str(path.parent)
    sys.path.insert(0, tools_path)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(tools_path)
    return module


def test_release_context_contains_only_verified_selected_artifact(tmp_path: Path) -> None:
    builder = _load_builder()
    output = tmp_path / "context"
    manifest = builder.build_context(ROOT, ARTIFACT, output)

    assert manifest["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256
    assert hashlib.sha256((output / "artifacts/pi1.pt").read_bytes()).hexdigest() == (
        EXPECTED_ARTIFACT_SHA256
    )
    assert not (output / "runs").exists()
    assert not (output / "tests").exists()
    assert not tuple(output.rglob("*.sqlite3"))
    assert not tuple(output.rglob("__pycache__"))
    assert json.loads((output / "release-manifest.json").read_text()) == manifest


def test_release_context_fails_closed_on_wrong_artifact(tmp_path: Path) -> None:
    builder = _load_builder()
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not pi1")
    with pytest.raises(builder.ReleaseBuildError, match="digest differs"):
        builder.build_context(ROOT, wrong, tmp_path / "context")


def test_release_context_tree_copy_ignores_generated_python_metadata(
    tmp_path: Path,
) -> None:
    builder = _load_builder()
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/module.pyc").write_bytes(b"generated")
    (source / "package.egg-info").mkdir()
    (source / "package.egg-info/PKG-INFO").write_text(
        "generated\n", encoding="utf-8"
    )

    destination = tmp_path / "destination"
    builder._copy_tree(source, destination)

    assert (destination / "module.py").is_file()
    assert not (destination / "__pycache__").exists()
    assert not (destination / "package.egg-info").exists()


def test_release_source_filter_rejects_generated_python_metadata(
    tmp_path: Path,
) -> None:
    builder = _load_release_builder()
    authored = tmp_path / "src/dracula/module.py"
    authored.parent.mkdir(parents=True)
    authored.write_text("VALUE = 1\n", encoding="utf-8")
    pycache = authored.parent / "__pycache__/module.cpython-312.pyc"
    pycache.parent.mkdir()
    pycache.write_bytes(b"generated")
    egg_info = tmp_path / "src/dracula_game.egg-info/PKG-INFO"
    egg_info.parent.mkdir()
    egg_info.write_text("generated\n", encoding="utf-8")

    assert builder._is_release_source(authored)
    assert not builder._is_release_source(pycache)
    assert not builder._is_release_source(egg_info)
    assert builder._display_path(authored, tmp_path) == "src/dracula/module.py"
    outside = Path("/tmp/dracula-release.json")
    assert builder._display_path(outside, tmp_path) == outside.as_posix()


def test_container_and_cloudformation_have_one_stateless_production_path() -> None:
    dockerfile = (ROOT / "deployment/container/Dockerfile").read_text()
    template = (ROOT / "infrastructure/application.yaml").read_text()

    assert "aws-lambda-adapter@sha256:" in dockerfile
    assert "DRACULA_POLICY_ARTIFACT=/opt/dracula/artifacts/pi1.pt" in dockerfile
    assert "DRACULA_GAMEPLAY_MODE" not in dockerfile
    assert "DRACULA_OPPONENT_MODE" not in dockerfile
    assert "DRACULA_BGC_PI0_ARTIFACT=/opt/dracula/artifacts/pi1.pt" not in dockerfile
    assert "artifacts/pi1.pt" in dockerfile
    assert "sqlite" not in dockerfile.lower()
    assert "sagemaker" not in template.lower()
    assert "dynamodb" not in template.lower()
    assert "DRACULA_GAMEPLAY_MODE" not in template
    assert "DRACULA_OPPONENT_MODE" not in template
    assert "AWS::ApiGatewayV2::Api" in template
    assert "PayloadFormatVersion: \"2.0\"" in template
    assert "EndpointType: REGIONAL" in template
    assert "bedrock:InvokeModel" in template
    assert "ReservedConcurrentExecutions" in template
    assert "CorsConfiguration" in template
    assert "AllowOrigins: [!Ref AllowedFrontendOrigin]" in template
    assert "DetailedMetricsEnabled: false" in template
    assert "AWS::CloudWatch::Alarm" in template
    assert "MetricName: Errors" in template
    assert "MetricName: 5xx" in template
    assert "AWS::Budgets::Budget" in template


def test_production_settings_have_one_neutral_policy_artifact_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DRACULA_POLICY_ARTIFACT", "/opt/dracula/artifacts/pi1.pt")
    monkeypatch.setenv("DRACULA_NARRATION_ENABLED", "true")
    monkeypatch.setenv("DRACULA_REPLAY_CACHE_ENTRIES", "17")

    assert ProductionSettings.from_environment() == ProductionSettings(
        policy_artifact="/opt/dracula/artifacts/pi1.pt",
        narration_enabled=True,
        replay_cache_entries=17,
    )


def test_environment_parameter_files_contain_no_account_or_model_identity() -> None:
    for environment in ("staging", "production"):
        path = ROOT / f"infrastructure/parameters/{environment}.json"
        values = {
            item["ParameterKey"]: item["ParameterValue"]
            for item in json.loads(path.read_text())
        }
        assert values["EnvironmentName"] == environment
        assert values["BedrockModelId"] == ""
        assert values["BedrockModelArn"] == ""
        assert values["CustomDomainCertificateArn"] == ""
        assert values["AlarmNotificationEmail"] == ""
        assert "arn:aws" not in path.read_text()


def test_production_log_formatter_emits_one_json_object() -> None:
    record = logging.LogRecord(
        "dracula.test", logging.INFO, __file__, 1, "message %s", ("value",), None
    )
    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "dracula.test"
    assert payload["message"] == "message value"
    assert payload["timestamp"].endswith("+00:00")
