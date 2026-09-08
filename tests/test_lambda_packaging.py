"""Verify the Lambda build context and static deployment contracts.

Tests protect the pinned policy digest, production import closure, exclusion of
local artifacts, structured logging, infrastructure parameters, and workflows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys

import pytest

from dracula.api.logging import JsonLogFormatter
from dracula.api.config import ProductionSettings


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACT_SHA256 = (
    "d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203"
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
    artifact_digest = builder.build_context(ROOT, output)

    assert artifact_digest == EXPECTED_ARTIFACT_SHA256
    assert hashlib.sha256((output / "artifacts/policy.pt").read_bytes()).hexdigest() == (
        EXPECTED_ARTIFACT_SHA256
    )
    assert not (output / "runs").exists()
    assert not (output / "tests").exists()
    assert not tuple(output.rglob("__pycache__"))
    assert not (output / "src/dracula/policy/training").exists()
    assert (output / "src/dracula/api/production.py").is_file()
    assert (output / "src/dracula/api/stateless/replay.py").is_file()
    assert not (output / "release-manifest.json").exists()
    assert not (output / "pyproject.toml").exists()
    assert (output / "frontend/index.html").is_file()
    assert not (output / "frontend/node_modules").exists()
    assert not (output / "frontend/src").exists()


def test_release_context_fails_closed_on_wrong_artifact(tmp_path: Path) -> None:
    builder = _load_builder()
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not selected policy")
    with pytest.raises(builder.ReleaseBuildError, match="digest differs"):
        builder._verified_artifact_digest(wrong)


def test_release_context_cli_reports_an_absolute_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = _load_builder()
    output = tmp_path / "context"

    assert builder.main(("--output", str(output))) == 0

    result = capsys.readouterr().out.strip()
    assert f"output={output.as_posix()}" in result
    assert f"policy_sha256={EXPECTED_ARTIFACT_SHA256}" in result


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


def test_release_inventory_includes_all_authored_build_inputs() -> None:
    builder = _load_release_builder()
    paths = {path.relative_to(ROOT).as_posix() for path in builder._release_paths(ROOT)}

    assert "frontend/scripts/prepare-card-assets.mjs" in paths
    assert "frontend/scripts/verify-production-build.mjs" in paths
    assert "src/dracula/api/stateless/replay.py" in paths
    assert "tools/lambda_validation_gameplay.py" in paths


def test_container_and_cloudformation_have_one_stateless_production_path() -> None:
    dockerfile = (ROOT / "deployment/container/Dockerfile").read_text()
    template = (ROOT / "infrastructure/application.yaml").read_text()

    assert "aws-lambda-adapter@sha256:" in dockerfile
    assert "DRACULA_POLICY_ARTIFACT=/opt/dracula/artifacts/policy.pt" in dockerfile
    assert "artifacts/policy.pt" in dockerfile
    assert "sagemaker" not in template.lower()
    assert "dynamodb" not in template.lower()
    assert "AWS::ApiGatewayV2::Api" in template
    assert "PayloadFormatVersion: \"2.0\"" in template
    assert "ANY /Dracula/{proxy+}" in template
    assert "ANY /Dracula" in template
    assert "DependsOn: [ApplicationRootRoute, ApplicationRoute, StaticGetRoute]" in template
    assert "bedrock:InvokeModel" in template
    assert "ReservedConcurrentExecutions" in template
    assert 'AWS_LWA_ASYNC_INIT: "true"' in template
    assert "CorsConfiguration" not in template
    assert "AWS::ApiGatewayV2::DomainName" not in template
    assert "DetailedMetricsEnabled: false" in template
    assert "AWS::CloudWatch::Alarm" in template
    assert "MetricName: Errors" in template
    assert "MetricName: 5xx" in template
    assert "AWS::Budgets::Budget" in template
    # CloudFormation rejected numeric Equals operands during stack creation,
    # even though validate-template accepted them. Compare parameter strings.
    assert re.search(r"!Equals \[[^\]]*, -?\d+\]", template) is None


def test_production_settings_have_one_neutral_policy_artifact_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DRACULA_POLICY_ARTIFACT", "/opt/dracula/artifacts/policy.pt")
    monkeypatch.setenv("DRACULA_NARRATION_ENABLED", "true")
    monkeypatch.setenv("DRACULA_REPLAY_CACHE_ENTRIES", "17")

    assert ProductionSettings.from_environment() == ProductionSettings(
        policy_artifact="/opt/dracula/artifacts/policy.pt",
        narration_enabled=True,
        replay_cache_entries=17,
    )


def test_environment_parameter_files_contain_selected_model_but_no_account_identity() -> None:
    for environment in ("staging", "production"):
        path = ROOT / f"infrastructure/parameters/{environment}.json"
        values = {
            item["ParameterKey"]: item["ParameterValue"]
            for item in json.loads(path.read_text())
        }
        assert values["EnvironmentName"] == environment
        assert values["BedrockModelId"] == "amazon.nova-lite-v1:0"
        # AWS foundation-model ARNs have an empty account field and are safe
        # release configuration, unlike an operator's account-owned ARN.
        expected_arn = (
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"
            if environment == "production" else ""
        )
        assert values["BedrockModelArn"] == expected_arn
        assert "CustomDomainCertificateArn" not in values
        assert values["AlarmNotificationEmail"] == ""
        assert re.search(r"arn:aws[^:]*:[^:]*:[^:]*:[0-9]{12}:", path.read_text()) is None
        if environment == "production":
            assert values["NarrationEnabled"] == "true"
            assert values["ReservedConcurrency"] == "-1"
            assert values["MonthlyBudgetUsd"] == "10"
            assert values["BudgetNotificationEmail"] == ""


def test_production_log_formatter_emits_one_json_object() -> None:
    record = logging.LogRecord(
        "dracula.test", logging.INFO, __file__, 1, "message %s", ("value",), None
    )
    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "dracula.test"
    assert payload["message"] == "message value"
    assert payload["timestamp"].endswith("+00:00")
