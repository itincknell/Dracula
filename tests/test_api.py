"""Verify the stateless development application and import boundaries."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import httpx

from dracula.api.development import create_app


def test_development_health_reports_unconfigured_dependencies() -> None:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(narration_enabled=False))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "opponent_configured": False,
        "narration_enabled": False,
    }


def test_api_import_does_not_load_training_or_aws_dependencies(tmp_path, monkeypatch) -> None:
    (tmp_path / "index.html").write_text("<html>test</html>")
    monkeypatch.setenv("DRACULA_FRONTEND_DIR", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dracula.api.development; import dracula.api.production; "
                "forbidden=('torch','boto3','botocore','sagemaker'); "
                "loaded={name for name in sys.modules if name.split('.')[0] in forbidden}; "
                "assert not loaded, loaded"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
