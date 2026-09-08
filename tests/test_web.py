"""Protect the combined application's file, API, caching, and privacy boundary."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from dracula.api.web import create_web_app


@pytest.fixture
def web(tmp_path: Path) -> TestClient:
    frontend = tmp_path / "frontend"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text("<html>Dracula</html>")
    (frontend / "assets/index-AbCd1234.js").write_text("export const game = true;")
    (frontend / "portrait.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "policy.pt").write_bytes(b"private model")
    (frontend / "escape.pt").symlink_to(tmp_path / "policy.pt")
    api = FastAPI()

    @api.post("/games")
    def game(body: dict):
        return body

    return TestClient(create_web_app(api, frontend))


def test_frontend_content_types_and_cache_headers(web: TestClient) -> None:
    index = web.get("/Dracula/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    assert index.headers["cache-control"] == "no-cache"
    script = web.get("/Dracula/assets/index-AbCd1234.js")
    assert "javascript" in script.headers["content-type"]
    assert script.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert web.head("/Dracula/portrait.png").headers["content-type"] == "image/png"
    redirect = web.get("/Dracula", follow_redirects=False)
    assert redirect.status_code == 308
    assert redirect.headers["location"] == "/Dracula/"


def test_api_has_priority_and_is_never_cached(web: TestClient) -> None:
    result = web.post("/Dracula/api/games", json={"seed": "visible"})
    assert result.json() == {"seed": "visible"}
    assert result.headers["cache-control"] == "no-store"
    missing = web.get("/Dracula/api/missing")
    assert missing.status_code == 404
    assert missing.headers["content-type"] == "application/json"
    assert missing.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", [
    "/Dracula/missing.js", "/Dracula/policy.pt", "/Dracula/escape.pt",
    "/Dracula/%2e%2e/policy.pt", "/Dracula/src/dracula/api/production.py",
    "/Dracula/game", "/projects/",
])
def test_files_cannot_escape_distribution_or_fall_back_to_html(web: TestClient, path: str) -> None:
    response = web.get(path)
    assert response.status_code == 404
    assert response.headers["cache-control"] == "no-store"
    assert "private model" not in response.text


def test_missing_frontend_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frontend build is missing"):
        create_web_app(FastAPI(), tmp_path)
