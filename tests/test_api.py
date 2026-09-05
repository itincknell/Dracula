"""Verify shared public API models and application import boundaries.

The tests protect public serialization, strict field handling, and side-effect-
free imports without duplicating complete gameplay transaction coverage.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from dracula.api.app import create_app
from dracula.api.contracts import (
    ApiErrorResponse,
    HealthResponse,
    HumanGameView,
    PublicEvent,
    PublicGameView,
)
from dracula.engine import canonical_state_data, create_game

FIXTURES = Path(__file__).parents[1] / "contracts" / "public"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("health.json", HealthResponse),
        ("human-game-view.json", HumanGameView),
        ("public-event.json", PublicEvent),
        ("api-error.json", ApiErrorResponse),
    ],
)
def test_public_contract_fixtures_validate_in_python(
    filename: str, model: type[HealthResponse | HumanGameView | PublicEvent | ApiErrorResponse]
) -> None:
    model.model_validate(_fixture(filename))


def test_health_returns_the_public_contract_with_narration_disabled() -> None:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(narration_enabled=False))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == _fixture("health.json")


def test_api_import_does_not_load_training_or_aws_dependencies() -> None:
    # The HTTP boundary must remain importable in a lightweight Lambda or local API process.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dracula.api.app; import dracula.api.production; "
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


def test_private_engine_fields_cannot_enter_human_or_public_contracts() -> None:
    engine_data = canonical_state_data(create_game("api-privacy-fixture"))["state"]
    assert isinstance(engine_data, dict)
    payload = _fixture("human-game-view.json")
    payload.update(
        {
            "seed": engine_data["seed"],
            "stock": engine_data["stock"],
            "opponent_hand": engine_data["hands"],
            "policy_hidden_state": [0.0] * 128,
        }
    )

    with pytest.raises(ValidationError):
        HumanGameView.model_validate(payload)

    human_view = HumanGameView.model_validate(_fixture("human-game-view.json"))
    public_payload = TypeAdapter(PublicGameView).dump_python(human_view, mode="json")
    assert "human_hand" not in public_payload
    assert "legal_moves" not in public_payload
    assert not {
        "seed",
        "stock",
        "hands",
        "opponent_hand",
        "policy_hidden_state",
    } & public_payload.keys()
