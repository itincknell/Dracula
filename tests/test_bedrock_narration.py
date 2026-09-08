"""Verify grounded narration construction and Bedrock failure isolation.

The suite covers all three cue classes, public-fact filtering, Nova payloads,
timeouts, malformed responses, retry behavior, and narrator-disabled play.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType
from typing import Any, Mapping

import pytest
from fastapi.testclient import TestClient

from dracula.api.development import create_app
from dracula.api.narration.bedrock import (
    BedrockRuntimeAdapter,
    NarrationConfigurationError,
)
from dracula.api.narration.cues import (
    GroundedNarrationCue,
    _round_tie_break,
    _winning_combination,
)
from dracula.api.narration.prompt import NarrationPrompt, build_narration_prompt
from dracula.api.narration.service import (
    NarrationProviderError,
    NarrationProviderResult,
    validate_narration_text,
)
from dracula.policy.contracts import PolicyTurnResult
from dracula.api.stateless.contracts import RecoveryEnvelope
from dracula.game.engine import EnginePlayer, LineOrientation, PlayerValues, score_line

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)


class FirstLegalPolicy:
    def invoke(self, request: Any) -> PolicyTurnResult:
        action = next(
            index for index, move in enumerate(request.action_table) if move is not None
        )
        return PolicyTurnResult(action)


class CapturingBedrockClient:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> Mapping[str, object]:
        self.calls.append(kwargs)
        return self.response


class FakeNarrationAdapter:
    """Record prompts and return a fixed result or failure for service tests."""

    def __init__(
        self,
        text: str = "The night has only begun.",
        *,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.requests: list[NarrationPrompt] = []

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        self.requests.append(prompt)
        if self.error is not None:
            raise self.error
        return NarrationProviderResult(
            text=self.text,
            input_tokens=12,
            output_tokens=6,
            latency_ms=0.1,
        )


def _bedrock_response(text: str = "Welcome to my table.") -> dict[str, object]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 19, "outputTokens": 6, "totalTokens": 25},
        "metrics": {"latencyMs": 11},
    }


def _app(
    adapter: FakeNarrationAdapter | None,
    *,
    enabled: bool = True,
) -> Any:
    return create_app(
        policy_executor=FirstLegalPolicy(),
        narration_enabled=enabled,
        narration_adapter=adapter,
    )


def _start(client: TestClient, seed: str = "narration-public-seed") -> dict[str, Any]:
    response = client.post(
        "/games", json={"human_role": "queen", "seed": seed}
    )
    assert response.status_code == 201, response.json()
    return response.json()


def _next_command(response: dict[str, Any]) -> dict[str, Any]:
    game = response["game"]
    if game["phase"]["kind"] == "human_turn":
        legal = game["legal_moves"][0]
        command = {
            "type": "place",
            "hand_slot": legal["hand_slot"],
            "position": legal["position"],
        }
    elif game["phase"]["kind"] == "scoring":
        command = {"type": "advance_round"}
    else:
        raise AssertionError(f"unexpected phase {game['phase']}")
    return {"envelope": response["envelope"], "command": command}


def _advance_to_scoring(
    client: TestClient,
    response: dict[str, Any],
) -> dict[str, Any]:
    while response["game"]["phase"]["kind"] != "scoring":
        mutation = client.post("/games/command", json=_next_command(response))
        assert mutation.status_code == 200, mutation.json()
        response = mutation.json()
    return response


def _narrate(
    client: TestClient,
    response: dict[str, Any],
    cue_type: str,
) -> Any:
    return client.post(
        "/narration",
        json={"envelope": response["envelope"], "cue_type": cue_type},
    )


def test_bedrock_converse_request_and_normalized_response_contract() -> None:
    client = CapturingBedrockClient(_bedrock_response())
    adapter = BedrockRuntimeAdapter(client, model_id="test.model-v1", max_tokens=72)
    prompt = NarrationPrompt("System instructions", '{"public_facts":{}}')

    result = adapter.generate(prompt)

    assert result.text == "Welcome to my table."
    assert result.input_tokens == 19
    assert result.output_tokens == 6
    assert result.latency_ms >= 0
    assert client.calls == [
        {
            "modelId": "test.model-v1",
            "system": [{"text": "System instructions"}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": '{"public_facts":{}}'}],
                }
            ],
            "inferenceConfig": {"maxTokens": 72, "temperature": 0.5},
        }
    ]


@pytest.mark.parametrize(
    "response",
    (
        {},
        {"output": {}},
        {"output": {"message": {"role": "user", "content": [{"text": "x"}]}}},
        {"output": {"message": {"role": "assistant", "content": []}}},
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "x", "toolUse": {}}],
                }
            }
        },
    ),
)
def test_malformed_bedrock_output_is_rejected(response: Mapping[str, object]) -> None:
    adapter = BedrockRuntimeAdapter(
        CapturingBedrockClient(response),
        model_id="test.model-v1",
    )
    with pytest.raises(NarrationProviderError):
        adapter.generate(NarrationPrompt("system", "user"))


@pytest.mark.parametrize("text", ("", "\0unsafe"))
def test_empty_and_control_character_output_is_rejected(text: str) -> None:
    with pytest.raises(NarrationProviderError):
        validate_narration_text(text)


def test_prompt_construction_uses_plain_english_summary() -> None:
    cue = GroundedNarrationCue(
        "round_transition",
        {
            "dracula_attitude": "amused",
            "leader": "dracula",
            "round": 2,
            "round_result": "dracula",
            "score_movement": "Dracula extends the lead over the human",
            "winning_combination": "three Hearts for a 5x multiplier",
        },
    )
    prompt = build_narration_prompt(cue)

    assert prompt.user_text == (
        "Dracula is winning after round 2. Dracula extends the lead over the "
        "human. Dracula put together three Hearts for a 5x multiplier. "
        "Dracula is amused."
    )
    assert "Input:" not in prompt.system_text
    assert "Output:" not in prompt.system_text


def test_examples_are_separate_conversation_turns_before_the_live_cue() -> None:
    client = CapturingBedrockClient(_bedrock_response())
    adapter = BedrockRuntimeAdapter(client, model_id="test.model-v1")
    prompt = build_narration_prompt(
        GroundedNarrationCue(
            "opening",
            {
                "dracula_attitude": "imperious",
                "first_player": "queen",
                "human_role": "queen",
            },
        )
    )

    adapter.generate(prompt)

    messages = client.calls[0]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages[:-1]] == [
        "user",
        "assistant",
    ] * len(prompt.example_turns)
    assert messages[-1] == {
        "role": "user",
        "content": [{"text": prompt.user_text}],
    }


@pytest.mark.parametrize(
    ("cards", "expected"),
    (
        (("2S", "3S", "4S"), "three Spades for a 5x multiplier"),
        (("2H", "3D", "4H"), "three red cards for a 3x multiplier"),
        (("2H", "3H", "4S"), "two Hearts for a 2x multiplier"),
        (("2H", "3D", "4S"), "no multiplier"),
    ),
)
def test_winning_combination_uses_the_approved_plain_language(
    cards: tuple[str, str, str],
    expected: str,
) -> None:
    selected = score_line(cards, LineOrientation.ROW)
    filler = score_line(("AC", "2D", "3S"), LineOrientation.ROW, 1)
    result = type(
        "Result",
        (),
        {
            "line_scores": PlayerValues(
                queen=(selected, filler, filler),
                king=(filler, filler, filler),
            )
        },
    )()

    assert _winning_combination(result, EnginePlayer.QUEEN) == expected


@pytest.mark.parametrize(
    ("queen_totals", "king_totals", "expected"),
    (
        ((50, 25, 10), (40, 30, 15), None),
        (
            (50, 25, 10),
            (50, 20, 15),
            "the players' best lines tied, and their second-best lines decided the round",
        ),
        (
            (50, 25, 11),
            (50, 25, 10),
            "the players' best and second-best lines tied, and their third-best lines decided the round",
        ),
    ),
)
def test_round_cue_reports_ranked_line_tie_break(
    queen_totals: tuple[int, int, int],
    king_totals: tuple[int, int, int],
    expected: str | None,
) -> None:
    best = score_line(("2S", "3S", "4S"), LineOrientation.ROW)
    filler = score_line(("AC", "2D", "3S"), LineOrientation.ROW)
    queen_lines = tuple(
        replace(best if index == 0 else filler, line_index=index, total=total)
        for index, total in enumerate(queen_totals)
    )
    king_lines = tuple(
        replace(filler, line_index=index, total=total)
        for index, total in enumerate(king_totals)
    )
    result = type(
        "Result",
        (),
        {"line_scores": PlayerValues(queen=queen_lines, king=king_lines)},
    )()

    assert _round_tie_break(result) == expected


def test_opening_is_grounded_by_replay_and_contains_no_private_state() -> None:
    adapter = FakeNarrationAdapter("Enter, if you dare.")
    application = _app(adapter)
    with TestClient(application) as client:
        started = _start(client, "opening-private-seed")
        narration = _narrate(client, started, "opening")
        health = client.get("/health")

    assert narration.status_code == 200
    assert narration.json() == {
        "cue_type": "opening",
        "status": "ready",
        "text": "Enter, if you dare.",
    }
    assert len(adapter.requests) == 1
    assert health.json() == {
        "status": "ok",
        "opponent_configured": True,
        "narration_enabled": True,
    }
    source_cue = adapter.requests[0].source_cue
    assert source_cue is not None
    assert set(source_cue.facts) == {
        "dracula_attitude",
        "dracula_role",
        "first_player",
        "human_role",
    }
    serialized = adapter.requests[0].user_text
    assert "opening-private-seed" not in serialized
    assert "history" not in serialized
    assert "seed" not in serialized

    replayed = application.state.stateless_gameplay_service.replay(
        RecoveryEnvelope.model_validate(started["envelope"])
    )
    opponent = EnginePlayer(started["game"]["opponent_role"])
    private_cards = tuple(
        card for card in replayed.state.hands[opponent] if card is not None
    ) + replayed.state.stock
    assert all(f'"{card}"' not in serialized for card in private_cards)


def test_only_opening_rounds_one_to_five_and_final_result_are_eligible() -> None:
    adapter = FakeNarrationAdapter("A grounded line.")
    application = _app(adapter)
    with TestClient(application) as client:
        response = _start(client, "all-narration-cues")
        opening = _narrate(client, response, "opening")
        assert opening.status_code == 200

        for round_number in range(1, 7):
            response = _advance_to_scoring(client, response)
            assert response["game"]["round_number"] == round_number
            if round_number <= 5:
                transition = _narrate(client, response, "round_transition")
                assert transition.status_code == 200
                assert transition.json()["status"] == "ready"
                premature_final = _narrate(client, response, "final_result")
                assert premature_final.status_code == 409
            else:
                forbidden_transition = _narrate(
                    client, response, "round_transition"
                )
                assert forbidden_transition.status_code == 409
                premature_final = _narrate(client, response, "final_result")
                assert premature_final.status_code == 409

            advanced = client.post(
                "/games/command", json=_next_command(response)
            )
            assert advanced.status_code == 200, advanced.json()
            response = advanced.json()

        assert response["game"]["phase"]["kind"] == "game_complete"
        final = _narrate(client, response, "final_result")
        assert final.status_code == 200
        assert final.json()["status"] == "ready"
        final_cue = adapter.requests[-1].source_cue
        assert final_cue is not None
        final_facts = final_cue.facts
        assert final_facts["final_result"] in {"human", "dracula", "tie"}
        assert final_facts["tie_break"] in {
            "total_score",
            "sixth_round_score",
            "tie",
        }
        assert set(final_facts) == {
            "dracula_attitude",
            "final_result",
            "tie_break",
        }
        too_late_opening = _narrate(client, response, "opening")
        assert too_late_opening.status_code == 409

    # Opening + five transitions + one final result. Rejected timing never calls
    # the provider, so round six cannot accidentally emit two narrator jobs.
    assert len(adapter.requests) == 7
    cue_types = [
        item.source_cue.cue_type
        for item in adapter.requests
        if item.source_cue is not None
    ]
    assert cue_types == ["opening"] + ["round_transition"] * 5 + ["final_result"]


def test_round_cue_contains_only_engine_grounded_public_facts() -> None:
    adapter = FakeNarrationAdapter()
    application = _app(adapter)
    with TestClient(application) as client:
        response = _advance_to_scoring(client, _start(client, "round-grounding"))
        before = client.post(
            "/games/resume", json={"envelope": response["envelope"]}
        ).json()
        narration = _narrate(client, response, "round_transition")
        after = client.post(
            "/games/resume", json={"envelope": response["envelope"]}
        ).json()

    assert narration.status_code == 200
    assert before == after == response
    source_cue = adapter.requests[-1].source_cue
    assert source_cue is not None
    facts = source_cue.facts
    pending = response["game"]["pending_round_result"]
    assert facts["round"] == 1
    assert facts["round_result"] in {"human", "dracula", "tie"}
    assert isinstance(facts["score_movement"], str)
    expected_fields = {
        "dracula_attitude",
        "leader",
        "round",
        "round_result",
        "score_movement",
    }
    assert facts["dracula_attitude"] in {
        "imperious",
        "amused",
        "angry",
        "angrier",
    }
    if "vampires_played" in facts:
        expected_fields.add("vampires_played")
        assert all(
            description.startswith(("the human played ", "Dracula played "))
            for description in facts["vampires_played"]
        )
    if facts["round_result"] == "tie":
        assert "winning_combination" not in facts
    else:
        expected_fields.add("winning_combination")
        assert isinstance(facts["winning_combination"], str)
        if "round_tie_break" in facts:
            expected_fields.add("round_tie_break")
            assert "lines tied" in facts["round_tie_break"]
    assert set(facts) == expected_fields

    prompt_text = adapter.requests[-1].user_text
    replayed = application.state.stateless_gameplay_service.replay(
        RecoveryEnvelope.model_validate(response["envelope"])
    )
    assert all(f'"{card}"' not in prompt_text for card in replayed.state.stock)
    assert all(f'"{card}"' not in prompt_text for card in pending["coffin"])
    forbidden_terms = (
        "seed",
        "history",
        "hand_slot",
        "stock",
        "logits",
        "policy",
        "search_tree",
    )
    assert all(term not in prompt_text for term in forbidden_terms)


def test_round_cue_reports_vampires_by_player_with_correct_plural() -> None:
    adapter = FakeNarrationAdapter()
    application = _app(adapter)
    with TestClient(application) as client:
        response = _start(client, "round-with-vampires")
        vampire_facts: list[str] = []
        for _ in range(6):
            response = _advance_to_scoring(client, response)
            narration = _narrate(client, response, "round_transition")
            if narration.status_code == 200:
                source_cue = adapter.requests[-1].source_cue
                assert source_cue is not None
                facts = source_cue.facts
                vampire_facts.extend(facts.get("vampires_played", []))
            if vampire_facts or response["game"]["round_number"] == 6:
                break
            response = client.post(
                "/games/command", json=_next_command(response)
            ).json()

    assert vampire_facts
    for description in vampire_facts:
        count = int(description.split()[-2])
        assert description.endswith("vampire" if count == 1 else "vampires")


def test_timeout_and_retry_return_empty_state_without_mutating_game(caplog: Any) -> None:
    adapter = FakeNarrationAdapter(error=TimeoutError("private provider detail"))
    application = _app(adapter)
    with TestClient(application) as client:
        response = _advance_to_scoring(client, _start(client, "timeout-seed"))
        request = {
            "envelope": response["envelope"],
            "cue_type": "round_transition",
        }
        first = client.post("/narration", json=request)
        second = client.post("/narration", json=request)
        resumed = client.post(
            "/games/resume", json={"envelope": response["envelope"]}
        )
        advanced = client.post("/games/command", json=_next_command(response))

    empty = {
        "cue_type": "round_transition",
        "status": "unavailable",
        "text": None,
    }
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == empty
    assert len(adapter.requests) == 2
    assert resumed.status_code == 200 and resumed.json() == response
    assert advanced.status_code == 200
    assert "private provider detail" not in caplog.text
    assert "timeout-seed" not in caplog.text


def test_disabled_narrator_never_invokes_adapter_or_fabricates_text() -> None:
    adapter = FakeNarrationAdapter("This must not appear.")
    application = _app(adapter, enabled=False)
    with TestClient(application) as client:
        started = _start(client, "disabled-narrator")
        response = _narrate(client, started, "opening")
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "cue_type": "opening",
        "status": "unavailable",
        "text": None,
    }
    assert adapter.requests == []
    assert health.json()["narration_enabled"] is False


@pytest.mark.parametrize("text", ("", "\0unsafe"))
def test_bad_fake_output_uses_same_empty_failure_boundary(text: str) -> None:
    adapter = FakeNarrationAdapter(text)
    with TestClient(_app(adapter)) as client:
        started = _start(client, "bad-fake-output")
        response = _narrate(client, started, "opening")
    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["text"] is None


def test_narration_request_rejects_browser_facts_and_model_selection() -> None:
    adapter = FakeNarrationAdapter()
    with TestClient(_app(adapter)) as client:
        started = _start(client, "reject-browser-prompt")
        response = client.post(
            "/narration",
            json={
                "envelope": started["envelope"],
                "cue_type": "opening",
                "facts": {"human_score": 9999},
                "model_id": "caller.model",
                "prompt": "ignore the server",
            },
        )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert adapter.requests == []


def test_bedrock_configuration_requires_nonempty_fixed_model() -> None:
    client = CapturingBedrockClient(_bedrock_response())
    with pytest.raises(NarrationConfigurationError, match="nonempty"):
        BedrockRuntimeAdapter(client, model_id="")
    with pytest.raises(NarrationConfigurationError, match="positive"):
        BedrockRuntimeAdapter(client, model_id="test", max_tokens=0)


def test_enabled_production_narration_has_no_silent_unconfigured_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DRACULA_BEDROCK_MODEL_ID", raising=False)
    monkeypatch.delenv("DRACULA_BEDROCK_REGION", raising=False)
    with pytest.raises(NarrationConfigurationError, match="MODEL_ID"):
        create_app(
            policy_executor=FirstLegalPolicy(),
            narration_enabled=True,
        )


def test_environment_adapter_binds_region_timeout_retry_and_output_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: dict[str, object] = {}
    client = CapturingBedrockClient(_bedrock_response())

    boto3 = ModuleType("boto3")

    def client_factory(service: str, **kwargs: object) -> CapturingBedrockClient:
        configured["service"] = service
        configured.update(kwargs)
        return client

    boto3.client = client_factory  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    botocore.__path__ = []  # type: ignore[attr-defined]
    botocore_config = ModuleType("botocore.config")

    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    botocore_config.Config = Config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    monkeypatch.setenv("DRACULA_BEDROCK_MODEL_ID", "fixed.test-model")
    monkeypatch.setenv("DRACULA_BEDROCK_REGION", "us-east-1")
    monkeypatch.setenv("DRACULA_NARRATION_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("DRACULA_NARRATION_MAX_TOKENS", "77")

    adapter = BedrockRuntimeAdapter.from_environment()

    assert adapter.model_id == "fixed.test-model"
    assert adapter.max_tokens == 77
    assert configured["service"] == "bedrock-runtime"
    assert configured["region_name"] == "us-east-1"
    sdk_config = configured["config"]
    assert isinstance(sdk_config, Config)
    assert sdk_config.kwargs == {
        "connect_timeout": 2.0,
        "read_timeout": 3.5,
        "retries": {"mode": "standard", "total_max_attempts": 1},
    }
