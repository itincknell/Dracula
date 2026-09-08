"""Run the stateless local app with predictable test dialogue.

Gameplay uses the configured standalone policy. Narration uses deterministic
grounded placeholder text so the browser's timing and layout can be exercised
without making Bedrock requests.
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI

from dracula.api.development import create_app
from dracula.api.narration.prompt import NarrationPrompt
from dracula.api.narration.service import NarrationProviderResult


def _sentence_case(value: object) -> str:
    """Capitalize one grounded phrase inserted after sentence punctuation."""

    text = str(value)
    return text if not text else text[0].upper() + text[1:]


def _dummy_dialogue(cue_type: str, facts: Mapping[str, object]) -> str:
    """Render predictable text from the same public facts sent to Bedrock."""

    if cue_type == "opening":
        return "Welcome, little mortal! I have already reserved a coffin in your size!"
    if cue_type == "round_transition":
        result = facts["round_result"]
        movement = _sentence_case(facts["score_movement"])
        if result == "dracula":
            combination = _sentence_case(facts["winning_combination"])
            return f"{combination}! {movement} while your miserable hopes decay!"
        if result == "human":
            combination = str(facts["winning_combination"])
            return (
                f"You found {combination}! {movement}; enjoy your tiny "
                "triumph while it lasts!"
            )
        return "A tie? How tedious! I will correct this insult next round!"
    if facts["final_result"] == "dracula":
        return "The night, the game, and your final humiliation all belong to me!"
    if facts["final_result"] == "human":
        return "Impossible! You miserable cheat—I demand another game!"
    return "A draw? Insolent mortal—return at once so I may correct it!"


class LocalDummyNarrationAdapter:
    """Turn grounded cue facts into predictable arcade-Dracula dialogue."""

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        """Use the originating public cue to return a zero-cost provider result."""

        cue = prompt.source_cue
        if cue is None:
            raise ValueError("local preview narration requires a grounded cue")
        return NarrationProviderResult(
            text=_dummy_dialogue(cue.cue_type, cue.facts),
            input_tokens=None,
            output_tokens=None,
            latency_ms=0.0,
        )


def create_local_preview_app() -> FastAPI:
    """Create the configured policy preview with deterministic narration.

    `create_app` reads and loads the selected artifact. The preview overrides
    only narration, leaving gameplay, replay, and cache behavior identical to
    ordinary local development.
    """

    return create_app(
        narration_enabled=True,
        narration_adapter=LocalDummyNarrationAdapter(),
    )


app = create_local_preview_app()
