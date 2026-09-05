"""Protect the deterministic narration used by the local browser preview.

These focused tests verify that placeholder lines use supplied public cue facts
and remain suitable for exercising the production frontend's dialogue timing.
"""

from __future__ import annotations

import json

from dracula.api.local_preview import LocalDummyNarrationAdapter
from dracula.api.narration import NarrationPrompt


def _generate(cue_type: str, facts: dict[str, object]) -> str:
    prompt = NarrationPrompt(
        system_text="unused local fixture",
        user_text=json.dumps(
            {"cue_type": cue_type, "public_facts": facts},
            sort_keys=True,
        ),
    )
    return LocalDummyNarrationAdapter().generate(prompt).text


def test_dummy_dialogue_exercises_opening_round_and_final_tones() -> None:
    assert "coffin" in _generate("opening", {})
    assert _generate(
        "round_transition",
        {
            "round_result": "dracula",
            "score_movement": "Dracula extends his lead over the human",
            "winning_combination": "three Spades for a 5x multiplier",
        },
    ).startswith("Three Spades for a 5x multiplier!")
    assert "tiny triumph" in _generate(
        "round_transition",
        {
            "round_result": "human",
            "score_movement": "the human takes the lead",
            "winning_combination": "two Hearts for a 2x multiplier",
        },
    )
    assert "final humiliation" in _generate(
        "final_result", {"final_result": "dracula"}
    )
    assert "Impossible" in _generate(
        "final_result", {"final_result": "human"}
    )
