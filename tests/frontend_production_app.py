"""Provide a production-shaped ASGI server for frontend browser tests.

The fixture uses stateless gameplay with deterministic local dependencies so
Playwright can exercise recovery and UI behavior without AWS or Bedrock.
"""

from __future__ import annotations

import json

from dracula.api.app import create_app
from dracula.api.narration import NarrationPrompt, NarrationProviderResult


class CueNarrationAdapter:
    @property
    def configured(self) -> bool:
        return True

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        cue = json.loads(prompt.user_text)["cue_type"]
        return NarrationProviderResult(
            text={
                "opening": "Opening cue",
                "round_transition": "Round transition cue",
                "final_result": "Final result cue",
            }[cue],
            input_tokens=8,
            output_tokens=3,
            latency_ms=0.1,
        )


app = create_app(
    gameplay_mode="stateless",
    narration_enabled=True,
    narration_adapter=CueNarrationAdapter(),
)
