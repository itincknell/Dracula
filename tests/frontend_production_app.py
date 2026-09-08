"""Provide a production-shaped ASGI server for frontend browser tests.

The fixture uses stateless gameplay with deterministic local dependencies so
Playwright can exercise recovery and UI behavior without AWS or Bedrock.
"""

from __future__ import annotations

from pathlib import Path

from dracula.api.development import create_app
from dracula.api.narration.prompt import NarrationPrompt
from dracula.api.narration.service import NarrationProviderResult
from dracula.api.web import create_web_app


class CueNarrationAdapter:
    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        # Production prompts contain English, so use the originating cue rather
        # than parsing presentation text to select a predictable test response.
        assert prompt.source_cue is not None
        cue = prompt.source_cue.cue_type
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


api = create_app(
    narration_enabled=True,
    narration_adapter=CueNarrationAdapter(),
)
app = create_web_app(api, Path(__file__).resolve().parents[1] / "frontend/dist")
