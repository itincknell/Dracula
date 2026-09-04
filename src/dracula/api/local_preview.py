"""Production-shaped local preview with deterministic placeholder narration."""

from __future__ import annotations

import json

from dracula.active_policy import ActivePolicyExecutor
from dracula.api.app import create_app
from dracula.api.narration import NarrationPrompt, NarrationProviderResult
from dracula.api.production_config import ProductionSettings


def _sentence_case(value: object) -> str:
    text = str(value)
    return text if not text else text[0].upper() + text[1:]


class LocalDummyNarrationAdapter:
    """Turn grounded cue facts into predictable arcade-Dracula test dialogue."""

    @property
    def configured(self) -> bool:
        """Return true because this deterministic adapter requires no service."""

        return True

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        """Render fixed dialogue from the same grounded facts used by Bedrock."""

        payload = json.loads(prompt.user_text)
        cue_type = payload["cue_type"]
        facts = payload["public_facts"]
        if cue_type == "opening":
            text = (
                "Welcome, little mortal! I have already reserved a coffin in "
                "your size!"
            )
        elif cue_type == "round_transition":
            result = facts["round_result"]
            movement = _sentence_case(facts["score_movement"])
            if result == "dracula":
                combination = _sentence_case(facts["winning_combination"])
                text = (
                    f"{combination}! {movement} while your miserable hopes decay!"
                )
            elif result == "human":
                combination = str(facts["winning_combination"])
                text = (
                    f"You found {combination}! {movement}; enjoy your tiny "
                    "triumph while it lasts!"
                )
            else:
                text = "A tie? How tedious! I will correct this insult next round!"
        elif facts["final_result"] == "dracula":
            text = (
                "The night, the game, and your final humiliation all belong to me!"
            )
        elif facts["final_result"] == "human":
            text = "Impossible! You miserable cheat—I demand another game!"
        else:
            text = "A draw? Insolent mortal—return at once so I may correct it!"
        return NarrationProviderResult(
            text=text,
            input_tokens=None,
            output_tokens=None,
            latency_ms=0.0,
        )


def create_local_preview_app():
    """Create the stateless π1 preview with deterministic dummy narration."""

    settings = ProductionSettings.from_environment()
    executor = (
        None
        if settings.policy_artifact is None
        else ActivePolicyExecutor(settings.policy_artifact)
    )
    return create_app(
        gameplay_mode="stateless",
        policy_executor=executor,
        policy_descriptor=None if executor is None else executor.descriptor,
        narration_enabled=True,
        narration_adapter=LocalDummyNarrationAdapter(),
        replay_cache_entries=settings.replay_cache_entries,
    )


app = create_local_preview_app()

__all__ = (
    "LocalDummyNarrationAdapter",
    "app",
    "create_local_preview_app",
)
