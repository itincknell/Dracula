"""Explicit stateless ASGI entry point for the production Lambda runtime."""

from dracula.api.bedrock import BedrockRuntimeAdapter
from dracula.api.production_config import ProductionSettings
from dracula.api.stateless_app import create_stateless_app


def create_production_app():
    """Assemble the stateless Lambda app with explicitly configured dependencies."""

    settings = ProductionSettings.from_environment()
    executor = None
    if settings.policy_artifact is not None:
        from dracula.active_policy import ActivePolicyExecutor

        executor = ActivePolicyExecutor(settings.policy_artifact)
    narration_adapter = (
        BedrockRuntimeAdapter.from_environment()
        if settings.narration_enabled
        else None
    )
    return create_stateless_app(
        policy_executor=executor,
        policy_descriptor=None if executor is None else executor.descriptor,
        narration_enabled=settings.narration_enabled,
        narration_adapter=narration_adapter,
        replay_cache_entries=settings.replay_cache_entries,
    )


app = create_production_app()

__all__ = ("app", "create_production_app")
