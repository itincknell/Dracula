"""Create the stateless FastAPI application loaded by the Lambda container.

Uvicorn imports the module-level ``app`` object below. During that import,
this module reads configuration,
loads the selected standalone policy, and optionally creates the Bedrock
client. Requests are handled only after this one-time startup work completes.
"""

import os
from pathlib import Path

from fastapi import FastAPI

from dracula.api.narration.bedrock import BedrockRuntimeAdapter
from dracula.api.config import ProductionSettings
from dracula.api.stateless.app import create_stateless_app
from dracula.api.web import create_web_app


def create_production_app() -> FastAPI:
    """Assemble the Lambda app from validated environment configuration.

    Policy and Bedrock objects are created here, once per warm Lambda process.
    The resulting FastAPI app remains stateless: only its bounded replay cache
    survives between requests handled by the same process.
    """

    # Route handlers never read process configuration after startup.
    settings = ProductionSettings.from_environment()
    executor = None
    if settings.policy_artifact is not None:
        # Importing the policy runtime loads PyTorch. Keep that dependency out
        # of policy-free API imports and tests, but load it once when configured.
        from dracula.policy.runtime import ActivePolicyExecutor

        executor = ActivePolicyExecutor(settings.policy_artifact)
    # Narration disabled means no AWS client is created and no Bedrock request
    # can be issued. Enabling it requires complete Bedrock configuration.
    narration_adapter = (
        BedrockRuntimeAdapter.from_environment()
        if settings.narration_enabled
        else None
    )
    # Routing and game replay remain inside the shared stateless application.
    api = create_stateless_app(
        policy_executor=executor,
        narration_adapter=narration_adapter,
        replay_cache_entries=settings.replay_cache_entries,
    )
    return create_web_app(
        api, Path(os.getenv("DRACULA_FRONTEND_DIR", "frontend/dist"))
    )


# Uvicorn and the Lambda Web Adapter refer to this object as
# ``dracula.api.production:app``; importing the module constructs it once.
app = create_production_app()
