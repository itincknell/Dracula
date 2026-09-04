"""HTTP boundaries for local and stateless Dracula applications."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ("app", "create_app")


def __getattr__(name: str) -> Any:
    """Load the historical package-level app exports only when requested."""

    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("dracula.api.app"), name)
