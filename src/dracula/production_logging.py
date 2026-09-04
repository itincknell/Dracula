"""Minimal JSON formatter for container stdout and CloudWatch ingestion."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonLogFormatter(logging.Formatter):
    """Render one log record as a bounded, single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


__all__ = ("JsonLogFormatter",)
