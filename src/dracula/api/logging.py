"""Format production log records as compact structured JSON.

The Lambda container writes these records to standard output for CloudWatch.
Callers remain responsible for excluding seeds, histories, and private state.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonLogFormatter(logging.Formatter):
    """Render one standard logging record as a single-line JSON object.

    CloudWatch stores each stdout line as one event. Compact JSON keeps the
    severity, logger, message, UTC timestamp, and optional formatted exception
    queryable without adding gameplay fields or inspecting arbitrary extras.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Build the fixed public-safe log field set for one record."""

        payload: dict[str, object] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
        }
        if record.exc_info:
            # Tracebacks are included only when the caller logged an exception;
            # ordinary records remain four small fields.
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
