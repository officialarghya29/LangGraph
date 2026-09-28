"""Structured logging.

Logs are JSON so they can be indexed, with a human-readable form for local
development. Every record passes through the same redaction step, because a log
line is one of the most common ways a credential escapes.

Caller-supplied context fields are merged into the record rather than dropped,
so ``task_id``, ``node``, ``tool``, and ``duration_ms`` stay indexable instead of
being buried inside a message string.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from app.core.config import Settings
from app.core.security import redact

__all__ = ["HumanFormatter", "JsonFormatter", "configure_logging"]

#: Attributes present on every LogRecord. Anything else in ``record.__dict__``
#: is caller-supplied context.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


def _context(record: logging.LogRecord, secrets: tuple[str, ...]) -> dict[str, str]:
    """Extract and redact caller-supplied context fields."""
    return {
        key: redact(str(value), secrets)
        for key, value in record.__dict__.items()
        if key not in _RESERVED and not key.startswith("_")
    }


def _exception_text(record: logging.LogRecord, secrets: tuple[str, ...]) -> str | None:
    """Render the exception attached to a record, if any."""
    if record.exc_info is None or record.exc_info[0] is None:
        return None
    return redact(f"{record.exc_info[0].__name__}: {record.exc_info[1]}", secrets)


class JsonFormatter(logging.Formatter):
    """Render a log record as a single JSON object."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """Serialise a record, redacting messages, context, and exceptions."""
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage(), self._secrets),
        }
        payload.update(_context(record, self._secrets))

        exception = _exception_text(record, self._secrets)
        if exception is not None:
            payload["exception"] = exception

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Render a log record for reading in a terminal."""

    def __init__(self, secrets: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with its context appended.

        The record's own ``msg`` is never mutated, so a shared record cannot
        leak a redacted-then-restored value to another handler.
        """
        timestamp = self.formatTime(record, self.datefmt)
        message = redact(record.getMessage(), self._secrets)
        line = f"{timestamp} {record.levelname:<8} {record.name}: {message}"

        context = _context(record, self._secrets)
        if context:
            rendered = " ".join(f"{key}={value}" for key, value in sorted(context.items()))
            line = f"{line}  {rendered}"

        exception = _exception_text(record, self._secrets)
        if exception is not None:
            line = f"{line}\n{exception}"

        return line


def configure_logging(settings: Settings) -> None:
    """Install the root logging handler.

    Safe to call more than once: existing handlers are removed first, so a
    reload does not duplicate every line.

    Args:
        settings: Application settings, for the level and the secrets to redact.
    """
    secrets = settings.secret_values()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter(secrets) if settings.is_production else HumanFormatter(secrets)
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)
