"""Span tracing.

Traces are emitted as structured logs and, when a registry is supplied, as
metrics. That combination is not a placeholder for a "real" tracing backend —
it is a deliberate choice about what this system needs to answer:

- **What happened, in what order, for one request.** Answered by the span log,
  correlated by the request id the middleware already assigns.
- **How long each phase takes, and how often it runs.** Answered by the
  histogram, which is what you actually look at when something is slow.

A distributed tracing backend adds cross-service propagation and a query UI,
which is a real benefit and a real dependency. OpenTelemetry is not installed
here, so rather than ship an adapter that cannot be exercised, this module
defines the seam and says plainly where the OTLP implementation would attach.

Spans nest through a ``ContextVar``, so no call site has to thread a parent
through its arguments — which is exactly the kind of plumbing that rots when it
is manual.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Protocol

from app.core.config import Settings
from app.observability.metrics import MetricRegistry

__all__ = ["LoggingTracer", "Span", "Tracer", "build_tracer", "current_span"]

logger = logging.getLogger(__name__)

#: The span currently being executed, if any. A ``ContextVar`` rather than a
#: global so that concurrent requests each see their own span stack.
_active_span: ContextVar[Span | None] = ContextVar("active_span", default=None)


def current_span() -> Span | None:
    """Return the span that is currently executing, if there is one."""
    return _active_span.get()


@dataclass
class Span:
    """One unit of work with a start, an end, and attributes."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "ok") -> None:
        """Mark the span complete and record how long it took.

        Idempotent: finishing twice keeps the first duration. Without that, a
        context manager and an explicit ``finish`` in a ``finally`` would
        overwrite a real measurement with a near-zero one.
        """
        if self.duration_ms:
            return
        self.status = status
        self.duration_ms = round((time.perf_counter() - self.started_at) * 1000, 3)


class Tracer(Protocol):
    """Records spans."""

    def span(self, name: str, **attributes: object) -> _SpanContext:
        """Return a context manager that records a span named ``name``."""
        ...


class LoggingTracer:
    """Emits every span as a structured log line, and as a metric.

    A bounded ring buffer of recent spans is kept for introspection — enough to
    answer "what did the last run do" in a test or a debug session, small enough
    that it cannot become a slow memory leak.
    """

    def __init__(
        self,
        *,
        service_name: str = "langgraph-multi-agent",
        metrics: MetricRegistry | None = None,
        history: int = 200,
    ) -> None:
        self.service_name = service_name
        self._metrics = metrics
        self._recent: deque[Span] = deque(maxlen=history)

    @property
    def recent(self) -> tuple[Span, ...]:
        """Return recently completed spans, oldest first."""
        return tuple(self._recent)

    def span(self, name: str, **attributes: object) -> _SpanContext:
        """Open a span and return the context manager that records it."""
        return _SpanContext(self, name, attributes)

    def record(self, span: Span) -> None:
        """Publish a finished span to the log, the metric registry, and history."""
        self._recent.append(span)
        logger.info(
            "trace.span",
            extra={
                "service": self.service_name,
                "span": span.name,
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_id": span.parent_id,
                "duration_ms": span.duration_ms,
                "status": span.status,
                **span.attributes,
            },
        )
        if self._metrics is not None:
            self._metrics.increment("spans_total", span=span.name, status=span.status)
            self._metrics.observe("span_duration_seconds", span.duration_ms / 1000, span=span.name)


class _SpanContext:
    """Context manager returned by :meth:`LoggingTracer.span`."""

    def __init__(self, tracer: LoggingTracer, name: str, attributes: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._token: Any = None
        self.span: Span | None = None

    def __enter__(self) -> Span:
        """Start the span and make it the active one."""
        parent = current_span()
        self.span = Span(
            name=self._name,
            # A child inherits its parent's trace id, which is what makes the
            # log lines for one request group together.
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:32],
            parent_id=parent.span_id if parent else None,
            attributes=dict(self._attributes),
        )
        self._token = _active_span.set(self.span)
        return self.span

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the span, marking it failed when the body raised."""
        if self.span is not None:
            self.span.finish(status="error" if exc_type is not None else "ok")
            if exc is not None:
                self.span.attributes.setdefault("error", type(exc).__name__)
            self._tracer.record(self.span)
        if self._token is not None:
            _active_span.reset(self._token)


def _module_exists(name: str) -> bool:
    """Return whether an importable module with this name exists.

    ``find_spec`` raises ``ModuleNotFoundError`` when a *parent* package is
    missing rather than returning ``None``, so the plain call is not a check for
    the thing it looks like it checks.
    """
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def build_tracer(settings: Settings, *, metrics: MetricRegistry | None = None) -> LoggingTracer:
    """Return the tracer this process should use.

    When an OTLP endpoint is configured but the OpenTelemetry packages are not
    installed, this logs that fact once and falls back. Refusing to start would
    turn an observability misconfiguration into an outage, which is the wrong
    trade; silently doing nothing would hide it, which is worse.

    The history size is taken from settings rather than left to the constructor's
    default. Until it was, ``TRACE_HISTORY_SIZE`` was a setting that configured
    nothing: an operator could lower it to bound memory and see no change.

    Args:
        settings: Application settings.
        metrics: Optional registry to feed span counts and durations into.

    Returns:
        The tracer.
    """
    tracer = LoggingTracer(
        service_name=settings.otel_service_name,
        metrics=metrics,
        history=settings.trace_history_size,
    )

    if settings.otel_exporter_otlp_endpoint:
        # Probed rather than imported: an import statement for a package that may
        # not exist needs a suppression, and a suppression that is correct in one
        # environment is an unused-ignore error in another.
        if not _module_exists("opentelemetry.sdk.trace"):
            logger.warning(
                "tracing.exporter_unavailable",
                extra={
                    "endpoint": settings.otel_exporter_otlp_endpoint,
                    "reason": "the opentelemetry packages are not installed",
                },
            )
        else:  # pragma: no cover - requires the optional dependency
            logger.info(
                "tracing.exporter_ignored",
                extra={"endpoint": settings.otel_exporter_otlp_endpoint},
            )

    return tracer


@contextmanager
def traced_span(tracer: Tracer | None, name: str, **attributes: object) -> Iterator[None]:
    """Open a span when a tracer exists, and do nothing when it does not.

    Lets a collaborator be traced or untraced without every call site growing an
    ``if tracer is not None``.
    """
    if tracer is None:
        yield
        return
    with tracer.span(name, **attributes):
        yield
