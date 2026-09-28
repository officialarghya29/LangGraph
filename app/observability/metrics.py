"""In-process metrics, exposed in the Prometheus text format.

Written rather than pulled in, for one reason: the client libraries that do this
well also want to own the process (a registry singleton, a background exporter,
their own HTTP handler). This application already has one composition root and
one HTTP surface, and a metrics library that reached around both would be the
only component in the codebase that did.

What is here is deliberately small. Three instrument types cover everything this
system actually needs to answer:

- **Counters** for things that only increase: requests, tool calls, task
  outcomes. A counter that can go down is a gauge wearing a disguise.
- **Gauges** for things that are true right now: registered agents, in-flight
  requests.
- **Histograms** for latency, with explicit bucket bounds. Buckets are fixed and
  named in the constructor rather than derived, because a histogram is only
  useful if the boundaries reflect the latencies you care about.

Labels are part of a metric's identity here, not decoration. ``http_requests_total``
with a ``status`` label is one metric with several series; two calls with
different labels produce two series under the same name, which is what a
Prometheus caller expects. Cardinality is the operator's problem, so the
recorded labels are always low-cardinality values — a route path, a status
class, a tool name — never a task id or a user id.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = ["MetricRegistry", "default_buckets", "render_prometheus"]

#: Label sets, as immutable pairs so they can key a dictionary.
Labels = tuple[tuple[str, str], ...]


def default_buckets() -> tuple[float, ...]:
    """Return the default latency buckets, in seconds.

    Chosen around this system's actual shape: a local cache hit is sub-millisecond,
    a database write is single-digit milliseconds, and anything past five seconds
    is a provider or a timeout, not a latency curve worth resolving.
    """
    return (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def _labels(pairs: Mapping[str, object] | None) -> Labels:
    """Normalise a label mapping into a sorted tuple.

    Sorted so that two callers naming the same labels in a different order land
    on the same series rather than on two that look identical.
    """
    if not pairs:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in pairs.items()))


def _escape(value: str) -> str:
    """Escape a label value for the exposition format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_number(value: float) -> str:
    """Render a float the way the exposition format expects.

    Prometheus has no ``inf`` or ``nan`` literal. ``+Inf`` is spelled out, and a
    non-finite value is clamped rather than emitted, because one bad observation
    should not make the whole scrape unparseable.
    """
    if math.isnan(value):
        return "0"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


@dataclass
class _Histogram:
    """A cumulative histogram with fixed buckets."""

    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        """Record one observation."""
        self.total += value
        self.count += 1
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[index] += 1


class MetricRegistry:
    """Holds every metric this process records.

    Thread-safe, because the ASGI server runs handlers in a thread pool as well
    as on the event loop. The lock is coarse on purpose: contention on a metrics
    registry is a signal that the hot path is doing something unusual, and a
    fine-grained lock would trade a real bug for a measurement of itself.
    """

    def __init__(self, buckets: Iterable[float] | None = None) -> None:
        self._buckets = tuple(buckets) if buckets is not None else default_buckets()
        self._counters: dict[tuple[str, Labels], float] = {}
        self._gauges: dict[tuple[str, Labels], float] = {}
        self._histograms: dict[tuple[str, Labels], _Histogram] = {}
        self._lock = threading.Lock()

    # -- Recording --------------------------------------------------------- #

    def increment(self, name: str, amount: float = 1.0, **labels: object) -> None:
        """Add ``amount`` to a counter series.

        Args:
            name: Metric name, conventionally suffixed ``_total``.
            amount: Value to add. Must not be negative; a counter that decreases
                is a gauge, and silently allowing it hides a bug.
            **labels: Label values for the series.
        """
        if amount < 0:
            raise ValueError("counters cannot decrease; use set_gauge for a value that can")
        key = (name, _labels(labels))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amount

    def set_gauge(self, name: str, value: float, **labels: object) -> None:
        """Set a gauge series to ``value``."""
        with self._lock:
            self._gauges[(name, _labels(labels))] = value

    def observe(self, name: str, value: float, **labels: object) -> None:
        """Record one observation in a histogram series.

        A non-numeric value is dropped before the series is created. Dropping it
        afterwards would leave an empty series behind, which renders as a metric
        that exists and is always zero — the sort of phantom that gets alerted
        on and then argued about.
        """
        if isinstance(value, float) and math.isnan(value):
            return
        key = (name, _labels(labels))
        with self._lock:
            histogram = self._histograms.get(key)
            if histogram is None:
                histogram = _Histogram(self._buckets)
                self._histograms[key] = histogram
            histogram.observe(value)

    # -- Introspection ----------------------------------------------------- #

    def counter(self, name: str, **labels: object) -> float:
        """Return the current value of a counter series, or zero."""
        return self._counters.get((name, _labels(labels)), 0.0)

    def gauge(self, name: str, **labels: object) -> float | None:
        """Return the current value of a gauge series, or ``None`` if unset."""
        return self._gauges.get((name, _labels(labels)))

    def histogram(self, name: str, **labels: object) -> _Histogram | None:
        """Return a histogram series, or ``None`` if nothing was recorded."""
        return self._histograms.get((name, _labels(labels)))

    @property
    def series_count(self) -> int:
        """Return how many distinct series are held.

        Reported by the readiness probe, where the useful reading is "metrics are
        flowing" rather than any particular value.
        """
        with self._lock:
            return len(self._counters) + len(self._gauges) + len(self._histograms)

    def reset(self) -> None:
        """Drop every series. Used between tests so counts never leak."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def render(self) -> str:
        """Render every series in the Prometheus text exposition format."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = {
                key: _Histogram(value.buckets, list(value.counts), value.total, value.count)
                for key, value in self._histograms.items()
            }
        return render_prometheus(counters, gauges, histograms)


def _series_name(name: str, labels: Labels, extra: tuple[tuple[str, str], ...] = ()) -> str:
    """Return ``name{labels}`` for the exposition format."""
    merged = tuple(sorted((*labels, *extra)))
    if not merged:
        return name
    rendered = ",".join(f'{key}="{_escape(value)}"' for key, value in merged)
    return f"{name}{{{rendered}}}"


def render_prometheus(
    counters: Mapping[tuple[str, Labels], float],
    gauges: Mapping[tuple[str, Labels], float],
    histograms: Mapping[tuple[str, Labels], _Histogram],
) -> str:
    """Render metric series as Prometheus text.

    Histograms are expanded the way the format requires: a cumulative ``_bucket``
    line per bound including ``+Inf``, then ``_sum`` and ``_count``. The ``+Inf``
    bucket is not optional — it is how the format represents the total, and a
    histogram without it is rejected by Prometheus.

    Args:
        counters: Counter series keyed by name and labels.
        gauges: Gauge series keyed by name and labels.
        histograms: Histogram series keyed by name and labels.

    Returns:
        The exposition payload, newline-terminated.
    """
    lines: list[str] = []

    # Sorted so the output is stable between scrapes, which makes diffs readable
    # and keeps the tests deterministic.
    for (name, labels), value in sorted(counters.items()):
        lines.append(f"{_series_name(name, labels)} {_format_number(value)}")

    for (name, labels), value in sorted(gauges.items()):
        lines.append(f"{_series_name(name, labels)} {_format_number(value)}")

    for (name, labels), histogram in sorted(histograms.items()):
        # ``counts`` is already cumulative: each observation fell into every
        # bucket at or above its bound.
        for bound, count in zip(histogram.buckets, histogram.counts, strict=True):
            lines.append(
                f"{_series_name(name + '_bucket', labels, (('le', _format_number(bound)),))}"
                f" {_format_number(count)}"
            )
        lines.append(
            f"{_series_name(name + '_bucket', labels, (('le', '+Inf'),))} "
            f"{_format_number(histogram.count)}"
        )
        lines.append(f"{_series_name(name + '_sum', labels)} {_format_number(histogram.total)}")
        lines.append(f"{_series_name(name + '_count', labels)} {_format_number(histogram.count)}")

    return "\n".join(lines) + "\n"
