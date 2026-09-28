"""Tests for the metrics registry and the span tracer.

The exposition format is a contract with something outside this repository, so
the tests check the rendered text rather than only the internal state: a
histogram without its ``+Inf`` bucket, or a label that is not escaped, produces a
payload that Prometheus rejects — and the failure would appear in production
monitoring, not here.
"""

from __future__ import annotations

import math

import pytest

from app.core.config import Settings
from app.observability.metrics import MetricRegistry, render_prometheus
from app.observability.tracing import LoggingTracer, build_tracer, current_span, traced_span


def registry() -> MetricRegistry:
    return MetricRegistry(buckets=(0.1, 0.5, 1.0))


# --------------------------------------------------------------------------- #
# Counters
# --------------------------------------------------------------------------- #


def test_a_counter_accumulates() -> None:
    metrics = registry()

    metrics.increment("requests_total")
    metrics.increment("requests_total", 2)

    assert metrics.counter("requests_total") == 3


def test_counters_separate_by_label() -> None:
    metrics = registry()

    metrics.increment("requests_total", method="GET")
    metrics.increment("requests_total", method="POST", attempts=3)

    assert metrics.counter("requests_total", method="GET") == 1
    assert metrics.counter("requests_total", method="POST", attempts=3) == 1
    assert metrics.counter("requests_total", method="POST") == 0


def test_label_order_does_not_create_a_second_series() -> None:
    """Two callers naming the same labels differently share one series."""
    metrics = registry()

    metrics.increment("requests_total", method="GET", path="/health")
    metrics.increment("requests_total", path="/health", method="GET")

    assert metrics.counter("requests_total", method="GET", path="/health") == 2


def test_a_counter_cannot_decrease() -> None:
    """A decreasing counter is a gauge, and allowing it hides a bug."""
    metrics = registry()

    with pytest.raises(ValueError, match="cannot decrease"):
        metrics.increment("requests_total", -1)


# --------------------------------------------------------------------------- #
# Gauges
# --------------------------------------------------------------------------- #


def test_a_gauge_replaces_rather_than_accumulates() -> None:
    metrics = registry()

    metrics.set_gauge("in_flight", 4)
    metrics.set_gauge("in_flight", 1)

    assert metrics.gauge("in_flight") == 1


def test_an_unset_gauge_is_none_not_zero() -> None:
    """Zero is a real reading; None means nothing has been observed."""
    assert registry().gauge("never_set") is None


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #


def test_histogram_buckets_are_cumulative() -> None:
    metrics = registry()

    metrics.observe("latency_seconds", 0.05)
    metrics.observe("latency_seconds", 0.75)

    histogram = metrics.histogram("latency_seconds")
    assert histogram is not None
    # Cumulative: 0.05 falls in all three buckets, 0.75 only in the last.
    assert histogram.counts == [1, 1, 2]
    assert histogram.count == 2
    assert histogram.total == pytest.approx(0.8)


def test_a_value_above_every_bucket_still_counts() -> None:
    """The total is the ``+Inf`` bucket and must never lose an observation."""
    metrics = registry()

    metrics.observe("latency_seconds", 99.0)

    histogram = metrics.histogram("latency_seconds")
    assert histogram is not None
    assert histogram.counts == [0, 0, 0]
    assert histogram.count == 1


def test_a_nan_observation_is_ignored() -> None:
    """A dropped sample must not leave an always-zero series behind."""
    metrics = registry()

    metrics.observe("latency_seconds", math.nan)

    assert metrics.histogram("latency_seconds") is None


def test_a_nan_after_a_real_observation_leaves_the_real_one() -> None:
    metrics = registry()
    metrics.observe("latency_seconds", 0.2)

    metrics.observe("latency_seconds", math.nan)

    histogram = metrics.histogram("latency_seconds")
    assert histogram is not None
    assert histogram.count == 1


# --------------------------------------------------------------------------- #
# Exposition
# --------------------------------------------------------------------------- #


def test_the_rendered_payload_is_parseable_by_a_scraper() -> None:
    metrics = registry()
    metrics.increment("requests_total", method="GET")
    metrics.observe("latency_seconds", 0.05)

    payload = metrics.render()

    assert 'requests_total{method="GET"} 1' in payload
    assert 'latency_seconds_bucket{le="0.1"} 1' in payload
    assert 'latency_seconds_bucket{le="+Inf"} 1' in payload
    assert "latency_seconds_sum 0.05" in payload
    assert "latency_seconds_count 1" in payload
    assert payload.endswith("\n")


def test_a_quote_in_a_label_is_escaped() -> None:
    """An unescaped quote would break the whole scrape, not just one series."""
    rendered = render_prometheus(
        {("requests_total", (("path", 'a"b'),)): 1.0},
        {},
        {},
    )

    assert 'path="a\\"b"' in rendered


def test_integers_render_without_a_decimal_point() -> None:
    """Prometheus accepts either, but ``1.0`` for a count reads as a mistake."""
    rendered = render_prometheus({("requests_total", ()): 7.0}, {}, {})

    assert rendered.startswith("requests_total 7\n")


def test_a_non_finite_counter_is_clamped_rather_than_emitted() -> None:
    """``nan`` is not a Prometheus literal, and one bad value must not break a scrape."""
    rendered = render_prometheus(
        {("requests_total", ()): math.inf, ("other_total", ()): math.nan},
        {},
        {},
    )

    assert "requests_total +Inf" in rendered
    assert "other_total 0" in rendered


def test_reset_drops_every_series() -> None:
    """Tests must not inherit each other's counts."""
    metrics = registry()
    metrics.increment("requests_total")
    metrics.set_gauge("in_flight", 1)
    metrics.observe("latency_seconds", 0.2)

    metrics.reset()

    assert metrics.counter("requests_total") == 0
    assert metrics.gauge("in_flight") is None
    assert metrics.render() == "\n"


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #


def test_a_span_records_its_duration() -> None:
    tracer = LoggingTracer()

    with tracer.span("node.plan") as span:
        pass

    assert span.duration_ms > 0
    assert span.status == "ok"
    assert tracer.recent == (span,)


def test_finishing_twice_keeps_the_first_duration() -> None:
    """A context manager plus an explicit finish must not overwrite a real number."""
    tracer = LoggingTracer()

    with tracer.span("node.plan") as span:
        pass
    measured = span.duration_ms

    span.finish()

    assert span.duration_ms == measured


def test_a_child_span_shares_its_parent_trace() -> None:
    """Sharing the trace id is what groups one request's spans together."""
    tracer = LoggingTracer()

    with tracer.span("task.execute") as parent, tracer.span("node.plan") as child:
        assert child.trace_id == parent.trace_id
        assert child.parent_id == parent.span_id


def test_a_span_nests_through_the_context_variable() -> None:
    """Callers must not have to thread a parent through their arguments."""
    tracer = LoggingTracer()

    assert current_span() is None
    with tracer.span("outer") as outer:
        assert current_span() is outer
    assert current_span() is None


def test_a_raised_error_marks_the_span_failed() -> None:
    tracer = LoggingTracer()

    with pytest.raises(RuntimeError), tracer.span("node.plan") as span:
        raise RuntimeError("boom")

    assert span.status == "error"
    assert span.attributes["error"] == "RuntimeError"


def test_spans_feed_the_metric_registry() -> None:
    metrics = registry()
    tracer = LoggingTracer(metrics=metrics)

    with tracer.span("node.plan"):
        pass

    assert metrics.counter("spans_total", span="node.plan", status="ok") == 1
    assert metrics.histogram("span_duration_seconds", span="node.plan") is not None


def test_the_tracer_records_at_startup_without_an_exporter() -> None:
    tracer = build_tracer(Settings(_env_file=None))

    assert isinstance(tracer, LoggingTracer)


def test_an_exporter_without_the_package_degrades_instead_of_failing() -> None:
    """A tracing misconfiguration must not become an outage."""
    settings = Settings(_env_file=None, otel_exporter_otlp_endpoint="http://collector:4317")

    tracer = build_tracer(settings)

    assert isinstance(tracer, LoggingTracer)


def test_a_traced_span_is_optional() -> None:
    """A collaborator can be traced or untraced without a branch at the call site."""
    with traced_span(None, "anything"):
        pass

    tracer = LoggingTracer()
    with traced_span(tracer, "named"):
        pass

    assert [span.name for span in tracer.recent] == ["named"]
