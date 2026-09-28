"""A timing harness for the hot paths.

Micro-benchmarks are easy to do badly in two opposite ways: report a single
``time.time()`` delta, which is mostly noise, or run for minutes, which nobody
does twice. This harness takes the middle path — warm up, take many samples, and
report percentiles — so a result is stable enough to act on and cheap enough to
keep running.

The comparisons here are *paired*: the same input is scored by the old and the
new implementation in the same process, on the same machine, in the same run.
Comparing against a number recorded earlier on a different host measures the host,
not the change.

Every comparison also asserts that the two implementations agree on their output.
A speed-up that changes the answer is not a speed-up, and without the check a
benchmark would happily report one.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from app.core.config import get_settings
from app.models.memory import MemoryItem, MemoryType
from app.services.memory import (
    MemoryQuery,
    _prepare_query,
    _tokens,
    build_memory_manager,
    estimate_importance,
    score_candidate,
)

__all__ = [
    "BenchmarkResult",
    "TimingResult",
    "agrees_within",
    "benchmark_suite",
    "compare",
    "measure",
    "percentile",
]

T = TypeVar("T")

#: How many candidate memories the scoring benchmarks scan. Matches the default
#: ``memory_scan_limit``, so the measurement describes the configuration the
#: service actually runs with.
_SCAN_SIZE = 500

#: A representative request, long enough that tokenisation is not free.
_SAMPLE_REQUEST = (
    "Compare the two checkpointing backends, explain which fails more safely, "
    "and write a short decision record covering recovery time and storage cost."
)


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``samples``, nearest-rank.

    Args:
        samples: Durations in milliseconds.
        fraction: The percentile to take, between 0.0 and 1.0.

    Returns:
        The selected sample, or ``0.0`` when there are none.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class TimingResult:
    """The distribution of one measured operation."""

    name: str
    samples_ms: tuple[float, ...]

    @property
    def iterations(self) -> int:
        """Return how many samples were taken."""
        return len(self.samples_ms)

    @property
    def mean_ms(self) -> float:
        """Return the mean duration in milliseconds."""
        if not self.samples_ms:
            return 0.0
        return sum(self.samples_ms) / len(self.samples_ms)

    @property
    def best_ms(self) -> float:
        """Return the fastest observed duration, a lower bound on the cost."""
        return min(self.samples_ms, default=0.0)

    @property
    def p50_ms(self) -> float:
        """Return the median duration in milliseconds."""
        return percentile(self.samples_ms, 0.5)

    @property
    def p95_ms(self) -> float:
        """Return the 95th percentile duration in milliseconds."""
        return percentile(self.samples_ms, 0.95)

    @property
    def stdev_ms(self) -> float:
        """Return the sample standard deviation, or ``0.0`` for fewer than two."""
        if len(self.samples_ms) < 2:
            return 0.0
        mean = self.mean_ms
        variance = sum((sample - mean) ** 2 for sample in self.samples_ms) / (
            len(self.samples_ms) - 1
        )
        return math.sqrt(variance)

    @property
    def ops_per_second(self) -> float:
        """Return how many operations per second the mean duration implies."""
        return 0.0 if self.mean_ms <= 0 else 1000.0 / self.mean_ms

    def summary(self) -> str:
        """Return a single-line summary of the distribution."""
        return (
            f"{self.name}: mean {self.mean_ms:.4f} ms, p50 {self.p50_ms:.4f} ms, "
            f"p95 {self.p95_ms:.4f} ms, sd {self.stdev_ms:.4f} ms "
            f"({self.ops_per_second:,.0f}/s)"
        )


def measure(
    operation: Callable[[], Any],
    *,
    name: str = "operation",
    iterations: int = 200,
    warmup: int = 20,
) -> TimingResult:
    """Time ``operation`` repeatedly and return the distribution.

    Garbage collection is disabled for the duration and re-enabled afterwards.
    A collection triggered by an unrelated allocation would otherwise land inside
    one sample and dominate the tail, which is the usual reason a benchmark's p95
    looks alarming and unrepeatable.

    Args:
        operation: A zero-argument callable. Whatever it returns is discarded.
        name: Label for the result.
        iterations: How many timed samples to take.
        warmup: How many untimed calls to make first, to let caches fill.

    Returns:
        The measured distribution.
    """
    for _ in range(max(0, warmup)):
        operation()

    samples: list[float] = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            operation()
            samples.append((time.perf_counter() - started) * 1000)
    finally:
        if enabled:
            gc.enable()

    return TimingResult(name=name, samples_ms=tuple(samples))


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """A paired comparison between two implementations of the same operation."""

    name: str
    baseline: TimingResult
    candidate: TimingResult
    #: Whether both implementations produced identical output on the sample input.
    equivalent: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def speedup(self) -> float:
        """Return how many times faster the candidate is than the baseline."""
        if self.candidate.mean_ms <= 0:
            return 0.0
        return self.baseline.mean_ms / self.candidate.mean_ms

    @property
    def improvement_percent(self) -> float:
        """Return the reduction in mean duration as a percentage."""
        return (1.0 - 1.0 / self.speedup) * 100.0 if self.speedup > 0 else 0.0

    def summary(self) -> str:
        """Return a single-line summary of the comparison."""
        verdict = "faster" if self.speedup >= 1.0 else "slower"
        factor = self.speedup if self.speedup >= 1.0 else 1.0 / self.speedup
        flag = "" if self.equivalent else "  [NOT EQUIVALENT]"
        return (
            f"{self.name}: {factor:.2f}x {verdict} than baseline "
            f"({self.baseline.mean_ms:.4f} ms → {self.candidate.mean_ms:.4f} ms){flag}"
        )


def compare(
    name: str,
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    iterations: int = 200,
    warmup: int = 20,
    notes: Sequence[str] = (),
    equivalent: Callable[[], bool] | None = None,
) -> BenchmarkResult:
    """Time two implementations of the same operation and check they agree.

    Args:
        name: Label for the comparison.
        baseline: The reference implementation.
        candidate: The implementation being evaluated.
        iterations: How many timed samples to take for each.
        warmup: How many untimed calls to make for each.
        notes: Free-text caveats to carry into the report.
        equivalent: Predicate deciding whether the two agree. Defaults to exact
            equality. Supply one where bit-identical output is not the right
            standard — scoring, for instance, reads the clock, so hoisting that
            reading changes the last bits of every recency term without changing
            the ranking. Requiring exact equality there would report a false
            alarm and, worse, hide a real difference behind a rejected check.

    Returns:
        The comparison, including whether the two agreed.
    """
    first = measure(baseline, name=f"{name} (baseline)", iterations=iterations, warmup=warmup)
    second = measure(candidate, name=f"{name} (candidate)", iterations=iterations, warmup=warmup)

    check = equivalent or (lambda: bool(baseline() == candidate()))
    return BenchmarkResult(
        name=name,
        baseline=first,
        candidate=second,
        equivalent=check(),
        notes=tuple(notes),
    )


def agrees_within(
    baseline: Callable[[], Sequence[float]],
    candidate: Callable[[], Sequence[float]],
    tolerance: float,
) -> bool:
    """Return whether two score sequences agree to within ``tolerance``.

    Args:
        baseline: Callable producing the reference scores.
        candidate: Callable producing the compared scores.
        tolerance: The largest absolute difference allowed per element.

    Returns:
        ``True`` if the two produce the same number of scores and every pair is
        within tolerance.
    """
    left, right = baseline(), candidate()
    if len(left) != len(right):
        return False
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance) for a, b in zip(left, right, strict=True)
    )


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #


def _sample_memories(count: int = _SCAN_SIZE) -> list[tuple[MemoryItem, list[float]]]:
    """Return synthetic memories and embeddings for the scoring benchmarks.

    The content is varied enough that lexical overlap is not degenerate, and the
    embeddings are low-dimensional so the benchmark measures the scoring loop
    rather than vector arithmetic on a realistic 1536-dimension model output.

    Args:
        count: How many memories to build.

    Returns:
        ``(memory, embedding)`` pairs, newest first.
    """
    now = datetime.now(UTC)
    topics = (
        "the checkpointing backend writes to postgres",
        "rate limiting uses a fixed window in redis",
        "the planner must be offered only authorised tools",
        "memory retrieval blends similarity importance and recency",
        "approvals are persisted so they survive a restart",
    )
    memories: list[tuple[MemoryItem, list[float]]] = []
    for index in range(count):
        content = f"{topics[index % len(topics)]} (note {index})"
        memories.append(
            (
                MemoryItem(
                    id=f"mem-{index:05d}",
                    user_id="bench-user",
                    type=MemoryType.LONG_TERM,
                    content=content,
                    importance=0.5 + (index % 5) / 10,
                    created_at=now - timedelta(days=index % 30),
                    metadata={"index": index},
                ),
                [float(index % 7), float((index + 1) % 5), float((index + 2) % 3)],
            )
        )
    return memories


# --------------------------------------------------------------------------- #
# The suite
# --------------------------------------------------------------------------- #


def scoring_comparison(*, iterations: int = 200) -> BenchmarkResult:
    """Compare preparing the query once against preparing it per candidate.

    The baseline calls the real scoring entry point once per candidate; the
    candidate prepares a single context and reuses it. Both run on the same input
    in the same process, and the comparison checks that they agree before
    reporting a speed-up.

    Args:
        iterations: How many timed samples to take per implementation.

    Returns:
        The paired comparison.
    """
    memories = _sample_memories()
    query_embedding = [0.5, 0.25, 0.75]
    manager = build_memory_manager(get_settings(), embeddings=None)
    query = MemoryQuery(user_id="bench-user", text=_SAMPLE_REQUEST, limit=10)

    def unprepared() -> list[float]:
        return [
            manager._score(query, item, embedding, query_embedding) for item, embedding in memories
        ]

    def prepared() -> list[float]:
        context = _prepare_query(_SAMPLE_REQUEST, query_embedding)
        return [score_candidate(context, item, embedding) for item, embedding in memories]

    return compare(
        f"memory scoring over {len(memories)} candidates",
        unprepared,
        prepared,
        iterations=iterations,
        notes=(
            "The prepared path hoists the query tokenisation, the query vector "
            "norm, and the clock reading out of the per-candidate loop.",
        ),
        # Exact equality is the wrong test here: the prepared path reads the
        # clock once instead of once per candidate, so the recency term of a
        # candidate scored late in the scan drifts by up to the scan's own
        # duration. That shows up as a difference around 2e-9 on a 500-candidate
        # scan. A tolerance four orders of magnitude below the smallest score gap
        # the ranking could depend on is still far tighter than any real
        # behavioural change would survive.
        equivalent=lambda: agrees_within(unprepared, prepared, 1e-6),
    )


def benchmark_suite(*, iterations: int = 200) -> tuple[TimingResult, ...]:
    """Time each hot path and return the results.

    Args:
        iterations: How many timed samples to take per measurement.

    Returns:
        One result per measured operation, in reporting order.
    """
    memories = _sample_memories()
    context = _prepare_query(_SAMPLE_REQUEST, [0.5, 0.25, 0.75])

    def tokenise() -> frozenset[str]:
        return frozenset(_tokens(_SAMPLE_REQUEST))

    def estimate() -> float:
        return estimate_importance(_SAMPLE_REQUEST)

    def prepare_query() -> object:
        return _prepare_query(_SAMPLE_REQUEST, [0.5, 0.25, 0.75])

    def score_prepared() -> list[float]:
        return [score_candidate(context, item, embedding) for item, embedding in memories]

    def lexical_score() -> list[float]:
        lexical = _prepare_query(_SAMPLE_REQUEST, None)
        return [score_candidate(lexical, item, None) for item, _embedding in memories]

    return (
        measure(tokenise, name="tokenise a request", iterations=iterations * 5),
        measure(estimate, name="estimate importance", iterations=iterations * 5),
        measure(prepare_query, name="prepare a query context", iterations=iterations * 5),
        measure(score_prepared, name="score 500 candidates (vectors)", iterations=iterations),
        measure(
            lexical_score,
            name="score 500 candidates (lexical fallback)",
            iterations=iterations,
        ),
    )
