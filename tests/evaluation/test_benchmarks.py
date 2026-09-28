"""Tests for the timing harness and the scoring optimisation.

The budgets here are regression guards, not benchmarks. They are set an order of
magnitude above the measured cost on the development host, because a tight bound
would fail on a loaded or slower machine and teach everyone to ignore it. What
they catch is a change that makes scoring an order of magnitude slower — the kind
that turns a scan into a stall.

The paired comparison is asserted separately because it is machine-independent:
both implementations run in the same process on the same input, so the ratio
holds regardless of how fast the host is.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.core.config import get_settings
from app.evaluation.benchmarks import (
    _SAMPLE_REQUEST,
    TimingResult,
    _sample_memories,
    agrees_within,
    benchmark_suite,
    compare,
    measure,
    percentile,
    scoring_comparison,
)
from app.services.memory import MemoryQuery, _prepare_query, build_memory_manager, score_candidate

#: A 500-candidate scan must stay in the low milliseconds. The measured mean on
#: the development host is roughly 2 ms; 40 ms is a regression alarm, not a
#: performance target.
SCORE_BUDGET_MS = 40.0
LEXICAL_BUDGET_MS = 60.0


def _counter() -> Callable[[], int]:
    """Return a callable that does a small, fixed amount of work."""
    total = 0

    def tick() -> int:
        nonlocal total
        total += 1
        return total

    return tick


def test_measure_takes_the_requested_number_of_samples() -> None:
    """The sample count is what was asked for, regardless of warmup."""
    result = measure(_counter(), name="tick", iterations=7, warmup=3)

    assert result.iterations == 7
    assert result.name == "tick"
    assert len(result.samples_ms) == 7


def test_percentiles_are_ordered_and_bounded() -> None:
    """p50 cannot exceed p95, and neither can exceed the worst sample."""
    result = measure(_counter(), iterations=25, warmup=2)

    assert result.best_ms <= result.p50_ms <= result.p95_ms
    assert result.p95_ms <= max(result.samples_ms)


def test_ops_per_second_matches_the_mean() -> None:
    """The derived rate is consistent with the mean it is derived from."""
    result = measure(_counter(), iterations=10, warmup=1)

    assert result.mean_ms > 0
    assert result.ops_per_second == pytest.approx(1000.0 / result.mean_ms)


def test_stdev_is_zero_for_a_single_sample() -> None:
    """One sample has no spread to report, and that is not a division by zero."""
    result = measure(_counter(), iterations=1, warmup=0)

    assert result.stdev_ms == 0.0


def test_an_empty_timing_result_reports_zero() -> None:
    """No samples is a valid state with defined answers."""
    empty = TimingResult(name="nothing", samples_ms=())

    assert empty.mean_ms == 0.0
    assert empty.ops_per_second == 0.0
    assert empty.p50_ms == 0.0


def test_percentile_selects_an_observed_sample() -> None:
    """Nearest-rank means the answer is always one of the inputs."""
    values = [5.0, 1.0, 4.0, 2.0, 3.0]

    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 5.0
    assert percentile(values, 0.5) in values


def test_compare_reports_a_faster_candidate() -> None:
    """A genuinely cheaper implementation is reported as faster."""
    result = compare(
        "trivial",
        lambda: sum(range(400)),
        lambda: sum(range(40)),
        iterations=20,
        warmup=2,
    )

    assert result.equivalent is False  # different sums, different answers
    assert result.speedup > 1.0
    assert "faster" in result.summary()


def test_compare_flags_an_equivalent_candidate() -> None:
    """Implementations that agree are reported as equivalent.

    The two callables must be deterministic and share no state, or the check
    would be comparing two different counters rather than two implementations.
    """

    def cheap() -> int:
        return sum(range(40))

    def also_cheap() -> int:
        return sum(range(40))

    result = compare("same", cheap, also_cheap, iterations=5, warmup=1)

    assert result.equivalent is True
    assert "NOT EQUIVALENT" not in result.summary()


def test_agrees_within_tolerates_float_drift_but_not_real_change() -> None:
    """The tolerance is a bound on noise, not a licence to differ."""
    stable = lambda: [1.0, 2.0]  # noqa: E731 - a named helper would say less
    drift = lambda: [1.0, 2.0 + 1e-9]  # noqa: E731
    change = lambda: [1.0, 2.5]  # noqa: E731

    assert agrees_within(stable, stable, 1e-6) is True
    assert agrees_within(stable, drift, 1e-6) is True
    assert agrees_within(stable, change, 1e-6) is False
    assert agrees_within(stable, lambda: [1.0], 1e-6) is False


def test_prepared_scoring_is_substantially_faster() -> None:
    """Hoisting the invariant work out of the loop is the optimisation claimed.

    The baseline prepares the query once per candidate and the candidate prepares
    it once per scan, so the ratio is structural rather than incidental. The
    bound is set well below the measured factor so it survives a noisy host.
    """
    result = scoring_comparison(iterations=25)

    assert result.equivalent, result.summary()
    assert result.speedup > 1.5, result.summary()
    assert result.improvement_percent > 30.0


def test_prepared_and_unprepared_scoring_agree_on_every_score() -> None:
    """The operationally important claim: retrieval produces the same ranking.

    Scores are compared after sorting, so the assertion is about the score
    distribution rather than about which of two tied rows came first.

    Two caveats are stated rather than hidden. The prepared path reads the clock
    once, so scores drift by around a nanosecond; and this sample data contains
    exact ties by construction — two different embeddings with the same cosine to
    the query vector — so the order *within* a tie is not part of the contract and
    is not asserted. The best-scoring memory is asserted, because that one is not
    a tie.
    """
    memories = _sample_memories(200)
    query_embedding = [0.5, 0.25, 0.75]
    manager = build_memory_manager(get_settings(), embeddings=None)
    query = MemoryQuery(user_id="bench-user", text=_SAMPLE_REQUEST, limit=10)

    unprepared = [
        (manager._score(query, item, embedding, query_embedding), item.id)
        for item, embedding in memories
    ]
    context = _prepare_query(_SAMPLE_REQUEST, query_embedding)
    prepared = [
        (score_candidate(context, item, embedding), item.id) for item, embedding in memories
    ]

    unprepared_scores = sorted(score for score, _id in unprepared)
    prepared_scores = sorted(score for score, _id in prepared)
    assert agrees_within(lambda: unprepared_scores, lambda: prepared_scores, 1e-6)

    best_unprepared = max(unprepared, key=lambda pair: pair[0])[1]
    best_prepared = max(prepared, key=lambda pair: pair[0])[1]
    assert best_prepared == best_unprepared


def test_a_500_candidate_scan_stays_within_budget() -> None:
    """Scoring a full scan is milliseconds, not seconds."""
    memories = _sample_memories()
    context = _prepare_query(_SAMPLE_REQUEST, [0.5, 0.25, 0.75])

    result = measure(
        lambda: [score_candidate(context, item, embedding) for item, embedding in memories],
        iterations=20,
        warmup=3,
    )

    assert result.mean_ms < SCORE_BUDGET_MS, result.summary()


def test_the_lexical_fallback_stays_within_budget() -> None:
    """The no-embedding path tokenises every candidate and is the slower one."""
    memories = _sample_memories()
    context = _prepare_query(_SAMPLE_REQUEST, None)

    result = measure(
        lambda: [score_candidate(context, item, None) for item, _embedding in memories],
        iterations=20,
        warmup=3,
    )

    assert result.mean_ms < LEXICAL_BUDGET_MS, result.summary()


def test_the_suite_measures_every_hot_path() -> None:
    """Each named operation is measured and reported."""
    results = benchmark_suite(iterations=10)

    assert len(results) == 5
    names = " ".join(result.name for result in results)
    assert "tokenise" in names
    assert "score 500 candidates (vectors)" in names
    assert all(result.iterations > 0 for result in results)
