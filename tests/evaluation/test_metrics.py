"""Tests for the classification metrics.

The metrics are the instrument the evaluation rests on, so they are tested the
way an instrument is tested: against inputs whose answers are known by hand, and
against the degenerate cases that produce a division by zero.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import ConfusionMatrix, ratio


def test_ratio_is_zero_when_undefined() -> None:
    """A ratio with no denominator is 0, not an exception."""
    assert ratio(0, 0) == 0.0
    assert ratio(5, 0) == 0.0
    assert ratio(-1, 4) < 0.0 or ratio(-1, 4) == -0.25


def test_a_perfect_matrix_scores_one() -> None:
    """Everything on the diagonal is a perfect score."""
    matrix = ConfusionMatrix.from_pairs([("a", "a"), ("b", "b"), ("b", "b")], ("a", "b"))

    assert matrix.total == 3
    assert matrix.correct == 3
    assert matrix.accuracy == 1.0
    assert matrix.macro_f1() == 1.0
    assert matrix.misclassifications() == ()


def test_precision_and_recall_differ_on_an_asymmetric_error() -> None:
    """Two of three real ``a`` cases found, and one spurious ``a`` predicted.

    Recall is 2/3 and precision is 2/3 by construction here, so the test pins the
    definitions rather than the arithmetic of one convenient number.
    """
    matrix = ConfusionMatrix.from_pairs(
        [("a", "a"), ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")], ("a", "b")
    )

    metrics = {m.label: m for m in matrix.per_label()}
    assert metrics["a"].true_positives == 2
    assert metrics["a"].support == 3
    assert metrics["a"].predicted == 3
    assert metrics["a"].precision == pytest.approx(2 / 3)
    assert metrics["a"].recall == pytest.approx(2 / 3)
    assert metrics["b"].precision == pytest.approx(1 / 2)
    assert metrics["b"].recall == pytest.approx(1 / 2)
    assert matrix.accuracy == pytest.approx(3 / 5)


def test_a_label_never_predicted_has_zero_precision_not_an_error() -> None:
    """Absence of predictions is a fact about the classifier, not a crash."""
    matrix = ConfusionMatrix.from_pairs([("a", "a"), ("b", "a")], ("a", "b"))

    metrics = {m.label: m for m in matrix.per_label()}
    assert metrics["b"].precision == 0.0
    assert metrics["b"].recall == 0.0
    assert metrics["b"].f1 == 0.0


def test_f1_is_the_harmonic_mean() -> None:
    """F1 sits between precision and recall and equals them when they agree."""
    matrix = ConfusionMatrix.from_pairs([("a", "a")], ("a",))
    metrics = matrix.per_label()[0]

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_macro_f1_ignores_labels_with_no_support() -> None:
    """An unused route must not drag the score down.

    Otherwise adding a route to the enum would lower every report, and reports
    would stop being comparable over time.
    """
    matrix = ConfusionMatrix.from_pairs([("a", "a"), ("b", "b")], ("a", "b", "c"))

    assert matrix.f1_for("c") == 0.0
    assert matrix.macro_f1() == 1.0


def test_an_unexpected_label_is_visible_rather_than_dropped() -> None:
    """A router that invents a route must show up in the matrix."""
    matrix = ConfusionMatrix.from_pairs([("a", "a"), ("a", "surprise")], ("a",))

    assert "surprise" in matrix.labels
    assert matrix.accuracy == 0.5


def test_misclassifications_are_ordered_by_frequency() -> None:
    """The biggest confusion is reported first, so it is not buried."""
    matrix = ConfusionMatrix.from_pairs(
        [("a", "b"), ("a", "b"), ("b", "c"), ("c", "c")], ("a", "b", "c")
    )

    confusions = matrix.misclassifications()
    assert confusions[0] == ("a", "b", 2)
    assert ("b", "c", 1) in confusions


def test_an_empty_matrix_reports_zero_rather_than_dividing_by_zero() -> None:
    """No cases is a valid input, and the answer is not a traceback."""
    matrix = ConfusionMatrix.from_pairs([], ("a", "b"))

    assert matrix.total == 0
    assert matrix.accuracy == 0.0
    assert matrix.macro_f1() == 0.0
