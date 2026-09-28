"""Classification metrics for the evaluation harness.

Accuracy alone is the wrong headline number for this system. Six of the seven
routes are rare relative to the others, so a router that answered ``direct`` for
everything would score well on a naive corpus while being useless on every
request that mattered. The metrics here are computed per label so that failure on
a minority route is visible instead of averaged away.

Every ratio is defined for a zero denominator, returning ``0.0`` rather than
raising. A label with no predictions has a precision of zero, not an undefined
one: the useful statement is "this route was never selected", and a traceback is
not a better way of saying it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrix",
    "ratio",
]


def ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or ``0.0`` when undefined.

    Args:
        numerator: The counted successes.
        denominator: The counted opportunities.

    Returns:
        The ratio, or ``0.0`` if there was nothing to divide by.
    """
    if denominator <= 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Precision, recall, and F1 for a single label."""

    label: str
    #: How many cases truly had this label.
    support: int
    #: How many cases were predicted to have it.
    predicted: int
    #: How many were both.
    true_positives: int

    @property
    def precision(self) -> float:
        """Return how often a prediction of this label was correct."""
        return ratio(self.true_positives, self.predicted)

    @property
    def recall(self) -> float:
        """Return what fraction of this label's cases were found."""
        return ratio(self.true_positives, self.support)

    @property
    def f1(self) -> float:
        """Return the harmonic mean of precision and recall."""
        precision, recall = self.precision, self.recall
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Counts of expected-versus-predicted labels.

    ``counts[expected][predicted]`` is the number of cases that were labelled
    ``expected`` and classified as ``predicted``. Keeping the full matrix rather
    than only the diagonal is what makes a systematic confusion visible — a
    router that sends every document request to research is a different bug from
    one that sends them randomly, and only the matrix distinguishes them.
    """

    labels: tuple[str, ...]
    counts: Mapping[str, Mapping[str, int]]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> ConfusionMatrix:
        """Build a matrix from ``(expected, predicted)`` pairs.

        Labels seen in the pairs but absent from ``labels`` are added, so a
        router that invents a new route shows up in the matrix instead of
        silently disappearing from it.

        Args:
            pairs: Expected and predicted labels, one per case.
            labels: The labels to track, in reporting order.

        Returns:
            The populated matrix.
        """
        ordered = list(labels)
        rows: dict[str, dict[str, int]] = {}

        for expected, predicted in pairs:
            for label in (expected, predicted):
                if label not in ordered:
                    ordered.append(label)
            row = rows.setdefault(expected, {})
            row[predicted] = row.get(predicted, 0) + 1

        filled = {
            expected: {predicted: rows.get(expected, {}).get(predicted, 0) for predicted in ordered}
            for expected in ordered
        }
        return cls(labels=tuple(ordered), counts=filled)

    @property
    def total(self) -> int:
        """Return the number of cases in the matrix."""
        return sum(sum(row.values()) for row in self.counts.values())

    @property
    def correct(self) -> int:
        """Return the number of cases classified as expected."""
        return sum(row.get(label, 0) for label, row in self.counts.items())

    @property
    def accuracy(self) -> float:
        """Return the fraction of cases classified correctly."""
        return ratio(self.correct, self.total)

    def per_label(self) -> tuple[ClassificationMetrics, ...]:
        """Return metrics for every tracked label, in reporting order."""
        results = []
        for label in self.labels:
            support = sum(self.counts.get(label, {}).values())
            predicted = sum(row.get(label, 0) for row in self.counts.values())
            results.append(
                ClassificationMetrics(
                    label=label,
                    support=support,
                    predicted=predicted,
                    true_positives=self.counts.get(label, {}).get(label, 0),
                )
            )
        return tuple(results)

    def macro_f1(self) -> float:
        """Return the unweighted mean F1 across labels with any support.

        Labels that never occur are excluded rather than counted as zero.
        Including them would make the score depend on how many routes exist
        rather than on how well the router performs.
        """
        scores = [m.f1 for m in self.per_label() if m.support > 0]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def f1_for(self, label: str) -> float:
        """Return F1 for one label, or ``0.0`` if it is not tracked.

        Args:
            label: The label to look up.

        Returns:
            The label's F1 score.
        """
        for metrics in self.per_label():
            if metrics.label == label:
                return metrics.f1
        return 0.0

    def misclassifications(self) -> tuple[tuple[str, str, int], ...]:
        """Return every off-diagonal cell, largest first.

        Returns:
            ``(expected, predicted, count)`` triples for each observed error.
        """
        errors = [
            (expected, predicted, count)
            for expected, row in self.counts.items()
            for predicted, count in row.items()
            if expected != predicted and count > 0
        ]
        return tuple(sorted(errors, key=lambda item: (-item[2], item[0], item[1])))
