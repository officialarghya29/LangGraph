"""The routing evaluation harness.

An evaluation is only useful if its baseline is honest. This module therefore
ships a **deterministic rule-based router** and measures it against the corpus.
The baseline is not a straw man: it encodes the same guidance the model router is
given, it is ordered so that irreversible requests are classified first, and it
carries a context guard so that a destructive verb inside a description of a test
is not mistaken for a request to destroy something.

Two things follow from having it:

- **The metric has a floor.** A model router that cannot beat keyword matching
  on this corpus is not earning its latency, and that is now a measurable claim
  rather than an opinion.
- **The harness itself is testable.** Feeding it a deliberately wrong router must
  move the score in the expected direction. A metric that cannot detect a broken
  subject measures nothing, so that is asserted in the test suite rather than
  assumed.

The evaluator is provider-agnostic: it takes anything with ``route(request)``, so
the same corpus scores the rule-based baseline, a fake provider, and a real one.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.evaluation.corpus import ROUTING_CORPUS, RoutingCase
from app.evaluation.corpus import labels as corpus_labels
from app.evaluation.metrics import ConfusionMatrix, ratio
from app.schemas.plans import Complexity, Route, RouteDecision

__all__ = [
    "EvaluationReport",
    "Mismatch",
    "RouteEvaluator",
    "RouterLike",
    "RuleBasedRouter",
    "evaluate_routes",
    "percentile",
]

#: The two labels used for the approval matrix. Missing an approval gate is a
#: different and more serious failure than choosing the wrong specialist, so it
#: is scored separately from routing accuracy.
_APPROVAL_LABELS: tuple[str, ...] = ("approval", "no_approval")

_APPROVAL = "approval"
_NO_APPROVAL = "no_approval"


@runtime_checkable
class RouterLike(Protocol):
    """Anything that can classify a request into a route decision."""

    async def route(self, user_request: str) -> RouteDecision:
        """Classify ``user_request``."""
        ...


# --------------------------------------------------------------------------- #
# The baseline router
# --------------------------------------------------------------------------- #

#: Verbs that ask for something irreversible. Matched first, because a request
#: that destroys data is never *only* a coding request.
_IRREVERSIBLE = re.compile(
    r"\b("
    r"delete|drop|wipe|purge|truncate|erase|destroy|revoke|uninstall|"
    r"force[-\s]?push|reset\s+--hard|rotate\s+the|rotate\s+our|rekey"
    r")\b",
    re.IGNORECASE,
)

#: Mentions of testing or fixtures. A destructive verb inside one of these is
#: normally describing what a test *does*, not asking for it to be done. Without
#: this guard the baseline classifies "a fixture that truncates tables" as a
#: request to truncate tables, which is the single most common false positive a
#: keyword router makes.
_DESCRIPTIVE = re.compile(
    r"\b(test|tests|pytest|fixture|fixtures|mock|stub|example|reproduce|repro)\b",
    re.IGNORECASE,
)

#: Domain markers. A request containing markers from three or more domains is a
#: genuinely multi-specialist job; fewer than three is a single-specialist job
#: with some incidental vocabulary.
_DOMAIN_MARKERS: dict[str, re.Pattern[str]] = {
    "research": re.compile(
        r"\b("
        r"research|investigate|sources?|survey|benchmark|comparing|compare|"
        r"comparison|latest|state of the art|who maintains|what changed|guidance|"
        r"find out|look up"
        r")\b",
        re.IGNORECASE,
    ),
    "document": re.compile(
        r"\b("
        r"report|readme|release notes|changelog|decision record|runbook|"
        r"document|documents|documentation|docs|guide|memo|proposal|spec|"
        r"write[-\s]?up|action list|one[-\s]page|incident note|notes|"
        r"extract|table of|into a table"
        r")\b",
        re.IGNORECASE,
    ),
    # ``mean`` is matched only where it is a statistic and not an English verb.
    # "What does idempotent mean?" is a definition, and a bare ``mean`` marker
    # classified it as data analysis until this corpus caught it.
    "data_analysis": re.compile(
        r"("
        r"\b("
        r"dataset|median|averages?|percentile|correlation|regression|"
        r"compute|calculate|aggregate|group|distribution|growth rate|"
        r"forecast|projection|profile|statistics|trend|csv|chart|histogram"
        r")\b|"
        r"\bmean\s+(?:of|value|duration|latency|time|cost|score|absolute|squared)\b"
        r")",
        re.IGNORECASE,
    ),
    "coding": re.compile(
        r"\b("
        r"implement|refactor|debug|fix|patch|traceback|exception|function|"
        r"class|method|endpoint|middleware|migration|schema|index|pytest|"
        r"fixture|code|coding|unit tests?|regression tests?|api|wire it|"
        r"script|module|handler"
        r")\b",
        re.IGNORECASE,
    ),
}

#: Single-domain resolution order. Longest-range capability first, because a
#: request that needs external evidence must gather it before anything else can
#: be decided from it.
_DOMAIN_ROUTES: tuple[tuple[str, Route], ...] = (
    ("research", Route.RESEARCH),
    ("document", Route.DOCUMENT),
    ("data_analysis", Route.DATA_ANALYSIS),
    ("coding", Route.CODING),
)

#: How many distinct domains must appear before a request counts as multi-agent.
_MULTI_AGENT_DOMAINS = 3

_COMPLEXITY_FOR: dict[Route, Complexity] = {
    Route.DIRECT: Complexity.SIMPLE,
    Route.RESEARCH: Complexity.MODERATE,
    Route.CODING: Complexity.MODERATE,
    Route.DATA_ANALYSIS: Complexity.MODERATE,
    Route.DOCUMENT: Complexity.MODERATE,
    Route.MULTI_AGENT: Complexity.COMPLEX,
    Route.HUMAN_APPROVAL: Complexity.SIMPLE,
}


class RuleBasedRouter:
    """A deterministic keyword router, used as the evaluation baseline.

    It satisfies the same contract as the model router, so it can be scored by
    the same harness and swapped in wherever a router is expected.
    """

    #: Human-readable name used in reports.
    name = "rule-based baseline"

    async def route(self, user_request: str) -> RouteDecision:
        """Classify a request with ordered rules.

        Args:
            user_request: The raw request text.

        Returns:
            A validated decision. Never raises, so the baseline never benefits
            from a fallback path the model router does not have.
        """
        route = self._classify(user_request)
        return RouteDecision(
            route=route,
            complexity=_COMPLEXITY_FOR[route],
            intent=route.value,
            reasoning_summary=f"matched {route.value} rules",
        )

    def _classify(self, text: str) -> Route:
        """Return the route the rules select for ``text``."""
        if _IRREVERSIBLE.search(text) and not _DESCRIPTIVE.search(text):
            return Route.HUMAN_APPROVAL

        domains = [name for name, pattern in _DOMAIN_MARKERS.items() if pattern.search(text)]
        if len(domains) >= _MULTI_AGENT_DOMAINS:
            return Route.MULTI_AGENT

        for name, route in _DOMAIN_ROUTES:
            if name in domains:
                return route

        return Route.DIRECT


# --------------------------------------------------------------------------- #
# The evaluator
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` percentile of ``values``.

    Uses nearest-rank rather than interpolation, so the returned number is always
    an observation that actually occurred. For latency reporting that matters:
    an interpolated p95 that no request ever experienced is a fiction.

    Args:
        values: The observations. Need not be sorted.
        fraction: The percentile to take, between 0.0 and 1.0.

    Returns:
        The selected observation, or ``0.0`` when there are none.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One case the system under test classified differently from the corpus."""

    request: str
    expected: str
    predicted: str
    rationale: str
    approval_expected: bool
    approval_predicted: bool

    @property
    def approval_miss(self) -> bool:
        """Return whether this mismatch lost an approval gate."""
        return self.approval_expected and not self.approval_predicted


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """The outcome of scoring one router against the corpus."""

    subject: str
    confusion: ConfusionMatrix
    approval: ConfusionMatrix
    latencies_ms: tuple[float, ...]
    mismatches: tuple[Mismatch, ...] = field(default_factory=tuple)
    #: Adversarial cases evaluated, and how many were routed correctly. Reported
    #: separately so easy shapes cannot mask hard ones.
    adversarial_total: int = 0
    adversarial_correct: int = 0

    @property
    def adversarial_accuracy(self) -> float:
        """Return accuracy over the adversarial cases, or ``0.0`` if there are none."""
        return ratio(self.adversarial_correct, self.adversarial_total)

    @property
    def total(self) -> int:
        """Return the number of cases evaluated."""
        return self.confusion.total

    @property
    def accuracy(self) -> float:
        """Return the fraction of cases routed correctly."""
        return self.confusion.accuracy

    @property
    def macro_f1(self) -> float:
        """Return the unweighted mean F1 across routes."""
        return self.confusion.macro_f1()

    @property
    def approval_recall(self) -> float:
        """Return how many irreversible requests were gated.

        This is the number that matters most: a request that should have been
        gated and was not has already done its damage by the time anyone reads
        the report.
        """
        for metrics in self.approval.per_label():
            if metrics.label == _APPROVAL:
                return metrics.recall
        return 0.0

    @property
    def approval_precision(self) -> float:
        """Return how often a gated request genuinely needed gating."""
        for metrics in self.approval.per_label():
            if metrics.label == _APPROVAL:
                return metrics.precision
        return 0.0

    @property
    def missed_approvals(self) -> tuple[Mismatch, ...]:
        """Return the cases where an approval gate was dropped."""
        return tuple(m for m in self.mismatches if m.approval_miss)

    @property
    def latency_p50_ms(self) -> float:
        """Return the median per-request latency in milliseconds."""
        return percentile(self.latencies_ms, 0.5)

    @property
    def latency_p95_ms(self) -> float:
        """Return the 95th percentile per-request latency in milliseconds."""
        return percentile(self.latencies_ms, 0.95)

    def headline(self) -> str:
        """Return a one-line summary suitable for a log or a console."""
        return (
            f"{self.subject}: accuracy={self.accuracy:.3f} "
            f"adversarial={self.adversarial_accuracy:.3f} "
            f"macro_f1={self.macro_f1:.3f} "
            f"approval_recall={self.approval_recall:.3f} "
            f"p50={self.latency_p50_ms:.2f}ms p95={self.latency_p95_ms:.2f}ms"
        )

    def to_markdown(self) -> str:
        """Render the report as a Markdown section.

        Returns:
            A table of per-route metrics, the approval matrix, the most common
            confusions, and any dropped approval gates.
        """
        lines = [
            f"### {self.subject}",
            "",
            f"- **Accuracy:** {self.accuracy:.3f} ({self.confusion.correct}/{self.total})",
            f"- **Adversarial accuracy:** {self.adversarial_accuracy:.3f} "
            f"({self.adversarial_correct}/{self.adversarial_total})",
            f"- **Macro F1:** {self.macro_f1:.3f}",
            f"- **Approval recall:** {self.approval_recall:.3f}",
            f"- **Latency:** p50 {self.latency_p50_ms:.2f} ms, p95 {self.latency_p95_ms:.2f} ms",
            "",
            "| Route | Support | Predicted | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for metrics in self.confusion.per_label():
            lines.append(
                f"| `{metrics.label}` | {metrics.support} | {metrics.predicted} | "
                f"{metrics.precision:.3f} | {metrics.recall:.3f} | {metrics.f1:.3f} |"
            )

        confusions = self.confusion.misclassifications()
        if confusions:
            lines += ["", "**Most common confusions**", ""]
            lines += [
                f"- `{expected}` → `{predicted}`: {count}"
                for expected, predicted, count in confusions[:5]
            ]

        if self.missed_approvals:
            lines += ["", "**Dropped approval gates**", ""]
            lines += [
                f"- {mismatch.request!r} — expected `{mismatch.expected}`, got "
                f"`{mismatch.predicted}`"
                for mismatch in self.missed_approvals
            ]

        return "\n".join(lines)


class RouteEvaluator:
    """Scores a router against a labelled corpus."""

    def __init__(
        self,
        cases: Iterable[RoutingCase] = ROUTING_CORPUS,
        *,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._cases = tuple(cases)
        self._labels = tuple(labels) if labels is not None else corpus_labels()

    @property
    def cases(self) -> tuple[RoutingCase, ...]:
        """Return the corpus being evaluated."""
        return self._cases

    async def evaluate(self, router: RouterLike, *, subject: str | None = None) -> EvaluationReport:
        """Score ``router`` over the corpus.

        Routes are evaluated sequentially rather than concurrently. The point of
        the timing numbers is to describe the routing cost a single request pays,
        and running candidates in parallel would make those numbers describe the
        host's scheduler instead.

        Args:
            router: Anything implementing :class:`RouterLike`.
            subject: Name for the report. Defaults to the router's own name.

        Returns:
            The report. Cases that raise are recorded as a ``error`` prediction
            rather than aborting the run, so one broken case cannot hide the
            score of the rest.
        """
        pairs: list[tuple[str, str]] = []
        approval_pairs: list[tuple[str, str]] = []
        latencies: list[float] = []
        mismatches: list[Mismatch] = []
        adversarial_total = 0
        adversarial_correct = 0

        for case in self._cases:
            started = time.perf_counter()
            # A case that raises is recorded as a failed prediction rather than
            # aborting the run: one broken case must not hide the score of the
            # other thirty-nine.
            try:
                decision = await router.route(case.request)
            except Exception:
                decision = None
            latencies.append((time.perf_counter() - started) * 1000)

            predicted = decision.route.value if decision is not None else "error"
            approval_predicted = bool(decision.requires_approval) if decision is not None else False

            pairs.append((case.route.value, predicted))
            approval_pairs.append(
                (
                    _APPROVAL if case.requires_approval else _NO_APPROVAL,
                    _APPROVAL if approval_predicted else _NO_APPROVAL,
                )
            )

            if case.adversarial:
                adversarial_total += 1
                adversarial_correct += int(predicted == case.route.value)

            if predicted != case.route.value or approval_predicted != case.requires_approval:
                mismatches.append(
                    Mismatch(
                        request=case.request,
                        expected=case.route.value,
                        predicted=predicted,
                        rationale=case.rationale,
                        approval_expected=case.requires_approval,
                        approval_predicted=approval_predicted,
                    )
                )

        resolved = subject or getattr(router, "name", None) or type(router).__name__
        return EvaluationReport(
            subject=str(resolved),
            confusion=ConfusionMatrix.from_pairs(pairs, self._labels),
            approval=ConfusionMatrix.from_pairs(approval_pairs, _APPROVAL_LABELS),
            latencies_ms=tuple(latencies),
            mismatches=tuple(mismatches),
            adversarial_total=adversarial_total,
            adversarial_correct=adversarial_correct,
        )

    async def compare(
        self, routers: Iterable[tuple[str, RouterLike]]
    ) -> tuple[EvaluationReport, ...]:
        """Score several routers and return their reports in the given order.

        Args:
            routers: ``(subject, router)`` pairs.

        Returns:
            One report per router.
        """
        return tuple([await self.evaluate(router, subject=name) for name, router in routers])


async def evaluate_routes(router: RouterLike, *, subject: str | None = None) -> EvaluationReport:
    """Score one router against the default corpus.

    Args:
        router: Anything implementing :class:`RouterLike`.
        subject: Name for the report.

    Returns:
        The report.
    """
    return await RouteEvaluator().evaluate(router, subject=subject)


def run(router: RouterLike) -> EvaluationReport:
    """Score a router from synchronous code.

    Args:
        router: Anything implementing :class:`RouterLike`.

    Returns:
        The report.
    """
    return asyncio.run(evaluate_routes(router))


#: Re-exported so callers can build a corpus-scoped evaluator without importing
#: the corpus module directly.
__all__ += ["ROUTING_CORPUS", "RoutingCase"]
