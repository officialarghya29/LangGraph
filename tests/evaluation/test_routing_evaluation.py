"""Tests for the routing evaluation harness.

Two things are being tested, and the second matters more than the first.

1. The **baseline** routes the core corpus correctly and never drops an approval
   gate. That fixes the floor a model router has to clear.
2. The **harness** detects routers that are broken. A metric that scores
   everything well measures nothing, so deliberately degenerate routers are fed
   through it and their scores are asserted to collapse in the expected way.

The adversarial failures of the baseline are asserted by name rather than by
count. If a later change fixes one, the test fails and says so — which is the
point, because the number in the report would otherwise silently go stale.
"""

from __future__ import annotations

from app.evaluation.corpus import ROUTING_CORPUS, RoutingCase
from app.evaluation.harness import (
    RouteEvaluator,
    RuleBasedRouter,
    evaluate_routes,
    percentile,
)
from app.schemas.plans import Complexity, Route, RouteDecision

#: Cases the keyword baseline gets wrong, recorded so the headroom in the report
#: is a known quantity instead of a surprise.
KNOWN_BASELINE_FAILURES = frozenset(
    {
        "How do I delete a row from the events table in SQL?",
        "Is it safe to drop an index while writes are in flight?",
        "Give me a Python function that computes the mean of a list of floats.",
        "What is a sensible way to structure retries so a flaky provider cannot "
        "exhaust the budget?",
    }
)


class _LooksUpCorpus:
    """Returns the label the corpus gives a request — a ceiling, not a router."""

    name = "corpus oracle"

    def __init__(self) -> None:
        self._by_request = {case.request: case for case in ROUTING_CORPUS}

    async def route(self, user_request: str) -> RouteDecision:
        """Return the corpus label for ``user_request``."""
        case = self._by_request[user_request]
        return RouteDecision(
            route=case.route,
            complexity=Complexity.SIMPLE,
            intent=case.route.value,
            requires_approval=case.requires_approval,
        )


class _AlwaysDirect:
    """The majority-class predictor: answers ``direct`` to everything."""

    name = "always direct"

    async def route(self, user_request: str) -> RouteDecision:
        """Return a trivial direct decision for any request."""
        return RouteDecision(route=Route.DIRECT, intent="direct")


class _AlwaysApproving:
    """Predicts the right route but asks for approval every time."""

    name = "over-approving"

    def __init__(self) -> None:
        self._inner = _LooksUpCorpus()

    async def route(self, user_request: str) -> RouteDecision:
        """Return the corpus route, forced through the approval gate."""
        decision = await self._inner.route(user_request)
        decision.requires_approval = True
        return decision


class _Raising:
    """Raises on every request, to prove a broken subject is contained."""

    name = "broken"

    async def route(self, user_request: str) -> RouteDecision:
        """Fail, always."""
        raise RuntimeError("provider unavailable")


class _EchoesOneRoute:
    """Always answers the same non-trivial route."""

    name = "stuck"

    async def route(self, user_request: str) -> RouteDecision:
        """Return ``coding`` for everything."""
        return RouteDecision(route=Route.CODING, intent="coding")


# --------------------------------------------------------------------------- #
# The baseline
# --------------------------------------------------------------------------- #


async def test_the_baseline_is_perfect_on_the_core_corpus() -> None:
    """Core cases are the ordinary shapes; the baseline must handle all of them."""
    report = await evaluate_routes(RuleBasedRouter())

    core_total = report.total - report.adversarial_total
    core_correct = report.confusion.correct - report.adversarial_correct

    assert core_total > 0
    assert core_correct == core_total, report.to_markdown()


async def test_the_baseline_never_drops_an_approval_gate() -> None:
    """Recall on irreversible requests must be total, whatever else slips."""
    report = await evaluate_routes(RuleBasedRouter())

    assert report.approval_recall == 1.0
    assert report.missed_approvals == ()


async def test_the_baseline_over_gates_a_question_about_deleting() -> None:
    """A destructive verb inside a question is a known false positive.

    Treating "how do I delete a row" as a request to delete one is wrong, and it
    is exactly the kind of mistake that makes an approval gate feel like noise.
    Asserting it as a known failure keeps it visible rather than averaging it
    into a headline number.
    """
    report = await evaluate_routes(RuleBasedRouter())

    assert report.approval_precision < 1.0
    requests = {mismatch.request for mismatch in report.mismatches if mismatch.approval_expected}
    assert not requests, "the baseline should never miss a gate, only add one"
    predicted_gates = {
        mismatch.request
        for mismatch in report.mismatches
        if mismatch.approval_predicted and not mismatch.approval_expected
    }
    assert predicted_gates, "the over-gating false positive has been fixed; update this test"


async def test_the_baseline_adversarial_failures_are_exactly_the_known_set() -> None:
    """Adversarial headroom is a tracked number, not an anonymous deficit."""
    report = await evaluate_routes(RuleBasedRouter())

    failed = {mismatch.request for mismatch in report.mismatches}

    assert failed == KNOWN_BASELINE_FAILURES
    assert report.adversarial_accuracy < 1.0


async def test_the_baseline_beats_the_majority_class_clearly() -> None:
    """A degenerate router must score far below the baseline, not near it."""
    evaluator = RouteEvaluator()
    baseline = await evaluator.evaluate(RuleBasedRouter())
    majority = await evaluator.evaluate(_AlwaysDirect())

    assert majority.accuracy < baseline.accuracy / 2
    assert majority.macro_f1 < 0.2


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


async def test_a_perfect_router_scores_perfectly() -> None:
    """The metric must be able to reach the top of its range."""
    report = await evaluate_routes(_LooksUpCorpus())

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.approval_recall == 1.0
    assert report.adversarial_accuracy == 1.0
    assert report.mismatches == ()


async def test_a_broken_router_is_recorded_rather_than_aborting_the_run() -> None:
    """One failing subject must not prevent the other cases being scored."""
    report = await evaluate_routes(_Raising())

    assert report.total == len(ROUTING_CORPUS)
    assert report.accuracy == 0.0
    assert len(report.mismatches) == len(ROUTING_CORPUS)


async def test_a_router_stuck_on_one_route_is_caught_by_macro_f1() -> None:
    """Accuracy can flatter a biased router; macro F1 does not."""
    report = await evaluate_routes(_EchoesOneRoute())

    assert report.confusion.f1_for("coding") < 0.5
    assert report.macro_f1 < 0.2


async def test_dropping_gates_is_reported_separately_from_accuracy() -> None:
    """The approval matrix catches what routing accuracy would hide.

    ``direct`` is a plausible route for a badly behaving router, so the approval
    score has to be computed from the approval flag rather than inferred from the
    route.
    """
    report = await evaluate_routes(_AlwaysDirect())

    assert report.approval_recall == 0.0
    assert len(report.missed_approvals) > 0
    assert all(mismatch.approval_miss for mismatch in report.missed_approvals)


async def test_over_approving_lowers_precision_without_touching_recall() -> None:
    """Gating everything is recalled perfectly and is still wrong."""
    report = await evaluate_routes(_AlwaysApproving())

    assert report.approval_recall == 1.0
    assert report.approval_precision < 0.5
    assert report.accuracy == 1.0


async def test_evaluation_is_deterministic() -> None:
    """Two runs over a deterministic router give identical counts."""
    evaluator = RouteEvaluator()
    first = await evaluator.evaluate(RuleBasedRouter())
    second = await evaluator.evaluate(RuleBasedRouter())

    assert first.confusion.counts == second.confusion.counts
    assert first.accuracy == second.accuracy
    assert first.mismatches == second.mismatches


async def test_the_report_renders_the_numbers_it_computed() -> None:
    """The Markdown must carry the score, the routes, and the confusions."""
    report = await evaluate_routes(RuleBasedRouter())
    markdown = report.to_markdown()

    assert "Adversarial accuracy" in markdown
    assert f"{report.accuracy:.3f}" in markdown
    assert "`human_approval`" in markdown
    assert "Most common confusions" in markdown
    assert "| Route | Support | Predicted | Precision | Recall | F1 |" in markdown


async def test_dropped_gates_are_named_in_the_report() -> None:
    """A missed gate is the one failure that must be shouted about."""
    markdown = (await evaluate_routes(_AlwaysDirect())).to_markdown()

    assert "Dropped approval gates" in markdown


async def test_the_subject_name_is_recorded() -> None:
    """Reports are compared side by side, so each must say who it measured."""
    report = await evaluate_routes(RuleBasedRouter(), subject="baseline v1")

    assert report.subject == "baseline v1"
    assert report.headline().startswith("baseline v1:")


def test_percentile_returns_an_observed_value() -> None:
    """Nearest-rank keeps latency percentiles to numbers that occurred."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 0.5) in values
    assert percentile([], 0.5) == 0.0


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #


def test_every_case_carries_a_rationale() -> None:
    """A label is only defensible if the reason is written next to it."""
    missing = [case.request for case in ROUTING_CORPUS if not case.rationale.strip()]

    assert not missing


def test_every_route_has_several_cases() -> None:
    """One case per route would measure luck, not behaviour."""
    counts: dict[str, int] = {}
    for case in ROUTING_CORPUS:
        counts[case.route.value] = counts.get(case.route.value, 0) + 1

    thin = {route: count for route, count in counts.items() if count < 2}
    assert not thin, thin


def test_adversarial_cases_are_a_minority() -> None:
    """Accuracy should describe the ordinary case, with the hard cases separate."""
    adversarial = [case for case in ROUTING_CORPUS if case.adversarial]

    assert 0 < len(adversarial) < len(ROUTING_CORPUS) / 2


def test_adversarial_requests_are_unique() -> None:
    """A duplicated request would silently double its weight in the score."""
    requests = [case.request for case in ROUTING_CORPUS]

    assert len(requests) == len(set(requests))


def test_approval_cases_are_labelled_as_such() -> None:
    """The approval flag and the approval route must agree in the corpus."""
    inconsistent = [
        case
        for case in ROUTING_CORPUS
        if case.requires_approval is not (case.route is Route.HUMAN_APPROVAL)
    ]

    assert not inconsistent, [case.request for case in inconsistent]


def test_a_custom_corpus_can_be_evaluated() -> None:
    """The evaluator accepts a narrowing of the corpus, for focused runs."""
    case = RoutingCase(
        request="Delete everything.",
        route=Route.HUMAN_APPROVAL,
        rationale="destructive",
        requires_approval=True,
    )
    evaluator = RouteEvaluator(cases=[case], labels=["human_approval"])

    assert evaluator.cases == (case,)
