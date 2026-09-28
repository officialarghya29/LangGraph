"""Evaluation and benchmarking for the agent system.

Two questions this package answers, and the reason both live together:

- **Is it right?** :mod:`app.evaluation.harness` scores routing against a
  hand-labelled corpus and reports per-route precision, recall, and F1, plus the
  separate question of whether irreversible requests were gated.
- **Is it fast enough?** :mod:`app.evaluation.benchmarks` times the hot paths
  with a percentile harness, so an optimisation has to prove itself against a
  measurement rather than against an impression.

Quality and cost are reported side by side deliberately. A change that improves
accuracy by adding a model call per candidate is a trade, and a trade cannot be
evaluated from one number.
"""

from app.evaluation.benchmarks import (
    BenchmarkResult,
    TimingResult,
    benchmark_suite,
    measure,
    scoring_comparison,
)
from app.evaluation.corpus import ROUTING_CORPUS, RoutingCase
from app.evaluation.harness import (
    EvaluationReport,
    RouteEvaluator,
    RouterLike,
    RuleBasedRouter,
    evaluate_routes,
    percentile,
)
from app.evaluation.metrics import ClassificationMetrics, ConfusionMatrix

__all__ = [
    "ROUTING_CORPUS",
    "BenchmarkResult",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "EvaluationReport",
    "RouteEvaluator",
    "RouterLike",
    "RoutingCase",
    "RuleBasedRouter",
    "TimingResult",
    "benchmark_suite",
    "evaluate_routes",
    "measure",
    "percentile",
    "scoring_comparison",
]
