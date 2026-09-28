#!/usr/bin/env python3
"""Run the quality evaluation and the performance suite.

Two numbers decide whether a change to the agent system was an improvement:
whether routing still classifies correctly, and whether the hot paths still cost
what they did. This script reports both, in that order, so a change that buys
accuracy with latency is visible as the trade it is rather than as a win.

    python scripts/evaluate.py
    python scripts/evaluate.py --iterations 500 --json report.json

The rule-based baseline is always included. It needs no credentials and no
network, so the report is reproducible anywhere the project runs, and a model
router's score is only meaningful next to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.evaluation.benchmarks import (  # noqa: E402
    BenchmarkResult,
    TimingResult,
    benchmark_suite,
    scoring_comparison,
)
from app.evaluation.harness import (  # noqa: E402
    EvaluationReport,
    RouteEvaluator,
    RuleBasedRouter,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return the parsed command line.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="timed samples per benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="also write the raw numbers to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the one-line headlines",
    )
    return parser.parse_args(argv)


def _report_markdown(
    report: EvaluationReport,
    suite: tuple[TimingResult, ...],
    comparison: BenchmarkResult,
) -> str:
    """Render the full report as Markdown.

    Args:
        report: The routing evaluation.
        suite: The measured hot paths.
        comparison: The prepared-versus-unprepared scoring comparison.

    Returns:
        The rendered Markdown.
    """
    lines = [
        "# Evaluation report",
        "",
        "## Routing",
        "",
        report.to_markdown(),
        "",
        "## Benchmarks",
        "",
        *[f"- {result.summary()}" for result in suite],
        "",
        f"- {comparison.summary()}",
        "",
    ]
    return "\n".join(lines)


def _raw_payload(
    report: EvaluationReport,
    suite: tuple[TimingResult, ...],
    comparison: BenchmarkResult,
) -> dict[str, Any]:
    """Return the machine-readable form of the report.

    Args:
        report: The routing evaluation.
        suite: The measured hot paths.
        comparison: The prepared-versus-unprepared scoring comparison.

    Returns:
        A JSON-serialisable dictionary.
    """
    return {
        "routing": {
            "subject": report.subject,
            "cases": report.total,
            "accuracy": report.accuracy,
            "adversarial_accuracy": report.adversarial_accuracy,
            "macro_f1": report.macro_f1,
            "approval_recall": report.approval_recall,
            "approval_precision": report.approval_precision,
            "latency_p50_ms": report.latency_p50_ms,
            "latency_p95_ms": report.latency_p95_ms,
            "per_route": {
                metrics.label: {
                    "support": metrics.support,
                    "predicted": metrics.predicted,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
                for metrics in report.confusion.per_label()
            },
            "mismatches": [
                {
                    "request": mismatch.request,
                    "expected": mismatch.expected,
                    "predicted": mismatch.predicted,
                    "rationale": mismatch.rationale,
                }
                for mismatch in report.mismatches
            ],
        },
        "benchmarks": {
            "suite": {
                result.name: {
                    "mean_ms": result.mean_ms,
                    "p50_ms": result.p50_ms,
                    "p95_ms": result.p95_ms,
                    "stdev_ms": result.stdev_ms,
                    "ops_per_second": result.ops_per_second,
                }
                for result in suite
            },
            "scoring": {
                "name": comparison.name,
                "equivalent": comparison.equivalent,
                "speedup": comparison.speedup,
                "baseline_mean_ms": comparison.baseline.mean_ms,
                "candidate_mean_ms": comparison.candidate.mean_ms,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and print the report.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code. Always zero: a low score is a finding, not a crash,
        and turning it into a non-zero exit would make the script unusable in a
        pipeline where the numbers are the output.
    """
    args = _parse_args(argv)
    report = asyncio.run(RouteEvaluator().evaluate(RuleBasedRouter()))
    suite = benchmark_suite(iterations=args.iterations)
    comparison = scoring_comparison(iterations=args.iterations)

    if args.quiet:
        print(report.headline())
        print(comparison.summary())
    else:
        print(_report_markdown(report, suite, comparison))

    if args.json is not None:
        args.json.write_text(
            json.dumps(_raw_payload(report, suite, comparison), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
