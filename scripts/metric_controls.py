#!/usr/bin/env python3
"""Synthetic checks of the pinned official BEIR evaluator.

This module never opens a dataset or test judgments.  It deliberately checks
the installed evaluator rather than replacing it with our own implementation.
The benchmark runner must separately reject omitted judged query IDs before
calling EvaluateRetrieval.evaluate: BEIR averages only the queries returned by
pytrec_eval.  The omission control below demonstrates that failure mode.

Sources for the graded-gain expectation:
  https://github.com/terrierteam/pytrec_eval/blob/v0.5.10/setup.py
  https://github.com/usnistgov/trec_eval/blob/v9.0.8/m_ndcg_cut.c
The former identifies trec_eval v9.0.8; the latter uses relevance itself as gain.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from importlib.metadata import version
from pathlib import Path
from typing import Any


EXPECTED_VERSIONS = {"beir": "2.2.0", "pytrec-eval-terrier": "0.5.10"}
K_VALUES = [10, 100]


def _scores(document_ids: list[str]) -> dict[str, float]:
    """Give an explicit order distinct descending scores, with no score ties."""
    return {doc_id: float(len(document_ids) - i) for i, doc_id in enumerate(document_ids)}


def _check_case(
    evaluator: Any,
    name: str,
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    expected_ndcg: float,
    expected_recall: float,
    explanation: str,
) -> dict[str, Any]:
    # Explicit False is essential: SciFact query IDs and document IDs are
    # unrelated namespaces and a coincidentally equal ID must remain retrieved.
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels, results, K_VALUES, ignore_identical_ids=False
    )
    observed = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"]}
    expected_exact = {"NDCG@10": expected_ndcg, "Recall@100": expected_recall}
    expected_rounded = {key: round(value, 5) for key, value in expected_exact.items()}
    for metric, expected in expected_rounded.items():
        assert math.isclose(observed[metric], expected, rel_tol=0.0, abs_tol=1e-10), (
            f"{name}: {metric}: observed {observed[metric]!r}, expected {expected!r}"
        )
    return {
        "name": name,
        "passed": True,
        "explanation": explanation,
        "qrels": qrels,
        "results": results,
        "expected_exact": expected_exact,
        "expected_beir_rounded_5dp": expected_rounded,
        "observed": observed,
        "official_all_metrics": {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
        },
    }


def run_controls() -> dict[str, Any]:
    """Assert synthetic metric expectations and return JSON-serializable evidence.

    Dependencies must already be installed in the project virtual environment.
    No package installation, network access, dataset read, or file write occurs.
    """
    from beir.retrieval.evaluation import EvaluateRetrieval

    observed_versions = {name: version(name) for name in EXPECTED_VERSIONS}
    assert observed_versions == EXPECTED_VERSIONS, (
        f"Evaluator dependency versions differ from the protocol: {observed_versions!r}"
    )

    discount2 = 1.0 / math.log2(3.0)
    cases = [
        _check_case(
            EvaluateRetrieval,
            "perfect_including_equal_query_document_id",
            {"same-id": {"same-id": 1}},
            {"same-id": {"same-id": 1.0}},
            1.0,
            1.0,
            "One relevant document at rank 1; identical IDs must not be removed.",
        ),
        _check_case(
            EvaluateRetrieval,
            "complete_miss",
            {"q-miss": {"relevant": 1, "judged-nonrelevant": 0}},
            {"q-miss": {"unjudged-document": 1.0}},
            0.0,
            0.0,
            "An unjudged retrieved document does not count as relevant.",
        ),
        _check_case(
            EvaluateRetrieval,
            "single_relevant_at_rank_2",
            {"q-rank2": {"relevant": 1}},
            {"q-rank2": _scores(["nonrelevant", "relevant"])},
            discount2,
            1.0,
            "nDCG@10 = 1/log2(3), while the document is within Recall@100.",
        ),
        _check_case(
            EvaluateRetrieval,
            "single_relevant_at_rank_100",
            {"q-rank100": {"relevant": 1}},
            {"q-rank100": _scores([f"filler-{i:03d}" for i in range(99)] + ["relevant"])},
            0.0,
            1.0,
            "Rank 100 is included by Recall@100 and excluded by nDCG@10.",
        ),
        _check_case(
            EvaluateRetrieval,
            "single_relevant_at_rank_101",
            {"q-rank101": {"relevant": 1}},
            {"q-rank101": _scores([f"filler-{i:03d}" for i in range(100)] + ["relevant"])},
            0.0,
            0.0,
            "Rank 101 is outside both requested cutoffs.",
        ),
        _check_case(
            EvaluateRetrieval,
            "multiple_relevant_recall_denominator",
            {"q-multi": {"relevant-a": 1, "relevant-b": 1, "relevant-c": 1, "zero": 0}},
            {"q-multi": _scores(["relevant-a", "relevant-b"])},
            (1.0 + discount2) / (1.0 + discount2 + 1.0 / math.log2(4.0)),
            2.0 / 3.0,
            "Recall denominator includes all three positive judgments, including the missing document.",
        ),
        _check_case(
            EvaluateRetrieval,
            "query_macro_average",
            {"q-perfect": {"a": 1}, "q-miss": {"b": 1, "c": 1, "d": 1}},
            {"q-perfect": {"a": 1.0}, "q-miss": {"unjudged": 1.0}},
            0.5,
            0.5,
            "Official aggregation weights the two evaluated queries equally.",
        ),
        _check_case(
            EvaluateRetrieval,
            "graded_linear_gain_reversed_ranking",
            {"q-graded": {"grade-3": 3, "grade-1": 1}},
            {"q-graded": _scores(["grade-1", "grade-3"])},
            (1.0 + 3.0 * discount2) / (3.0 + discount2),
            1.0,
            "trec_eval ndcg_cut uses linear gains 3 and 1, rather than 2**rel-1.",
        ),
    ]

    graded = cases[-1]
    exponential_gain_alternative = (1.0 + 7.0 * discount2) / (7.0 + discount2)
    assert not math.isclose(
        graded["observed"]["NDCG@10"],
        round(exponential_gain_alternative, 5),
        rel_tol=0.0,
        abs_tol=1e-5,
    ), "The graded control must distinguish linear from exponential gains."
    graded["rejected_exponential_gain_alternative"] = exponential_gain_alternative

    omitted_qrels = {"q-present": {"relevant-a": 1}, "q-omitted": {"relevant-b": 1}}
    omission_case = _check_case(
        EvaluateRetrieval,
        "missing_query_is_not_automatically_scored_as_zero",
        omitted_qrels,
        {"q-present": {"relevant-a": 1.0}},
        1.0,
        1.0,
        "This deliberately incomplete synthetic run demonstrates the need for the runner's coverage guard.",
    )
    complete_case = _check_case(
        EvaluateRetrieval,
        "missing_query_represented_as_a_complete_miss",
        omitted_qrels,
        {"q-present": {"relevant-a": 1.0}, "q-omitted": {"unjudged": 1.0}},
        0.5,
        0.5,
        "When both query IDs are supplied, the missing relevant document contributes zero.",
    )
    assert omission_case["observed"]["Recall@100"] > complete_case["observed"]["Recall@100"]

    evaluator_source = Path(inspect.getfile(EvaluateRetrieval))
    return {
        "passed": True,
        "scope": "Synthetic official-evaluator controls only; no SciFact data or judgments were opened.",
        "official_evaluator": "beir.retrieval.evaluation.EvaluateRetrieval.evaluate",
        "ignore_identical_ids": False,
        "k_values": K_VALUES,
        "dependency_versions": observed_versions,
        "evaluator_source_sha256": hashlib.sha256(evaluator_source.read_bytes()).hexdigest(),
        "rounding": "BEIR aggregate outputs are rounded to 5 decimal places.",
        "graded_gain_convention": "linear relevance gain; discount 1/log2(rank+1)",
        "graded_gain_sources": [
            "https://github.com/terrierteam/pytrec_eval/blob/v0.5.10/setup.py",
            "https://github.com/usnistgov/trec_eval/blob/v9.0.8/m_ndcg_cut.c",
        ],
        "metric_cases": cases,
        "query_coverage_failure_demonstration": {
            "external_guard_required": True,
            "required_guard": "Assert exact qrels/results query-ID coverage and nonempty retrievals before official evaluation.",
            "incomplete_run": omission_case,
            "complete_miss_run": complete_case,
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_controls(), indent=2, sort_keys=True, allow_nan=False))
