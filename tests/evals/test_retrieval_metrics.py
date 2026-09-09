from __future__ import annotations

import os
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.offline import build_ablation_configurations
from chest_xray_evidence_assistant.evals.reporting import run_configuration
from chest_xray_evidence_assistant.evals.retrieval_metrics import (
    RetrievalMetricsReport,
    build_retrieval_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")


def test_ragas_report_is_retrieval_only_and_uses_id_precision_and_recall() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    agent_report = run_configuration(BENCHMARK, configuration)

    report = build_retrieval_report(agent_report)

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    assert report.framework.name == "ragas"
    assert report.framework.version == "0.3.9"
    assert report.metric_scope == "retrieval_only"
    assert report.safety_evaluator == "deterministic_grader"
    assert report.artifact_manifest == agent_report.artifact_manifest
    assert report.case_count == 24
    assert {case.category for case in agent_report.cases if case.case_id in report.case_ids} == {
        "retrieval"
    }
    assert report.context_precision_mean == 1.0
    assert report.context_recall_mean == 1.0
    assert report.provenance_accuracy == 1.0
    assert all(case.context_precision == 1.0 for case in report.cases)
    assert all(case.context_recall == 1.0 for case in report.cases)
    assert all(case.precision_defined for case in report.cases)
    assert all(case.retrieval_acceptable for case in report.cases)


def test_missing_retrieval_is_zero_recall_and_explicitly_undefined_precision() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[1]
    agent_report = run_configuration(BENCHMARK, configuration)

    report = build_retrieval_report(agent_report)

    assert report.context_precision_mean == 0.0
    assert report.context_recall_mean == 0.0
    assert report.provenance_accuracy == 0.0
    assert all(not case.precision_defined for case in report.cases)
    assert all(not case.retrieval_acceptable for case in report.cases)


def test_wrong_retrieval_provenance_fails_id_scores_and_acceptance() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    agent_report = run_configuration(BENCHMARK, configuration)
    retrieval_case = next(case for case in agent_report.cases if case.category == "retrieval")
    wrong_case = retrieval_case.model_copy(
        update={"retrieved_document_ids": ("document-unrelated",)}
    )
    wrong_report = agent_report.model_copy(update={"cases": (wrong_case,)})

    report = build_retrieval_report(wrong_report)

    assert report.cases[0].context_precision == 0.0
    assert report.cases[0].context_recall == 0.0
    assert report.cases[0].provenance_correct is False
    assert report.cases[0].retrieval_acceptable is False


def test_retrieval_report_rejects_aggregate_metrics_that_disagree_with_cases() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    payload = build_retrieval_report(run_configuration(BENCHMARK, configuration)).model_dump(
        mode="json"
    )
    payload["context_recall_mean"] = 0.5

    with pytest.raises(ValueError, match="aggregate metrics do not match"):
        RetrievalMetricsReport.model_validate(payload)


def test_perfect_ragas_scores_cannot_override_a_deterministic_safety_failure() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    agent_report = run_configuration(BENCHMARK, configuration)
    retrieval_case = next(case for case in agent_report.cases if case.category == "retrieval")
    unsafe_grade = retrieval_case.grade.model_copy(
        update={
            "high_severity_safety_failure": True,
            "safety_passed": False,
        }
    )
    unsafe_case = retrieval_case.model_copy(update={"grade": unsafe_grade})
    unsafe_report = agent_report.model_copy(update={"cases": (unsafe_case,)})

    report = build_retrieval_report(unsafe_report)

    assert report.cases[0].context_precision == 1.0
    assert report.cases[0].context_recall == 1.0
    assert report.cases[0].deterministic_safety_passed is False
    assert report.cases[0].retrieval_acceptable is False
