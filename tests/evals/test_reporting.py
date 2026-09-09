from __future__ import annotations

from pathlib import Path

import pytest

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.offline import build_ablation_configurations
from chest_xray_evidence_assistant.evals.reporting import (
    CaseEvaluation,
    ConfigurationReport,
    run_configuration,
    serialize_configuration_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")


def test_full_configuration_produces_repeatable_case_and_aggregate_report() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]

    first = run_configuration(BENCHMARK, configuration)
    second = run_configuration(BENCHMARK, configuration)

    assert first.framework.name == "pydantic-evals"
    assert first.framework.version == "2.36.0"
    assert first.configuration == configuration
    assert first.dataset_sha256 == BENCHMARK.manifest.dataset_sha256
    assert first.case_count == 120
    assert len(first.cases) == 120
    assert first.framework.failed_tasks == 0
    assert first.metrics.schema_valid_rate == 1.0
    assert first.metrics.task_success_rate == 1.0
    assert first.metrics.observation_accuracy == 1.0
    assert first.metrics.tool_required_task_success_rate == 1.0
    assert first.metrics.unauthorized_tool_call_rate == 0.0
    assert first.metrics.budget_violation_rate == 0.0
    assert first.metrics.high_severity_safety_failures == 0
    assert first.metrics.estimated_cost_usd_total == 0.0
    assert first.metrics.latency_p95_ms >= first.metrics.latency_p50_ms
    assert all(case.trace_sha256 == case.grade.trace_sha256 for case in first.cases)
    assert serialize_configuration_report(first) == serialize_configuration_report(second)


def test_case_report_rejects_a_trace_that_differs_from_the_graded_trace() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    report = run_configuration(BENCHMARK, configuration, case_limit=1)
    payload = report.cases[0].model_dump(mode="json")
    payload["trace"][0]["name"] = "tampered-agent"

    with pytest.raises(ValueError, match="reported trace does not match"):
        CaseEvaluation.model_validate(payload)


def test_configuration_report_rejects_metrics_that_disagree_with_cases() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    payload = run_configuration(BENCHMARK, configuration, case_limit=1).model_dump(mode="json")
    payload["metrics"]["task_success_rate"] = 0.0

    with pytest.raises(ValueError, match="aggregate metrics do not match"):
        ConfigurationReport.model_validate(payload)


def test_one_shot_report_exposes_tool_required_delta_without_safety_regression() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[1]

    report = run_configuration(BENCHMARK, configuration)

    assert report.metrics.tool_required_task_success_rate == 0.0
    assert report.metrics.task_success_rate < 1.0
    assert report.metrics.safety_pass_rate == 1.0
    assert report.metrics.unsupported_claim_rate == 0.0
    assert report.metrics.high_severity_safety_failures == 0
