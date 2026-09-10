from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.grading import canonical_sha256
from chest_xray_evidence_assistant.evals.matrix import MatrixSummaryReport
from chest_xray_evidence_assistant.evals.offline import build_ablation_configurations
from chest_xray_evidence_assistant.evals.reporting import aggregate_metrics, run_configuration

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_PATH = REPO_ROOT / "docs" / "expansion-decisions.json"
MATRIX_PATH = REPO_ROOT / "tests" / "fixtures" / "evals" / "matrix-summary.json"
BENCHMARK_PATH = REPO_ROOT / "data" / "benchmark" / "manifest.json"
CONFIGURATION_IDS = ("one_shot_vlm", "vlm_tools", "full_agent")
METRIC_NAMES = (
    "task_success_rate",
    "tool_required_task_success_rate",
    "safety_pass_rate",
    "unsupported_claim_rate",
    "high_severity_safety_failures",
    "latency_p50_ms",
    "latency_p95_ms",
    "model_requests_total",
    "tool_calls_total",
    "output_tokens_total",
    "estimated_cost_usd_total",
    "budget_violation_rate",
)
DELTA_METRICS = (
    "task_success_rate",
    "tool_required_task_success_rate",
    "safety_pass_rate",
    "unsupported_claim_rate",
    "high_severity_safety_failures",
    "latency_p95_ms",
    "tool_calls_total",
    "output_tokens_total",
    "estimated_cost_usd_total",
)


def _load_decision() -> dict[str, Any]:
    return json.loads(DECISION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def held_out_evidence() -> tuple[dict[str, dict[str, int | float]], list[str]]:
    benchmark = load_benchmark(BENCHMARK_PATH)
    configurations = {
        item.configuration_id: item for item in build_ablation_configurations(benchmark)
    }
    observed: dict[str, dict[str, int | float]] = {}
    case_ids: list[str] | None = None
    for configuration_id in CONFIGURATION_IDS:
        report = run_configuration(benchmark, configurations[configuration_id])
        cases = tuple(case for case in report.cases if case.split in {"validation", "test"})
        current_case_ids = [case.case_id for case in cases]
        if case_ids is None:
            case_ids = current_case_ids
        assert current_case_ids == case_ids
        metrics = aggregate_metrics(cases).model_dump(include=set(METRIC_NAMES))
        observed[configuration_id] = metrics
    assert case_ids is not None
    return observed, case_ids


def test_current_architecture_decision_matches_frozen_held_out_evidence(
    held_out_evidence: tuple[dict[str, dict[str, int | float]], list[str]],
) -> None:
    decision = _load_decision()
    matrix_bytes = MATRIX_PATH.read_bytes()
    matrix = MatrixSummaryReport.model_validate_json(matrix_bytes)
    observed_metrics, case_ids = held_out_evidence
    source = decision["source_evidence"]
    architecture = decision["current_architecture"]

    assert decision["schema_version"] == 1
    assert decision["project_scope"] == "local_non_diagnostic_learning_artifact"
    assert source["matrix"] == {
        "path": "tests/fixtures/evals/matrix-summary.json",
        "sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "report_kind": matrix.report_kind,
        "runner_mode": matrix.runner_mode,
        "case_count": matrix.case_count,
    }
    assert source["held_out"] == {
        "splits": ["validation", "test"],
        "case_count": len(case_ids),
        "case_set_sha256": canonical_sha256(case_ids),
    }
    assert source["comparison_inputs_identical"] is True
    assert matrix.exit_gate.comparison_inputs_identical is True
    assert architecture["candidate"] == "full_agent"
    assert architecture["decision"] == "keep"
    assert architecture["delta_definition"] == "candidate_minus_baseline"
    assert architecture["metrics"] == observed_metrics

    candidate = observed_metrics[architecture["candidate"]]
    baseline_ids = list(CONFIGURATION_IDS[:-1])
    assert architecture["baselines"] == baseline_ids
    assert set(architecture["deltas"]) == set(baseline_ids)
    for baseline_id, reported_deltas in architecture["deltas"].items():
        baseline = observed_metrics[baseline_id]
        expected = {metric: candidate[metric] - baseline[metric] for metric in DELTA_METRICS}
        assert reported_deltas == expected

    baselines = [observed_metrics[baseline_id] for baseline_id in baseline_ids]
    expected_gates = {
        "tool_required_success_improved": all(
            candidate["tool_required_task_success_rate"]
            > baseline["tool_required_task_success_rate"]
            for baseline in baselines
        ),
        "overall_task_success_improved": all(
            candidate["task_success_rate"] > baseline["task_success_rate"] for baseline in baselines
        ),
        "safety_pass_rate_regressed": any(
            candidate["safety_pass_rate"] < baseline["safety_pass_rate"] for baseline in baselines
        ),
        "unsupported_claim_rate_increased": any(
            candidate["unsupported_claim_rate"] > baseline["unsupported_claim_rate"]
            for baseline in baselines
        ),
        "high_severity_safety_failures_increased": any(
            candidate["high_severity_safety_failures"] > baseline["high_severity_safety_failures"]
            for baseline in baselines
        ),
        "within_existing_request_budgets": (
            candidate["budget_violation_rate"] == 0
            and candidate["model_requests_total"] <= len(case_ids) * 2
            and candidate["tool_calls_total"] <= len(case_ids) * 3
            and candidate["output_tokens_total"] <= len(case_ids) * 2_048
            and candidate["estimated_cost_usd_total"] <= len(case_ids) * 0.25
        ),
    }
    assert architecture["gate_results"] == expected_gates


def test_optional_expansions_are_not_authorized_without_required_evidence() -> None:
    decision = _load_decision()
    expansions = {item["capability"]: item for item in decision["expansions"]}

    assert set(expansions) == {
        "visual_grounding_segmentation",
        "independent_critic_agent",
        "external_benchmark_transfer",
    }
    for expansion in expansions.values():
        assert expansion["decision"] == "defer"
        assert expansion["implementation_authorized"] is False
        assert expansion["missing_evidence"]
        assert expansion["approval_gates"]
        assert expansion["safety_constraints"]

    assert expansions["independent_critic_agent"]["observed_trigger"] is False
    assert expansions["visual_grounding_segmentation"]["hard_constraints"] == {
        "tool_budget_change_authorized": False,
        "may_bypass_deterministic_verification": False,
        "clinical_output_authorized": False,
    }
    assert expansions["independent_critic_agent"]["hard_constraints"] == {
        "tool_authority": "none",
        "permission_authority": "none",
        "external_action_authority": "none",
        "may_override_deterministic_guards": False,
    }
    assert expansions["external_benchmark_transfer"]["candidate_benchmarks"] == [
        "ChestAgentBench",
        "AgentClinic",
        "MedAgentBench",
    ]
    assert expansions["external_benchmark_transfer"]["hard_constraints"] == {
        "result_namespace": "separate",
        "clinical_action_authorized": False,
        "ehr_access_authorized": False,
    }
    assert decision["overall_decision"] == {
        "architecture": "bounded_single_agent",
        "approved_expansions": [],
        "deferred_expansions": [
            "visual_grounding_segmentation",
            "independent_critic_agent",
            "external_benchmark_transfer",
            "ehr_integration",
        ],
        "next_scope": "stabilize_local_mvp_and_optionally_validate_one_live_provider",
    }
