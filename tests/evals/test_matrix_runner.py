from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.matrix import (
    MatrixSummaryReport,
    run_matrix,
    write_matrix_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data" / "benchmark" / "cxr-agent-bench-v0.jsonl"
BENCHMARK = load_benchmark(DATASET_PATH.parent / "manifest.json")
SUMMARY_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "evals" / "matrix-summary.json"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


def test_fixed_matrix_uses_identical_comparison_inputs_and_passes_exit_gate() -> None:
    bundle = run_matrix(BENCHMARK)
    summary = bundle.summary

    assert [item.configuration_id for item in summary.configurations] == [
        "text_only",
        "one_shot_vlm",
        "vlm_tools",
        "vlm_rag",
        "full_agent",
    ]
    assert len(bundle.agent_reports) == 5
    assert len(bundle.retrieval_reports) == 5
    assert len(bundle.manifest_index.manifests) == 5
    assert tuple(item.artifact_manifest_sha256 for item in summary.configurations) == tuple(
        manifest.manifest_sha256 for manifest in bundle.manifest_index.manifests
    )
    assert summary.runner_mode == "offline_scripted"
    assert summary.token_accounting == "alphanumeric-span-estimate-v1"
    assert summary.latency_accounting == "deterministic-estimate-v1"
    assert summary.frameworks.pydantic_evals == "2.36.0"
    assert summary.frameworks.ragas == "0.3.9"
    assert {report.case_set_sha256 for report in bundle.agent_reports} == {
        bundle.agent_reports[0].case_set_sha256
    }
    assert {report.configuration.seed_set_sha256 for report in bundle.agent_reports} == {
        bundle.agent_reports[0].configuration.seed_set_sha256
    }
    assert {report.configuration.limits_sha256 for report in bundle.agent_reports} == {
        bundle.agent_reports[0].configuration.limits_sha256
    }
    assert {report.configuration.prompt_set_sha256 for report in bundle.agent_reports} == {
        bundle.agent_reports[0].configuration.prompt_set_sha256
    }
    assert {report.configuration.rubric_sha256 for report in bundle.agent_reports} == {
        bundle.agent_reports[0].configuration.rubric_sha256
    }
    assert summary.exit_gate.full_agent_beats_one_shot_on_tool_required is True
    assert summary.exit_gate.full_agent_unauthorized_tool_calls_zero is True
    assert summary.exit_gate.full_agent_high_severity_safety_failures_zero is True
    assert summary.exit_gate.comparison_inputs_identical is True
    assert summary.exit_gate.passed is True
    assert all(
        report.configuration.provider == "offline-scripted"
        and report.configuration.model == "deterministic-capability-policy"
        and report.configuration.token_accounting == "alphanumeric-span-estimate-v1"
        and report.configuration.latency_accounting == "deterministic-estimate-v1"
        for report in bundle.agent_reports
    )
    assert MatrixSummaryReport.model_validate_json(SUMMARY_FIXTURE.read_bytes()) == summary


def test_matrix_reports_bounded_performance_and_cost_samples() -> None:
    summary = MatrixSummaryReport.model_validate_json(SUMMARY_FIXTURE.read_bytes())

    for configuration in summary.configurations:
        metrics = configuration.agent_metrics
        assert metrics.latency_p95_ms >= metrics.latency_p50_ms
        assert metrics.model_requests_total <= summary.case_count * 2
        assert metrics.tool_calls_total <= summary.case_count * 3
        assert metrics.output_tokens_total <= summary.case_count * 2_048
        assert metrics.estimated_cost_usd_total <= summary.case_count * 0.25
        assert metrics.budget_violation_rate == 0.0


def test_matrix_bundle_is_machine_readable_and_byte_repeatable(tmp_path: Path) -> None:
    bundle = run_matrix(BENCHMARK)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first_summary = write_matrix_bundle(bundle, first_root)
    second_summary = write_matrix_bundle(bundle, second_root)

    assert first_summary == first_root / "matrix-summary.json"
    assert second_summary == second_root / "matrix-summary.json"
    assert len(_tree_hashes(first_root)) == 12
    assert _tree_hashes(first_root) == _tree_hashes(second_root)
    loaded = MatrixSummaryReport.model_validate_json(first_summary.read_bytes())
    assert loaded == bundle.summary
    assert (first_root / "evaluation-manifest.json").is_file()
    serialized = first_summary.read_text(encoding="utf-8").casefold()
    assert "raw_prompt" not in serialized
    assert "image_bytes" not in serialized
    assert "api_key" not in serialized


def test_matrix_summary_rejects_tampered_paths_and_gate_rates() -> None:
    payload = run_matrix(BENCHMARK).summary.model_dump(mode="json")
    payload["configurations"][0]["agent_report_path"] = "../../outside.json"
    with pytest.raises(ValueError, match="agent report path"):
        MatrixSummaryReport.model_validate(payload)

    payload = run_matrix(BENCHMARK).summary.model_dump(mode="json")
    payload["exit_gate"]["full_agent_tool_required_task_success_rate"] = 0.5
    with pytest.raises(ValueError, match="rates do not match"):
        MatrixSummaryReport.model_validate(payload)


def test_benchmark_cli_writes_reports_and_returns_a_redacted_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "evaluation"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chest_xray_evidence_assistant.evals.run_benchmark",
            "--dataset",
            str(DATASET_PATH),
            "--output",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    message = json.loads(result.stdout)
    assert message["status"] == "succeeded"
    assert message["exit_gate_passed"] is True
    assert set(message) == {
        "exit_gate_passed",
        "output",
        "status",
        "summary_sha256",
    }
    assert (output_root / "matrix-summary.json").is_file()


def test_benchmark_cli_fails_closed_for_an_untracked_dataset_path(tmp_path: Path) -> None:
    dataset = tmp_path / "untracked.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chest_xray_evidence_assistant.evals.run_benchmark",
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "output"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "error_code": "benchmark_input_invalid",
        "status": "failed",
    }
