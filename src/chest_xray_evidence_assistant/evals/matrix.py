"""Comparable five-configuration offline benchmark matrix and artifact bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from .datasets import BenchmarkDataset
from .manifest import EvaluationManifestIndex
from .offline import CONFIGURATION_IDS, ConfigurationId, build_ablation_configurations
from .reporting import (
    AggregateMetrics,
    ConfigurationReport,
    run_configuration,
    serialize_configuration_report,
)
from .retrieval_metrics import RetrievalMetricsReport, build_retrieval_report


class ComparisonFingerprints(ContractModel):
    dataset_sha256: Sha256Digest
    case_set_sha256: Sha256Digest
    seed_set_sha256: Sha256Digest
    prompt_set_sha256: Sha256Digest
    limits_sha256: Sha256Digest
    rubric_sha256: Sha256Digest
    response_schema_sha256: Sha256Digest
    retrieval_index_sha256: Sha256Digest
    corpus_sha256: Sha256Digest


class MatrixConfigurationSummary(ContractModel):
    configuration_id: ConfigurationId
    configuration_sha256: Sha256Digest
    artifact_manifest_sha256: Sha256Digest
    agent_report_path: ShortText
    agent_report_sha256: Sha256Digest
    retrieval_report_path: ShortText
    retrieval_report_sha256: Sha256Digest
    agent_metrics: AggregateMetrics
    context_precision_mean: float = Field(ge=0, le=1)
    context_recall_mean: float = Field(ge=0, le=1)
    provenance_accuracy: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_artifact_paths(self) -> MatrixConfigurationSummary:
        if self.agent_report_path != f"agent/{self.configuration_id}.json":
            raise ValueError("agent report path does not match its configuration")
        if self.retrieval_report_path != f"retrieval/{self.configuration_id}.json":
            raise ValueError("retrieval report path does not match its configuration")
        return self


class MatrixExitGate(ContractModel):
    one_shot_tool_required_task_success_rate: float = Field(ge=0, le=1)
    full_agent_tool_required_task_success_rate: float = Field(ge=0, le=1)
    full_agent_beats_one_shot_on_tool_required: bool
    full_agent_unauthorized_tool_calls_zero: bool
    full_agent_high_severity_safety_failures_zero: bool
    comparison_inputs_identical: bool
    passed: bool


class MatrixFrameworks(ContractModel):
    pydantic_evals: ShortText
    ragas: ShortText


class MatrixSummaryReport(ContractModel):
    schema_version: Literal[1] = 1
    report_kind: Literal["offline_scripted_ablation_matrix"] = "offline_scripted_ablation_matrix"
    runner_mode: Literal["offline_scripted"] = "offline_scripted"
    token_accounting: Literal["alphanumeric-span-estimate-v1"] = "alphanumeric-span-estimate-v1"
    latency_accounting: Literal["deterministic-estimate-v1"] = "deterministic-estimate-v1"
    dataset_id: Identifier
    dataset_version: Identifier
    case_count: int = Field(gt=0)
    comparison: ComparisonFingerprints
    frameworks: MatrixFrameworks
    evaluation_manifest_path: Literal["evaluation-manifest.json"]
    evaluation_manifest_sha256: Sha256Digest
    configurations: tuple[MatrixConfigurationSummary, ...]
    exit_gate: MatrixExitGate

    @model_validator(mode="after")
    def validate_fixed_matrix_and_gate(self) -> MatrixSummaryReport:
        if tuple(item.configuration_id for item in self.configurations) != CONFIGURATION_IDS:
            raise ValueError("matrix must contain the five configurations in fixed order")
        one_shot = self.configurations[1].agent_metrics
        full_agent = self.configurations[-1].agent_metrics
        if any(item.agent_metrics.case_count != self.case_count for item in self.configurations):
            raise ValueError("matrix configuration case counts do not match")
        if (
            self.exit_gate.one_shot_tool_required_task_success_rate
            != one_shot.tool_required_task_success_rate
            or self.exit_gate.full_agent_tool_required_task_success_rate
            != full_agent.tool_required_task_success_rate
        ):
            raise ValueError("matrix exit-gate rates do not match configuration metrics")
        expected_values = (
            full_agent.tool_required_task_success_rate > one_shot.tool_required_task_success_rate,
            full_agent.unauthorized_tool_calls == 0,
            full_agent.high_severity_safety_failures == 0,
        )
        actual_values = (
            self.exit_gate.full_agent_beats_one_shot_on_tool_required,
            self.exit_gate.full_agent_unauthorized_tool_calls_zero,
            self.exit_gate.full_agent_high_severity_safety_failures_zero,
        )
        if actual_values != expected_values:
            raise ValueError("matrix exit-gate evidence does not match its metrics")
        if self.exit_gate.passed != all(
            (*actual_values, self.exit_gate.comparison_inputs_identical)
        ):
            raise ValueError("matrix aggregate gate does not match its component gates")
        return self


@dataclass(frozen=True, slots=True)
class MatrixBundle:
    summary: MatrixSummaryReport
    agent_reports: tuple[ConfigurationReport, ...]
    retrieval_reports: tuple[RetrievalMetricsReport, ...]
    manifest_index: EvaluationManifestIndex

    def __post_init__(self) -> None:
        if (
            _text_sha256(_serialize_model(self.manifest_index))
            != self.summary.evaluation_manifest_sha256
        ):
            raise ValueError("matrix evaluation manifest fingerprint does not match")
        summary_manifests = tuple(
            item.artifact_manifest_sha256 for item in self.summary.configurations
        )
        indexed_manifests = tuple(
            manifest.manifest_sha256 for manifest in self.manifest_index.manifests
        )
        if summary_manifests != indexed_manifests:
            raise ValueError("matrix summaries do not match the evaluation manifest index")


def _serialize_model(model: ContractModel) -> str:
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def run_matrix(benchmark: BenchmarkDataset) -> MatrixBundle:
    """Run all configurations over the same cases, seeds, limits, and rubric."""

    configurations = build_ablation_configurations(benchmark)
    agent_reports = tuple(
        run_configuration(benchmark, configuration) for configuration in configurations
    )
    retrieval_reports = tuple(build_retrieval_report(report) for report in agent_reports)
    common_sets = {
        "case_set": {report.case_set_sha256 for report in agent_reports},
        "corpus": {report.configuration.corpus_sha256 for report in agent_reports},
        "dataset": {report.configuration.dataset_sha256 for report in agent_reports},
        "limits": {report.configuration.limits_sha256 for report in agent_reports},
        "prompts": {report.configuration.prompt_set_sha256 for report in agent_reports},
        "retrieval": {report.configuration.retrieval_index_sha256 for report in agent_reports},
        "rubric": {report.configuration.rubric_sha256 for report in agent_reports},
        "schema": {report.configuration.response_schema_sha256 for report in agent_reports},
        "seeds": {report.configuration.seed_set_sha256 for report in agent_reports},
    }
    comparison_identical = all(len(values) == 1 for values in common_sets.values())
    first_configuration = configurations[0]
    comparison = ComparisonFingerprints(
        dataset_sha256=first_configuration.dataset_sha256,
        case_set_sha256=agent_reports[0].case_set_sha256,
        seed_set_sha256=first_configuration.seed_set_sha256,
        prompt_set_sha256=first_configuration.prompt_set_sha256,
        limits_sha256=first_configuration.limits_sha256,
        rubric_sha256=first_configuration.rubric_sha256,
        response_schema_sha256=first_configuration.response_schema_sha256,
        retrieval_index_sha256=first_configuration.retrieval_index_sha256,
        corpus_sha256=first_configuration.corpus_sha256,
    )
    manifest_index = EvaluationManifestIndex.build(
        dataset_id=benchmark.manifest.dataset_id,
        dataset_version=benchmark.manifest.dataset_version,
        manifests=tuple(report.artifact_manifest for report in agent_reports),
    )
    summaries: list[MatrixConfigurationSummary] = []
    for configuration, agent_report, retrieval_report in zip(
        configurations,
        agent_reports,
        retrieval_reports,
        strict=True,
    ):
        agent_content = serialize_configuration_report(agent_report)
        retrieval_content = _serialize_model(retrieval_report)
        summaries.append(
            MatrixConfigurationSummary(
                configuration_id=configuration.configuration_id,
                configuration_sha256=configuration.configuration_sha256,
                artifact_manifest_sha256=agent_report.artifact_manifest.manifest_sha256,
                agent_report_path=f"agent/{configuration.configuration_id}.json",
                agent_report_sha256=_text_sha256(agent_content),
                retrieval_report_path=f"retrieval/{configuration.configuration_id}.json",
                retrieval_report_sha256=_text_sha256(retrieval_content),
                agent_metrics=agent_report.metrics,
                context_precision_mean=retrieval_report.context_precision_mean,
                context_recall_mean=retrieval_report.context_recall_mean,
                provenance_accuracy=retrieval_report.provenance_accuracy,
            )
        )
    one_shot = agent_reports[1].metrics
    full_agent = agent_reports[-1].metrics
    beats_one_shot = (
        full_agent.tool_required_task_success_rate > one_shot.tool_required_task_success_rate
    )
    unauthorized_zero = full_agent.unauthorized_tool_calls == 0
    safety_zero = full_agent.high_severity_safety_failures == 0
    gate = MatrixExitGate(
        one_shot_tool_required_task_success_rate=one_shot.tool_required_task_success_rate,
        full_agent_tool_required_task_success_rate=full_agent.tool_required_task_success_rate,
        full_agent_beats_one_shot_on_tool_required=beats_one_shot,
        full_agent_unauthorized_tool_calls_zero=unauthorized_zero,
        full_agent_high_severity_safety_failures_zero=safety_zero,
        comparison_inputs_identical=comparison_identical,
        passed=all((beats_one_shot, unauthorized_zero, safety_zero, comparison_identical)),
    )
    summary = MatrixSummaryReport(
        dataset_id=benchmark.manifest.dataset_id,
        dataset_version=benchmark.manifest.dataset_version,
        case_count=len(benchmark.cases),
        comparison=comparison,
        frameworks=MatrixFrameworks(
            pydantic_evals=agent_reports[0].framework.version,
            ragas=retrieval_reports[0].framework.version,
        ),
        evaluation_manifest_path="evaluation-manifest.json",
        evaluation_manifest_sha256=_text_sha256(_serialize_model(manifest_index)),
        configurations=tuple(summaries),
        exit_gate=gate,
    )
    return MatrixBundle(
        summary=summary,
        agent_reports=agent_reports,
        retrieval_reports=retrieval_reports,
        manifest_index=manifest_index,
    )


def _text_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_matrix_bundle(bundle: MatrixBundle, output_root: Path) -> Path:
    """Write stable reports to explicit paths and return the summary path."""

    root = output_root.resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("benchmark output must be a directory")
    (root / "agent").mkdir(parents=True, exist_ok=True)
    (root / "retrieval").mkdir(parents=True, exist_ok=True)
    manifest_content = _serialize_model(bundle.manifest_index)
    if _text_sha256(manifest_content) != bundle.summary.evaluation_manifest_sha256:
        raise ValueError("evaluation manifest changed after matrix assembly")
    (root / bundle.summary.evaluation_manifest_path).write_text(
        manifest_content,
        encoding="utf-8",
    )
    for summary, agent_report, retrieval_report in zip(
        bundle.summary.configurations,
        bundle.agent_reports,
        bundle.retrieval_reports,
        strict=True,
    ):
        agent_content = serialize_configuration_report(agent_report)
        retrieval_content = _serialize_model(retrieval_report)
        if _text_sha256(agent_content) != summary.agent_report_sha256:
            raise ValueError("agent report changed after matrix assembly")
        if _text_sha256(retrieval_content) != summary.retrieval_report_sha256:
            raise ValueError("retrieval report changed after matrix assembly")
        (root / summary.agent_report_path).write_text(agent_content, encoding="utf-8")
        (root / summary.retrieval_report_path).write_text(
            retrieval_content,
            encoding="utf-8",
        )
    summary_path = root / "matrix-summary.json"
    summary_path.write_text(_serialize_model(bundle.summary), encoding="utf-8")
    return summary_path
