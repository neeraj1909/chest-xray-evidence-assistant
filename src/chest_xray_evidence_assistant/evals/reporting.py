"""Stable machine-readable reports produced through Pydantic Evals."""

from __future__ import annotations

import json
import math
import statistics
from importlib.metadata import version
from typing import Literal, TypeAlias, cast

from pydantic import Field, model_validator

from ..models import AgentTraceEvent, ContractModel, Identifier, Sha256Digest, ShortText
from .datasets import BenchmarkDataset, BenchmarkSplit, ExpectedStatus, TaskCategory
from .grading import RunGrade, canonical_sha256, grade_run
from .manifest import EvaluationArtifactManifest, build_evaluation_manifest
from .offline import AblationConfiguration, OfflineAblationTask
from .pydantic_adapter import build_pydantic_dataset

NumericScore: TypeAlias = int | float


class FrameworkEvidence(ContractModel):
    name: Literal["pydantic-evals"] = "pydantic-evals"
    version: ShortText
    evaluators: tuple[Identifier, ...]
    failed_tasks: int = Field(ge=0)
    evaluator_failures: int = Field(ge=0)
    report_sha256: Sha256Digest


class CaseEvaluation(ContractModel):
    """Case outcome carrying the exact normalized trace used by the graders."""

    case_id: Identifier
    split: BenchmarkSplit
    category: TaskCategory
    expected_status: ExpectedStatus
    actual_status: ExpectedStatus
    expected_document_ids: tuple[Identifier, ...] = Field(max_length=5)
    retrieved_document_ids: tuple[Identifier, ...] = Field(max_length=20)
    configuration_id: Identifier
    grade: RunGrade
    trace: tuple[AgentTraceEvent, ...] = Field(max_length=32)
    trace_sha256: Sha256Digest
    pydantic_assertions: dict[Identifier, bool]
    pydantic_scores: dict[Identifier, NumericScore]

    @model_validator(mode="after")
    def bind_reported_trace_to_grade(self) -> CaseEvaluation:
        trace_sha256 = canonical_sha256([event.model_dump(mode="json") for event in self.trace])
        if trace_sha256 != self.trace_sha256 or trace_sha256 != self.grade.trace_sha256:
            raise ValueError("reported trace does not match the graded trace")
        if self.case_id != self.grade.case_id:
            raise ValueError("reported case does not match its grade")
        if self.configuration_id != self.grade.configuration_id:
            raise ValueError("reported configuration does not match its grade")
        return self


class AggregateMetrics(ContractModel):
    case_count: int = Field(gt=0)
    schema_valid_rate: float = Field(ge=0, le=1)
    required_fields_complete_rate: float = Field(ge=0, le=1)
    answer_properties_accuracy: float = Field(ge=0, le=1)
    observation_accuracy: float = Field(ge=0, le=1)
    evidence_locator_accuracy: float = Field(ge=0, le=1)
    citation_accuracy: float = Field(ge=0, le=1)
    tool_selection_accuracy: float = Field(ge=0, le=1)
    tool_argument_accuracy: float = Field(ge=0, le=1)
    trajectory_success_rate: float = Field(ge=0, le=1)
    task_success_rate: float = Field(ge=0, le=1)
    tool_required_task_success_rate: float = Field(ge=0, le=1)
    recovery_rate: float = Field(ge=0, le=1)
    safety_pass_rate: float = Field(ge=0, le=1)
    unsupported_claim_rate: float = Field(ge=0, le=1)
    false_confident_answer_rate: float = Field(ge=0, le=1)
    abstention_precision: float = Field(ge=0, le=1)
    abstention_recall: float = Field(ge=0, le=1)
    clarification_precision: float = Field(ge=0, le=1)
    clarification_recall: float = Field(ge=0, le=1)
    unauthorized_tool_calls: int = Field(ge=0)
    unauthorized_tool_call_rate: float = Field(ge=0, le=1)
    budget_violation_rate: float = Field(ge=0, le=1)
    high_severity_safety_failures: int = Field(ge=0)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)
    model_requests_total: int = Field(ge=0)
    model_requests_mean: float = Field(ge=0)
    tool_calls_total: int = Field(ge=0)
    tool_calls_mean: float = Field(ge=0)
    input_tokens_total: int = Field(ge=0)
    input_tokens_mean: float = Field(ge=0)
    output_tokens_total: int = Field(ge=0)
    output_tokens_mean: float = Field(ge=0)
    estimated_cost_usd_total: float = Field(ge=0)
    estimated_cost_usd_mean: float = Field(ge=0)
    category_task_success_rate: dict[TaskCategory, float]


def _framework_case_payload(
    cases: tuple[CaseEvaluation, ...],
) -> list[dict[str, object]]:
    return [
        {
            "assertions": case.pydantic_assertions,
            "case_id": case.case_id,
            "observation_sha256": case.grade.observation_sha256,
            "scores": case.pydantic_scores,
        }
        for case in cases
    ]


class ConfigurationReport(ContractModel):
    schema_version: Literal[1] = 1
    report_kind: Literal["offline_scripted_pydantic_evals"] = "offline_scripted_pydantic_evals"
    dataset_id: Identifier
    dataset_version: Identifier
    dataset_sha256: Sha256Digest
    scope: Identifier
    case_count: int = Field(gt=0)
    case_set_sha256: Sha256Digest
    configuration: AblationConfiguration
    artifact_manifest: EvaluationArtifactManifest
    framework: FrameworkEvidence
    metrics: AggregateMetrics
    cases: tuple[CaseEvaluation, ...] = Field(min_length=1, max_length=2_048)

    @model_validator(mode="after")
    def validate_report(self) -> ConfigurationReport:
        if self.case_count != len(self.cases):
            raise ValueError("report case count does not match case records")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("report case IDs must be unique")
        if self.case_set_sha256 != canonical_sha256([case.case_id for case in self.cases]):
            raise ValueError("report case-set fingerprint does not match")
        if self.configuration.dataset_sha256 != self.dataset_sha256:
            raise ValueError("report and configuration target different datasets")
        if not self.artifact_manifest.matches_configuration(self.configuration):
            raise ValueError("artifact manifest does not match report configuration")
        if self.artifact_manifest.framework_versions["pydantic-evals"] != self.framework.version:
            raise ValueError("artifact manifest has a stale Pydantic Evals version")
        if any(case.configuration_id != self.configuration.configuration_id for case in self.cases):
            raise ValueError("report contains a case from another configuration")
        if self.metrics.case_count != self.case_count:
            raise ValueError("aggregate count does not match report cases")
        if self.metrics != aggregate_metrics(self.cases):
            raise ValueError("aggregate metrics do not match report cases")
        if self.framework.report_sha256 != canonical_sha256(_framework_case_payload(self.cases)):
            raise ValueError("framework report fingerprint does not match report cases")
        return self


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _precision_recall(
    cases: tuple[CaseEvaluation, ...],
    status: ExpectedStatus,
) -> tuple[float, float]:
    predicted = [case for case in cases if case.actual_status == status]
    expected = [case for case in cases if case.expected_status == status]
    true_positive = sum(
        case.expected_status == status and case.actual_status == status for case in cases
    )
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    return precision, recall


def _p95(values: list[int]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)])


def aggregate_metrics(cases: tuple[CaseEvaluation, ...]) -> AggregateMetrics:
    if not cases:
        raise ValueError("cannot aggregate an empty evaluation")
    grades = [case.grade for case in cases]
    usages = [grade.usage for grade in grades]
    observation = [grade for grade in grades if grade.category == "observation"]
    tool_required = [grade for grade in grades if grade.category in {"tool_use", "retrieval"}]
    abstention_precision, abstention_recall = _precision_recall(cases, "abstain")
    clarification_precision, clarification_recall = _precision_recall(
        cases,
        "needs_clarification",
    )
    category_rates = {
        category: _rate([grade.task_success for grade in grades if grade.category == category])
        for category in (
            "observation",
            "tool_use",
            "retrieval",
            "contradiction",
            "abstention",
        )
    }
    count = len(cases)
    duration_values = [usage.duration_ms for usage in usages]
    model_requests = sum(usage.model_requests for usage in usages)
    tool_calls = sum(usage.tool_calls for usage in usages)
    input_tokens = sum(usage.input_tokens for usage in usages)
    output_tokens = sum(usage.output_tokens for usage in usages)
    cost = sum(float(usage.estimated_cost_usd) for usage in usages)
    unauthorized = sum(grade.unauthorized_tool_calls for grade in grades)
    return AggregateMetrics(
        case_count=count,
        schema_valid_rate=_rate([grade.schema_valid for grade in grades]),
        required_fields_complete_rate=_rate([grade.required_fields_complete for grade in grades]),
        answer_properties_accuracy=_rate([grade.answer_properties_correct for grade in grades]),
        observation_accuracy=_rate([grade.task_success for grade in observation]),
        evidence_locator_accuracy=_rate([grade.locators_correct for grade in grades]),
        citation_accuracy=_rate([grade.citations_correct for grade in grades]),
        tool_selection_accuracy=_rate([grade.tool_selection_correct for grade in grades]),
        tool_argument_accuracy=_rate([grade.tool_arguments_correct for grade in grades]),
        trajectory_success_rate=_rate([grade.trajectory_correct for grade in grades]),
        task_success_rate=_rate([grade.task_success for grade in grades]),
        tool_required_task_success_rate=_rate([grade.task_success for grade in tool_required]),
        recovery_rate=_rate([grade.recovered for grade in grades]),
        safety_pass_rate=_rate([grade.safety_passed for grade in grades]),
        unsupported_claim_rate=_rate([grade.unsupported_claims > 0 for grade in grades]),
        false_confident_answer_rate=_rate([grade.false_confident_answer for grade in grades]),
        abstention_precision=abstention_precision,
        abstention_recall=abstention_recall,
        clarification_precision=clarification_precision,
        clarification_recall=clarification_recall,
        unauthorized_tool_calls=unauthorized,
        unauthorized_tool_call_rate=_rate([grade.unauthorized_tool_calls > 0 for grade in grades]),
        budget_violation_rate=_rate([not grade.budget_compliant for grade in grades]),
        high_severity_safety_failures=sum(grade.high_severity_safety_failure for grade in grades),
        latency_p50_ms=float(statistics.median(duration_values)),
        latency_p95_ms=_p95(duration_values),
        model_requests_total=model_requests,
        model_requests_mean=model_requests / count,
        tool_calls_total=tool_calls,
        tool_calls_mean=tool_calls / count,
        input_tokens_total=input_tokens,
        input_tokens_mean=input_tokens / count,
        output_tokens_total=output_tokens,
        output_tokens_mean=output_tokens / count,
        estimated_cost_usd_total=cost,
        estimated_cost_usd_mean=cost / count,
        category_task_success_rate=category_rates,
    )


def run_configuration(
    benchmark: BenchmarkDataset,
    configuration: AblationConfiguration,
    *,
    case_limit: int | None = None,
) -> ConfigurationReport:
    """Evaluate one fixed capability projection through Pydantic Evals."""

    dataset = build_pydantic_dataset(benchmark)
    if case_limit is not None:
        if isinstance(case_limit, bool) or not 1 <= case_limit <= len(dataset.cases):
            raise ValueError("case limit is outside the dataset boundary")
        dataset.cases = list(dataset.cases[:case_limit])
    task = OfflineAblationTask(benchmark, configuration)
    framework_report = dataset.evaluate_sync(
        task,
        name=configuration.configuration_id,
        max_concurrency=1,
        progress=False,
        metadata={
            "configuration_sha256": configuration.configuration_sha256,
            "dataset_sha256": benchmark.manifest.dataset_sha256,
        },
    )
    evaluator_failure_count = sum(
        len(case.evaluator_failures) for case in framework_report.cases
    ) + len(framework_report.report_evaluator_failures)
    if framework_report.failures or evaluator_failure_count:
        raise ValueError("Pydantic Evals reported task or evaluator failures")

    case_evaluations: list[CaseEvaluation] = []
    for result in framework_report.cases:
        if result.metadata is None:
            raise ValueError("Pydantic Evals omitted benchmark metadata")
        benchmark_case = result.metadata.benchmark_case(result.inputs)
        grade = grade_run(benchmark_case, result.output)
        response = result.output.response_payload
        trace = tuple(AgentTraceEvent.model_validate(event) for event in response.get("trace", []))
        assertions = {
            name: bool(evaluation.value) for name, evaluation in sorted(result.assertions.items())
        }
        scores = {name: evaluation.value for name, evaluation in sorted(result.scores.items())}
        case_evaluations.append(
            CaseEvaluation(
                case_id=benchmark_case.case_id,
                split=benchmark_case.split,
                category=benchmark_case.category,
                expected_status=benchmark_case.expected.status,
                actual_status=_response_status(response["status"]),
                expected_document_ids=benchmark_case.expected.source_document_ids,
                retrieved_document_ids=tuple(
                    evidence["document_id"] for evidence in response.get("source_evidence", [])
                ),
                configuration_id=configuration.configuration_id,
                grade=grade,
                trace=trace,
                trace_sha256=grade.trace_sha256,
                pydantic_assertions=assertions,
                pydantic_scores=scores,
            )
        )
    cases = tuple(case_evaluations)
    framework = FrameworkEvidence(
        version=version("pydantic-evals"),
        evaluators=tuple(type(evaluator).__name__ for evaluator in dataset.evaluators),
        failed_tasks=len(framework_report.failures),
        evaluator_failures=evaluator_failure_count,
        report_sha256=canonical_sha256(_framework_case_payload(cases)),
    )
    return ConfigurationReport(
        dataset_id=benchmark.manifest.dataset_id,
        dataset_version=benchmark.manifest.dataset_version,
        dataset_sha256=benchmark.manifest.dataset_sha256,
        scope="all" if case_limit is None else f"first-{case_limit}",
        case_count=len(cases),
        case_set_sha256=canonical_sha256([case.case_id for case in cases]),
        configuration=configuration,
        artifact_manifest=build_evaluation_manifest(
            configuration,
            framework_versions={
                "pydantic-evals": version("pydantic-evals"),
                "ragas": version("ragas"),
            },
        ),
        framework=framework,
        metrics=aggregate_metrics(cases),
        cases=cases,
    )


def serialize_configuration_report(report: ConfigurationReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _response_status(value: object) -> ExpectedStatus:
    if value not in {"answered", "needs_clarification", "abstain"}:
        raise ValueError("evaluated response has an invalid status")
    return cast(ExpectedStatus, value)
