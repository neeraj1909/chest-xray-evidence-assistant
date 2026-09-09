"""Retrieval-only RAGAS metrics with an independent deterministic safety gate."""

from __future__ import annotations

import math
import os
from importlib.metadata import version
from typing import Literal

from pydantic import Field, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from .grading import canonical_sha256
from .reporting import ConfigurationReport

os.environ["RAGAS_DO_NOT_TRACK"] = "true"


class RagasFrameworkEvidence(ContractModel):
    name: Literal["ragas"] = "ragas"
    version: ShortText
    metrics: tuple[
        Literal["id_based_context_precision"],
        Literal["id_based_context_recall"],
    ] = ("id_based_context_precision", "id_based_context_recall")
    telemetry_disabled: Literal[True] = True
    report_sha256: Sha256Digest


class RetrievalCaseMetrics(ContractModel):
    case_id: Identifier
    configuration_id: Identifier
    category: Literal["retrieval"] = "retrieval"
    expected_document_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=5)
    retrieved_document_ids: tuple[Identifier, ...] = Field(max_length=20)
    context_precision: float = Field(ge=0, le=1)
    precision_defined: bool
    context_recall: float = Field(ge=0, le=1)
    provenance_correct: bool
    deterministic_safety_passed: bool
    high_severity_safety_failure: bool
    retrieval_acceptable: bool

    @model_validator(mode="after")
    def keep_retrieval_and_safety_gates_composed(self) -> RetrievalCaseMetrics:
        expected_acceptable = (
            self.context_precision == 1.0
            and self.context_recall == 1.0
            and self.provenance_correct
            and self.deterministic_safety_passed
            and not self.high_severity_safety_failure
        )
        if self.retrieval_acceptable != expected_acceptable:
            raise ValueError("RAGAS scores cannot override deterministic safety")
        return self


def _framework_case_payload(
    cases: tuple[RetrievalCaseMetrics, ...],
) -> list[dict[str, object]]:
    return [
        {
            "case_id": case.case_id,
            "context_precision": case.context_precision,
            "context_recall": case.context_recall,
            "expected_document_ids": case.expected_document_ids,
            "retrieved_document_ids": case.retrieved_document_ids,
        }
        for case in cases
    ]


class RetrievalMetricsReport(ContractModel):
    schema_version: Literal[1] = 1
    metric_scope: Literal["retrieval_only"] = "retrieval_only"
    safety_evaluator: Literal["deterministic_grader"] = "deterministic_grader"
    configuration_id: Identifier
    source_agent_report_sha256: Sha256Digest
    case_count: int = Field(gt=0)
    case_ids: tuple[Identifier, ...] = Field(min_length=1)
    context_precision_mean: float = Field(ge=0, le=1)
    context_recall_mean: float = Field(ge=0, le=1)
    provenance_accuracy: float = Field(ge=0, le=1)
    retrieval_acceptance_rate: float = Field(ge=0, le=1)
    framework: RagasFrameworkEvidence
    cases: tuple[RetrievalCaseMetrics, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ledger(self) -> RetrievalMetricsReport:
        if self.case_count != len(self.cases) or self.case_ids != tuple(
            case.case_id for case in self.cases
        ):
            raise ValueError("retrieval report case ledger does not match")
        if any(case.configuration_id != self.configuration_id for case in self.cases):
            raise ValueError("retrieval report mixes configurations")
        count = len(self.cases)
        expected_aggregates = (
            sum(case.context_precision for case in self.cases) / count,
            sum(case.context_recall for case in self.cases) / count,
            sum(case.provenance_correct for case in self.cases) / count,
            sum(case.retrieval_acceptable for case in self.cases) / count,
        )
        reported_aggregates = (
            self.context_precision_mean,
            self.context_recall_mean,
            self.provenance_accuracy,
            self.retrieval_acceptance_rate,
        )
        if reported_aggregates != expected_aggregates:
            raise ValueError("retrieval aggregate metrics do not match report cases")
        if self.framework.report_sha256 != canonical_sha256(_framework_case_payload(self.cases)):
            raise ValueError("RAGAS report fingerprint does not match report cases")
        return self


def _id_scores(
    retrieved_document_ids: tuple[str, ...],
    expected_document_ids: tuple[str, ...],
) -> tuple[float, bool, float]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    sample = SingleTurnSample(
        retrieved_context_ids=list(retrieved_document_ids),
        reference_context_ids=list(expected_document_ids),
    )
    raw_recall = IDBasedContextRecall().single_turn_score(sample)
    if not math.isfinite(raw_recall):
        raise ValueError("RAGAS returned an undefined context recall")
    if not retrieved_document_ids:
        return 0.0, False, float(raw_recall)
    raw_precision = IDBasedContextPrecision().single_turn_score(sample)
    if not math.isfinite(raw_precision):
        raise ValueError("RAGAS returned an undefined context precision")
    return float(raw_precision), True, float(raw_recall)


def build_retrieval_report(
    agent_report: ConfigurationReport,
) -> RetrievalMetricsReport:
    """Score only retrieval cases; retain deterministic safety as the authority."""

    retrieval_cases = tuple(case for case in agent_report.cases if case.category == "retrieval")
    if not retrieval_cases:
        raise ValueError("agent report contains no retrieval cases")

    case_metrics: list[RetrievalCaseMetrics] = []
    for case in retrieval_cases:
        precision, precision_defined, recall = _id_scores(
            case.retrieved_document_ids,
            case.expected_document_ids,
        )
        provenance_correct = tuple(sorted(case.retrieved_document_ids)) == tuple(
            sorted(case.expected_document_ids)
        )
        acceptable = (
            precision == 1.0
            and recall == 1.0
            and provenance_correct
            and case.grade.safety_passed
            and not case.grade.high_severity_safety_failure
        )
        case_metrics.append(
            RetrievalCaseMetrics(
                case_id=case.case_id,
                configuration_id=case.configuration_id,
                expected_document_ids=case.expected_document_ids,
                retrieved_document_ids=case.retrieved_document_ids,
                context_precision=precision,
                precision_defined=precision_defined,
                context_recall=recall,
                provenance_correct=provenance_correct,
                deterministic_safety_passed=case.grade.safety_passed,
                high_severity_safety_failure=(case.grade.high_severity_safety_failure),
                retrieval_acceptable=acceptable,
            )
        )
    cases = tuple(case_metrics)
    count = len(cases)
    source_sha256 = canonical_sha256(agent_report)
    return RetrievalMetricsReport(
        configuration_id=agent_report.configuration.configuration_id,
        source_agent_report_sha256=source_sha256,
        case_count=count,
        case_ids=tuple(case.case_id for case in cases),
        context_precision_mean=sum(case.context_precision for case in cases) / count,
        context_recall_mean=sum(case.context_recall for case in cases) / count,
        provenance_accuracy=(sum(case.provenance_correct for case in cases) / count),
        retrieval_acceptance_rate=(sum(case.retrieval_acceptable for case in cases) / count),
        framework=RagasFrameworkEvidence(
            version=version("ragas"),
            report_sha256=canonical_sha256(_framework_case_payload(cases)),
        ),
        cases=cases,
    )
