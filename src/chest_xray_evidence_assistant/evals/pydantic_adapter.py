"""Pydantic Evals adapters backed only by deterministic benchmark graders."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from ..models import ContractModel, Identifier, ImageAsset, QuestionText, RunLimits
from .datasets import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkExpected,
    BenchmarkSplit,
    TaskCategory,
)
from .grading import ObservedRun, RunGrade, grade_run


class EvaluationInput(ContractModel):
    """Task-visible inputs, deliberately excluding labels and expected outputs."""

    case_id: Identifier
    image: ImageAsset
    prompt: QuestionText
    seed: int = Field(ge=0, le=2**31 - 1)
    limits: RunLimits


class EvaluationCaseMetadata(ContractModel):
    """Evaluator-only labels and expectations that never enter the task callable."""

    dataset_id: Identifier
    split: BenchmarkSplit
    category: TaskCategory
    prompt_family: Identifier
    expected: BenchmarkExpected

    def benchmark_case(self, inputs: EvaluationInput) -> BenchmarkCase:
        return BenchmarkCase(
            case_id=inputs.case_id,
            image_case_id=inputs.image.image_id,
            split=self.split,
            category=self.category,
            prompt_family=self.prompt_family,
            prompt=inputs.prompt,
            seed=inputs.seed,
            expected=self.expected,
        )


EvaluationContext = EvaluatorContext[
    EvaluationInput,
    ObservedRun,
    EvaluationCaseMetadata,
]


def _grade(ctx: EvaluationContext) -> RunGrade:
    if ctx.metadata is None:
        raise ValueError("evaluation case metadata is required")
    return grade_run(ctx.metadata.benchmark_case(ctx.inputs), ctx.output)


@dataclass
class SchemaEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool]:
        grade = _grade(ctx)
        return {
            "schema_valid": grade.schema_valid,
            "required_fields_complete": grade.required_fields_complete,
            "status_correct": grade.status_correct,
            "answer_properties_correct": grade.answer_properties_correct,
        }


@dataclass
class EvidenceEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool]:
        grade = _grade(ctx)
        return {
            "evidence_correct": grade.evidence_correct,
            "locators_correct": grade.locators_correct,
            "citations_correct": grade.citations_correct,
        }


@dataclass
class ToolEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool | int]:
        grade = _grade(ctx)
        return {
            "tool_selection_correct": grade.tool_selection_correct,
            "tool_arguments_correct": grade.tool_arguments_correct,
            "unauthorized_tool_calls": grade.unauthorized_tool_calls,
        }


@dataclass
class TrajectoryEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool]:
        grade = _grade(ctx)
        return {
            "trajectory_correct": grade.trajectory_correct,
            "task_success": grade.task_success,
        }


@dataclass
class BudgetEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool | int | float]:
        grade = _grade(ctx)
        usage = grade.usage
        return {
            "budget_compliant": grade.budget_compliant,
            "duration_ms": usage.duration_ms,
            "model_requests": usage.model_requests,
            "tool_calls": usage.tool_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": float(usage.estimated_cost_usd),
        }


@dataclass
class SafetyEvaluator(Evaluator[EvaluationInput, ObservedRun, EvaluationCaseMetadata]):
    def evaluate(self, ctx: EvaluationContext) -> dict[str, bool | int]:
        grade = _grade(ctx)
        return {
            "fallback_correct": grade.fallback_correct,
            "safety_passed": grade.safety_passed,
            "no_false_confident_answer": not grade.false_confident_answer,
            "unsupported_claims": grade.unsupported_claims,
            "high_severity_safety_failure": grade.high_severity_safety_failure,
        }


DEFAULT_EVALUATORS = (
    SchemaEvaluator(),
    EvidenceEvaluator(),
    ToolEvaluator(),
    TrajectoryEvaluator(),
    BudgetEvaluator(),
    SafetyEvaluator(),
)


def build_pydantic_dataset(
    benchmark: BenchmarkDataset,
    *,
    split: BenchmarkSplit | None = None,
) -> Dataset[EvaluationInput, ObservedRun, EvaluationCaseMetadata]:
    """Translate owned cases into the code-first Pydantic Evals dataset API."""

    cases = benchmark.cases if split is None else benchmark.cases_for_split(split)
    return Dataset(
        name=benchmark.manifest.dataset_id,
        cases=[
            Case(
                name=case.case_id,
                inputs=EvaluationInput(
                    case_id=case.case_id,
                    image=benchmark.image_for(case).asset,
                    prompt=case.prompt,
                    seed=case.seed,
                    limits=RunLimits(
                        max_model_requests=case.expected.max_model_requests,
                        max_tool_calls=case.expected.max_tool_calls,
                    ),
                ),
                metadata=EvaluationCaseMetadata(
                    dataset_id=benchmark.manifest.dataset_id,
                    split=case.split,
                    category=case.category,
                    prompt_family=case.prompt_family,
                    expected=case.expected,
                ),
            )
            for case in cases
        ],
        evaluators=DEFAULT_EVALUATORS,
    )
