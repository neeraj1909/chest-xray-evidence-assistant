from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.grading import (
    ObservedRun,
    RunUsage,
    canonical_sha256,
)
from chest_xray_evidence_assistant.evals.pydantic_adapter import (
    EvaluationInput,
    build_pydantic_dataset,
)
from chest_xray_evidence_assistant.models import RunLimits, VisualResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")
ZERO_DIGEST = "0" * 64


def _observation_task(case: EvaluationInput) -> ObservedRun:
    response = {
        "answer": (
            "The synthetic full-frame pattern is a horizontal gradient, darker "
            "on the left and brighter on the right."
        ),
        "confidence": 0.8,
        "observations": ["A deterministic synthetic pattern is visible."],
        "source_evidence": [],
        "status": "answered",
        "trace": [
            {
                "input_sha256": ZERO_DIGEST,
                "kind": "request",
                "name": "cxr-evidence-agent",
                "sequence": 0,
                "status": "started",
            },
            {
                "attributes": {
                    "model_requests": 1,
                    "provider": "offline-eval",
                    "tool_calls": 0,
                },
                "input_sha256": ZERO_DIGEST,
                "kind": "model",
                "name": "deterministic-policy",
                "output_sha256": ZERO_DIGEST,
                "sequence": 1,
                "status": "succeeded",
            },
            {
                "input_sha256": ZERO_DIGEST,
                "kind": "validation",
                "name": "visual-response",
                "output_sha256": ZERO_DIGEST,
                "sequence": 2,
                "status": "succeeded",
            },
            {
                "kind": "final",
                "name": "answered",
                "output_sha256": ZERO_DIGEST,
                "sequence": 3,
                "status": "succeeded",
            },
        ],
        "uncertainty": ["Synthetic benchmark evidence only."],
        "visual_evidence": [
            {
                "confidence": 0.8,
                "description": "The complete synthetic fixture frame.",
                "locator": {
                    "box": None,
                    "image_id": case.image.image_id,
                    "kind": "full_frame",
                },
            }
        ],
    }
    response_sha256 = canonical_sha256(
        VisualResponse.model_validate(response).model_dump(mode="json", exclude={"trace"})
    )
    response["trace"][1] = {
        **response["trace"][1],
        "name": "adapter-test",
        "output_sha256": response_sha256,
    }
    response["trace"][2] = {
        **response["trace"][2],
        "input_sha256": response_sha256,
        "output_sha256": response_sha256,
    }
    response["trace"][3] = {
        **response["trace"][3],
        "output_sha256": response_sha256,
    }
    return ObservedRun(
        case_id=case.case_id,
        configuration_id="adapter-test",
        response_payload=response,
        limits=RunLimits(),
        usage=RunUsage(
            duration_ms=1,
            estimated_cost_usd=Decimal("0"),
            image_bytes=case.image.byte_size,
            input_tokens=12,
            model_requests=1,
            output_tokens=24,
            recovered=False,
            tool_calls=0,
        ),
    )


def test_pydantic_evals_dataset_preserves_cases_metadata_and_custom_scores() -> None:
    observation_case = next(case for case in BENCHMARK.cases if case.category == "observation")
    dataset = build_pydantic_dataset(BENCHMARK, split=observation_case.split)
    dataset.cases = [case for case in dataset.cases if case.name == observation_case.case_id]

    report = dataset.evaluate_sync(
        _observation_task,
        name=dataset.name,
        progress=False,
    )

    assert report.name == "cxr-agent-bench-v0"
    assert not report.failures
    assert len(report.cases) == 1
    result = report.cases[0]
    assert result.name == observation_case.case_id
    assert result.metadata.category == "observation"
    assert result.metadata.dataset_id == "cxr-agent-bench-v0"
    assert result.metadata.split == "development"
    assert not hasattr(result.inputs, "expected")
    assert not hasattr(result.inputs, "category")
    assert result.assertions["schema_valid"].value is True
    assert result.assertions["evidence_correct"].value is True
    assert result.assertions["tool_selection_correct"].value is True
    assert result.assertions["trajectory_correct"].value is True
    assert result.assertions["budget_compliant"].value is True
    assert result.assertions["safety_passed"].value is True
    assert result.assertions["no_false_confident_answer"].value is True
    assert result.assertions["task_success"].value is True
    assert result.scores["model_requests"].value == 1
    assert result.scores["tool_calls"].value == 0


def test_pydantic_adapter_uses_only_requested_split() -> None:
    dataset = build_pydantic_dataset(BENCHMARK, split="test")

    assert len(dataset.cases) == 24
    assert {case.metadata.split for case in dataset.cases if case.metadata} == {"test"}
