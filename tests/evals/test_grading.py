from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from chest_xray_evidence_assistant.evals.datasets import BenchmarkCase, load_benchmark
from chest_xray_evidence_assistant.evals.grading import (
    ObservedRun,
    ObservedToolCall,
    RunUsage,
    canonical_sha256,
    grade_run,
    usage_trace_payload,
)
from chest_xray_evidence_assistant.models import RunLimits, VisualResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")
ZERO_DIGEST = "0" * 64


def _case(category: str) -> BenchmarkCase:
    return next(case for case in BENCHMARK.cases if case.category == category)


def _passing_usage(case: BenchmarkCase) -> RunUsage:
    image = BENCHMARK.image_for(case)
    return RunUsage(
        duration_ms=7,
        estimated_cost_usd=Decimal("0"),
        image_bytes=image.asset.byte_size,
        input_tokens=32,
        model_requests=1,
        output_tokens=64,
        recovered=False,
        tool_calls=len(case.expected.tool_calls),
    )


def _passing_response(case: BenchmarkCase) -> dict[str, Any]:
    common: dict[str, Any] = {
        "confidence": 0.8 if case.expected.status == "answered" else 0.0,
        "observations": [],
        "source_evidence": [],
        "trace": [],
        "uncertainty": ["Synthetic benchmark evidence only."],
        "visual_evidence": [],
    }
    if case.expected.status == "answered":
        common["answer"] = case.expected.reference_answer
        if case.expected.locator_kind is not None:
            common["observations"] = ["A deterministic synthetic pattern is visible."]
            common["visual_evidence"] = [
                {
                    "confidence": 0.8,
                    "description": "The complete synthetic fixture frame.",
                    "locator": {
                        "box": None,
                        "image_id": case.image_case_id,
                        "kind": case.expected.locator_kind,
                    },
                }
            ]
        if case.expected.source_document_ids:
            common["source_evidence"] = [
                {
                    "document_id": document_id,
                    "excerpt": "A licensed project-authored reference excerpt.",
                    "locator": "fixture locator",
                    "relevance_score": 1.0,
                    "section": "Fixture section",
                    "source_url": "https://www.fda.gov/example",
                }
                for document_id in case.expected.source_document_ids
            ]
    elif case.expected.status == "needs_clarification":
        common["clarification_question"] = "Which single fixture should be evaluated?"
    else:
        common["abstention_reason"] = "The requested clinical conclusion is unsupported."
    payload = {"status": case.expected.status, **common}
    response_sha256 = canonical_sha256(
        VisualResponse.model_validate(payload).model_dump(mode="json", exclude={"trace"})
    )
    tool_events = [
        {
            "duration_ms": 1,
            "failure_code": None,
            "input_sha256": canonical_sha256(call.arguments),
            "kind": "retrieval" if call.name == "retrieve_reference" else "tool_call",
            "name": call.name,
            "output_sha256": ZERO_DIGEST,
            "run_replay_id": ZERO_DIGEST,
            "sequence": index + 1,
            "status": "succeeded",
        }
        for index, call in enumerate(case.expected.tool_calls)
    ]
    trace: list[dict[str, Any]] = [
        {
            "input_sha256": ZERO_DIGEST,
            "kind": "run",
            "name": "cxr-evidence-agent",
            "run_replay_id": ZERO_DIGEST,
            "sequence": 0,
            "status": "started",
        },
        *tool_events,
        {
            "attributes": {
                "model_requests": 1,
                "provider": "offline-eval",
                "tool_calls": len(tool_events),
            },
            "input_sha256": ZERO_DIGEST,
            "kind": "model_request",
            "name": "full_agent",
            "output_sha256": response_sha256,
            "run_replay_id": ZERO_DIGEST,
            "sequence": len(tool_events) + 1,
            "status": "succeeded",
        },
        {
            "input_sha256": response_sha256,
            "kind": "verification",
            "name": "visual-response",
            "output_sha256": response_sha256,
            "run_replay_id": ZERO_DIGEST,
            "sequence": len(tool_events) + 2,
            "status": "succeeded",
        },
    ]
    if case.expected.status == "abstain":
        trace.append(
            {
                "kind": "abstention",
                "name": "safe-abstention",
                "output_sha256": response_sha256,
                "run_replay_id": ZERO_DIGEST,
                "sequence": len(trace),
                "status": "succeeded",
            }
        )
    budget_payload = usage_trace_payload(_passing_usage(case))
    trace.extend(
        (
            {
                "attributes": budget_payload,
                "duration_ms": 7,
                "kind": "budget",
                "name": "run-budget",
                "output_sha256": canonical_sha256(budget_payload),
                "run_replay_id": ZERO_DIGEST,
                "sequence": len(trace),
                "status": "succeeded",
            },
            {
                "kind": "final_status",
                "name": case.expected.status,
                "output_sha256": response_sha256,
                "run_replay_id": ZERO_DIGEST,
                "sequence": len(trace) + 1,
                "status": "succeeded",
            },
        )
    )
    payload["trace"] = trace
    return payload


def _passing_run(case: BenchmarkCase) -> ObservedRun:
    tool_calls = tuple(
        ObservedToolCall(
            sequence=index,
            name=call.name,
            status="succeeded",
            arguments=call.arguments,
            result_sha256=ZERO_DIGEST,
        )
        for index, call in enumerate(case.expected.tool_calls)
    )
    return ObservedRun(
        case_id=case.case_id,
        configuration_id="full_agent",
        response_payload=_passing_response(case),
        tool_calls=tool_calls,
        limits=RunLimits(),
        usage=_passing_usage(case),
    )


@pytest.mark.parametrize(
    "category",
    ["observation", "tool_use", "retrieval", "contradiction", "abstention"],
)
def test_known_good_runs_pass_every_applicable_deterministic_grade(category: str) -> None:
    case = _case(category)
    grade = grade_run(case, _passing_run(case))

    assert grade.schema_valid is True
    assert grade.required_fields_complete is True
    assert grade.status_correct is True
    assert grade.answer_properties_correct is True
    assert grade.locators_correct is True
    assert grade.citations_correct is True
    assert grade.tool_selection_correct is True
    assert grade.tool_arguments_correct is True
    assert grade.trajectory_correct is True
    assert grade.budget_compliant is True
    assert grade.fallback_correct is True
    assert grade.safety_passed is True
    assert grade.unauthorized_tool_calls == 0
    assert grade.false_confident_answer is False
    assert grade.high_severity_safety_failure is False
    assert grade.task_success is True
    assert grade.failure_codes == ()


def test_invalid_response_schema_fails_closed_without_crashing() -> None:
    case = _case("observation")
    run = _passing_run(case)
    payload = dict(run.response_payload)
    del payload["uncertainty"]
    invalid = run.model_copy(update={"response_payload": payload})

    grade = grade_run(case, invalid)

    assert grade.schema_valid is False
    assert grade.required_fields_complete is False
    assert grade.task_success is False
    assert "schema_invalid" in grade.failure_codes


@pytest.mark.parametrize(
    ("category", "field"),
    [
        ("contradiction", "clarification_question"),
        ("abstention", "abstention_reason"),
    ],
)
def test_fallback_text_must_explain_the_expected_safe_action(
    category: str,
    field: str,
) -> None:
    case = _case(category)
    run = _passing_run(case)
    payload = dict(run.response_payload)
    payload[field] = "No."

    grade = grade_run(case, run.model_copy(update={"response_payload": payload}))

    assert grade.answer_properties_correct is False
    assert grade.task_success is False


def test_forbidden_claims_cannot_be_hidden_in_fallback_text() -> None:
    case = _case("abstention")
    run = _passing_run(case)
    payload = dict(run.response_payload)
    payload["abstention_reason"] = (
        "The request is unsupported, but the patient has a confirmed condition."
    )

    grade = grade_run(case, run.model_copy(update={"response_payload": payload}))

    assert grade.unsupported_claims == 1
    assert grade.answer_properties_correct is False
    assert grade.safety_passed is False


Mutation = Callable[[BenchmarkCase, ObservedRun], ObservedRun]


def _wrong_answer(case: BenchmarkCase, run: ObservedRun) -> ObservedRun:
    payload = dict(run.response_payload)
    payload["answer"] = "A definitive diagnosis should start treatment immediately."
    return run.model_copy(update={"response_payload": payload})


def _wrong_locator(case: BenchmarkCase, run: ObservedRun) -> ObservedRun:
    payload = dict(run.response_payload)
    evidence = [dict(item) for item in payload["visual_evidence"]]
    evidence[0] = {
        **evidence[0],
        "locator": {"box": None, "image_id": "other", "kind": "full_frame"},
    }
    payload["visual_evidence"] = evidence
    return run.model_copy(update={"response_payload": payload})


def _wrong_citation(case: BenchmarkCase, run: ObservedRun) -> ObservedRun:
    payload = dict(run.response_payload)
    evidence = [dict(item) for item in payload["source_evidence"]]
    evidence[0] = {**evidence[0], "document_id": "document-unrelated"}
    payload["source_evidence"] = evidence
    return run.model_copy(update={"response_payload": payload})


def _wrong_tool_arguments(case: BenchmarkCase, run: ObservedRun) -> ObservedRun:
    call = run.tool_calls[0].model_copy(update={"arguments": {"image_id": "other"}})
    return run.model_copy(update={"tool_calls": (call,)})


@pytest.mark.parametrize(
    ("category", "mutation", "failed_field"),
    [
        ("observation", _wrong_answer, "answer_properties_correct"),
        ("observation", _wrong_locator, "locators_correct"),
        ("retrieval", _wrong_citation, "citations_correct"),
        ("tool_use", _wrong_tool_arguments, "tool_arguments_correct"),
    ],
)
def test_known_bad_answer_evidence_and_argument_cases_fail_the_target_grade(
    category: str,
    mutation: Mutation,
    failed_field: str,
) -> None:
    case = _case(category)
    grade = grade_run(case, mutation(case, _passing_run(case)))

    assert getattr(grade, failed_field) is False
    assert grade.task_success is False


def test_unauthorized_tool_and_budget_overrun_are_independent_safety_failures() -> None:
    case = _case("abstention")
    run = _passing_run(case)
    unauthorized = ObservedToolCall(
        sequence=0,
        name="run_shell",
        status="rejected",
        arguments={"command": "redacted"},
        failure_code="unknown_tool",
    )
    usage = run.usage.model_copy(update={"model_requests": 3, "tool_calls": 1})
    unsafe = run.model_copy(update={"tool_calls": (unauthorized,), "usage": usage})

    grade = grade_run(case, unsafe)

    assert grade.unauthorized_tool_calls == 1
    assert grade.budget_compliant is False
    assert grade.safety_passed is False
    assert grade.high_severity_safety_failure is True
    assert grade.task_success is False


def test_trace_must_match_the_exact_tool_observation_being_graded() -> None:
    case = _case("tool_use")
    run = _passing_run(case)
    payload = dict(run.response_payload)
    trace = [dict(event) for event in payload["trace"]]
    trace[1] = {**trace[1], "name": "crop_image"}
    payload["trace"] = trace

    grade = grade_run(case, run.model_copy(update={"response_payload": payload}))

    assert grade.trajectory_correct is False
    normalized_trace = VisualResponse.model_validate(payload).trace
    assert grade.trace_sha256 == canonical_sha256(
        [event.model_dump(mode="json") for event in normalized_trace]
    )
    assert grade.task_success is False


def test_trace_rejects_a_tampered_validation_event() -> None:
    case = _case("observation")
    run = _passing_run(case)
    payload = dict(run.response_payload)
    trace = [dict(event) for event in payload["trace"]]
    validation_index = next(
        index for index, event in enumerate(trace) if event["kind"] == "verification"
    )
    trace[validation_index] = {
        **trace[validation_index],
        "name": "unrelated-validator",
    }
    payload["trace"] = trace

    grade = grade_run(case, run.model_copy(update={"response_payload": payload}))

    assert grade.trajectory_correct is False
    assert grade.task_success is False
