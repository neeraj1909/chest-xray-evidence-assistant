from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.models import AgentTraceEvent, RunTrace
from chest_xray_evidence_assistant.observability import (
    RunTraceCollector,
    TraceRedactionReport,
    audit_trace_redaction,
)

RUN_REPLAY_ID = "a" * 64
TRACE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "traces"
EVENT_KINDS = (
    "run",
    "model_request",
    "tool_call",
    "retrieval",
    "verification",
    "abstention",
    "error",
    "budget",
    "final_status",
)


@pytest.mark.parametrize("kind", EVENT_KINDS)
def test_structured_event_vocabulary_carries_one_run_replay_identity(kind: str) -> None:
    event = AgentTraceEvent(
        sequence=0,
        run_replay_id=RUN_REPLAY_ID,
        kind=kind,
        status="failed" if kind == "error" else "succeeded",
        name="test-event",
        failure_code="dependency_unavailable" if kind == "error" else None,
    )

    assert event.kind == kind
    assert event.run_replay_id == RUN_REPLAY_ID


@pytest.mark.parametrize(
    "attributes",
    [
        {"raw_prompt": "do not retain this"},
        {"patientName": "Example Person"},
        {"api-key": "example credential"},
        {"image_bytes": "89504e47"},
        {"patient_name": "Example Person"},
        {"safe_detail": "Bearer should-not-survive"},
        {"safe_detail": "sk-examplecredential123456"},
        {"safe_detail": "person@example.org"},
    ],
)
def test_structured_events_reject_sensitive_attribute_keys_and_values(
    attributes: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="sensitive"):
        AgentTraceEvent(
            sequence=0,
            run_replay_id=RUN_REPLAY_ID,
            kind="error",
            status="failed",
            name="run-error",
            failure_code="dependency_unavailable",
            attributes=attributes,
        )


def test_error_event_requires_a_stable_failure_code() -> None:
    with pytest.raises(ValidationError, match="failure code"):
        AgentTraceEvent(
            sequence=0,
            run_replay_id=RUN_REPLAY_ID,
            kind="error",
            status="failed",
            name="run-error",
        )


def test_run_trace_rejects_mixed_replay_identities() -> None:
    events = (
        AgentTraceEvent(
            sequence=0,
            run_replay_id=RUN_REPLAY_ID,
            kind="run",
            status="started",
            name="cxr-evidence-agent",
            input_sha256=RUN_REPLAY_ID,
        ),
        AgentTraceEvent(
            sequence=1,
            run_replay_id="b" * 64,
            kind="final_status",
            status="failed",
            name="failed",
            failure_code="run_failed",
        ),
    )

    with pytest.raises(ValidationError, match="replay identity"):
        RunTrace(run_replay_id=RUN_REPLAY_ID, events=events)


def test_trace_collector_records_exactly_one_terminal_trace() -> None:
    trace = RunTrace(
        run_replay_id=RUN_REPLAY_ID,
        events=(
            AgentTraceEvent(
                sequence=0,
                run_replay_id=RUN_REPLAY_ID,
                kind="run",
                status="started",
                name="cxr-evidence-agent",
                input_sha256=RUN_REPLAY_ID,
            ),
            AgentTraceEvent(
                sequence=1,
                run_replay_id=RUN_REPLAY_ID,
                kind="budget",
                status="succeeded",
                name="run-budget",
            ),
            AgentTraceEvent(
                sequence=2,
                run_replay_id=RUN_REPLAY_ID,
                kind="error",
                status="failed",
                name="run-error",
                failure_code="run_failed",
            ),
            AgentTraceEvent(
                sequence=3,
                run_replay_id=RUN_REPLAY_ID,
                kind="final_status",
                status="failed",
                name="failed",
                failure_code="run_failed",
            ),
        ),
    )
    collector = RunTraceCollector()

    collector.record(trace)

    assert collector.trace == trace
    with pytest.raises(RuntimeError, match="already contains"):
        collector.record(trace)


@pytest.mark.parametrize("duplicate_kind", ["run", "budget", "final_status"])
def test_run_trace_rejects_duplicate_singleton_events(duplicate_kind: str) -> None:
    trace = RunTrace.model_validate_json((TRACE_ROOT / "successful-run.json").read_bytes())
    events = list(trace.events)
    duplicate = next(event for event in events if event.kind == duplicate_kind)
    events.insert(-1, duplicate)
    renumbered = tuple(
        event.model_copy(update={"sequence": sequence}) for sequence, event in enumerate(events)
    )

    with pytest.raises(ValidationError, match="requires one run, budget"):
        RunTrace(run_replay_id=trace.run_replay_id, events=renumbered)


def test_failed_run_trace_requires_matching_terminal_failure_metadata() -> None:
    trace = RunTrace.model_validate_json((TRACE_ROOT / "failed-run.json").read_bytes())
    events = list(trace.events)
    events[-1] = events[-1].model_copy(update={"failure_code": "different_failure"})

    with pytest.raises(ValidationError, match="inconsistent failure metadata"):
        RunTrace(run_replay_id=trace.run_replay_id, events=tuple(events))


def test_trace_fixtures_cover_the_complete_event_vocabulary() -> None:
    traces = tuple(
        RunTrace.model_validate_json((TRACE_ROOT / filename).read_bytes())
        for filename in ("successful-run.json", "failed-run.json")
    )

    observed_kinds = {event.kind for trace in traces for event in trace.events}

    assert observed_kinds == set(EVENT_KINDS)


def test_redaction_report_is_reproducible_and_retains_only_probe_hashes() -> None:
    traces = tuple(
        RunTrace.model_validate_json((TRACE_ROOT / filename).read_bytes())
        for filename in ("successful-run.json", "failed-run.json")
    )
    probes = {
        "raw_prompt": "unique dictated prompt must not survive",
        "api_key": "sk-examplecredential123456",
        "image_bytes": b"\x89PNG\r\n\x1a\nprivate-image-body",
        "patient_email": "person@example.org",
    }

    report = audit_trace_redaction(traces, probes)
    expected = TraceRedactionReport.model_validate_json(
        (TRACE_ROOT / "redaction-report.json").read_bytes()
    )

    assert report == expected
    assert report.passed is True
    assert report.violations == ()
    serialized = report.model_dump_json()
    assert all(
        (value.hex() if isinstance(value, bytes) else value) not in serialized
        for value in probes.values()
    )
