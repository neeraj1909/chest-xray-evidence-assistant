from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.models import RunLimits
from chest_xray_evidence_assistant.retrieval import (
    InMemoryBM25Index,
    LexicalReferenceService,
    load_reference_corpus,
)
from chest_xray_evidence_assistant.runtime import BudgetExceeded, RunBudget
from chest_xray_evidence_assistant.tools import (
    BoundedToolExecutor,
    FixtureImageTools,
    ToolCallRecord,
    ToolCallRejected,
    ToolExecutionRejected,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "fixtures" / "manifest.json"
REFERENCE_MANIFEST_PATH = REPO_ROOT / "data" / "references" / "manifest.json"
TRACE_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "traces" / "tool-trajectory.json"
RUN_SHA256 = "a" * 64


def make_executor(
    *,
    max_tool_calls: int = 3,
    retriever: object | None = None,
    allowed_image_ids: frozenset[str] = frozenset(
        {"synthetic-full-frame", "synthetic-crop-target"}
    ),
    monotonic: Callable[[], float] = lambda: 10.0,
) -> BoundedToolExecutor:
    return BoundedToolExecutor(
        budget=RunBudget(
            RunLimits(max_tool_calls=max_tool_calls),
            monotonic=monotonic,
        ),
        image_tools=FixtureImageTools.from_manifest(MANIFEST_PATH),
        reference_retriever=retriever,
        run_sha256=RUN_SHA256,
        allowed_image_ids=allowed_image_ids,
    )


def test_successful_image_tool_records_safe_deterministic_evidence() -> None:
    executor = make_executor()

    metadata = asyncio.run(
        executor.execute("get_image_metadata", {"image_id": "synthetic-crop-target"})
    )
    crop = asyncio.run(
        executor.execute(
            "crop_image",
            {
                "image_id": "synthetic-crop-target",
                "box": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
            },
        )
    )

    assert metadata.image.image_id == "synthetic-crop-target"
    assert crop.output_sha256 == (
        "14aba6019240908536da3a24fd6df302ef176102fc16c02cf701da374500942f"
    )
    records = executor.records
    assert [record.status for record in records] == ["succeeded", "succeeded"]
    assert [record.name for record in records] == ["get_image_metadata", "crop_image"]
    assert records[1].budget_after.tool_calls == 2
    assert records[1].budget_after.image_bytes == 79
    assert records[1].safe_result["output_sha256"] == crop.output_sha256
    assert all(len(record.replay_id) == 64 for record in records)
    serialized = json.dumps([record.model_dump(mode="json") for record in records])
    assert '"content"' not in serialized
    assert "89504e47" not in serialized


def test_unauthorized_call_is_counted_and_redacted() -> None:
    executor = make_executor()

    with pytest.raises(ToolCallRejected, match="^unknown_tool$"):
        asyncio.run(
            executor.execute(
                "run_shell",
                {"command": "cat /etc/passwd", "api_key": "must-not-leak"},
            )
        )

    record = executor.records[0]
    serialized = record.model_dump_json()
    assert record.name == "unrecognized"
    assert record.status == "rejected"
    assert record.failure_code == "unknown_tool"
    assert record.safe_arguments == {}
    assert record.budget_after.tool_calls == 1
    assert "run_shell" not in serialized
    assert "/etc/passwd" not in serialized
    assert "must-not-leak" not in serialized


def test_exhausted_parent_budget_records_rejection_without_incrementing_past_limit() -> None:
    executor = make_executor(max_tool_calls=1)
    asyncio.run(executor.execute("get_image_metadata", {"image_id": "synthetic-full-frame"}))

    with pytest.raises(BudgetExceeded, match="^tool_call_limit$"):
        asyncio.run(executor.execute("get_image_metadata", {"image_id": "synthetic-full-frame"}))

    assert [record.status for record in executor.records] == ["succeeded", "rejected"]
    assert executor.records[-1].failure_code == "tool_call_limit"
    assert executor.records[-1].budget_after.tool_calls == 1


def test_image_tool_cannot_cross_the_run_authorization_boundary() -> None:
    executor = make_executor(allowed_image_ids=frozenset({"synthetic-full-frame"}))

    with pytest.raises(ToolExecutionRejected, match="^image_not_authorized$"):
        asyncio.run(executor.execute("get_image_metadata", {"image_id": "synthetic-crop-target"}))

    assert executor.records[0].status == "rejected"
    assert executor.records[0].failure_code == "image_not_authorized"


def test_retrieval_projection_hashes_query_and_result_without_retaining_text() -> None:
    retriever = LexicalReferenceService(
        InMemoryBM25Index.from_corpus(load_reference_corpus(REFERENCE_MANIFEST_PATH))
    )
    executor = make_executor(retriever=retriever)
    sensitive_query = "private patient phrase must not survive"

    result = asyncio.run(
        executor.execute("retrieve_reference", {"query": sensitive_query, "top_k": 1})
    )

    assert result[0].chunk.document_id.startswith("document-")
    serialized = executor.records[0].model_dump_json()
    assert sensitive_query not in serialized
    assert executor.records[0].safe_arguments["top_k"] == 1
    assert len(str(executor.records[0].safe_arguments["query_sha256"])) == 64
    assert executor.records[0].safe_result["result_count"] == 1


def test_successful_and_rejected_calls_replay_to_the_same_records() -> None:
    original = make_executor()
    success_arguments = {"image_id": "synthetic-full-frame"}
    rejected_arguments = {"authorization": "Bearer secret"}
    asyncio.run(original.execute("get_image_metadata", success_arguments))
    with pytest.raises(ToolCallRejected):
        asyncio.run(original.execute("open_browser", rejected_arguments))

    replay = make_executor()
    replayed_success = asyncio.run(
        replay.replay(original.records[0], "get_image_metadata", success_arguments)
    )
    replayed_rejection = asyncio.run(
        replay.replay(original.records[1], "open_browser", rejected_arguments)
    )

    assert replayed_success == original.records[0]
    assert replayed_rejection == original.records[1]


def test_replay_ignores_wall_clock_drift_but_preserves_stable_evidence() -> None:
    def ticking_clock(step: float) -> Callable[[], float]:
        current = [10.0]

        def tick() -> float:
            current[0] += step
            return current[0]

        return tick

    arguments = {"image_id": "synthetic-full-frame"}
    original = make_executor(monotonic=ticking_clock(0.001))
    asyncio.run(original.execute("get_image_metadata", arguments))
    replay = make_executor(monotonic=ticking_clock(0.01))

    actual = asyncio.run(replay.replay(original.records[0], "get_image_metadata", arguments))

    assert actual.replay_id == original.records[0].replay_id
    assert actual.result_sha256 == original.records[0].result_sha256
    assert actual.budget_after.elapsed_ms != original.records[0].budget_after.elapsed_ms


def test_trace_fixture_round_trips_and_matches_live_deterministic_records() -> None:
    raw = json.loads(TRACE_FIXTURE.read_text(encoding="utf-8"))
    expected = tuple(ToolCallRecord.model_validate(item) for item in raw["records"])
    executor = make_executor()
    asyncio.run(executor.execute("get_image_metadata", {"image_id": "synthetic-full-frame"}))
    with pytest.raises(ToolCallRejected):
        asyncio.run(executor.execute("open_browser", {"api_key": "fixture-secret"}))

    assert executor.records == expected
    assert "fixture-secret" not in TRACE_FIXTURE.read_text(encoding="utf-8")
