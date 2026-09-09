from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import chest_xray_evidence_assistant.ui as ui_module
from chest_xray_evidence_assistant.fixtures import load_fixture_manifest
from chest_xray_evidence_assistant.models import (
    AgentTraceEvent,
    ImageLocator,
    NormalizedBoundingBox,
    VisualEvidence,
    VisualResponse,
)
from chest_xray_evidence_assistant.ui import (
    DEMO_QUESTION,
    UIInputRejected,
    build_demo_request,
    launch_app,
    process_submission,
    render_response,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "data" / "fixtures"


def fixture_bytes(image_id: str) -> bytes:
    manifest = load_fixture_manifest(FIXTURE_ROOT / "manifest.json")
    fixture = next(item for item in manifest.fixtures if item.asset.image_id == image_id)
    return (FIXTURE_ROOT / fixture.path).read_bytes()


def test_demo_request_accepts_only_attested_registered_fixture_bytes() -> None:
    image_bytes = fixture_bytes("synthetic-full-frame")

    request = build_demo_request(
        image_bytes=image_bytes,
        question=f"  {DEMO_QUESTION}  ",
        additional_context="   ",
        data_confirmed=True,
    )

    assert request.image.image_id == "synthetic-full-frame"
    assert request.question.question == DEMO_QUESTION
    assert request.question.additional_context is None
    assert request.image.license_status == "synthetic"
    assert request.image.contains_phi is False
    assert request.limits.max_tool_calls == 3

    with pytest.raises(UIInputRejected, match="^unsupported_image$"):
        build_demo_request(
            image_bytes=b"not a registered fixture",
            question=DEMO_QUESTION,
            additional_context=None,
            data_confirmed=True,
        )


@pytest.mark.parametrize(
    ("image_bytes", "question", "data_confirmed", "code"),
    [
        (None, DEMO_QUESTION, True, "missing_image"),
        (b"", DEMO_QUESTION, True, "missing_image"),
        (b"registered later", " ", True, "invalid_question"),
        (b"registered later", DEMO_QUESTION, False, "data_not_confirmed"),
    ],
)
def test_invalid_ui_inputs_have_stable_non_echoing_codes(
    image_bytes: bytes | None,
    question: str,
    data_confirmed: bool,
    code: str,
) -> None:
    with pytest.raises(UIInputRejected, match=f"^{code}$") as caught:
        build_demo_request(
            image_bytes=image_bytes,
            question=question,
            additional_context=None,
            data_confirmed=data_confirmed,
        )

    assert caught.value.code == code
    if question.strip():
        assert question.strip() not in str(caught.value)


def test_oversized_whitespace_context_is_not_silently_accepted() -> None:
    with pytest.raises(UIInputRejected, match="^invalid_context$"):
        build_demo_request(
            image_bytes=fixture_bytes("synthetic-full-frame"),
            question=DEMO_QUESTION,
            additional_context=" " * 8_001,
            data_confirmed=True,
        )


def test_offline_demo_exposes_answer_clarification_and_abstention_states() -> None:
    answered = asyncio.run(
        process_submission(
            fixture_bytes("synthetic-full-frame"),
            DEMO_QUESTION,
            None,
            True,
        )
    )
    clarification = asyncio.run(
        process_submission(
            fixture_bytes("synthetic-full-frame"),
            "Can you diagnose this image?",
            None,
            True,
        )
    )
    abstention = asyncio.run(
        process_submission(
            fixture_bytes("synthetic-abstention"),
            DEMO_QUESTION,
            None,
            True,
        )
    )

    assert answered.state == "answered"
    assert answered.answer is not None
    assert answered.visual_evidence[0].locator == "full frame: synthetic-full-frame"
    assert any(
        event.kind == "tool" and event.name == "get_image_metadata" and event.status == "succeeded"
        for event in answered.trace
    )
    assert clarification.state == "needs_clarification"
    assert "non-diagnostic observation" in clarification.message
    assert abstention.state == "abstain"
    assert "Insufficient visual evidence" in abstention.message
    assert all(result.latency_ms >= 0 for result in (answered, clarification, abstention))


def test_rendered_trace_omits_prompts_bytes_digests_and_attributes() -> None:
    response = VisualResponse(
        status="answered",
        answer="A bounded synthetic observation.",
        confidence=0.75,
        observations=["One abstract region is visible."],
        visual_evidence=[
            VisualEvidence(
                locator=ImageLocator(image_id="synthetic-full-frame", kind="full_frame"),
                description="The complete synthetic frame.",
                confidence=0.75,
            )
        ],
        uncertainty=["No clinical interpretation was performed."],
        trace=[
            AgentTraceEvent(
                sequence=0,
                kind="tool",
                status="succeeded",
                name="get_image_metadata",
                duration_ms=17,
                input_sha256="a" * 64,
                output_sha256="b" * 64,
                attributes={"provider": "private-provider", "replay_id": "c" * 64},
            )
        ],
    )

    rendered = render_response(response, latency_ms=19)
    serialized = rendered.model_dump_json()

    assert rendered.trace[0].model_dump() == {
        "sequence": 0,
        "kind": "tool",
        "name": "get_image_metadata",
        "status": "succeeded",
        "duration_ms": 17,
    }
    assert rendered.latency_ms == 19
    assert "private-provider" not in serialized
    assert "a" * 64 not in serialized
    assert "b" * 64 not in serialized
    assert "c" * 64 not in serialized


def test_rendered_bounding_box_uses_normalized_coordinates() -> None:
    response = VisualResponse(
        status="answered",
        answer="A bounded synthetic observation.",
        confidence=0.75,
        observations=["One abstract region is visible."],
        visual_evidence=[
            VisualEvidence(
                locator=ImageLocator(
                    image_id="synthetic-crop-target",
                    kind="bounding_box",
                    box=NormalizedBoundingBox(
                        x_min=0.25,
                        y_min=0.125,
                        x_max=0.75,
                        y_max=0.875,
                    ),
                ),
                description="A normalized synthetic region.",
                confidence=0.75,
            )
        ],
        uncertainty=["No clinical interpretation was performed."],
    )

    rendered = render_response(response, latency_ms=1)

    assert rendered.visual_evidence[0].locator == (
        "bounding box: synthetic-crop-target (0.250, 0.125)–(0.750, 0.875)"
    )


def test_internal_runner_failure_returns_a_fixed_recovery_message() -> None:
    async def failing_runner(request: object, image_bytes: bytes) -> VisualResponse:
        del request, image_bytes
        raise RuntimeError("credential=should-never-be-rendered")

    result = asyncio.run(
        process_submission(
            fixture_bytes("synthetic-full-frame"),
            DEMO_QUESTION,
            None,
            True,
            runner=failing_runner,
        )
    )

    assert result.state == "error"
    assert result.error_code == "run_failed"
    assert "credential" not in result.model_dump_json()
    assert "Try again" in result.message


def test_launch_boundary_is_local_private_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.launch_arguments: dict[str, object] = {}

        def launch(self, **kwargs: object) -> None:
            self.launch_arguments = kwargs

    app = FakeApp()
    monkeypatch.setattr(ui_module, "build_app", lambda: app)

    assert launch_app(port=7_861, prevent_thread_lock=True) is app
    assert app.launch_arguments["server_name"] == "127.0.0.1"
    assert app.launch_arguments["server_port"] == 7_861
    assert app.launch_arguments["share"] is False
    assert app.launch_arguments["run_history"] is False
    assert app.launch_arguments["enable_monitoring"] is False
    assert app.launch_arguments["strict_cors"] is True
    assert app.launch_arguments["max_file_size"] == 20_000_000
    assert app.launch_arguments["allowed_paths"] == []
    assert app.launch_arguments["blocked_paths"] == [str(Path.cwd().resolve())]
    assert app.launch_arguments["mcp_server"] is False

    with pytest.raises(ValueError, match="port"):
        launch_app(port=80)
