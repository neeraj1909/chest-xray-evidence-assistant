"""Safe local UI service and Gradio composition for the offline demo."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import Field, HttpUrl

from .agent import AgentToolServices, run_evidence_request
from .fixtures import load_fixture_manifest
from .models import (
    ContractModel,
    EvidenceRequest,
    Identifier,
    ImageLocator,
    QuestionContext,
    RunLimits,
    ShortText,
    VisualResponse,
)
from .tools import FixtureImageTools

DEMO_QUESTION = "What can be observed in this synthetic fixture?"
MAX_UPLOAD_BYTES = 20_000_000
UIInputFailureCode: TypeAlias = Literal[
    "data_not_confirmed",
    "fixture_unavailable",
    "invalid_context",
    "invalid_question",
    "missing_image",
    "unsupported_image",
]
UIState: TypeAlias = Literal[
    "answered",
    "needs_clarification",
    "abstain",
    "error",
]


class UIInputRejected(ValueError):
    """Reject local UI input with a stable code and no user-content echo."""

    def __init__(self, code: UIInputFailureCode) -> None:
        self.code = code
        super().__init__(code)


class UIVisualEvidence(ContractModel):
    locator: ShortText
    description: ShortText
    confidence: float = Field(ge=0, le=1)


class UISourceEvidence(ContractModel):
    document_id: Identifier
    section: ShortText
    source_url: HttpUrl
    locator: ShortText
    excerpt: ShortText
    relevance_score: float = Field(ge=0, le=1)


class UITraceEvent(ContractModel):
    """The exact allow-list of trace fields safe for browser display."""

    sequence: int = Field(ge=0)
    kind: Literal["request", "model", "tool", "validation", "final"]
    name: Identifier | None = None
    status: Literal["started", "succeeded", "rejected", "failed"]
    duration_ms: int | None = Field(default=None, ge=0)


class UIRenderedResult(ContractModel):
    state: UIState
    title: ShortText
    message: ShortText
    answer: ShortText | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    observations: tuple[ShortText, ...] = ()
    uncertainty: tuple[ShortText, ...] = ()
    visual_evidence: tuple[UIVisualEvidence, ...] = ()
    source_evidence: tuple[UISourceEvidence, ...] = ()
    trace: tuple[UITraceEvent, ...] = ()
    latency_ms: int = Field(ge=0)
    error_code: Identifier | None = None


UIRunner = Callable[[EvidenceRequest, bytes], Awaitable[VisualResponse]]


def _validated_question(question: object, additional_context: object) -> QuestionContext:
    if not isinstance(question, str) or not question.strip() or len(question) > 4_000:
        raise UIInputRejected("invalid_question")
    if additional_context is not None:
        if not isinstance(additional_context, str) or len(additional_context) > 8_000:
            raise UIInputRejected("invalid_context")
        if not additional_context.strip():
            additional_context = None
    return QuestionContext(
        question=question,
        additional_context=additional_context or None,
    )


def build_demo_request(
    *,
    image_bytes: bytes | None,
    question: object,
    additional_context: object,
    data_confirmed: bool,
) -> EvidenceRequest:
    """Build a typed request only for exact, repository-verified demo bytes."""

    if data_confirmed is not True:
        raise UIInputRejected("data_not_confirmed")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise UIInputRejected("missing_image")
    context = _validated_question(question, additional_context)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise UIInputRejected("unsupported_image")

    try:
        manifest = load_fixture_manifest()
    except (OSError, ValueError):
        raise UIInputRejected("fixture_unavailable") from None
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    fixture = next(
        (
            item
            for item in manifest.fixtures
            if item.asset.sha256 == image_sha256 and item.asset.byte_size == len(image_bytes)
        ),
        None,
    )
    if fixture is None:
        raise UIInputRejected("unsupported_image")

    request_material = b"\x00".join(
        (
            image_sha256.encode("ascii"),
            context.model_dump_json().encode("utf-8"),
        )
    )
    request_sha256 = hashlib.sha256(request_material).hexdigest()
    return EvidenceRequest(
        request_id=f"ui-demo-{request_sha256[:32]}",
        image=fixture.asset,
        question=context,
        limits=RunLimits(),
    )


def _offline_payload(request: EvidenceRequest) -> dict[str, object]:
    if request.image.image_id == "synthetic-abstention":
        return {
            "status": "abstain",
            "confidence": 1.0,
            "observations": [],
            "visual_evidence": [],
            "source_evidence": [],
            "uncertainty": [
                "The blank synthetic fixture does not provide sufficient visual evidence."
            ],
            "abstention_reason": "Insufficient visual evidence for a grounded response.",
            "trace": [],
        }
    if request.question.question != DEMO_QUESTION:
        return {
            "status": "needs_clarification",
            "confidence": 0.0,
            "observations": [],
            "visual_evidence": [],
            "source_evidence": [],
            "uncertainty": ["Offline mode supports one explicit synthetic-demo question."],
            "clarification_question": (
                f"For this offline demo, ask this non-diagnostic observation: {DEMO_QUESTION}"
            ),
            "trace": [],
        }
    return {
        "status": "answered",
        "answer": "The verified synthetic fixture contains one abstract grayscale image.",
        "confidence": 1.0,
        "observations": ["One registered synthetic image is available for inspection."],
        "visual_evidence": [
            {
                "locator": {
                    "image_id": request.image.image_id,
                    "kind": "full_frame",
                    "box": None,
                },
                "description": "The complete verified synthetic fixture frame.",
                "confidence": 1.0,
            }
        ],
        "source_evidence": [],
        "uncertainty": [
            "This deterministic fake run does not inspect anatomy or produce clinical findings."
        ],
        "trace": [],
    }


async def run_offline_demo(request: EvidenceRequest, image_bytes: bytes) -> VisualResponse:
    """Exercise the real bounded agent and metadata-tool path without network access."""

    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    request_count = 0

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        nonlocal request_count
        del messages
        request_count += 1
        if request_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "get_image_metadata",
                        {"image_id": request.image.image_id},
                        tool_call_id="ui-metadata-1",
                    )
                ]
            )
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name,
                    _offline_payload(request),
                    tool_call_id="ui-final-1",
                )
            ]
        )

    return await run_evidence_request(
        request,
        image_bytes,
        model=FunctionModel(respond, model_name="offline-ui-fixture-v1"),
        tool_services=AgentToolServices(
            image_tools=FixtureImageTools.from_manifest(),
        ),
    )


def _visual_locator_text(locator: ImageLocator) -> str:
    if locator.kind == "full_frame":
        return f"full frame: {locator.image_id}"
    box = locator.box
    if box is None:
        raise ValueError("bounding-box locator has no coordinates")
    return (
        f"bounding box: {locator.image_id} "
        f"({box.x_min:.3f}, {box.y_min:.3f})–({box.x_max:.3f}, {box.y_max:.3f})"
    )


def render_response(response: VisualResponse, *, latency_ms: int) -> UIRenderedResult:
    """Project a response to the browser-safe display allow-list."""

    titles = {
        "answered": "Evidence-linked observation",
        "needs_clarification": "A narrower question is needed",
        "abstain": "No grounded answer",
    }
    if response.status == "answered":
        message = "Completed in bounded offline-demo mode. This is not a diagnosis."
    elif response.status == "needs_clarification":
        message = response.clarification_question or "Please narrow the question."
    else:
        message = response.abstention_reason or "The available evidence is insufficient."

    return UIRenderedResult(
        state=response.status,
        title=titles[response.status],
        message=message,
        answer=response.answer,
        confidence=response.confidence,
        observations=tuple(response.observations),
        uncertainty=tuple(response.uncertainty),
        visual_evidence=tuple(
            UIVisualEvidence(
                locator=_visual_locator_text(item.locator),
                description=item.description,
                confidence=item.confidence,
            )
            for item in response.visual_evidence
        ),
        source_evidence=tuple(
            UISourceEvidence.model_validate(item.model_dump(mode="json"))
            for item in response.source_evidence
        ),
        trace=tuple(
            UITraceEvent(
                sequence=item.sequence,
                kind=item.kind,
                name=item.name,
                status=item.status,
                duration_ms=item.duration_ms,
            )
            for item in response.trace
        ),
        latency_ms=latency_ms,
    )


def _error_result(code: str, *, latency_ms: int) -> UIRenderedResult:
    messages = {
        "data_not_confirmed": (
            "Confirm that the upload is a bundled synthetic fixture with no patient data."
        ),
        "fixture_unavailable": "The verified demo fixtures are unavailable. Restart the app.",
        "invalid_context": "Additional context must contain 1–8,000 characters or be empty.",
        "invalid_question": "Enter a question between 1 and 4,000 characters.",
        "missing_image": "Choose one bundled synthetic PNG before running the demo.",
        "unsupported_image": (
            "This offline demo accepts only the exact bundled synthetic PNG fixtures."
        ),
        "run_failed": "The offline run did not complete. Try again or restart the app.",
    }
    return UIRenderedResult(
        state="error",
        title="Request not run",
        message=messages.get(code, messages["run_failed"]),
        uncertainty=("No result was produced from this request.",),
        latency_ms=latency_ms,
        error_code=code,
    )


async def process_submission(
    image_bytes: bytes | None,
    question: object,
    additional_context: object,
    data_confirmed: bool,
    *,
    runner: UIRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> UIRenderedResult:
    """Validate, run, time, and safely render one local UI submission."""

    started = clock()
    try:
        request = build_demo_request(
            image_bytes=image_bytes,
            question=question,
            additional_context=additional_context,
            data_confirmed=data_confirmed,
        )
        response = await (runner or run_offline_demo)(request, image_bytes)
        latency_ms = max(0, round((clock() - started) * 1_000))
        return render_response(response, latency_ms=latency_ms)
    except UIInputRejected as error:
        latency_ms = max(0, round((clock() - started) * 1_000))
        return _error_result(error.code, latency_ms=latency_ms)
    except Exception:
        latency_ms = max(0, round((clock() - started) * 1_000))
        return _error_result("run_failed", latency_ms=latency_ms)


APP_CSS = """
:root {
  --cxr-ink: #17211f;
  --cxr-muted: #52615d;
  --cxr-surface: #f4f6f3;
  --cxr-paper: #ffffff;
  --cxr-line: #cbd5d1;
  --cxr-accent: #0d6b62;
  --cxr-warning: #865e18;
}

body { background: var(--cxr-surface); color: var(--cxr-ink); }
.gradio-container {
  max-width: 1160px !important;
  margin: 0 auto !important;
  padding: 28px 22px 48px !important;
}
#cxr-masthead { border-bottom: 1px solid var(--cxr-line); padding-bottom: 14px; }
#cxr-masthead h1 { font-size: clamp(1.9rem, 4vw, 3.2rem); letter-spacing: -0.035em; }
#cxr-scope {
  border-left: 5px solid var(--cxr-warning);
  background: #fffaf0;
  padding: 14px 18px;
  margin: 18px 0 22px;
}
.cxr-panel { background: var(--cxr-paper); border: 1px solid var(--cxr-line); padding: 18px; }
#cxr-run { min-height: 48px; background: var(--cxr-accent); border-color: var(--cxr-accent); }
#cxr-run:focus-visible, .gradio-container input:focus-visible,
.gradio-container textarea:focus-visible, .gradio-container button:focus-visible {
  outline: 3px solid #efb64d !important;
  outline-offset: 3px !important;
}
.cxr-kicker {
  color: var(--cxr-accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.cxr-help { color: var(--cxr-muted); font-size: 0.92rem; }
@media (max-width: 700px) {
  .gradio-container { padding: 18px 12px 32px !important; }
  .cxr-panel { padding: 12px; }
  #cxr-upload .filename { padding-inline: 6px; }
  #cxr-upload .download {
    min-width: 5.5rem !important;
    padding-inline: 6px;
  }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition: none !important;
    animation: none !important;
  }
}
"""


def _component_values(result: UIRenderedResult) -> tuple[object, ...]:
    state_labels = {
        "answered": "Answered",
        "needs_clarification": "Needs clarification",
        "abstain": "Abstained",
        "error": "Not run",
    }
    return (
        state_labels[result.state],
        result.title,
        result.message,
        result.answer or "",
        result.confidence,
        "\n".join(f"• {item}" for item in result.observations),
        "\n".join(f"• {item}" for item in result.uncertainty),
        {
            "visual": [item.model_dump(mode="json") for item in result.visual_evidence],
            "sources": [item.model_dump(mode="json") for item in result.source_evidence],
        },
        [item.model_dump(mode="json") for item in result.trace],
        result.latency_ms,
    )


def build_app() -> object:
    """Compose the local Gradio screen without enabling a hosted provider."""

    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("install the UI dependency group") from error

    async def submit(
        image_bytes: bytes | None,
        question: str,
        additional_context: str,
        data_confirmed: bool,
    ) -> tuple[object, ...]:
        return _component_values(
            await process_submission(
                image_bytes,
                question,
                additional_context,
                data_confirmed,
            )
        )

    with gr.Blocks(
        title="Chest X-ray Evidence Assistant — Offline Demo",
        analytics_enabled=False,
    ) as app:
        gr.Markdown(
            """
<span class="cxr-kicker">Learning tool · offline fixture mode</span>

# Chest X-ray Evidence Assistant

Traceable observations over a deliberately small synthetic test boundary.
""",
            elem_id="cxr-masthead",
            sanitize_html=True,
        )
        gr.Markdown(
            """
**Not for diagnosis, treatment, triage, or emergencies.** Do not upload patient
images or protected health information. This build accepts only the exact
bundled synthetic PNGs in `data/fixtures/images/`; all other files fail closed.
""",
            elem_id="cxr-scope",
            sanitize_html=True,
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes="cxr-panel"):
                gr.Markdown("## 1 · Prepare the request")
                upload = gr.File(
                    label="Synthetic PNG",
                    file_count="single",
                    file_types=[".png"],
                    type="binary",
                    height=150,
                    elem_id="cxr-upload",
                )
                question = gr.Textbox(
                    value=DEMO_QUESTION,
                    label="Question",
                    info="Offline mode supports the displayed synthetic-demo question.",
                    lines=2,
                    max_lines=4,
                    max_length=4_000,
                    elem_id="cxr-question",
                )
                context = gr.Textbox(
                    label="Additional context (optional)",
                    info="Do not enter names, identifiers, or clinical records.",
                    lines=2,
                    max_lines=5,
                    max_length=8_000,
                    elem_id="cxr-context",
                )
                confirmation = gr.Checkbox(
                    value=False,
                    label="I confirm this is a bundled synthetic fixture with no patient data.",
                    elem_id="cxr-confirm",
                )
                run_button = gr.Button(
                    "Run offline evidence check",
                    variant="primary",
                    size="lg",
                    elem_id="cxr-run",
                )
                gr.Markdown(
                    "The deterministic fake model uses no credentials, network, or paid call.",
                    elem_classes="cxr-help",
                )

            with gr.Column(scale=7, min_width=360, elem_classes="cxr-panel"):
                gr.Markdown("## 2 · Review the bounded result")
                with gr.Row():
                    outcome = gr.Textbox(label="Outcome", interactive=False, scale=1)
                    confidence = gr.Number(
                        label="Confidence",
                        interactive=False,
                        precision=2,
                        scale=1,
                    )
                    latency = gr.Number(
                        label="Total latency (ms)",
                        interactive=False,
                        precision=0,
                        scale=1,
                    )
                title = gr.Textbox(label="Result", interactive=False)
                message = gr.Textbox(label="Status detail", interactive=False, lines=2)
                answer = gr.Textbox(label="Answer", interactive=False, lines=3)
                observations = gr.Textbox(
                    label="Observations",
                    interactive=False,
                    lines=2,
                    max_lines=4,
                )
                uncertainty = gr.Textbox(
                    label="Uncertainty",
                    interactive=False,
                    lines=3,
                    max_lines=4,
                )
                with gr.Accordion("Evidence details", open=False):
                    evidence = gr.JSON(
                        label="Visual and source evidence",
                        value={"visual": [], "sources": []},
                        buttons=[],
                    )
                with gr.Accordion("Execution details", open=False):
                    trace = gr.JSON(
                        label="Redacted execution trace",
                        value=[],
                        buttons=[],
                    )

        run_button.click(
            submit,
            inputs=[upload, question, context, confirmation],
            outputs=[
                outcome,
                title,
                message,
                answer,
                confidence,
                observations,
                uncertainty,
                evidence,
                trace,
                latency,
            ],
            api_name="offline_demo",
            api_visibility="private",
            show_progress="minimal",
            concurrency_limit=1,
            trigger_mode="once",
        )

    return app.queue(api_open=False, max_size=8, default_concurrency_limit=1)


def launch_app(*, port: int = 7860, prevent_thread_lock: bool = False) -> object:
    """Launch one local-only UI with file, history, and monitoring limits."""

    if isinstance(port, bool) or not 1_024 <= port <= 65_535:
        raise ValueError("port must be between 1024 and 65535")
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=False,
        prevent_thread_lock=prevent_thread_lock,
        show_error=False,
        quiet=False,
        footer_links=[],
        run_history=False,
        allowed_paths=[],
        blocked_paths=[str(Path.cwd().resolve())],
        max_file_size=MAX_UPLOAD_BYTES,
        enable_monitoring=False,
        strict_cors=True,
        app_kwargs={"docs_url": None, "redoc_url": None, "openapi_url": None},
        state_session_capacity=32,
        mcp_server=False,
        pwa=False,
        ssr_mode=False,
        css=APP_CSS,
    )
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local offline fixture UI.")
    parser.add_argument("--port", type=int, default=7_860)
    arguments = parser.parse_args(argv)
    launch_app(port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
