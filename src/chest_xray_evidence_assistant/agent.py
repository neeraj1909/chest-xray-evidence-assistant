"""Bounded single-agent request path for the evidence assistant."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, UsageLimits

from .models import (
    AgentTraceEvent,
    EvidenceRequest,
    Identifier,
    NormalizedBoundingBox,
    TraceValue,
    VisualResponse,
)
from .retrieval.records import ScoredChunk
from .runtime import BudgetExceeded, RunBudget
from .tools import (
    BoundedToolExecutor,
    CropImageResult,
    ImageMetadataResult,
    ImageToolRejected,
    ImageToolsPort,
    ReferenceQuery,
    ReferenceRetrievalPort,
    ReferenceTopK,
    ToolCallRecord,
    ToolCallRejected,
    ToolExecutionRejected,
)

AGENT_NAME = "cxr-evidence-agent"
SYSTEM_INSTRUCTIONS = (
    "Return only a non-diagnostic VisualResponse grounded in the supplied image. "
    "Use abstain or needs_clarification when evidence is insufficient."
)


@dataclass(frozen=True, slots=True)
class AgentToolServices:
    """Explicit server-owned capabilities available to one agent run."""

    image_tools: ImageToolsPort
    reference_retriever: ReferenceRetrievalPort | None = None


async def crop_image(
    context: RunContext[BoundedToolExecutor],
    image_id: Identifier,
    box: NormalizedBoundingBox,
) -> CropImageResult:
    """Create a deterministic crop from an authorized server-owned image ID."""

    try:
        result = await context.deps.execute(
            "crop_image",
            {"image_id": image_id, "box": box.model_dump(mode="json")},
        )
    except (BudgetExceeded, ImageToolRejected, ToolCallRejected, ToolExecutionRejected) as error:
        raise ModelRetry(error.code) from None
    if not isinstance(result, CropImageResult):
        raise ModelRetry("tool_result_mismatch")
    return result


async def get_image_metadata(
    context: RunContext[BoundedToolExecutor],
    image_id: Identifier,
) -> ImageMetadataResult:
    """Return verified metadata for an authorized server-owned image ID."""

    try:
        result = await context.deps.execute("get_image_metadata", {"image_id": image_id})
    except (BudgetExceeded, ImageToolRejected, ToolCallRejected, ToolExecutionRejected) as error:
        raise ModelRetry(error.code) from None
    if not isinstance(result, ImageMetadataResult):
        raise ModelRetry("tool_result_mismatch")
    return result


async def retrieve_reference(
    context: RunContext[BoundedToolExecutor],
    query: ReferenceQuery,
    top_k: ReferenceTopK = 5,
) -> tuple[ScoredChunk, ...]:
    """Retrieve at most five provenance-bearing public reference records."""

    try:
        result = await context.deps.execute(
            "retrieve_reference",
            {"query": query, "top_k": top_k},
        )
    except (BudgetExceeded, ImageToolRejected, ToolCallRejected, ToolExecutionRejected) as error:
        raise ModelRetry(error.code) from None
    if not isinstance(result, tuple):
        raise ModelRetry("tool_result_mismatch")
    return result


AGENT_TOOLS = (crop_image, get_image_metadata, retrieve_reference)


def create_agent(
    model: Any,
    *,
    enable_tools: bool = False,
) -> Agent[Any, VisualResponse]:
    """Create one provider-injected agent without selecting a live model implicitly."""

    if model is None:
        raise ValueError("an explicit model or deterministic test model is required")

    return Agent(
        model=model,
        output_type=VisualResponse,
        instructions=SYSTEM_INSTRUCTIONS,
        name=AGENT_NAME,
        retries=1,
        deps_type=BoundedToolExecutor,
        tools=AGENT_TOOLS if enable_tools else (),
        max_concurrency=1,
        defer_model_check=True,
    )


def _validate_image_bytes(request: EvidenceRequest, image_bytes: bytes) -> None:
    if len(image_bytes) != request.image.byte_size:
        raise ValueError("image bytes do not match the declared asset size")
    if hashlib.sha256(image_bytes).hexdigest() != request.image.sha256:
        raise ValueError("image bytes do not match the declared asset digest")


def _prompt_parts(request: EvidenceRequest, image_bytes: bytes) -> list[Any]:
    question = request.question.question
    if request.question.additional_context is not None:
        question = f"{question}\nAdditional context: {request.question.additional_context}"

    return [
        question,
        BinaryContent(data=image_bytes, media_type=request.image.media_type),
    ]


def _validate_provenance(
    request: EvidenceRequest,
    response: VisualResponse,
    tool_records: tuple[ToolCallRecord, ...],
    authorized_retrieval_results: tuple[ScoredChunk, ...],
) -> VisualResponse:
    allowed_image_ids = {request.image.image_id}
    allowed_image_ids.update(
        str(record.safe_result["output_image_id"])
        for record in tool_records
        if "output_image_id" in record.safe_result
    )
    for evidence in response.visual_evidence:
        if evidence.locator.image_id not in allowed_image_ids:
            raise ValueError("visual evidence references an image outside the request")
    has_retrieval_trace = any(
        record.name == "retrieve_reference" and record.status == "succeeded"
        for record in tool_records
    )
    for evidence in response.source_evidence:
        is_authorized = any(
            evidence.document_id == result.chunk.document_id
            and evidence.section == result.chunk.source_span.section
            and evidence.source_url == result.chunk.source_span.source_url
            and evidence.locator == result.chunk.source_span.locator
            and evidence.excerpt in result.chunk.text
            for result in authorized_retrieval_results
        )
        if not has_retrieval_trace or not is_authorized:
            raise ValueError("source evidence is not present in an authorized retrieval result")
    return response


def _digest_request(request: EvidenceRequest, image_bytes: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    hasher.update(b"\x00")
    hasher.update(request.model_dump_json().encode("utf-8"))
    return hasher.hexdigest()


def _trace_identifier(value: str, fallback: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9._:-]+", "-", value).strip("-._:")
    return identifier[:128] or fallback


def _tool_trace_attributes(record: ToolCallRecord) -> dict[str, TraceValue]:
    attributes: dict[str, TraceValue] = {
        "replay_id": record.replay_id,
        "tool_calls": record.budget_after.tool_calls,
        "image_byte_count": record.budget_after.image_bytes,
    }
    projection_names = {
        "output_sha256": "artifact_sha256",
        "chunk_ids_sha256": "chunk_ids_sha256",
        "document_ids_sha256": "document_ids_sha256",
        "result_count": "result_count",
        "width_px": "width_px",
        "height_px": "height_px",
    }
    for source_name, trace_name in projection_names.items():
        if source_name in record.safe_result:
            attributes[trace_name] = record.safe_result[source_name]
    return attributes


def _with_runtime_trace(
    response: VisualResponse,
    *,
    request_digest: str,
    model_name: str,
    provider_name: str,
    model_requests: int,
    tool_records: tuple[ToolCallRecord, ...] = (),
) -> VisualResponse:
    output_digest = hashlib.sha256(
        response.model_dump_json(exclude={"trace"}).encode("utf-8")
    ).hexdigest()
    trace = [
        AgentTraceEvent(
            sequence=0,
            kind="request",
            status="started",
            name=AGENT_NAME,
            input_sha256=request_digest,
        ),
    ]
    trace.extend(
        AgentTraceEvent(
            sequence=index + 1,
            kind="tool",
            status=record.status,
            name=record.name,
            duration_ms=max(
                0,
                record.budget_after.elapsed_ms - record.budget_before.elapsed_ms,
            ),
            input_sha256=record.arguments_sha256,
            output_sha256=record.result_sha256,
            failure_code=record.failure_code,
            attributes=_tool_trace_attributes(record),
        )
        for index, record in enumerate(tool_records)
    )
    model_sequence = len(trace)
    trace.extend(
        [
            AgentTraceEvent(
                sequence=model_sequence,
                kind="model",
                status="succeeded",
                name=_trace_identifier(model_name, "model"),
                input_sha256=request_digest,
                output_sha256=output_digest,
                attributes={
                    "provider": provider_name,
                    "model_requests": model_requests,
                    "tool_calls": len(tool_records),
                },
            ),
            AgentTraceEvent(
                sequence=model_sequence + 1,
                kind="validation",
                status="succeeded",
                name="visual-response",
                input_sha256=output_digest,
                output_sha256=output_digest,
            ),
            AgentTraceEvent(
                sequence=model_sequence + 2,
                kind="final",
                status="succeeded",
                name=response.status,
                output_sha256=output_digest,
            ),
        ]
    )
    payload = response.model_dump(mode="json")
    payload["trace"] = [event.model_dump(mode="json") for event in trace]
    return VisualResponse.model_validate(payload)


async def run_evidence_request(
    request: EvidenceRequest,
    image_bytes: bytes,
    *,
    model: Any,
    tool_services: AgentToolServices | None = None,
) -> VisualResponse:
    """Run one bounded image-plus-question request and return a safe response."""

    _validate_image_bytes(request, image_bytes)
    budget = RunBudget(request.limits)
    budget.consume_image_bytes(len(image_bytes))
    request_digest = _digest_request(request, image_bytes)
    executor = (
        BoundedToolExecutor(
            budget=budget,
            image_tools=tool_services.image_tools,
            reference_retriever=tool_services.reference_retriever,
            run_sha256=request_digest,
            allowed_image_ids=frozenset({request.image.image_id}),
        )
        if tool_services is not None
        else None
    )
    agent = create_agent(model, enable_tools=executor is not None)
    cost_limit = (
        None
        if getattr(model, "system", None) in {"test", "function"}
        else Decimal(str(request.limits.max_estimated_cost_usd))
    )
    try:
        async with asyncio.timeout(budget.remaining_seconds()):
            result = await agent.run(
                _prompt_parts(request, image_bytes),
                deps=executor,
                usage_limits=UsageLimits(
                    request_limit=request.limits.max_model_requests,
                    tool_calls_limit=request.limits.max_tool_calls,
                    output_tokens_limit=request.limits.max_output_tokens,
                    cost_limit=cost_limit,
                ),
            )
    except TimeoutError:
        raise BudgetExceeded("timeout") from None
    usage = result.usage
    if usage.requests:
        budget.consume_model_requests(usage.requests)
    if usage.tool_calls and executor is None:
        budget.consume_tool_calls(usage.tool_calls)
    if usage.output_tokens:
        budget.consume_output_tokens(usage.output_tokens)
    if usage.cost:
        budget.consume_estimated_cost(usage.cost)
    response = VisualResponse.model_validate(result.output.model_dump(mode="json"))
    tool_records = executor.records if executor is not None else ()
    authorized_retrieval_results = (
        executor.authorized_retrieval_results if executor is not None else ()
    )
    response = _validate_provenance(
        request,
        response,
        tool_records,
        authorized_retrieval_results,
    )
    return _with_runtime_trace(
        response,
        request_digest=request_digest,
        model_name=result.response.model_name or "unknown-model",
        provider_name=result.response.provider_name or "unknown-provider",
        model_requests=result.usage.requests,
        tool_records=tool_records,
    )
