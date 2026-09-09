"""Bounded single-agent request path for the evidence assistant."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Never

from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from .models import (
    AgentTraceEvent,
    EvidenceRequest,
    Identifier,
    NormalizedBoundingBox,
    RunTrace,
    TraceValue,
    VisualResponse,
)
from .observability import RunTraceCollector
from .retrieval.records import ScoredChunk
from .runtime import BudgetExceeded, BudgetSnapshot, DependencyUnavailable, RunBudget
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


class _BudgetedModel(WrapperModel):
    """Charge the parent budget immediately before every provider request."""

    def __init__(self, model: Any, budget: RunBudget) -> None:
        super().__init__(model)
        self._budget = budget

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._budget.consume_model_requests()
        return await self.wrapped.request(
            messages,
            model_settings,
            model_request_parameters,
        )


def _raise_tool_error(
    error: BudgetExceeded | ImageToolRejected | ToolCallRejected | ToolExecutionRejected,
) -> Never:
    if isinstance(error, ToolExecutionRejected) and error.code in {
        "image_tool_unavailable",
        "retrieval_unavailable",
    }:
        raise DependencyUnavailable(error.code) from None
    raise ModelRetry(error.code) from None


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
        _raise_tool_error(error)
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
        _raise_tool_error(error)
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
        _raise_tool_error(error)
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


def _budget_trace_payload(snapshot: BudgetSnapshot) -> dict[str, TraceValue]:
    return {
        "estimated_cost_usd": str(snapshot.estimated_cost_usd),
        "image_byte_count": snapshot.image_bytes,
        "model_requests": snapshot.model_requests,
        "output_tokens": snapshot.output_tokens,
        "repairs": snapshot.repairs,
        "tool_calls": snapshot.tool_calls,
    }


def _digest_trace_payload(payload: dict[str, TraceValue]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _with_runtime_trace(
    response: VisualResponse,
    *,
    request_digest: str,
    model_name: str,
    provider_name: str,
    model_requests: int,
    budget_snapshot: BudgetSnapshot,
    tool_records: tuple[ToolCallRecord, ...] = (),
) -> VisualResponse:
    output_digest = hashlib.sha256(
        response.model_dump_json(exclude={"trace"}).encode("utf-8")
    ).hexdigest()
    trace = [
        AgentTraceEvent(
            sequence=0,
            run_replay_id=request_digest,
            kind="run",
            status="started",
            name=AGENT_NAME,
            input_sha256=request_digest,
        ),
    ]
    trace.extend(
        AgentTraceEvent(
            sequence=index + 1,
            run_replay_id=request_digest,
            kind="retrieval" if record.name == "retrieve_reference" else "tool_call",
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
                run_replay_id=request_digest,
                kind="model_request",
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
                run_replay_id=request_digest,
                kind="verification",
                status="succeeded",
                name="visual-response",
                input_sha256=output_digest,
                output_sha256=output_digest,
            ),
        ]
    )
    if response.status == "abstain":
        trace.append(
            AgentTraceEvent(
                sequence=len(trace),
                run_replay_id=request_digest,
                kind="abstention",
                status="succeeded",
                name="safe-abstention",
                output_sha256=output_digest,
            )
        )
    budget_payload = _budget_trace_payload(budget_snapshot)
    trace.extend(
        (
            AgentTraceEvent(
                sequence=len(trace),
                run_replay_id=request_digest,
                kind="budget",
                status="succeeded",
                name="run-budget",
                output_sha256=_digest_trace_payload(budget_payload),
                attributes=budget_payload,
            ),
            AgentTraceEvent(
                sequence=len(trace) + 1,
                run_replay_id=request_digest,
                kind="final_status",
                status="succeeded",
                name=response.status,
                output_sha256=output_digest,
            ),
        )
    )
    payload = response.model_dump(mode="json")
    payload["trace"] = [event.model_dump(mode="json") for event in trace]
    return VisualResponse.model_validate(payload)


def _failure_code(
    error: Exception,
    *,
    phase: str,
    tool_records: tuple[ToolCallRecord, ...],
) -> str:
    if isinstance(error, BudgetExceeded):
        return error.code
    if isinstance(error, DependencyUnavailable):
        return error.code
    if isinstance(error, ModelAPIError):
        return "model_unavailable"
    if tool_records and tool_records[-1].failure_code is not None:
        return tool_records[-1].failure_code
    return {
        "input": "input_validation_failed",
        "model_request": "model_request_failed",
        "verification": "verification_failed",
    }.get(phase, "run_failed")


def _failure_runtime_trace(
    *,
    request_digest: str,
    budget_snapshot: BudgetSnapshot,
    error: Exception,
    phase: str,
    tool_records: tuple[ToolCallRecord, ...],
) -> RunTrace:
    failure_code = _failure_code(error, phase=phase, tool_records=tool_records)
    events: list[AgentTraceEvent] = [
        AgentTraceEvent(
            sequence=0,
            run_replay_id=request_digest,
            kind="run",
            status="started",
            name=AGENT_NAME,
            input_sha256=request_digest,
        )
    ]
    events.extend(
        AgentTraceEvent(
            sequence=index + 1,
            run_replay_id=request_digest,
            kind="retrieval" if record.name == "retrieve_reference" else "tool_call",
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
    budget_payload = _budget_trace_payload(budget_snapshot)
    budget_failed = isinstance(error, BudgetExceeded)
    events.extend(
        (
            AgentTraceEvent(
                sequence=len(events),
                run_replay_id=request_digest,
                kind="budget",
                status="rejected" if budget_failed else "succeeded",
                name="run-budget",
                output_sha256=_digest_trace_payload(budget_payload),
                failure_code=failure_code if budget_failed else None,
                attributes=budget_payload,
            ),
            AgentTraceEvent(
                sequence=len(events) + 1,
                run_replay_id=request_digest,
                kind="error",
                status="failed",
                name=f"{phase}-error",
                failure_code=failure_code,
            ),
            AgentTraceEvent(
                sequence=len(events) + 2,
                run_replay_id=request_digest,
                kind="final_status",
                status="failed",
                name="failed",
                failure_code=failure_code,
            ),
        )
    )
    return RunTrace(run_replay_id=request_digest, events=tuple(events))


async def run_evidence_request(
    request: EvidenceRequest,
    image_bytes: bytes,
    *,
    model: Any,
    tool_services: AgentToolServices | None = None,
    trace_collector: RunTraceCollector | None = None,
) -> VisualResponse:
    """Run one bounded image-plus-question request and return a safe response."""

    request_digest = _digest_request(request, image_bytes)
    budget = RunBudget(request.limits)
    executor: BoundedToolExecutor | None = None
    phase = "input"
    try:
        _validate_image_bytes(request, image_bytes)
        budget.consume_image_bytes(len(image_bytes))
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
        agent = create_agent(
            _BudgetedModel(model, budget),
            enable_tools=executor is not None,
        )
        cost_limit = (
            None
            if getattr(model, "system", None) in {"test", "function"}
            else Decimal(str(request.limits.max_estimated_cost_usd))
        )
        phase = "model_request"
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
        if usage.requests != budget.snapshot().model_requests:
            raise BudgetExceeded("invalid_usage")
        if usage.tool_calls and executor is None:
            budget.consume_tool_calls(usage.tool_calls)
        if usage.output_tokens:
            budget.consume_output_tokens(usage.output_tokens)
        if usage.cost:
            budget.consume_estimated_cost(usage.cost)
        phase = "verification"
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
        response = _with_runtime_trace(
            response,
            request_digest=request_digest,
            model_name=result.response.model_name or "unknown-model",
            provider_name=result.response.provider_name or "unknown-provider",
            model_requests=result.usage.requests,
            budget_snapshot=budget.snapshot(),
            tool_records=tool_records,
        )
    except Exception as error:
        tool_records = executor.records if executor is not None else ()
        if trace_collector is not None:
            trace_collector.record(
                _failure_runtime_trace(
                    request_digest=request_digest,
                    budget_snapshot=budget.snapshot(),
                    error=error,
                    phase=phase,
                    tool_records=tool_records,
                )
            )
        raise

    if trace_collector is not None:
        trace_collector.record(
            RunTrace(
                run_replay_id=request_digest,
                events=tuple(response.trace),
            )
        )
    return response
