"""Bounded single-agent request path for the evidence assistant."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal
from typing import Any

from pydantic_ai import Agent, BinaryContent, UsageLimits

from .models import AgentTraceEvent, EvidenceRequest, VisualResponse

AGENT_NAME = "cxr-evidence-agent"
SYSTEM_INSTRUCTIONS = (
    "Return only a non-diagnostic VisualResponse grounded in the supplied image. "
    "Use abstain or needs_clarification when evidence is insufficient."
)


def create_agent(model: Any) -> Agent[None, VisualResponse]:
    """Create one provider-injected agent without selecting a live model implicitly."""

    if model is None:
        raise ValueError("an explicit model or deterministic test model is required")

    return Agent(
        model=model,
        output_type=VisualResponse,
        instructions=SYSTEM_INSTRUCTIONS,
        name=AGENT_NAME,
        retries=0,
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


def _validate_provenance(request: EvidenceRequest, response: VisualResponse) -> VisualResponse:
    for evidence in response.visual_evidence:
        if evidence.locator.image_id != request.image.image_id:
            raise ValueError("visual evidence references an image outside the request")
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


def _with_runtime_trace(
    response: VisualResponse,
    *,
    request_digest: str,
    model_name: str,
    provider_name: str,
    model_requests: int,
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
        AgentTraceEvent(
            sequence=1,
            kind="model",
            status="succeeded",
            name=_trace_identifier(model_name, "model"),
            input_sha256=request_digest,
            output_sha256=output_digest,
            attributes={
                "provider": provider_name,
                "model_requests": model_requests,
                "tool_calls": 0,
            },
        ),
        AgentTraceEvent(
            sequence=2,
            kind="validation",
            status="succeeded",
            name="visual-response",
            input_sha256=output_digest,
            output_sha256=output_digest,
        ),
        AgentTraceEvent(
            sequence=3,
            kind="final",
            status="succeeded",
            name=response.status,
            output_sha256=output_digest,
        ),
    ]
    payload = response.model_dump(mode="json")
    payload["trace"] = [event.model_dump(mode="json") for event in trace]
    return VisualResponse.model_validate(payload)


async def run_evidence_request(
    request: EvidenceRequest,
    image_bytes: bytes,
    *,
    model: Any,
) -> VisualResponse:
    """Run one bounded image-plus-question request and return a safe response."""

    _validate_image_bytes(request, image_bytes)
    agent = create_agent(model)
    result = await agent.run(
        _prompt_parts(request, image_bytes),
        usage_limits=UsageLimits(
            request_limit=request.limits.max_model_requests,
            tool_calls_limit=0,
            output_tokens_limit=request.limits.max_output_tokens,
            cost_limit=Decimal(str(request.limits.max_estimated_cost_usd)),
        ),
    )
    response = VisualResponse.model_validate(result.output.model_dump(mode="json"))
    response = _validate_provenance(request, response)
    return _with_runtime_trace(
        response,
        request_digest=_digest_request(request, image_bytes),
        model_name=result.response.model_name or "unknown-model",
        provider_name=result.response.provider_name or "unknown-provider",
        model_requests=result.usage.requests,
    )
