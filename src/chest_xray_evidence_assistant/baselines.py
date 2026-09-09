"""Repeatable one-shot/no-tool baseline capture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pydantic_ai
from pydantic import Field, model_validator

from .agent import AGENT_NAME, SYSTEM_INSTRUCTIONS, run_evidence_request
from .fixtures import load_fixture_manifest
from .models import (
    ContractModel,
    EvidenceRequest,
    Identifier,
    QuestionContext,
    RunLimits,
    Sha256Digest,
    ShortText,
    VisualResponse,
)

BASELINE_ID = "one-shot-vlm-no-tools-v1"
BASELINE_QUESTION = "What can be observed in this synthetic fixture?"
OFFLINE_BASELINE_MODEL = "fixture-model-v1"
BaselineMode = Literal["offline_fake", "live"]


class BaselineConfiguration(ContractModel):
    """Non-secret inputs that identify a comparable baseline run."""

    agent_name: Identifier
    provider: ShortText
    model: ShortText
    pydantic_ai_version: ShortText
    system_instructions_sha256: Sha256Digest
    response_schema_sha256: Sha256Digest
    question_sha256: Sha256Digest
    request_id: Identifier
    image_id: Identifier
    image_sha256: Sha256Digest
    max_model_requests: Literal[1]
    max_image_bytes: int = Field(gt=0, le=20_000_000)
    max_output_tokens: int = Field(gt=0, le=4096)
    timeout_seconds: float = Field(gt=0, le=120)
    max_estimated_cost_usd: float = Field(ge=0, le=1)
    tool_calls_limit: Literal[0] = 0


def configuration_fingerprint(configuration: BaselineConfiguration) -> str:
    """Hash canonical non-secret configuration for later comparisons."""

    payload = json.dumps(
        configuration.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BaselineArtifact(ContractModel):
    """A replayable baseline result without raw image bytes or credentials."""

    schema_version: Literal[1] = 1
    baseline_id: Literal["one-shot-vlm-no-tools-v1"] = BASELINE_ID
    mode: BaselineMode
    configuration: BaselineConfiguration
    configuration_sha256: Sha256Digest
    request: EvidenceRequest
    response: VisualResponse

    @model_validator(mode="after")
    def validate_fingerprints_and_boundary(self) -> BaselineArtifact:
        expected = configuration_fingerprint(self.configuration)
        if self.configuration_sha256 != expected:
            raise ValueError("configuration fingerprint does not match configuration")
        if self.configuration.request_id != self.request.request_id:
            raise ValueError("baseline configuration references a different request")
        if self.configuration.image_id != self.request.image.image_id:
            raise ValueError("baseline configuration references a different image")
        if self.configuration.image_sha256 != self.request.image.sha256:
            raise ValueError("baseline configuration has a stale image digest")

        model_events = [event for event in self.response.trace if event.kind == "model"]
        if len(model_events) != 1 or model_events[0].attributes.get("tool_calls") != 0:
            raise ValueError("one-shot baseline must record exactly one zero-tool model event")
        if model_events[0].attributes.get("model_requests") != 1:
            raise ValueError("one-shot baseline must record exactly one model request")
        if any(
            evidence.locator.image_id != self.request.image.image_id
            for evidence in self.response.visual_evidence
        ):
            raise ValueError(
                "baseline visual evidence references an image outside the baseline request"
            )
        return self


def load_baseline_case(project_root: Path | None = None) -> tuple[EvidenceRequest, bytes]:
    """Load the fixed synthetic image and prompt shared by baseline/live smoke."""

    root = project_root or Path(__file__).resolve().parents[2]
    fixture_root = root / "data" / "fixtures"
    manifest = load_fixture_manifest(fixture_root / "manifest.json")
    fixture = next(
        (item for item in manifest.fixtures if item.asset.image_id == "synthetic-full-frame"),
        None,
    )
    if fixture is None:
        raise ValueError("baseline fixture synthetic-full-frame is unavailable")

    image_bytes = (fixture_root / fixture.path).read_bytes()
    request = EvidenceRequest(
        request_id="one-shot-baseline-001",
        image=fixture.asset,
        question=QuestionContext(question=BASELINE_QUESTION),
        limits=RunLimits(max_model_requests=1, max_tool_calls=0),
    )
    return request, image_bytes


async def capture_one_shot_baseline(
    request: EvidenceRequest,
    image_bytes: bytes,
    *,
    model: Any,
    mode: BaselineMode,
    provider: str,
    model_name: str,
) -> BaselineArtifact:
    """Run the fixed zero-tool path and return its fingerprinted artifact."""

    response = await run_evidence_request(request, image_bytes, model=model)
    configuration = BaselineConfiguration(
        agent_name=AGENT_NAME,
        provider=provider,
        model=model_name,
        pydantic_ai_version=pydantic_ai.__version__,
        system_instructions_sha256=hashlib.sha256(SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        response_schema_sha256=hashlib.sha256(
            json.dumps(
                VisualResponse.model_json_schema(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        question_sha256=hashlib.sha256(
            request.question.model_dump_json().encode("utf-8")
        ).hexdigest(),
        request_id=request.request_id,
        image_id=request.image.image_id,
        image_sha256=request.image.sha256,
        max_model_requests=request.limits.max_model_requests,
        max_image_bytes=request.limits.max_image_bytes,
        max_output_tokens=request.limits.max_output_tokens,
        timeout_seconds=request.limits.timeout_seconds,
        max_estimated_cost_usd=request.limits.max_estimated_cost_usd,
        tool_calls_limit=0,
    )
    return BaselineArtifact(
        mode=mode,
        configuration=configuration,
        configuration_sha256=configuration_fingerprint(configuration),
        request=request,
        response=response,
    )


def serialize_baseline(artifact: BaselineArtifact) -> str:
    """Serialize an artifact canonically for byte-for-byte comparisons."""

    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
