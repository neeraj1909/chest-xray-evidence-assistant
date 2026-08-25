"""Provider-neutral contracts for the first project slice."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
]
QuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]
ContextText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
]
Sha256Digest = Annotated[
    str,
    StringConstraints(pattern=r"^[a-f0-9]{64}$"),
]
TraceText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
TraceValue = TraceText | int | float | bool | None


class ContractModel(BaseModel):
    """Strict base for every external, model, and tool boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NormalizedBoundingBox(ContractModel):
    """A crop rectangle expressed as normalized image coordinates."""

    x_min: float = Field(ge=0, le=1)
    y_min: float = Field(ge=0, le=1)
    x_max: float = Field(ge=0, le=1)
    y_max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self) -> NormalizedBoundingBox:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("bounding-box minima must be less than maxima")
        return self


class ImageLocator(ContractModel):
    """A truthful locator for either a full image or a normalized crop."""

    image_id: Identifier
    kind: Literal["full_frame", "bounding_box"]
    box: NormalizedBoundingBox | None = None

    @model_validator(mode="after")
    def validate_box(self) -> ImageLocator:
        if self.kind == "full_frame" and self.box is not None:
            raise ValueError("full-frame locators cannot include a box")
        if self.kind == "bounding_box" and self.box is None:
            raise ValueError("bounding-box locators require a box")
        return self


class ImageAsset(ContractModel):
    """Metadata for one allowed, bounded, non-PHI image asset."""

    image_id: Identifier
    sha256: Sha256Digest
    media_type: Literal["image/png", "image/jpeg"]
    byte_size: int = Field(gt=0, le=20_000_000)
    width_px: int = Field(gt=0, le=8192)
    height_px: int = Field(gt=0, le=8192)
    origin: ShortText
    origin_url: HttpUrl | None = None
    license_status: Literal["synthetic", "public_domain", "license_cleared"]
    contains_phi: Literal[False] = False


class QuestionContext(ContractModel):
    question: QuestionText
    additional_context: ContextText | None = None


class VisualEvidence(ContractModel):
    locator: ImageLocator
    description: ShortText
    confidence: float = Field(ge=0, le=1)


class SourceEvidence(ContractModel):
    document_id: Identifier
    section: ShortText
    source_url: HttpUrl
    locator: ShortText
    excerpt: ShortText
    relevance_score: float = Field(ge=0, le=1)


class RunLimits(ContractModel):
    max_tool_calls: int = Field(default=3, ge=0, le=3)
    max_model_requests: int = Field(default=2, ge=1, le=2)
    max_image_bytes: int = Field(
        default=20_000_000,
        gt=0,
        le=20_000_000,
    )
    max_output_tokens: int = Field(default=2048, gt=0, le=4096)
    timeout_seconds: float = Field(default=60, gt=0, le=120)
    max_estimated_cost_usd: float = Field(default=0.25, ge=0, le=1)


class EvidenceRequest(ContractModel):
    request_id: Identifier
    image: ImageAsset
    question: QuestionContext
    limits: RunLimits = Field(default_factory=RunLimits)


class AgentTraceEvent(ContractModel):
    """Redacted, deterministic trace metadata; never raw prompts or image bytes."""

    sequence: int = Field(ge=0)
    kind: Literal["request", "model", "tool", "validation", "final"]
    status: Literal["started", "succeeded", "rejected", "failed"]
    name: Identifier | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    input_sha256: Sha256Digest | None = None
    output_sha256: Sha256Digest | None = None
    failure_code: Identifier | None = None
    attributes: dict[Identifier, TraceValue] = Field(
        default_factory=dict,
        max_length=16,
    )

    @field_validator("attributes")
    @classmethod
    def reject_sensitive_attributes(
        cls,
        attributes: dict[str, TraceValue],
    ) -> dict[str, TraceValue]:
        forbidden = {
            "prompt",
            "raw_prompt",
            "image_bytes",
            "authorization",
            "api_key",
            "secret",
            "access_token",
        }
        if forbidden.intersection(key.lower() for key in attributes):
            raise ValueError("trace attributes cannot contain sensitive payloads")
        return attributes


class VisualResponse(ContractModel):
    status: Literal["answered", "needs_clarification", "abstain"]
    answer: ShortText | None = None
    confidence: float = Field(ge=0, le=1)
    observations: list[ShortText] = Field(default_factory=list, max_length=20)
    visual_evidence: list[VisualEvidence] = Field(
        default_factory=list,
        max_length=20,
    )
    source_evidence: list[SourceEvidence] = Field(
        default_factory=list,
        max_length=20,
    )
    uncertainty: list[ShortText] = Field(min_length=1, max_length=10)
    clarification_question: QuestionText | None = None
    abstention_reason: ShortText | None = None
    trace: list[AgentTraceEvent] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_status_payload(self) -> VisualResponse:
        if self.status == "answered":
            if self.answer is None:
                raise ValueError("answered responses require an answer")
            if not self.visual_evidence and not self.source_evidence:
                raise ValueError("answered responses require evidence")
            if self.clarification_question is not None or self.abstention_reason is not None:
                raise ValueError("answered responses cannot include fallback fields")

        elif self.status == "needs_clarification":
            if self.answer is not None or self.clarification_question is None:
                raise ValueError("clarification responses require only a clarification question")
            if self.abstention_reason is not None:
                raise ValueError("clarification responses cannot include an abstention reason")

        else:
            if self.answer is not None or self.abstention_reason is None:
                raise ValueError("abstentions require only an abstention reason")
            if self.clarification_question is not None:
                raise ValueError("abstentions cannot include a clarification question")

        return self
