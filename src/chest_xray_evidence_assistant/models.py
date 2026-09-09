"""Provider-neutral contracts for the first project slice."""

from __future__ import annotations

import re
from typing import Annotated, Literal, TypeAlias

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
RunEventKind: TypeAlias = Literal[
    "run",
    "model_request",
    "tool_call",
    "retrieval",
    "verification",
    "abstention",
    "error",
    "budget",
    "final_status",
]

_SENSITIVE_TRACE_KEYS = frozenset(
    {
        "access_token",
        "accession_number",
        "address",
        "api_key",
        "authorization",
        "command",
        "date_of_birth",
        "dicom_header",
        "dob",
        "email",
        "excerpt",
        "image_bytes",
        "image_data",
        "mrn",
        "path",
        "patient_id",
        "patient_name",
        "phone",
        "prompt",
        "query",
        "raw_prompt",
        "secret",
    }
)
_SENSITIVE_TRACE_KEY_FORMS = frozenset(key.replace("_", "") for key in _SENSITIVE_TRACE_KEYS)
_SENSITIVE_TRACE_VALUE = re.compile(
    r"(?i)(?:bearer\s+\S+|sk-[a-z0-9_-]{16,}|-----BEGIN[^-]*PRIVATE KEY-----|"
    r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+)"
)


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
    run_replay_id: Sha256Digest
    kind: RunEventKind
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
        normalized_keys = {re.sub(r"[^a-z0-9]+", "", key.casefold()) for key in attributes}
        if _SENSITIVE_TRACE_KEY_FORMS.intersection(normalized_keys):
            raise ValueError("trace attributes cannot contain sensitive keys")
        if any(
            isinstance(value, str) and _SENSITIVE_TRACE_VALUE.search(value)
            for value in attributes.values()
        ):
            raise ValueError("trace attributes cannot contain sensitive values")
        return attributes

    @model_validator(mode="after")
    def require_safe_outcome_shape(self) -> AgentTraceEvent:
        if self.kind == "error" and self.failure_code is None:
            raise ValueError("error events require a stable failure code")
        if self.status == "succeeded" and self.failure_code is not None:
            raise ValueError("successful events cannot include a failure code")
        return self


class RunTrace(ContractModel):
    """One terminal, replay-identifiable event ledger for a run attempt."""

    schema_version: Literal[1] = 1
    run_replay_id: Sha256Digest
    events: tuple[AgentTraceEvent, ...] = Field(min_length=2, max_length=64)

    @model_validator(mode="after")
    def validate_event_ledger(self) -> RunTrace:
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError("run trace event sequences must be contiguous")
        if any(event.run_replay_id != self.run_replay_id for event in self.events):
            raise ValueError("run trace events must share one replay identity")
        if self.events[0].kind != "run" or self.events[0].status != "started":
            raise ValueError("run trace must start with a started run event")
        if self.events[-1].kind != "final_status":
            raise ValueError("run trace must end with a final-status event")
        kinds = [event.kind for event in self.events]
        if any(kinds.count(kind) != 1 for kind in ("run", "budget", "final_status")):
            raise ValueError("run trace requires one run, budget, and final-status event")
        terminal_status = self.events[-1].status
        if terminal_status == "succeeded":
            if (
                kinds.count("model_request") != 1
                or kinds.count("verification") != 1
                or "error" in kinds
                or next(event for event in self.events if event.kind == "budget").status
                != "succeeded"
            ):
                raise ValueError("successful run trace is incomplete")
        elif terminal_status == "failed" and kinds.count("error") == 1:
            error_event = next(event for event in self.events if event.kind == "error")
            if (
                error_event.status != "failed"
                or self.events[-1].failure_code is None
                or error_event.failure_code != self.events[-1].failure_code
            ):
                raise ValueError("failed run trace has inconsistent failure metadata")
        else:
            raise ValueError("failed run trace requires budget and error events")
        return self


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

        if self.trace:
            RunTrace(
                run_replay_id=self.trace[0].run_replay_id,
                events=tuple(self.trace),
            )
            if self.trace[-1].name != self.status:
                raise ValueError("final trace status does not match the response")
            abstentions = sum(event.kind == "abstention" for event in self.trace)
            if abstentions != int(self.status == "abstain"):
                raise ValueError("trace abstention events do not match the response")

        return self
