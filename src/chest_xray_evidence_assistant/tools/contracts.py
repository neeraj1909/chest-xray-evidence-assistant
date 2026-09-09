"""Strict arguments and calls for the initial three-tool boundary."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..models import (
    ContractModel,
    Identifier,
    ImageAsset,
    NormalizedBoundingBox,
    Sha256Digest,
    TraceValue,
)
from ..runtime import BudgetSnapshot

ToolName: TypeAlias = Literal[
    "crop_image",
    "get_image_metadata",
    "retrieve_reference",
]
ReferenceQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
ReferenceTopK = Annotated[int, Field(ge=1, le=5)]


class CropImageArguments(ContractModel):
    image_id: Identifier
    box: NormalizedBoundingBox


class GetImageMetadataArguments(ContractModel):
    image_id: Identifier


class RetrieveReferenceArguments(ContractModel):
    query: ReferenceQuery
    top_k: ReferenceTopK = 5


class CropImageResult(ContractModel):
    """Safe metadata for a deterministic server-owned crop."""

    source_image_id: Identifier
    box: NormalizedBoundingBox
    image: ImageAsset
    input_sha256: Sha256Digest
    output_sha256: Sha256Digest


class ImageMetadataResult(ContractModel):
    """Verified image metadata with request and response fingerprints."""

    image: ImageAsset
    input_sha256: Sha256Digest
    output_sha256: Sha256Digest


class CropImageCall(ContractModel):
    name: Literal["crop_image"] = "crop_image"
    arguments: CropImageArguments


class GetImageMetadataCall(ContractModel):
    name: Literal["get_image_metadata"] = "get_image_metadata"
    arguments: GetImageMetadataArguments


class RetrieveReferenceCall(ContractModel):
    name: Literal["retrieve_reference"] = "retrieve_reference"
    arguments: RetrieveReferenceArguments


ToolCall: TypeAlias = CropImageCall | GetImageMetadataCall | RetrieveReferenceCall
ToolRejectionCode: TypeAlias = Literal["unknown_tool", "invalid_arguments"]
ImageToolRejectionCode: TypeAlias = Literal[
    "image_not_found",
    "image_too_large",
    "invalid_image",
    "invalid_manifest",
    "unsupported_media_type",
]


class ToolCallRejected(ValueError):
    """Reject an unsafe call with a stable code and no echoed payload."""

    def __init__(self, code: ToolRejectionCode) -> None:
        self.code = code
        super().__init__(code)


class ImageToolRejected(ValueError):
    """Reject an image operation with a stable code and no payload echo."""

    def __init__(self, code: ImageToolRejectionCode) -> None:
        self.code = code
        super().__init__(code)


ToolRecordName: TypeAlias = Literal[
    "crop_image",
    "get_image_metadata",
    "retrieve_reference",
    "unrecognized",
]
ToolRecordStatus: TypeAlias = Literal["succeeded", "rejected", "failed"]


class ToolCallRecord(ContractModel):
    """Redacted tool-attempt evidence with a deterministic replay identity."""

    schema_version: Literal[1] = 1
    sequence: int = Field(ge=0)
    replay_id: Sha256Digest
    name: ToolRecordName
    status: ToolRecordStatus
    arguments_sha256: Sha256Digest
    result_sha256: Sha256Digest | None = None
    failure_code: Identifier | None = None
    safe_arguments: dict[Identifier, TraceValue] = Field(default_factory=dict, max_length=16)
    safe_result: dict[Identifier, TraceValue] = Field(default_factory=dict, max_length=16)
    budget_before: BudgetSnapshot
    budget_after: BudgetSnapshot

    @field_validator("safe_arguments", "safe_result")
    @classmethod
    def reject_sensitive_projection_keys(
        cls,
        projection: dict[str, TraceValue],
    ) -> dict[str, TraceValue]:
        forbidden = {
            "api_key",
            "authorization",
            "command",
            "excerpt",
            "image_bytes",
            "path",
            "prompt",
            "query",
            "secret",
            "token",
        }
        if forbidden.intersection(key.lower() for key in projection):
            raise ValueError("tool records cannot retain sensitive payload fields")
        return projection

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolCallRecord:
        if self.status == "succeeded":
            if self.result_sha256 is None or self.failure_code is not None:
                raise ValueError("successful tool records require only a result digest")
        elif self.result_sha256 is not None or self.failure_code is None:
            raise ValueError("unsuccessful tool records require only a failure code")
        if self.budget_after.tool_calls < self.budget_before.tool_calls:
            raise ValueError("tool-call usage cannot move backwards")
        if self.budget_after.image_bytes < self.budget_before.image_bytes:
            raise ValueError("image-byte usage cannot move backwards")
        return self
