"""Strict arguments and calls for the initial three-tool boundary."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints

from ..models import ContractModel, Identifier, NormalizedBoundingBox

ToolName: TypeAlias = Literal[
    "crop_image",
    "get_image_metadata",
    "retrieve_reference",
]
ReferenceQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]


class CropImageArguments(ContractModel):
    image_id: Identifier
    box: NormalizedBoundingBox


class GetImageMetadataArguments(ContractModel):
    image_id: Identifier


class RetrieveReferenceArguments(ContractModel):
    query: ReferenceQuery
    top_k: int = Field(default=5, ge=1, le=5)


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


class ToolCallRejected(ValueError):
    """Reject an unsafe call with a stable code and no echoed payload."""

    def __init__(self, code: ToolRejectionCode) -> None:
        self.code = code
        super().__init__(code)
