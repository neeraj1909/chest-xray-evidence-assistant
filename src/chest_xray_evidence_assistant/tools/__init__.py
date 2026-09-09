"""Typed, allow-listed tool contracts for the evidence assistant."""

from .contracts import (
    CropImageArguments,
    CropImageCall,
    GetImageMetadataArguments,
    GetImageMetadataCall,
    RetrieveReferenceArguments,
    RetrieveReferenceCall,
    ToolCall,
    ToolCallRejected,
    ToolName,
)
from .ports import CropImagePort, ImageMetadataPort, ReferenceRetrievalPort
from .registry import INITIAL_TOOL_REGISTRY, ToolRegistry

__all__ = [
    "CropImageArguments",
    "CropImageCall",
    "CropImagePort",
    "GetImageMetadataArguments",
    "GetImageMetadataCall",
    "INITIAL_TOOL_REGISTRY",
    "ImageMetadataPort",
    "ReferenceRetrievalPort",
    "RetrieveReferenceArguments",
    "RetrieveReferenceCall",
    "ToolCall",
    "ToolCallRejected",
    "ToolName",
    "ToolRegistry",
]
