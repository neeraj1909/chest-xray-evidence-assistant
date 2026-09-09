"""Typed, allow-listed tool contracts for the evidence assistant."""

from .contracts import (
    CropImageArguments,
    CropImageCall,
    CropImageResult,
    GetImageMetadataArguments,
    GetImageMetadataCall,
    ImageMetadataResult,
    ImageToolRejected,
    ReferenceQuery,
    ReferenceTopK,
    RetrieveReferenceArguments,
    RetrieveReferenceCall,
    ToolCall,
    ToolCallRecord,
    ToolCallRejected,
    ToolName,
)
from .executor import BoundedToolExecutor, ToolExecutionRejected, ToolResult
from .image_tools import FixtureImageTools
from .ports import CropImagePort, ImageMetadataPort, ImageToolsPort, ReferenceRetrievalPort
from .registry import INITIAL_TOOL_REGISTRY, ToolRegistry

__all__ = [
    "BoundedToolExecutor",
    "CropImageArguments",
    "CropImageCall",
    "CropImagePort",
    "CropImageResult",
    "FixtureImageTools",
    "GetImageMetadataArguments",
    "GetImageMetadataCall",
    "INITIAL_TOOL_REGISTRY",
    "ImageMetadataPort",
    "ImageMetadataResult",
    "ImageToolRejected",
    "ImageToolsPort",
    "ReferenceRetrievalPort",
    "ReferenceQuery",
    "ReferenceTopK",
    "RetrieveReferenceArguments",
    "RetrieveReferenceCall",
    "ToolCall",
    "ToolCallRecord",
    "ToolCallRejected",
    "ToolExecutionRejected",
    "ToolName",
    "ToolRegistry",
    "ToolResult",
]
