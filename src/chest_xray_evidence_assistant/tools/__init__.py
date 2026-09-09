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
from .image_tools import FixtureImageTools, decode_grayscale_png
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
    "decode_grayscale_png",
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
