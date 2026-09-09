"""Narrow provider-neutral ports for future tool implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..retrieval.records import ScoredChunk
from .contracts import (
    CropImageArguments,
    CropImageResult,
    GetImageMetadataArguments,
    ImageMetadataResult,
    RetrieveReferenceArguments,
)


@runtime_checkable
class CropImagePort(Protocol):
    async def crop_image(self, arguments: CropImageArguments) -> CropImageResult:
        """Create a server-owned crop and return only its safe metadata."""


@runtime_checkable
class ImageMetadataPort(Protocol):
    async def get_image_metadata(
        self,
        arguments: GetImageMetadataArguments,
    ) -> ImageMetadataResult:
        """Return verified metadata for one server-owned image."""


@runtime_checkable
class ImageToolsPort(CropImagePort, ImageMetadataPort, Protocol):
    """Combined image capability used by the bounded dispatcher."""


@runtime_checkable
class ReferenceRetrievalPort(Protocol):
    async def retrieve_reference(
        self,
        arguments: RetrieveReferenceArguments,
    ) -> tuple[ScoredChunk, ...]:
        """Return a bounded set of provenance-bearing reference records."""
