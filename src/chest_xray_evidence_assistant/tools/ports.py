"""Narrow provider-neutral ports for future tool implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import ImageAsset, SourceEvidence
from .contracts import (
    CropImageArguments,
    GetImageMetadataArguments,
    RetrieveReferenceArguments,
)


@runtime_checkable
class CropImagePort(Protocol):
    async def crop_image(self, arguments: CropImageArguments) -> ImageAsset:
        """Create a server-owned crop and return only its safe metadata."""


@runtime_checkable
class ImageMetadataPort(Protocol):
    async def get_image_metadata(self, arguments: GetImageMetadataArguments) -> ImageAsset:
        """Return verified metadata for one server-owned image."""


@runtime_checkable
class ReferenceRetrievalPort(Protocol):
    async def retrieve_reference(
        self,
        arguments: RetrieveReferenceArguments,
    ) -> tuple[SourceEvidence, ...]:
        """Return a bounded set of provenance-bearing reference records."""
