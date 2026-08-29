"""Provider-neutral multimodal model ports and an offline fixture adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .models import EvidenceRequest, VisualResponse


@runtime_checkable
class ModelAdapter(Protocol):
    """The narrow async boundary used by the future agent runtime."""

    @property
    def model_id(self) -> str:
        """Return a stable, non-secret model identifier for traces."""

    async def complete(self, request: EvidenceRequest) -> VisualResponse:
        """Complete one image-plus-question request with a typed response."""


@dataclass(frozen=True, slots=True)
class DeterministicModelAdapter:
    """Return a validated response without credentials, network, or model calls."""

    response: VisualResponse
    model_id: str = "deterministic-fixture-v1"

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        model_id: str = "deterministic-fixture-v1",
    ) -> DeterministicModelAdapter:
        """Build a fixture adapter only after validating the response payload."""

        return cls(
            response=VisualResponse.model_validate(payload),
            model_id=model_id,
        )

    async def complete(self, request: EvidenceRequest) -> VisualResponse:
        """Return an independent validated copy for one typed request."""

        del request
        return VisualResponse.model_validate(self.response.model_dump(mode="json"))
