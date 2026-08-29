from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant import (
    DeterministicModelAdapter,
    EvidenceRequest,
    ImageAsset,
    ModelAdapter,
    QuestionContext,
)

RESPONSE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "responses"


def request() -> EvidenceRequest:
    return EvidenceRequest(
        request_id="request-001",
        image=ImageAsset(
            image_id="synthetic-001",
            sha256="0" * 64,
            media_type="image/png",
            byte_size=1024,
            width_px=64,
            height_px=64,
            origin="Locally generated synthetic fixture",
            license_status="synthetic",
        ),
        question=QuestionContext(question="What can be observed in this fixture?"),
    )


def load_response(name: str) -> dict[str, object]:
    return json.loads((RESPONSE_FIXTURES / name).read_text())


def test_deterministic_adapter_satisfies_provider_neutral_protocol() -> None:
    adapter = DeterministicModelAdapter.from_payload(load_response("answered.json"))

    assert isinstance(adapter, ModelAdapter)
    result = asyncio.run(adapter.complete(request()))

    assert adapter.model_id == "deterministic-fixture-v1"
    assert result.status == "answered"
    assert result is not adapter.response
    assert result.model_dump(mode="json") == adapter.response.model_dump(mode="json")


def test_adapter_revalidates_fixture_payload_before_composition() -> None:
    adapter = DeterministicModelAdapter.from_payload(
        load_response("needs-clarification.json"),
        model_id="fixture-clarification-v1",
    )

    result = asyncio.run(adapter.complete(request()))

    assert adapter.model_id == "fixture-clarification-v1"
    assert result.status == "needs_clarification"
    assert result.answer is None


def test_adapter_rejects_malformed_model_payload() -> None:
    with pytest.raises(ValidationError):
        DeterministicModelAdapter.from_payload(load_response("malformed.json"))
