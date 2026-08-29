from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from chest_xray_evidence_assistant.agent import create_agent, run_evidence_request
from chest_xray_evidence_assistant.fixtures import load_fixture_manifest
from chest_xray_evidence_assistant.models import EvidenceRequest, QuestionContext

models.ALLOW_MODEL_REQUESTS = False

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "data" / "fixtures"
RESPONSE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "responses"


def fixture_request() -> tuple[EvidenceRequest, bytes]:
    manifest = load_fixture_manifest(FIXTURE_ROOT / "manifest.json")
    fixture = next(
        item for item in manifest.fixtures if item.asset.image_id == "synthetic-full-frame"
    )
    image_bytes = (FIXTURE_ROOT / fixture.path).read_bytes()
    request = EvidenceRequest(
        request_id="request-001",
        image=fixture.asset,
        question=QuestionContext(question="What can be observed in this fixture?"),
    )
    return request, image_bytes


def response_payload(name: str) -> dict[str, object]:
    return json.loads((RESPONSE_ROOT / name).read_text())


def test_agent_requires_explicit_model_injection() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        create_agent(None)


def test_fake_model_completes_one_bounded_multimodal_request() -> None:
    request, image_bytes = fixture_request()
    model = TestModel(
        custom_output_args=response_payload("answered.json"),
        model_name="fixture-model-v1",
    )

    response = asyncio.run(run_evidence_request(request, image_bytes, model=model))

    assert response.status == "answered"
    assert response.visual_evidence[0].locator.image_id == request.image.image_id
    assert [event.kind for event in response.trace] == [
        "request",
        "model",
        "validation",
        "final",
    ]
    assert response.trace[1].attributes["model_requests"] == 1
    assert response.trace[1].attributes["tool_calls"] == 0
    assert all(event.input_sha256 or event.output_sha256 for event in response.trace)


def test_malformed_model_output_is_rejected_without_retry() -> None:
    request, image_bytes = fixture_request()
    model = TestModel(custom_output_args=response_payload("malformed.json"))

    with pytest.raises(UnexpectedModelBehavior, match="maximum output retries"):
        asyncio.run(run_evidence_request(request, image_bytes, model=model))


def test_visual_provenance_cannot_escape_the_request_image() -> None:
    request, image_bytes = fixture_request()
    payload = response_payload("answered.json")
    payload["visual_evidence"][0]["locator"]["image_id"] = "other-image"
    model = TestModel(custom_output_args=payload)

    with pytest.raises(ValueError, match="outside the request"):
        asyncio.run(run_evidence_request(request, image_bytes, model=model))


def test_declared_image_digest_and_size_are_checked_before_model_call() -> None:
    request, image_bytes = fixture_request()

    with pytest.raises(ValueError, match="declared asset size"):
        asyncio.run(
            run_evidence_request(
                request,
                image_bytes + b"unexpected",
                model=TestModel(custom_output_args=response_payload("answered.json")),
            )
        )
