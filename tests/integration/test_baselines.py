from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pydantic_ai
import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from chest_xray_evidence_assistant.baselines import (
    OFFLINE_BASELINE_MODEL,
    BaselineArtifact,
    capture_one_shot_baseline,
    configuration_fingerprint,
    load_baseline_case,
    serialize_baseline,
)

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_FIXTURE = ROOT / "tests" / "fixtures" / "responses" / "answered.json"


def capture_offline_baseline() -> BaselineArtifact:
    request, image_bytes = load_baseline_case(ROOT)
    response_payload = json.loads(RESPONSE_FIXTURE.read_text())
    model = TestModel(
        custom_output_args=response_payload,
        model_name=OFFLINE_BASELINE_MODEL,
    )
    return asyncio.run(
        capture_one_shot_baseline(
            request,
            image_bytes,
            model=model,
            mode="offline_fake",
            provider="test",
            model_name=OFFLINE_BASELINE_MODEL,
        )
    )


def test_baseline_uses_the_fixed_synthetic_image_and_prompt() -> None:
    request, image_bytes = load_baseline_case(ROOT)

    assert request.request_id == "one-shot-baseline-001"
    assert request.image.image_id == "synthetic-full-frame"
    assert request.question.question == "What can be observed in this synthetic fixture?"
    assert request.limits.max_model_requests == 1
    assert request.limits.max_tool_calls == 0
    assert len(image_bytes) == request.image.byte_size


def test_offline_baseline_records_a_truthful_no_tool_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-never-enter-the-artifact")

    artifact = capture_offline_baseline()
    serialized = serialize_baseline(artifact)

    assert artifact.mode == "offline_fake"
    assert artifact.configuration.provider == "test"
    assert artifact.configuration.model == OFFLINE_BASELINE_MODEL
    assert artifact.configuration.pydantic_ai_version == pydantic_ai.__version__
    assert len(artifact.configuration.response_schema_sha256) == 64
    assert artifact.configuration.tool_calls_limit == 0
    assert artifact.configuration.max_model_requests == 1
    assert artifact.configuration.image_sha256 == artifact.request.image.sha256
    assert artifact.configuration_sha256 == configuration_fingerprint(artifact.configuration)
    assert artifact.response.status == "answered"
    assert artifact.response.trace[1].attributes["model_requests"] == 1
    assert artifact.response.trace[1].attributes["tool_calls"] == 0
    assert "must-never-enter-the-artifact" not in serialized


def test_baseline_serialization_is_byte_for_byte_repeatable() -> None:
    first = capture_offline_baseline()
    second = capture_offline_baseline()

    assert serialize_baseline(first) == serialize_baseline(second)


def test_baseline_rejects_a_stale_configuration_fingerprint() -> None:
    artifact = capture_offline_baseline()
    payload = artifact.model_dump(mode="json")
    payload["configuration"]["model"] = "silently-changed-model"

    with pytest.raises(ValidationError, match="configuration fingerprint"):
        type(artifact).model_validate(payload)


def test_baseline_rejects_a_non_one_shot_trace() -> None:
    artifact = capture_offline_baseline()
    payload = artifact.model_dump(mode="json")
    payload["response"]["trace"][1]["attributes"]["model_requests"] = 2

    with pytest.raises(ValidationError, match="exactly one model request"):
        type(artifact).model_validate(payload)


def test_baseline_rejects_visual_evidence_for_another_image() -> None:
    artifact = capture_offline_baseline()
    payload = artifact.model_dump(mode="json")
    payload["response"]["visual_evidence"][0]["locator"]["image_id"] = "other-image"

    with pytest.raises(ValidationError, match="outside the baseline request"):
        type(artifact).model_validate(payload)
