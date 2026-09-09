from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import CostNotFoundWarning

from chest_xray_evidence_assistant.agent import (
    AgentToolServices,
    create_agent,
    run_evidence_request,
)
from chest_xray_evidence_assistant.fixtures import load_fixture_manifest
from chest_xray_evidence_assistant.models import (
    EvidenceRequest,
    QuestionContext,
    RunLimits,
    VisualResponse,
)
from chest_xray_evidence_assistant.retrieval import (
    InMemoryBM25Index,
    LexicalReferenceService,
    ScoredChunk,
    load_reference_corpus,
)
from chest_xray_evidence_assistant.runtime import BudgetExceeded
from chest_xray_evidence_assistant.tools import FixtureImageTools

models.ALLOW_MODEL_REQUESTS = False

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "data" / "fixtures"
RESPONSE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "responses"


class CountingTestModel(TestModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.request_count = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        self.request_count += 1
        return await super().request(*args, **kwargs)


class SlowTestModel(TestModel):
    async def request(self, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.05)
        return await super().request(*args, **kwargs)


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


def exact_tool_model(
    tool_name: str,
    arguments: dict[str, object],
    final_payload: dict[str, object],
) -> tuple[FunctionModel, list[tuple[str, ...]]]:
    calls = 0
    observed_tools: list[tuple[str, ...]] = []

    def respond(messages: list[object], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        del messages
        observed_tools.append(tuple(tool.name for tool in info.function_tools))
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, arguments, tool_call_id="call-1")])
        output_tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, final_payload, tool_call_id="final-1")]
        )

    return FunctionModel(respond, model_name="exact-tool-model"), observed_tools


def source_evidence_payload(result: ScoredChunk) -> dict[str, object]:
    chunk = result.chunk
    return {
        "document_id": chunk.document_id,
        "section": chunk.source_span.section,
        "source_url": str(chunk.source_span.source_url),
        "locator": chunk.source_span.locator,
        "excerpt": chunk.text,
        "relevance_score": 0.8,
    }


def test_agent_requires_explicit_model_injection() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        create_agent(None)


def test_fake_model_completes_one_bounded_multimodal_request() -> None:
    request, image_bytes = fixture_request()
    model = TestModel(
        custom_output_args=response_payload("answered.json"),
        model_name="fixture-model-v1",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", CostNotFoundWarning)
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


def test_malformed_model_output_gets_only_one_bounded_repair() -> None:
    request, image_bytes = fixture_request()
    model = CountingTestModel(custom_output_args=response_payload("malformed.json"))

    with pytest.raises(UnexpectedModelBehavior, match="maximum output retries"):
        asyncio.run(run_evidence_request(request, image_bytes, model=model))
    assert model.request_count == 2


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


def test_request_image_must_fit_the_parent_image_budget() -> None:
    request, image_bytes = fixture_request()
    request = request.model_copy(
        update={"limits": RunLimits(max_image_bytes=request.image.byte_size - 1)}
    )

    with pytest.raises(BudgetExceeded, match="^image_byte_limit$"):
        asyncio.run(
            run_evidence_request(
                request,
                image_bytes,
                model=TestModel(custom_output_args=response_payload("answered.json")),
            )
        )


def test_request_timeout_cancels_slow_model_work() -> None:
    request, image_bytes = fixture_request()
    request = request.model_copy(update={"limits": RunLimits(timeout_seconds=0.01)})

    with pytest.raises(BudgetExceeded, match="^timeout$"):
        asyncio.run(
            run_evidence_request(
                request,
                image_bytes,
                model=SlowTestModel(custom_output_args=response_payload("answered.json")),
            )
        )


def test_tool_enabled_agent_dispatches_through_exact_registry_and_records_trace() -> None:
    request, image_bytes = fixture_request()
    model, observed_tools = exact_tool_model(
        "get_image_metadata",
        {"image_id": request.image.image_id},
        response_payload("answered.json"),
    )

    response = asyncio.run(
        run_evidence_request(
            request,
            image_bytes,
            model=model,
            tool_services=AgentToolServices(
                image_tools=FixtureImageTools.from_manifest(FIXTURE_ROOT / "manifest.json")
            ),
        )
    )

    assert observed_tools[0] == (
        "crop_image",
        "get_image_metadata",
        "retrieve_reference",
    )
    tool_events = [event for event in response.trace if event.kind == "tool"]
    assert len(tool_events) == 1
    assert tool_events[0].name == "get_image_metadata"
    assert tool_events[0].status == "succeeded"
    assert tool_events[0].duration_ms is not None
    assert tool_events[0].duration_ms >= 0
    assert tool_events[0].attributes["tool_calls"] == 1
    assert response.trace[-1].kind == "final"


def test_tool_required_path_beats_the_zero_tool_path_on_expected_trajectory() -> None:
    request, image_bytes = fixture_request()
    one_shot = asyncio.run(
        run_evidence_request(
            request,
            image_bytes,
            model=TestModel(custom_output_args=response_payload("answered.json")),
        )
    )
    manifest = load_fixture_manifest(FIXTURE_ROOT / "manifest.json")
    crop_fixture = next(
        item for item in manifest.fixtures if item.asset.image_id == "synthetic-crop-target"
    )
    crop_request = request.model_copy(update={"image": crop_fixture.asset})
    crop_bytes = (FIXTURE_ROOT / crop_fixture.path).read_bytes()
    crop_sha256 = "14aba6019240908536da3a24fd6df302ef176102fc16c02cf701da374500942f"
    crop_image_id = f"crop-{crop_sha256[:32]}"
    crop_response = response_payload("answered.json")
    crop_response["visual_evidence"][0]["locator"]["image_id"] = crop_image_id  # type: ignore[index]
    tool_model, _ = exact_tool_model(
        "crop_image",
        {
            "image_id": "synthetic-crop-target",
            "box": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
        },
        crop_response,
    )
    tool_run = asyncio.run(
        run_evidence_request(
            crop_request,
            crop_bytes,
            model=tool_model,
            tool_services=AgentToolServices(
                image_tools=FixtureImageTools.from_manifest(FIXTURE_ROOT / "manifest.json")
            ),
        )
    )

    def tool_required_score(response: VisualResponse) -> int:
        has_crop_artifact = any(
            event.kind == "tool"
            and event.name == "crop_image"
            and event.status == "succeeded"
            and event.attributes.get("artifact_sha256") == crop_sha256
            and event.attributes.get("width_px") == 32
            and event.attributes.get("height_px") == 32
            for event in response.trace
        )
        cites_crop = any(
            evidence.locator.image_id == crop_image_id for evidence in response.visual_evidence
        )
        return int(response.status == "answered" and has_crop_artifact and cites_crop)

    assert tool_required_score(one_shot) == 0
    assert tool_required_score(tool_run) == 1


def test_retrieved_source_evidence_requires_the_authorized_result_and_trace() -> None:
    request, image_bytes = fixture_request()
    lexical_index = InMemoryBM25Index.from_corpus(
        load_reference_corpus(REPO_ROOT / "data" / "references" / "manifest.json")
    )
    service = LexicalReferenceService(lexical_index)
    query = "computed tomography slices"
    expected = lexical_index.search(query, top_k=1)[0]
    final_payload = response_payload("answered.json")
    final_payload["source_evidence"] = [source_evidence_payload(expected)]
    model, _ = exact_tool_model(
        "retrieve_reference",
        {"query": query, "top_k": 1},
        final_payload,
    )

    response = asyncio.run(
        run_evidence_request(
            request,
            image_bytes,
            model=model,
            tool_services=AgentToolServices(
                image_tools=FixtureImageTools.from_manifest(FIXTURE_ROOT / "manifest.json"),
                reference_retriever=service,
            ),
        )
    )

    assert response.source_evidence[0].document_id == expected.chunk.document_id
    assert any(
        event.kind == "tool" and event.name == "retrieve_reference" and event.status == "succeeded"
        for event in response.trace
    )
    retrieval_event = next(event for event in response.trace if event.name == "retrieve_reference")
    assert retrieval_event.attributes["result_count"] == 1
    assert len(str(retrieval_event.attributes["chunk_ids_sha256"])) == 64


def test_fabricated_source_evidence_is_rejected_without_retrieval() -> None:
    request, image_bytes = fixture_request()
    lexical_index = InMemoryBM25Index.from_corpus(
        load_reference_corpus(REPO_ROOT / "data" / "references" / "manifest.json")
    )
    fabricated = response_payload("answered.json")
    fabricated["source_evidence"] = [
        source_evidence_payload(lexical_index.search("radiography", top_k=1)[0])
    ]

    with pytest.raises(ValueError, match="authorized retrieval result"):
        asyncio.run(
            run_evidence_request(
                request,
                image_bytes,
                model=TestModel(custom_output_args=fabricated),
            )
        )
