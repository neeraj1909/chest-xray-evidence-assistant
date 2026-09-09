"""Leakage-resistant deterministic policies for the fixed offline ablation matrix."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal, TypeAlias

from pydantic import model_validator

from ..models import (
    AgentTraceEvent,
    ContractModel,
    ImageAsset,
    Sha256Digest,
    VisualResponse,
)
from ..retrieval.corpus import LoadedReferenceCorpus, load_reference_corpus
from ..retrieval.index import InMemoryBM25Index
from ..tools import decode_grayscale_png
from .datasets import BenchmarkDataset
from .grading import (
    ObservedRun,
    ObservedToolCall,
    RunGrade,
    RunUsage,
    canonical_sha256,
)
from .pydantic_adapter import EvaluationInput

ConfigurationId: TypeAlias = Literal[
    "text_only",
    "one_shot_vlm",
    "vlm_tools",
    "vlm_rag",
    "full_agent",
]
Intent: TypeAlias = Literal[
    "observation",
    "metadata",
    "retrieval",
    "contradiction",
    "unsafe",
]
CONFIGURATION_IDS: tuple[ConfigurationId, ...] = (
    "text_only",
    "one_shot_vlm",
    "vlm_tools",
    "vlm_rag",
    "full_agent",
)
POLICY_VERSION = "deterministic-capability-policy-v1"
TOPIC_QUERIES = {
    "how a static radiograph is formed": (
        "static radiography detector absorb scatter recorded image"
    ),
    "the difference between projection radiography and ct": (
        "computed tomography many views cross-sectional slices overlap"
    ),
    "the purpose and radiation context of medical x-ray imaging": (
        "purpose ionizing radiation justification optimization medical x-ray"
    ),
}


class AblationConfiguration(ContractModel):
    """One capability projection with immutable common-evaluation fingerprints."""

    configuration_id: ConfigurationId
    runner_mode: Literal["offline_scripted"] = "offline_scripted"
    provider: Literal["offline-scripted"] = "offline-scripted"
    model: Literal["deterministic-capability-policy"] = "deterministic-capability-policy"
    policy_version: Literal["deterministic-capability-policy-v1"] = POLICY_VERSION
    token_accounting: Literal["alphanumeric-span-estimate-v1"] = "alphanumeric-span-estimate-v1"
    latency_accounting: Literal["deterministic-estimate-v1"] = "deterministic-estimate-v1"
    image_access: bool
    metadata_tool: bool
    retrieval_tool: bool
    dataset_sha256: Sha256Digest
    case_set_sha256: Sha256Digest
    seed_set_sha256: Sha256Digest
    prompt_set_sha256: Sha256Digest
    limits_sha256: Sha256Digest
    rubric_sha256: Sha256Digest
    response_schema_sha256: Sha256Digest
    retrieval_index_sha256: Sha256Digest
    configuration_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_configuration(self) -> AblationConfiguration:
        expected_capabilities = {
            "text_only": (False, False, False),
            "one_shot_vlm": (True, False, False),
            "vlm_tools": (True, True, False),
            "vlm_rag": (True, False, True),
            "full_agent": (True, True, True),
        }[self.configuration_id]
        actual = (self.image_access, self.metadata_tool, self.retrieval_tool)
        if actual != expected_capabilities:
            raise ValueError("ablation configuration has an invalid capability projection")
        expected_fingerprint = configuration_fingerprint(self)
        if self.configuration_sha256 != expected_fingerprint:
            raise ValueError("ablation configuration fingerprint does not match")
        return self


def configuration_fingerprint(configuration: AblationConfiguration | dict[str, object]) -> str:
    if isinstance(configuration, AblationConfiguration):
        payload = configuration.model_dump(mode="json", exclude={"configuration_sha256"})
    else:
        payload = dict(configuration)
        payload.pop("configuration_sha256", None)
    return canonical_sha256(payload)


def build_ablation_configurations(
    benchmark: BenchmarkDataset,
) -> tuple[AblationConfiguration, ...]:
    corpus = load_reference_corpus()
    index = InMemoryBM25Index.from_corpus(corpus)
    common = {
        "dataset_sha256": benchmark.manifest.dataset_sha256,
        "case_set_sha256": canonical_sha256([case.case_id for case in benchmark.cases]),
        "seed_set_sha256": canonical_sha256([case.seed for case in benchmark.cases]),
        "prompt_set_sha256": canonical_sha256(
            [{"case_id": case.case_id, "prompt": case.prompt} for case in benchmark.cases]
        ),
        "limits_sha256": canonical_sha256(
            {
                "max_estimated_cost_usd": 0.25,
                "max_image_bytes": 20_000_000,
                "max_model_requests": 2,
                "max_output_tokens": 2048,
                "max_tool_calls": 3,
                "timeout_seconds": 60,
            }
        ),
        "rubric_sha256": canonical_sha256(RunGrade.model_json_schema()),
        "response_schema_sha256": canonical_sha256(VisualResponse.model_json_schema()),
        "retrieval_index_sha256": index.fingerprint,
    }
    capabilities = {
        "text_only": (False, False, False),
        "one_shot_vlm": (True, False, False),
        "vlm_tools": (True, True, False),
        "vlm_rag": (True, False, True),
        "full_agent": (True, True, True),
    }
    configurations: list[AblationConfiguration] = []
    for configuration_id in CONFIGURATION_IDS:
        image_access, metadata_tool, retrieval_tool = capabilities[configuration_id]
        payload = {
            "configuration_id": configuration_id,
            "runner_mode": "offline_scripted",
            "provider": "offline-scripted",
            "model": "deterministic-capability-policy",
            "policy_version": POLICY_VERSION,
            "token_accounting": "alphanumeric-span-estimate-v1",
            "latency_accounting": "deterministic-estimate-v1",
            "image_access": image_access,
            "metadata_tool": metadata_tool,
            "retrieval_tool": retrieval_tool,
            **common,
        }
        configurations.append(
            AblationConfiguration(
                **payload,
                configuration_sha256=configuration_fingerprint(payload),
            )
        )
    return tuple(configurations)


def _intent(prompt: str) -> Intent:
    normalized = prompt.casefold()
    if any(
        marker in normalized
        for marker in (
            "both labels",
            "conflict",
            "contradict",
            "different image",
            "incompatible",
            "mismatched",
            "mutually exclusive",
        )
    ):
        return "contradiction"
    if any(
        marker in normalized
        for marker in (
            "clinical finding",
            "condition",
            "diagnos",
            "disease",
            "patient",
            "prescribe",
            "real person",
            "therapy",
            "treatment",
        )
    ):
        return "unsafe"
    if any(
        marker in normalized
        for marker in ("cite", "corpus", "indexed", "licensed source", "reference")
    ):
        return "retrieval"
    if any(
        marker in normalized
        for marker in (
            "allowed tool",
            "byte count",
            "byte size",
            "dimensions",
            "inspection operation",
            "media type",
            "metadata",
            "png properties",
            "tool-backed",
            "width and height",
        )
    ):
        return "metadata"
    return "observation"


def _retrieval_query(prompt: str) -> str:
    normalized = prompt.casefold()
    for topic, query in TOPIC_QUERIES.items():
        if topic in normalized:
            return query
    raise ValueError("retrieval prompt has no supported benchmark topic")


def _fallback_payload(status: Literal["needs_clarification", "abstain"]) -> dict[str, object]:
    payload: dict[str, object] = {
        "confidence": 0.0,
        "observations": [],
        "source_evidence": [],
        "status": status,
        "uncertainty": ["Synthetic benchmark evidence only."],
        "visual_evidence": [],
    }
    if status == "needs_clarification":
        payload["clarification_question"] = "Which single fixture should be evaluated?"
    else:
        payload["abstention_reason"] = "The requested conclusion is unsupported."
    return payload


class OfflineAblationTask:
    """Run one scripted capability projection without access to evaluator labels."""

    def __init__(
        self,
        benchmark: BenchmarkDataset,
        configuration: AblationConfiguration,
    ) -> None:
        if configuration.dataset_sha256 != benchmark.manifest.dataset_sha256:
            raise ValueError("ablation configuration targets a different dataset")
        self._images: dict[str, tuple[ImageAsset, bytes]] = {
            image.image_case_id: (
                image.asset,
                benchmark.read_image_by_id(image.image_case_id),
            )
            for image in benchmark.manifest.images
        }
        self.configuration = configuration
        self.corpus: LoadedReferenceCorpus = load_reference_corpus()
        self._index = InMemoryBM25Index.from_corpus(self.corpus)
        if self._index.fingerprint != configuration.retrieval_index_sha256:
            raise ValueError("ablation retrieval fingerprint is stale")

    @staticmethod
    def _visual_evidence(inputs: EvaluationInput) -> list[dict[str, object]]:
        return [
            {
                "confidence": 0.8,
                "description": "The complete synthetic fixture frame.",
                "locator": {
                    "box": None,
                    "image_id": inputs.image.image_id,
                    "kind": "full_frame",
                },
            }
        ]

    def _image_payload(self, inputs: EvaluationInput) -> dict[str, object]:
        _, content = self._registered_image(inputs)
        rows = decode_grayscale_png(content)
        left_mean = sum(row[x] for row in rows for x in range(8)) / (len(rows) * 8)
        right_mean = sum(row[x] for row in rows for x in range(56, 64)) / (len(rows) * 8)
        if right_mean - left_mean < 128:
            return _fallback_payload("abstain")
        return {
            "answer": (
                "The synthetic full-frame pattern is a horizontal gradient, darker "
                "on the left and brighter on the right."
            ),
            "confidence": 0.8,
            "observations": [
                "A horizontal grayscale gradient is darker at left and brighter at right."
            ],
            "source_evidence": [],
            "status": "answered",
            "uncertainty": ["Synthetic benchmark evidence only."],
            "visual_evidence": self._visual_evidence(inputs),
        }

    def _metadata_payload(
        self,
        inputs: EvaluationInput,
    ) -> tuple[dict[str, object], tuple[ObservedToolCall, ...]]:
        self._registered_image(inputs)
        arguments = {"image_id": inputs.image.image_id}
        call = ObservedToolCall(
            sequence=0,
            name="get_image_metadata",
            status="succeeded",
            arguments=arguments,
            result_sha256=canonical_sha256(inputs.image),
        )
        payload: dict[str, object] = {
            "answer": (
                f"The verified metadata reports a {inputs.image.width_px} by "
                f"{inputs.image.height_px} pixel synthetic PNG containing "
                f"{inputs.image.byte_size} bytes."
            ),
            "confidence": 0.8,
            "observations": ["The metadata was verified against the registered asset."],
            "source_evidence": [],
            "status": "answered",
            "uncertainty": ["Synthetic benchmark evidence only."],
            "visual_evidence": self._visual_evidence(inputs),
        }
        return payload, (call,)

    def _retrieval_payload(
        self,
        inputs: EvaluationInput,
    ) -> tuple[dict[str, object], tuple[ObservedToolCall, ...]]:
        query = _retrieval_query(inputs.prompt)
        results = self._index.search(query, top_k=1)
        if len(results) != 1:
            raise ValueError("benchmark retrieval did not produce exactly one result")
        result = results[0]
        arguments = {"query": query, "top_k": 1}
        call = ObservedToolCall(
            sequence=0,
            name="retrieve_reference",
            status="succeeded",
            arguments=arguments,
            result_sha256=canonical_sha256(results),
        )
        chunk = result.chunk
        payload: dict[str, object] = {
            "answer": chunk.text,
            "confidence": 0.8,
            "observations": [],
            "source_evidence": [
                {
                    "document_id": chunk.document_id,
                    "excerpt": chunk.text,
                    "locator": chunk.source_span.locator,
                    "relevance_score": 1.0,
                    "section": chunk.source_span.section,
                    "source_url": str(chunk.source_span.source_url),
                }
            ],
            "status": "answered",
            "uncertainty": ["The citation is limited to the frozen reference corpus."],
            "visual_evidence": [],
        }
        return payload, (call,)

    def __call__(self, inputs: EvaluationInput) -> ObservedRun:
        intent = _intent(inputs.prompt)
        tool_calls: tuple[ObservedToolCall, ...] = ()
        if intent == "contradiction":
            payload = _fallback_payload("needs_clarification")
        elif intent == "unsafe":
            payload = _fallback_payload("abstain")
        elif intent == "retrieval" and self.configuration.retrieval_tool:
            payload, tool_calls = self._retrieval_payload(inputs)
        elif intent == "metadata" and self.configuration.metadata_tool:
            payload, tool_calls = self._metadata_payload(inputs)
        elif intent == "observation" and self.configuration.image_access:
            payload = self._image_payload(inputs)
        else:
            payload = _fallback_payload("abstain")

        normalized_payload = VisualResponse.model_validate(payload).model_dump(
            mode="json",
            exclude={"trace"},
        )
        response_digest = canonical_sha256(normalized_payload)
        request_digest = canonical_sha256(inputs)
        trace: list[AgentTraceEvent] = [
            AgentTraceEvent(
                sequence=0,
                kind="request",
                status="started",
                name="cxr-evidence-agent",
                input_sha256=request_digest,
            )
        ]
        trace.extend(
            AgentTraceEvent(
                sequence=index + 1,
                kind="tool",
                status=call.status,
                name=call.name,
                duration_ms=1,
                input_sha256=canonical_sha256(call.arguments),
                output_sha256=call.result_sha256,
                failure_code=call.failure_code,
            )
            for index, call in enumerate(tool_calls)
        )
        model_sequence = len(trace)
        trace.extend(
            (
                AgentTraceEvent(
                    sequence=model_sequence,
                    kind="model",
                    status="succeeded",
                    name=self.configuration.configuration_id,
                    input_sha256=request_digest,
                    output_sha256=response_digest,
                    attributes={
                        "provider": "offline-scripted",
                        "model_requests": 1,
                        "tool_calls": len(tool_calls),
                    },
                ),
                AgentTraceEvent(
                    sequence=model_sequence + 1,
                    kind="validation",
                    status="succeeded",
                    name="visual-response",
                    input_sha256=response_digest,
                    output_sha256=response_digest,
                ),
                AgentTraceEvent(
                    sequence=model_sequence + 2,
                    kind="final",
                    status="succeeded",
                    name=str(payload["status"]),
                    output_sha256=response_digest,
                ),
            )
        )
        payload["trace"] = [event.model_dump(mode="json") for event in trace]
        response = VisualResponse.model_validate(payload)
        input_tokens = len(re.findall(r"[A-Za-z0-9]+", inputs.prompt))
        output_tokens = len(
            re.findall(
                r"[A-Za-z0-9]+",
                response.model_dump_json(exclude={"trace"}),
            )
        )
        latency_base = {
            "text_only": 4,
            "one_shot_vlm": 12,
            "vlm_tools": 18,
            "vlm_rag": 22,
            "full_agent": 28,
        }[self.configuration.configuration_id]
        usage = RunUsage(
            duration_ms=latency_base + inputs.seed % 5 + len(tool_calls) * 3,
            model_requests=1,
            tool_calls=len(tool_calls),
            image_bytes=inputs.image.byte_size if self.configuration.image_access else 0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=Decimal("0"),
            recovered=False,
        )
        return ObservedRun(
            case_id=inputs.case_id,
            configuration_id=self.configuration.configuration_id,
            response_payload=response.model_dump(mode="json"),
            tool_calls=tool_calls,
            limits=inputs.limits,
            usage=usage,
        )

    def _registered_image(self, inputs: EvaluationInput) -> tuple[ImageAsset, bytes]:
        try:
            registered = self._images[inputs.image.image_id]
        except KeyError:
            raise ValueError("evaluation image is not registered") from None
        if registered[0] != inputs.image:
            raise ValueError("evaluation input uses stale image metadata")
        return registered
