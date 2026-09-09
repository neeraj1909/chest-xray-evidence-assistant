"""Manifest-first loading for the owned, synthetic CXR-AgentBench-v0 dataset."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..models import (
    ContractModel,
    Identifier,
    ImageAsset,
    QuestionText,
    Sha256Digest,
    ShortText,
)

BenchmarkSplit: TypeAlias = Literal["development", "validation", "test"]
TaskCategory: TypeAlias = Literal[
    "observation",
    "tool_use",
    "retrieval",
    "contradiction",
    "abstention",
]
ExpectedStatus: TypeAlias = Literal["answered", "needs_clarification", "abstain"]
ExpectedToolName: TypeAlias = Literal[
    "crop_image",
    "get_image_metadata",
    "retrieve_reference",
]
ToolArgumentValue: TypeAlias = str | int | float | bool
RelativePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]

BENCHMARK_ID = "cxr-agent-bench-v0"
DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "data" / "benchmark"
MAX_MANIFEST_BYTES = 1_000_000
MAX_DATASET_BYTES = 5_000_000
MAX_CASE_LINE_BYTES = 32_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ExpectedToolCall(ContractModel):
    name: ExpectedToolName
    arguments: dict[Identifier, ToolArgumentValue] = Field(max_length=8)


class BenchmarkExpected(ContractModel):
    status: ExpectedStatus
    reference_answer: ShortText | None = None
    required_answer_terms: tuple[ShortText, ...] = Field(default=(), max_length=8)
    forbidden_claims: tuple[ShortText, ...] = Field(min_length=1, max_length=8)
    tool_calls: tuple[ExpectedToolCall, ...] = Field(default=(), max_length=3)
    locator_kind: Literal["full_frame", "bounding_box"] | None = None
    source_document_ids: tuple[Identifier, ...] = Field(default=(), max_length=5)
    safety_critical: bool = False
    max_tool_calls: int = Field(default=3, ge=0, le=3)
    max_model_requests: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def require_status_shape(self) -> BenchmarkExpected:
        if self.status == "answered" and self.reference_answer is None:
            raise ValueError("answered expectations require a reference answer")
        if self.status != "answered" and self.reference_answer is not None:
            raise ValueError("fallback expectations cannot include a reference answer")
        if len(self.tool_calls) > self.max_tool_calls:
            raise ValueError("expected tool calls exceed the declared budget")
        return self


class BenchmarkCase(ContractModel):
    case_id: Identifier
    image_case_id: Identifier
    split: BenchmarkSplit
    category: TaskCategory
    prompt_family: Identifier
    prompt: QuestionText
    seed: int = Field(ge=0, le=2**31 - 1)
    expected: BenchmarkExpected

    @model_validator(mode="after")
    def require_category_contract(self) -> BenchmarkCase:
        tool_names = {tool.name for tool in self.expected.tool_calls}
        if self.category == "observation":
            if self.expected.status != "answered" or self.expected.locator_kind is None:
                raise ValueError("observation cases require an answered image locator")
        elif self.category == "tool_use":
            if self.expected.status != "answered" or not tool_names:
                raise ValueError("tool-use cases require an answered tool call")
        elif self.category == "retrieval":
            if (
                self.expected.status != "answered"
                or "retrieve_reference" not in tool_names
                or not self.expected.source_document_ids
            ):
                raise ValueError("retrieval cases require retrieval and source provenance")
        elif self.category == "contradiction":
            if self.expected.status != "needs_clarification" or not self.expected.safety_critical:
                raise ValueError("contradiction cases require safety-critical clarification")
        elif self.expected.status != "abstain" or not self.expected.safety_critical:
            raise ValueError("abstention cases require a safety-critical abstention")
        return self


class BenchmarkImageCase(ContractModel):
    image_case_id: Identifier
    source_group_id: Identifier
    path: RelativePath
    split: BenchmarkSplit
    description: ShortText
    asset: ImageAsset

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("benchmark image paths must remain relative to the manifest")
        return value

    @model_validator(mode="after")
    def require_matching_image_id(self) -> BenchmarkImageCase:
        if self.image_case_id != self.asset.image_id:
            raise ValueError("benchmark image case ID must match the asset image ID")
        return self


class BenchmarkManifest(ContractModel):
    schema_version: Literal[1]
    dataset_id: Literal["cxr-agent-bench-v0"]
    dataset_version: Literal["v0"]
    generated_by: Literal["scripts/generate_benchmark.py"]
    seed: int = Field(ge=0, le=2**31 - 1)
    dataset_path: RelativePath
    dataset_sha256: Sha256Digest
    no_patient_data: Literal[True]
    license_status: Literal["synthetic"]
    image_count: int = Field(ge=40, le=256)
    prompt_count: int = Field(ge=120, le=2_048)
    split_image_counts: dict[BenchmarkSplit, int]
    split_prompt_counts: dict[BenchmarkSplit, int]
    category_counts: dict[TaskCategory, int]
    images: list[BenchmarkImageCase] = Field(min_length=40, max_length=256)

    @field_validator("dataset_path")
    @classmethod
    def require_jsonl_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".jsonl":
            raise ValueError("benchmark dataset path must be a relative JSONL file")
        return value

    @model_validator(mode="after")
    def require_image_ledger_consistency(self) -> BenchmarkManifest:
        image_ids = [image.image_case_id for image in self.images]
        source_groups = [image.source_group_id for image in self.images]
        paths = [image.path for image in self.images]
        digests = [image.asset.sha256 for image in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("benchmark image IDs must be unique")
        if len(source_groups) != len(set(source_groups)):
            raise ValueError("benchmark source groups must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("benchmark image paths must be unique")
        if len(digests) != len(set(digests)):
            raise ValueError("benchmark image bytes must be unique")
        if self.image_count != len(self.images):
            raise ValueError("benchmark image count does not match the image ledger")
        if self.split_image_counts != dict(Counter(image.split for image in self.images)):
            raise ValueError("benchmark split image counts do not match the image ledger")
        return self


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    manifest: BenchmarkManifest
    cases: tuple[BenchmarkCase, ...]
    root: Path

    def cases_for_split(self, split: BenchmarkSplit) -> tuple[BenchmarkCase, ...]:
        return tuple(case for case in self.cases if case.split == split)

    def image_ids(self, split: BenchmarkSplit) -> frozenset[str]:
        return frozenset(
            image.image_case_id for image in self.manifest.images if image.split == split
        )

    def source_group_ids(self, split: BenchmarkSplit) -> frozenset[str]:
        return frozenset(
            image.source_group_id for image in self.manifest.images if image.split == split
        )

    def prompt_families(self, split: BenchmarkSplit) -> frozenset[str]:
        return frozenset(case.prompt_family for case in self.cases if case.split == split)

    def image_for(self, case: BenchmarkCase) -> BenchmarkImageCase:
        return self.image_for_id(case.image_case_id)

    def image_for_id(self, image_case_id: str) -> BenchmarkImageCase:
        try:
            return next(
                image for image in self.manifest.images if image.image_case_id == image_case_id
            )
        except StopIteration:
            raise ValueError("benchmark image ID is not registered") from None

    def read_image(self, case: BenchmarkCase) -> bytes:
        return self.read_image_by_id(case.image_case_id)

    def read_image_by_id(self, image_case_id: str) -> bytes:
        image = self.image_for_id(image_case_id)
        path = _resolve_inside(self.root, image.path, "benchmark image")
        content = _read_bounded(path, image.asset.byte_size, "benchmark image")
        if len(content) != image.asset.byte_size:
            raise ValueError("benchmark image byte size mismatch")
        if hashlib.sha256(content).hexdigest() != image.asset.sha256:
            raise ValueError("benchmark image SHA-256 mismatch")
        return content


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        raise ValueError(f"{label} is unavailable") from None
    if size <= 0 or size > limit:
        raise ValueError(f"{label} size is outside the allowed boundary")
    with path.open("rb") as source:
        content = source.read(limit + 1)
    if len(content) != size:
        raise ValueError(f"{label} changed while it was read")
    return content


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"{label} path escapes the benchmark root") from None
    return candidate


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or not content.startswith(PNG_SIGNATURE) or content[12:16] != b"IHDR":
        raise ValueError("benchmark image is not a supported PNG")
    return struct.unpack(">II", content[16:24])


def _verify_images(root: Path, manifest: BenchmarkManifest) -> None:
    tracked_paths: set[Path] = set()
    for image in manifest.images:
        candidate = _resolve_inside(root, image.path, "benchmark image")
        tracked_paths.add(candidate)
        try:
            size = candidate.stat().st_size
        except OSError:
            raise ValueError(f"benchmark image is unavailable: {image.path}") from None
        if size != image.asset.byte_size:
            raise ValueError(f"benchmark image byte size mismatch: {image.path}")
        with candidate.open("rb") as source:
            content = source.read(image.asset.byte_size + 1)
        if len(content) != image.asset.byte_size:
            raise ValueError(f"benchmark image byte size mismatch: {image.path}")
        if hashlib.sha256(content).hexdigest() != image.asset.sha256:
            raise ValueError(f"benchmark image SHA-256 mismatch: {image.path}")
        if _png_dimensions(content) != (image.asset.width_px, image.asset.height_px):
            raise ValueError(f"benchmark image dimensions mismatch: {image.path}")

    image_root = (root / "images").resolve()
    actual_paths = {path.resolve() for path in image_root.glob("*.png") if path.is_file()}
    untracked = actual_paths - tracked_paths
    if untracked:
        raise ValueError("untracked benchmark image")


def _prompt_signature(case: BenchmarkCase) -> str:
    normalized = re.sub(r"\bcxr-bench-\d+\b", "<image>", case.prompt.lower())
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = " ".join(re.findall(r"[a-z]+|<image>|<n>", normalized))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prompt_tokens(case: BenchmarkCase) -> frozenset[str]:
    normalized = re.sub(r"\bcxr-bench-\d+\b", "<image>", case.prompt.lower())
    return frozenset(re.findall(r"[a-z]+|<image>", normalized))


def _validate_cases(manifest: BenchmarkManifest, cases: tuple[BenchmarkCase, ...]) -> None:
    if len(cases) != manifest.prompt_count:
        raise ValueError("benchmark prompt count does not match the dataset")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark case IDs must be unique")

    images = {image.image_case_id: image for image in manifest.images}
    cases_by_image: dict[str, list[BenchmarkCase]] = defaultdict(list)
    family_splits: dict[str, set[str]] = defaultdict(set)
    signature_splits: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        image = images.get(case.image_case_id)
        if image is None or image.split != case.split:
            raise ValueError("benchmark case violates the image-level split")
        cases_by_image[case.image_case_id].append(case)
        family_splits[case.prompt_family].add(case.split)
        signature_splits[_prompt_signature(case)].add(case.split)

        for expected_call in case.expected.tool_calls:
            if expected_call.name == "get_image_metadata" and expected_call.arguments != {
                "image_id": case.image_case_id
            }:
                raise ValueError("metadata expectation targets the wrong image")

    if set(cases_by_image) != set(images) or any(
        len(image_cases) != 3 for image_cases in cases_by_image.values()
    ):
        raise ValueError("every benchmark image must own exactly three prompts")
    if any(len(splits) > 1 for splits in family_splits.values()):
        raise ValueError("benchmark prompt family leakage across splits")
    if any(len(splits) > 1 for splits in signature_splits.values()):
        raise ValueError("benchmark normalized prompt leakage across splits")
    for index, left in enumerate(cases):
        left_tokens = _prompt_tokens(left)
        for right in cases[index + 1 :]:
            if left.split == right.split:
                continue
            right_tokens = _prompt_tokens(right)
            similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
            if similarity >= 0.8:
                raise ValueError("benchmark near-duplicate prompt leakage across splits")

    prompt_split_counts = dict(Counter(case.split for case in cases))
    category_counts = dict(Counter(case.category for case in cases))
    if manifest.split_prompt_counts != prompt_split_counts:
        raise ValueError("benchmark split prompt counts do not match the dataset")
    if manifest.category_counts != category_counts:
        raise ValueError("benchmark category counts do not match the dataset")


def load_benchmark(path: Path | None = None) -> BenchmarkDataset:
    """Load and fully verify the benchmark manifest, JSONL cases, and image bytes."""

    manifest_path = (path or DEFAULT_BENCHMARK_ROOT / "manifest.json").resolve()
    root = manifest_path.parent.resolve()
    manifest_content = _read_bounded(manifest_path, MAX_MANIFEST_BYTES, "benchmark manifest")
    try:
        manifest = BenchmarkManifest.model_validate_json(manifest_content)
    except (ValueError, json.JSONDecodeError):
        raise ValueError("benchmark manifest is invalid") from None

    dataset_path = _resolve_inside(root, manifest.dataset_path, "benchmark dataset")
    dataset_content = _read_bounded(dataset_path, MAX_DATASET_BYTES, "benchmark dataset")
    if hashlib.sha256(dataset_content).hexdigest() != manifest.dataset_sha256:
        raise ValueError("benchmark dataset SHA-256 mismatch")
    try:
        lines = dataset_content.splitlines()
        if any(not line or len(line) > MAX_CASE_LINE_BYTES for line in lines):
            raise ValueError
        cases = tuple(BenchmarkCase.model_validate_json(line) for line in lines)
    except ValueError:
        raise ValueError("benchmark dataset JSONL is invalid") from None

    _verify_images(root, manifest)
    _validate_cases(manifest, cases)
    return BenchmarkDataset(manifest=manifest, cases=cases, root=root)
