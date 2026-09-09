from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.tools import decode_grayscale_png

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "data" / "benchmark"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.json"


def _copied_benchmark(tmp_path: Path) -> Path:
    destination = tmp_path / "benchmark"
    shutil.copytree(BENCHMARK_ROOT, destination)
    return destination / "manifest.json"


def _rewrite_cases(manifest_path: Path, cases: list[dict[str, object]]) -> None:
    dataset_path = manifest_path.parent / "cxr-agent-bench-v0.jsonl"
    content = "".join(
        json.dumps(case, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for case in cases
    ).encode("utf-8")
    dataset_path.write_bytes(content)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def test_owned_benchmark_has_frozen_image_level_splits_and_categories() -> None:
    benchmark = load_benchmark(MANIFEST_PATH)

    assert benchmark.manifest.dataset_id == "cxr-agent-bench-v0"
    assert benchmark.manifest.no_patient_data is True
    assert len(benchmark.manifest.images) == 40
    assert len(benchmark.cases) == 120
    assert Counter(image.split for image in benchmark.manifest.images) == {
        "development": 24,
        "validation": 8,
        "test": 8,
    }
    assert Counter(case.split for case in benchmark.cases) == {
        "development": 72,
        "validation": 24,
        "test": 24,
    }
    assert {case.category for case in benchmark.cases} == {
        "observation",
        "tool_use",
        "retrieval",
        "contradiction",
        "abstention",
    }
    assert all(image.asset.license_status == "synthetic" for image in benchmark.manifest.images)
    assert all(image.asset.contains_phi is False for image in benchmark.manifest.images)

    cases_by_image: dict[str, list[str]] = defaultdict(list)
    for case in benchmark.cases:
        cases_by_image[case.image_case_id].append(case.case_id)
    assert set(cases_by_image) == {image.image_case_id for image in benchmark.manifest.images}
    assert {len(case_ids) for case_ids in cases_by_image.values()} == {3}


def test_split_views_share_no_images_source_groups_or_prompt_families() -> None:
    benchmark = load_benchmark(MANIFEST_PATH)
    split_names = ("development", "validation", "test")

    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            assert benchmark.image_ids(left).isdisjoint(benchmark.image_ids(right))
            assert benchmark.source_group_ids(left).isdisjoint(benchmark.source_group_ids(right))
            assert benchmark.prompt_families(left).isdisjoint(benchmark.prompt_families(right))


def test_loader_rejects_case_assigned_to_an_image_from_another_split(
    tmp_path: Path,
) -> None:
    manifest_path = _copied_benchmark(tmp_path)
    dataset_path = manifest_path.parent / "cxr-agent-bench-v0.jsonl"
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    development = next(case for case in cases if case["split"] == "development")
    held_out = next(case for case in cases if case["split"] == "test")
    held_out["image_case_id"] = development["image_case_id"]
    _rewrite_cases(manifest_path, cases)

    with pytest.raises(ValueError, match="image-level split"):
        load_benchmark(manifest_path)


def test_loader_rejects_cross_split_duplicate_prompt_content(tmp_path: Path) -> None:
    manifest_path = _copied_benchmark(tmp_path)
    dataset_path = manifest_path.parent / "cxr-agent-bench-v0.jsonl"
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    development = next(case for case in cases if case["split"] == "development")
    held_out = next(case for case in cases if case["split"] == "test")
    held_out["prompt"] = development["prompt"]
    _rewrite_cases(manifest_path, cases)

    with pytest.raises(ValueError, match="prompt leakage"):
        load_benchmark(manifest_path)


def test_loader_rejects_cross_split_near_duplicate_prompt_content(tmp_path: Path) -> None:
    manifest_path = _copied_benchmark(tmp_path)
    dataset_path = manifest_path.parent / "cxr-agent-bench-v0.jsonl"
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    development = next(case for case in cases if case["split"] == "development")
    held_out = next(case for case in cases if case["split"] == "test")
    held_out["prompt"] = f"{development['prompt']} Please."
    _rewrite_cases(manifest_path, cases)

    with pytest.raises(ValueError, match="near-duplicate prompt leakage"):
        load_benchmark(manifest_path)


def test_loader_rejects_untracked_or_modified_image_bytes(tmp_path: Path) -> None:
    manifest_path = _copied_benchmark(tmp_path)
    image_root = manifest_path.parent / "images"
    (image_root / "untracked.png").write_bytes(b"not a benchmark image")

    with pytest.raises(ValueError, match="untracked benchmark image"):
        load_benchmark(manifest_path)

    (image_root / "untracked.png").unlink()
    tracked = next(image_root.glob("*.png"))
    tracked.write_bytes(tracked.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="byte size mismatch"):
        load_benchmark(manifest_path)


def test_generator_is_byte_reproducible(tmp_path: Path) -> None:
    output_root = tmp_path / "generated"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_benchmark.py"),
        "--output-root",
        str(output_root),
    ]

    subprocess.run(command, check=True)
    first = {
        path.relative_to(output_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    subprocess.run(command, check=True)
    second = {
        path.relative_to(output_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    committed = {
        path.relative_to(BENCHMARK_ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(BENCHMARK_ROOT.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl", ".png"}
    }

    assert len(first) == 42
    assert first == second
    assert first == committed
    assert load_benchmark(output_root / "manifest.json").manifest.seed == 20260910


def test_dataset_lookup_rejects_an_unregistered_image_id() -> None:
    benchmark = load_benchmark(MANIFEST_PATH)

    with pytest.raises(ValueError, match="not registered"):
        benchmark.image_for_id("unknown-image")


def test_observation_ground_truth_is_present_in_the_image_pixels() -> None:
    benchmark = load_benchmark(MANIFEST_PATH)
    case = next(case for case in benchmark.cases if case.category == "observation")

    rows = decode_grayscale_png(benchmark.read_image(case))
    left_mean = sum(row[x] for row in rows for x in range(8)) / (len(rows) * 8)
    right_mean = sum(row[x] for row in rows for x in range(56, 64)) / (len(rows) * 8)

    assert right_mean - left_mean >= 128
    assert len(set(rows)) == 1


def test_metadata_expectations_require_the_registered_byte_count() -> None:
    benchmark = load_benchmark(MANIFEST_PATH)

    for case in (case for case in benchmark.cases if case.category == "tool_use"):
        image = benchmark.image_for(case)
        assert str(image.asset.byte_size) in case.expected.required_answer_terms
        assert str(image.asset.byte_size) in (case.expected.reference_answer or "")
