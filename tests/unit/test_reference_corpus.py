from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.retrieval.corpus import (
    CorpusRejected,
    ReferenceCorpusManifest,
    corpus_fingerprint_for,
    load_reference_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "data" / "references"
MANIFEST_PATH = REFERENCE_ROOT / "manifest.json"


def copy_corpus(tmp_path: Path) -> Path:
    destination = tmp_path / "references"
    shutil.copytree(REFERENCE_ROOT, destination)
    return destination / "manifest.json"


def test_manifest_first_ingestion_returns_only_tracked_provenance_chunks() -> None:
    corpus = load_reference_corpus(MANIFEST_PATH)

    assert len(corpus.manifest.sources) == 3
    assert len(corpus.chunks) == 3
    assert corpus.manifest.logical_index == "sparse"
    assert corpus.manifest.chunking.strategy == "source_section"
    assert corpus.manifest.corpus_fingerprint == corpus_fingerprint_for(corpus.manifest)
    assert all(source.asset.contains_patient_data is False for source in corpus.manifest.sources)
    assert all(
        source.asset.license_status == "license_cleared" for source in corpus.manifest.sources
    )
    assert all(chunk.source_span.start_char == 0 for chunk in corpus.chunks)
    assert all(chunk.source_span.end_char == len(chunk.text) for chunk in corpus.chunks)
    assert {chunk.asset_id for chunk in corpus.chunks} == {
        source.asset.asset_id for source in corpus.manifest.sources
    }


def test_corpus_is_small_and_every_document_file_is_manifest_tracked() -> None:
    corpus = load_reference_corpus(MANIFEST_PATH)
    tracked = {source.path for source in corpus.manifest.sources}
    actual = {
        path.relative_to(REFERENCE_ROOT).as_posix()
        for path in (REFERENCE_ROOT / "documents").iterdir()
        if path.is_file()
    }

    assert tracked == actual
    assert sum(source.asset.byte_size for source in corpus.manifest.sources) < 10_000


def test_changed_source_bytes_are_rejected_before_ingestion(tmp_path: Path) -> None:
    manifest_path = copy_corpus(tmp_path)
    source_path = manifest_path.parent / "documents" / "radiography-formation.md"
    source_path.write_text(source_path.read_text() + "\nChanged after review.\n", encoding="utf-8")

    with pytest.raises(CorpusRejected, match="^content_mismatch$"):
        load_reference_corpus(manifest_path)


def test_untracked_document_is_rejected(tmp_path: Path) -> None:
    manifest_path = copy_corpus(tmp_path)
    (manifest_path.parent / "documents" / "untracked.md").write_text(
        "Not authorized by the manifest.",
        encoding="utf-8",
    )

    with pytest.raises(CorpusRejected, match="^untracked_source$"):
        load_reference_corpus(manifest_path)


@pytest.mark.parametrize(
    "mutation",
    ["path_traversal", "missing_license", "duplicate_asset", "patient_data"],
)
def test_unsafe_manifest_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    manifest_path = copy_corpus(tmp_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = raw["sources"][0]
    if mutation == "path_traversal":
        first["path"] = "../outside.md"
    elif mutation == "missing_license":
        del first["asset"]["license_name"]
    elif mutation == "duplicate_asset":
        raw["sources"][1]["asset"] = first["asset"]
    else:
        first["asset"]["contains_patient_data"] = True
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(CorpusRejected, match="^invalid_manifest$"):
        load_reference_corpus(manifest_path)


def test_manifest_rejects_a_stale_corpus_fingerprint() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["corpus_version"] = "v2"

    with pytest.raises(ValueError, match="corpus fingerprint"):
        ReferenceCorpusManifest.model_validate(raw)
