from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.retrieval.records import (
    CitationRecord,
    ReferenceAsset,
    ReferenceChunk,
    ReferenceDocument,
    ScoredChunk,
    TextSourceSpan,
    asset_id_for,
    chunk_id_for,
    citation_from_scored_chunk,
    document_id_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "provenance-records.json"
ASSET_SHA256 = hashlib.sha256(b"Synthetic public reference source.\n").hexdigest()
RENDITION_SHA256 = hashlib.sha256(b"Acquisition quality\nSynthetic reference text.").hexdigest()
EXTRACTOR_SHA256 = hashlib.sha256(b"plain-text-extractor:v1").hexdigest()
CHUNK_TEXT = "Synthetic reference text."


def records() -> tuple[
    ReferenceAsset,
    ReferenceDocument,
    ReferenceChunk,
    ScoredChunk,
    CitationRecord,
]:
    asset = ReferenceAsset(
        asset_id=asset_id_for(ASSET_SHA256),
        content_sha256=ASSET_SHA256,
        byte_size=35,
        media_type="text/plain",
        source_url="https://example.org/public-reference.txt",
        title="Synthetic public reference",
        publisher="Example publisher",
        accessed_on="2026-09-10",
        license_status="license_cleared",
        license_name="Synthetic test fixture license",
        contains_patient_data=False,
    )
    document = ReferenceDocument(
        document_id=document_id_for(
            asset_id=asset.asset_id,
            rendition_sha256=RENDITION_SHA256,
            extractor_fingerprint=EXTRACTOR_SHA256,
            language="en",
        ),
        asset_id=asset.asset_id,
        rendition_sha256=RENDITION_SHA256,
        extractor_name="plain-text-extractor",
        extractor_version="v1",
        extractor_fingerprint=EXTRACTOR_SHA256,
        title=asset.title,
        language="en",
    )
    span = TextSourceSpan(
        document_id=document.document_id,
        section="Acquisition quality",
        locator="lines 2-2",
        start_char=20,
        end_char=45,
        source_url=asset.source_url,
    )
    chunk_payload = {
        "document_id": document.document_id,
        "asset_id": asset.asset_id,
        "ordinal": 0,
        "text": CHUNK_TEXT,
        "text_sha256": hashlib.sha256(CHUNK_TEXT.encode()).hexdigest(),
        "rendition_sha256": document.rendition_sha256,
        "extractor_fingerprint": document.extractor_fingerprint,
        "source_span": span,
    }
    chunk = ReferenceChunk(
        chunk_id=chunk_id_for(**chunk_payload),
        **chunk_payload,
    )
    scored = ScoredChunk(
        chunk=chunk,
        rank=1,
        score_kind="bm25",
        raw_score=1.75,
        canonical_relevance=1.75,
        score_conversion="identity",
        higher_is_better=True,
        query_sha256=hashlib.sha256(b"synthetic query").hexdigest(),
        index_fingerprint="f" * 64,
    )
    citation = citation_from_scored_chunk(scored, start_char=0, end_char=9)
    return asset, document, chunk, scored, citation


def test_serialized_provenance_records_round_trip() -> None:
    raw = json.loads(RECORD_FIXTURE.read_text(encoding="utf-8"))
    model_types = (
        ReferenceAsset,
        ReferenceDocument,
        ReferenceChunk,
        ScoredChunk,
        CitationRecord,
    )

    parsed = tuple(
        model_type.model_validate(raw[name])
        for model_type, name in zip(
            model_types,
            ("asset", "document", "chunk", "scored_chunk", "citation"),
            strict=True,
        )
    )

    assert parsed == records()
    assert all(
        item.model_dump(mode="json") == model_type.model_validate(item).model_dump(mode="json")
        for item, model_type in zip(parsed, model_types, strict=True)
    )


def test_records_are_immutable_and_reject_extra_fields() -> None:
    asset, _, _, _, _ = records()

    with pytest.raises(ValidationError):
        asset.title = "Changed title"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ReferenceAsset.model_validate({**asset.model_dump(), "patient_name": "forbidden"})


@pytest.mark.parametrize("record_index", [0, 1, 2])
def test_content_addressed_identity_rejects_stale_ids(record_index: int) -> None:
    asset, document, chunk, _, _ = records()
    payloads = [
        {**asset.model_dump(), "content_sha256": "1" * 64},
        {**document.model_dump(), "rendition_sha256": "2" * 64},
        {**chunk.model_dump(), "text": "Altered reference text."},
    ]
    model_types = (ReferenceAsset, ReferenceDocument, ReferenceChunk)

    with pytest.raises(ValidationError, match="identity|digest"):
        model_types[record_index].model_validate(payloads[record_index])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_scored_chunk_rejects_non_finite_scores(score: float) -> None:
    *_, scored, _ = records()

    with pytest.raises(ValidationError, match="finite"):
        ScoredChunk.model_validate({**scored.model_dump(), "raw_score": score})


def test_canonical_relevance_is_higher_is_better_but_not_a_probability() -> None:
    *_, scored, _ = records()
    payload = {**scored.model_dump(), "raw_score": 9.5, "canonical_relevance": 9.5}

    high_bm25_score = ScoredChunk.model_validate(payload)

    assert high_bm25_score.canonical_relevance == 9.5
    with pytest.raises(ValidationError):
        ScoredChunk.model_validate({**payload, "higher_is_better": False})


def test_citation_must_be_an_exact_span_from_its_authorized_scored_chunk() -> None:
    *_, citation = records()
    assert citation.excerpt == "Synthetic"
    assert citation.scored_chunk.chunk.source_span.locator == "lines 2-2"

    with pytest.raises(ValidationError, match="exact chunk span"):
        CitationRecord.model_validate({**citation.model_dump(), "excerpt": "Invented"})
    with pytest.raises(ValueError, match="citation span"):
        citation_from_scored_chunk(citation.scored_chunk, start_char=9, end_char=9)


def test_source_span_and_chunk_provenance_must_reference_the_same_document() -> None:
    _, _, chunk, _, _ = records()
    payload = chunk.model_dump()
    payload["source_span"]["document_id"] = "document-other"  # type: ignore[index]

    with pytest.raises(ValidationError, match="same document"):
        ReferenceChunk.model_validate(payload)
