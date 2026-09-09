from __future__ import annotations

import asyncio
import hashlib
import json
import math
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.retrieval import (
    BM25Configuration,
    IndexRejected,
    InMemoryBM25Index,
    LexicalReferenceService,
    ReferenceChunk,
    RetrievalReport,
    TextSourceSpan,
    chunk_id_for,
    load_reference_corpus,
)
from chest_xray_evidence_assistant.tools import RetrieveReferenceArguments

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "references" / "manifest.json"
REPORT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "retrieval" / "lexical-report.json"


def index() -> InMemoryBM25Index:
    return InMemoryBM25Index.from_corpus(load_reference_corpus(MANIFEST_PATH))


@pytest.mark.parametrize(
    ("query", "title_fragment"),
    [
        ("detector static projection", "radiography-formation"),
        ("computed tomography cross-sectional slices", "projection-vs-ct"),
        ("justification optimization exposure", "radiation-purpose"),
    ],
)
def test_bm25_returns_the_expected_source_first(query: str, title_fragment: str) -> None:
    corpus = load_reference_corpus(MANIFEST_PATH)
    lexical_index = InMemoryBM25Index.from_corpus(corpus)
    expected_asset = next(
        source.asset.asset_id for source in corpus.manifest.sources if title_fragment in source.path
    )

    results = lexical_index.search(query, top_k=3)

    assert results
    assert results[0].chunk.asset_id == expected_asset
    assert all(result.score_kind == "bm25" for result in results)
    assert all(result.score_conversion == "identity" for result in results)
    assert all(result.raw_score == result.canonical_relevance for result in results)
    assert all(math.isfinite(result.raw_score) and result.raw_score > 0 for result in results)
    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_top_k_is_bounded_and_zero_match_queries_return_no_filler() -> None:
    lexical_index = index()

    assert len(lexical_index.search("x-ray radiography image", top_k=2)) <= 2
    assert lexical_index.search("---", top_k=5) == ()
    with pytest.raises(IndexRejected, match="^invalid_top_k$"):
        lexical_index.search("radiography", top_k=6)


def test_equal_scores_use_stable_chunk_identity_as_the_tie_break() -> None:
    template = load_reference_corpus(MANIFEST_PATH).chunks[0]

    def tied_chunk(document_id: str) -> ReferenceChunk:
        span = TextSourceSpan(
            document_id=document_id,
            section="Tie fixture",
            locator="tie fixture",
            start_char=0,
            end_char=len(template.text),
            source_url=template.source_span.source_url,
        )
        payload = {
            **template.model_dump(),
            "document_id": document_id,
            "source_span": span,
        }
        payload["chunk_id"] = chunk_id_for(
            document_id=document_id,
            asset_id=template.asset_id,
            ordinal=template.ordinal,
            text=template.text,
            text_sha256=template.text_sha256,
            rendition_sha256=template.rendition_sha256,
            extractor_fingerprint=template.extractor_fingerprint,
            source_span=span,
        )
        return ReferenceChunk.model_validate(payload)

    chunks = (tied_chunk("document-tie-b"), tied_chunk("document-tie-a"))
    lexical_index = InMemoryBM25Index(chunks=chunks, corpus_fingerprint="c" * 64)

    results = lexical_index.search("radiography", top_k=2)

    assert results[0].raw_score == results[1].raw_score
    assert [result.chunk.chunk_id for result in results] == sorted(
        result.chunk.chunk_id for result in results
    )


def test_index_rejects_duplicate_chunk_identity() -> None:
    corpus = load_reference_corpus(MANIFEST_PATH)

    with pytest.raises(IndexRejected, match="^duplicate_chunk$"):
        InMemoryBM25Index(
            chunks=(corpus.chunks[0], corpus.chunks[0]),
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )


def test_index_rejects_empty_or_untokenizable_chunk_sets() -> None:
    corpus = load_reference_corpus(MANIFEST_PATH)

    with pytest.raises(IndexRejected, match="^empty_index$"):
        InMemoryBM25Index(
            chunks=(),
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )

    template = corpus.chunks[0]
    text = "---"
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    span = TextSourceSpan(
        document_id=template.document_id,
        section="Untokenizable fixture",
        locator="untokenizable fixture",
        start_char=0,
        end_char=len(text),
        source_url=template.source_span.source_url,
    )
    payload = {
        **template.model_dump(),
        "text": text,
        "text_sha256": text_sha256,
        "source_span": span,
    }
    payload["chunk_id"] = chunk_id_for(
        document_id=template.document_id,
        asset_id=template.asset_id,
        ordinal=template.ordinal,
        text=text,
        text_sha256=text_sha256,
        rendition_sha256=template.rendition_sha256,
        extractor_fingerprint=template.extractor_fingerprint,
        source_span=span,
    )
    chunk = ReferenceChunk.model_validate(payload)

    with pytest.raises(IndexRejected, match="^invalid_chunk$"):
        InMemoryBM25Index(
            chunks=(chunk,),
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        )


def test_configuration_and_index_fingerprints_change_with_behavior() -> None:
    default = BM25Configuration.create()
    changed = BM25Configuration.create(k1=1.2)
    corpus = load_reference_corpus(MANIFEST_PATH)

    assert default.fingerprint != changed.fingerprint
    assert (
        InMemoryBM25Index.from_corpus(corpus).fingerprint
        != InMemoryBM25Index.from_corpus(corpus, configuration=changed).fingerprint
    )


def test_reference_service_implements_the_bounded_tool_port() -> None:
    service = LexicalReferenceService(index())

    results = asyncio.run(
        service.retrieve_reference(
            RetrieveReferenceArguments(query="computed tomography slices", top_k=1)
        )
    )

    assert len(results) == 1
    assert results[0].chunk.source_span.source_url.host == "www.fda.gov"
    assert results[0].rank == 1


def test_retrieval_report_fixture_is_exact_and_contains_no_raw_query() -> None:
    fixture = json.loads(REPORT_FIXTURE.read_text(encoding="utf-8"))
    expected = RetrievalReport.model_validate(fixture)
    report = index().report("computed tomography cross-sectional slices", top_k=3)

    assert report == expected
    assert "computed tomography cross-sectional slices" not in report.model_dump_json()
    assert all(match.source_url.host == "www.fda.gov" for match in report.matches)
