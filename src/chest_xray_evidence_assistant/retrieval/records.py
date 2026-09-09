"""Immutable, content-addressed records for provenance-preserving text retrieval."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, StringConstraints, field_validator, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from ._canonical import canonical_sha256

ReferenceText = Annotated[str, StringConstraints(min_length=1, max_length=16_000)]
LanguageCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z]{2,3}$"),
]
ReferenceMediaType = Literal["text/plain", "text/markdown"]
ReferenceLicenseStatus = Literal["public_domain", "open_access", "license_cleared"]
ScoreKind = Literal[
    "bm25",
    "cosine_similarity",
    "inner_product",
    "euclidean_distance",
    "reciprocal_rank_fusion",
    "reranker",
]
ScoreConversion = Literal[
    "identity",
    "negated_distance",
    "reciprocal_rank_fusion",
    "reranker_identity",
]


def asset_id_for(content_sha256: Sha256Digest) -> str:
    """Derive a stable asset identity from the exact acquired bytes."""

    return f"asset-{content_sha256[:40]}"


def document_id_for(
    *,
    asset_id: Identifier,
    rendition_sha256: Sha256Digest,
    extractor_fingerprint: Sha256Digest,
    language: str,
) -> str:
    """Derive a document identity from its rendition and extraction behavior."""

    digest = canonical_sha256(
        {
            "asset_id": asset_id,
            "extractor_fingerprint": extractor_fingerprint,
            "language": language,
            "rendition_sha256": rendition_sha256,
        }
    )
    return f"document-{digest[:40]}"


class ReferenceAsset(ContractModel):
    """One bounded, licensed source asset acquired for the reference corpus."""

    asset_id: Identifier
    content_sha256: Sha256Digest
    byte_size: int = Field(gt=0, le=2_000_000)
    media_type: ReferenceMediaType
    source_url: HttpUrl
    title: ShortText
    publisher: ShortText
    accessed_on: date
    license_status: ReferenceLicenseStatus
    license_name: ShortText
    license_url: HttpUrl | None = None
    contains_patient_data: Literal[False] = False

    @model_validator(mode="after")
    def validate_identity(self) -> ReferenceAsset:
        if self.asset_id != asset_id_for(self.content_sha256):
            raise ValueError("asset identity does not match the content digest")
        return self


class ReferenceDocument(ContractModel):
    """A normalized text rendition tied to an acquired asset and extractor."""

    document_id: Identifier
    asset_id: Identifier
    rendition_sha256: Sha256Digest
    extractor_name: Identifier
    extractor_version: Identifier
    extractor_fingerprint: Sha256Digest
    title: ShortText
    language: LanguageCode

    @model_validator(mode="after")
    def validate_identity(self) -> ReferenceDocument:
        expected = document_id_for(
            asset_id=self.asset_id,
            rendition_sha256=self.rendition_sha256,
            extractor_fingerprint=self.extractor_fingerprint,
            language=self.language,
        )
        if self.document_id != expected:
            raise ValueError("document identity does not match its rendition")
        return self


class TextSourceSpan(ContractModel):
    """Exact half-open character span in a normalized text rendition."""

    document_id: Identifier
    section: ShortText
    locator: ShortText
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    source_url: HttpUrl

    @model_validator(mode="after")
    def validate_span(self) -> TextSourceSpan:
        if self.start_char >= self.end_char:
            raise ValueError("source spans must be non-empty and ordered")
        return self


def chunk_id_for(
    *,
    document_id: Identifier,
    asset_id: Identifier,
    ordinal: int,
    text: str,
    text_sha256: Sha256Digest,
    rendition_sha256: Sha256Digest,
    extractor_fingerprint: Sha256Digest,
    source_span: TextSourceSpan,
) -> str:
    """Derive a chunk identity from content plus its exact provenance."""

    if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_sha256:
        raise ValueError("chunk text digest does not match its text")
    digest = canonical_sha256(
        {
            "asset_id": asset_id,
            "document_id": document_id,
            "extractor_fingerprint": extractor_fingerprint,
            "ordinal": ordinal,
            "rendition_sha256": rendition_sha256,
            "source_span": source_span,
            "text_sha256": text_sha256,
        }
    )
    return f"chunk-{digest[:40]}"


class ReferenceChunk(ContractModel):
    """A searchable text chunk whose ID changes with content or provenance."""

    chunk_id: Identifier
    document_id: Identifier
    asset_id: Identifier
    ordinal: int = Field(ge=0, le=1_000_000)
    text: ReferenceText
    text_sha256: Sha256Digest
    rendition_sha256: Sha256Digest
    extractor_fingerprint: Sha256Digest
    source_span: TextSourceSpan

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_content_and_identity(self) -> ReferenceChunk:
        if self.source_span.document_id != self.document_id:
            raise ValueError("chunk and source span must reference the same document")
        expected_text_sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_sha256 != expected_text_sha256:
            raise ValueError("chunk text digest does not match its text")
        expected_id = chunk_id_for(
            document_id=self.document_id,
            asset_id=self.asset_id,
            ordinal=self.ordinal,
            text=self.text,
            text_sha256=self.text_sha256,
            rendition_sha256=self.rendition_sha256,
            extractor_fingerprint=self.extractor_fingerprint,
            source_span=self.source_span,
        )
        if self.chunk_id != expected_id:
            raise ValueError("chunk identity does not match content and provenance")
        return self


class ScoredChunk(ContractModel):
    """One ranked chunk with explicit native and canonical score semantics."""

    chunk: ReferenceChunk
    rank: int = Field(gt=0, le=1_000)
    score_kind: ScoreKind
    raw_score: float
    canonical_relevance: float = Field(ge=0)
    score_conversion: ScoreConversion
    higher_is_better: Literal[True] = True
    query_sha256: Sha256Digest
    index_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_scores(self) -> ScoredChunk:
        if not math.isfinite(self.raw_score) or not math.isfinite(self.canonical_relevance):
            raise ValueError("retrieval scores must be finite")
        return self


def citation_id_for(
    scored_chunk: ScoredChunk,
    *,
    start_char: int,
    end_char: int,
    excerpt_sha256: Sha256Digest,
) -> str:
    digest = canonical_sha256(
        {
            "chunk_id": scored_chunk.chunk.chunk_id,
            "end_char": end_char,
            "excerpt_sha256": excerpt_sha256,
            "index_fingerprint": scored_chunk.index_fingerprint,
            "query_sha256": scored_chunk.query_sha256,
            "start_char": start_char,
        }
    )
    return f"citation-{digest[:40]}"


class CitationRecord(ContractModel):
    """An exact excerpt from one authorized, scored retrieval result."""

    citation_id: Identifier
    scored_chunk: ScoredChunk
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    excerpt: ReferenceText
    excerpt_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_excerpt_and_identity(self) -> CitationRecord:
        text = self.scored_chunk.chunk.text
        if self.start_char >= self.end_char or self.end_char > len(text):
            raise ValueError("citation span must be non-empty and inside the chunk")
        if self.excerpt != text[self.start_char : self.end_char]:
            raise ValueError("citation excerpt must match the exact chunk span")
        if self.excerpt_sha256 != hashlib.sha256(self.excerpt.encode("utf-8")).hexdigest():
            raise ValueError("citation excerpt digest does not match its text")
        expected_id = citation_id_for(
            self.scored_chunk,
            start_char=self.start_char,
            end_char=self.end_char,
            excerpt_sha256=self.excerpt_sha256,
        )
        if self.citation_id != expected_id:
            raise ValueError("citation identity does not match its authorized span")
        return self


def citation_from_scored_chunk(
    scored_chunk: ScoredChunk,
    *,
    start_char: int,
    end_char: int,
) -> CitationRecord:
    """Build a citation only from text present in an authorized scored chunk."""

    if start_char < 0 or start_char >= end_char or end_char > len(scored_chunk.chunk.text):
        raise ValueError("citation span must be non-empty and inside the chunk")
    excerpt = scored_chunk.chunk.text[start_char:end_char]
    excerpt_sha256 = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    return CitationRecord(
        citation_id=citation_id_for(
            scored_chunk,
            start_char=start_char,
            end_char=end_char,
            excerpt_sha256=excerpt_sha256,
        ),
        scored_chunk=scored_chunk,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        excerpt_sha256=excerpt_sha256,
    )
