"""Deterministic in-memory BM25 retrieval over validated reference chunks."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal, TypeAlias

from pydantic import Field, HttpUrl, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from ._canonical import canonical_sha256
from .corpus import LoadedReferenceCorpus
from .ports import RetrievalAdapterDescriptor, retrieval_adapter_fingerprint
from .records import ReferenceChunk, ScoredChunk

TOKEN_PATTERN = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"
IndexFailureCode: TypeAlias = Literal[
    "duplicate_chunk",
    "empty_index",
    "invalid_chunk",
    "invalid_query",
    "invalid_top_k",
]


class IndexRejected(ValueError):
    """Reject invalid index/search input with a stable code."""

    def __init__(self, code: IndexFailureCode) -> None:
        self.code = code
        super().__init__(code)


def bm25_configuration_fingerprint(
    *,
    algorithm: str,
    version: str,
    tokenizer_pattern: str,
    lowercase: bool,
    k1: float,
    b: float,
) -> str:
    return canonical_sha256(
        {
            "algorithm": algorithm,
            "b": b,
            "k1": k1,
            "lowercase": lowercase,
            "tokenizer_pattern": tokenizer_pattern,
            "version": version,
        }
    )


class BM25Configuration(ContractModel):
    algorithm: Literal["bm25_robertson"] = "bm25_robertson"
    version: Literal["v1"] = "v1"
    tokenizer_pattern: Literal[TOKEN_PATTERN] = TOKEN_PATTERN
    lowercase: Literal[True] = True
    k1: float = Field(default=1.5, gt=0, le=3)
    b: float = Field(default=0.75, ge=0, le=1)
    fingerprint: Sha256Digest

    @classmethod
    def create(cls, *, k1: float = 1.5, b: float = 0.75) -> BM25Configuration:
        values = {
            "algorithm": "bm25_robertson",
            "version": "v1",
            "tokenizer_pattern": TOKEN_PATTERN,
            "lowercase": True,
            "k1": k1,
            "b": b,
        }
        return cls(
            **values,
            fingerprint=bm25_configuration_fingerprint(**values),
        )

    @model_validator(mode="after")
    def validate_fingerprint(self) -> BM25Configuration:
        expected = bm25_configuration_fingerprint(
            algorithm=self.algorithm,
            version=self.version,
            tokenizer_pattern=self.tokenizer_pattern,
            lowercase=self.lowercase,
            k1=self.k1,
            b=self.b,
        )
        if self.fingerprint != expected:
            raise ValueError("BM25 configuration fingerprint does not match behavior")
        return self


def sparse_index_fingerprint(
    *,
    corpus_fingerprint: str,
    configuration_fingerprint: str,
    chunk_ids: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "chunk_ids": sorted(chunk_ids),
            "configuration_fingerprint": configuration_fingerprint,
            "corpus_fingerprint": corpus_fingerprint,
            "schema_version": 1,
        }
    )


class SparseIndexManifest(ContractModel):
    schema_version: Literal[1] = 1
    corpus_fingerprint: Sha256Digest
    configuration: BM25Configuration
    chunk_ids: tuple[Identifier, ...] = Field(min_length=1, max_length=100_000)
    index_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_identity(self) -> SparseIndexManifest:
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("index manifest chunk identities must be unique")
        expected = sparse_index_fingerprint(
            corpus_fingerprint=self.corpus_fingerprint,
            configuration_fingerprint=self.configuration.fingerprint,
            chunk_ids=self.chunk_ids,
        )
        if self.index_fingerprint != expected:
            raise ValueError("index fingerprint does not match manifest")
        return self


class RetrievalReportMatch(ContractModel):
    rank: int = Field(gt=0, le=5)
    chunk_id: Identifier
    document_id: Identifier
    asset_id: Identifier
    raw_score: float = Field(gt=0)
    canonical_relevance: float = Field(gt=0)
    score_kind: Literal["bm25"]
    score_conversion: Literal["identity"]
    source_url: HttpUrl
    locator: ShortText

    @model_validator(mode="after")
    def validate_scores(self) -> RetrievalReportMatch:
        if (
            not math.isfinite(self.raw_score)
            or not math.isfinite(self.canonical_relevance)
            or self.raw_score != self.canonical_relevance
        ):
            raise ValueError("BM25 report scores must be finite identity values")
        return self


class RetrievalReport(ContractModel):
    schema_version: Literal[1] = 1
    query_sha256: Sha256Digest
    index_fingerprint: Sha256Digest
    top_k: int = Field(ge=1, le=5)
    matches: tuple[RetrievalReportMatch, ...] = Field(max_length=5)

    @model_validator(mode="after")
    def validate_ranks(self) -> RetrievalReport:
        if [match.rank for match in self.matches] != list(range(1, len(self.matches) + 1)):
            raise ValueError("retrieval report ranks must be consecutive")
        return self


class InMemoryBM25Index:
    """A dependency-free sparse index with explicit score and tie semantics."""

    def __init__(
        self,
        *,
        chunks: Sequence[ReferenceChunk],
        corpus_fingerprint: Sha256Digest,
        configuration: BM25Configuration | None = None,
    ) -> None:
        if not chunks:
            raise IndexRejected("empty_index")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise IndexRejected("duplicate_chunk")
        self._configuration = configuration or BM25Configuration.create()
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._term_frequencies = {
            chunk.chunk_id: Counter(self._tokenize(chunk.text)) for chunk in chunks
        }
        if any(not frequencies for frequencies in self._term_frequencies.values()):
            raise IndexRejected("invalid_chunk")
        self._document_lengths = {
            chunk_id: sum(frequencies.values())
            for chunk_id, frequencies in self._term_frequencies.items()
        }
        self._average_document_length = sum(self._document_lengths.values()) / len(chunks)
        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies.values():
            document_frequencies.update(frequencies.keys())
        self._document_frequencies = document_frequencies
        self._manifest = SparseIndexManifest(
            corpus_fingerprint=corpus_fingerprint,
            configuration=self._configuration,
            chunk_ids=tuple(sorted(chunk_ids)),
            index_fingerprint=sparse_index_fingerprint(
                corpus_fingerprint=corpus_fingerprint,
                configuration_fingerprint=self._configuration.fingerprint,
                chunk_ids=chunk_ids,
            ),
        )
        descriptor_values = {
            "adapter_name": "in-memory-bm25",
            "adapter_version": "v1",
            "document_family": "text",
            "chunking_strategy": "source_section",
            "logical_strategy": "sparse",
            "corpus_fingerprint": corpus_fingerprint,
            "index_fingerprint": self._manifest.index_fingerprint,
            "score_kind": "bm25",
            "score_conversion": "identity",
            "evidence_level": "local_in_process",
        }
        self._descriptor = RetrievalAdapterDescriptor(
            **descriptor_values,
            descriptor_fingerprint=retrieval_adapter_fingerprint(**descriptor_values),
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: LoadedReferenceCorpus,
        *,
        configuration: BM25Configuration | None = None,
    ) -> InMemoryBM25Index:
        return cls(
            chunks=corpus.chunks,
            corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            configuration=configuration,
        )

    @property
    def manifest(self) -> SparseIndexManifest:
        return self._manifest

    @property
    def descriptor(self) -> RetrievalAdapterDescriptor:
        return self._descriptor

    @property
    def fingerprint(self) -> str:
        return self._manifest.index_fingerprint

    def _tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(match.group(0).lower() for match in re.finditer(TOKEN_PATTERN, text))

    def _score(self, chunk_id: str, query_terms: tuple[str, ...]) -> float:
        frequencies = self._term_frequencies[chunk_id]
        document_length = self._document_lengths[chunk_id]
        score = 0.0
        document_count = len(self._chunks)
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalization = self._configuration.k1 * (
                1
                - self._configuration.b
                + self._configuration.b * document_length / self._average_document_length
            )
            score += (
                inverse_document_frequency
                * (term_frequency * (self._configuration.k1 + 1))
                / (term_frequency + length_normalization)
            )
        return score

    def search(self, query: str, *, top_k: int = 5) -> tuple[ScoredChunk, ...]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise IndexRejected("invalid_top_k")
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise IndexRejected("invalid_query")
        normalized_query = query.strip()
        query_terms = tuple(sorted(set(self._tokenize(normalized_query))))
        if not query_terms:
            return ()
        ranked = [(self._score(chunk_id, query_terms), chunk_id) for chunk_id in self._chunks]
        ranked = [item for item in ranked if item[0] > 0]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        query_sha256 = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        return tuple(
            ScoredChunk(
                chunk=self._chunks[chunk_id],
                rank=rank,
                score_kind="bm25",
                raw_score=score,
                canonical_relevance=score,
                score_conversion="identity",
                higher_is_better=True,
                query_sha256=query_sha256,
                index_fingerprint=self.fingerprint,
            )
            for rank, (score, chunk_id) in enumerate(ranked[:top_k], start=1)
        )

    def report(self, query: str, *, top_k: int = 5) -> RetrievalReport:
        normalized_query = query.strip()
        results = self.search(normalized_query, top_k=top_k)
        return RetrievalReport(
            query_sha256=hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(),
            index_fingerprint=self.fingerprint,
            top_k=top_k,
            matches=tuple(
                RetrievalReportMatch(
                    rank=result.rank,
                    chunk_id=result.chunk.chunk_id,
                    document_id=result.chunk.document_id,
                    asset_id=result.chunk.asset_id,
                    raw_score=result.raw_score,
                    canonical_relevance=result.canonical_relevance,
                    score_kind="bm25",
                    score_conversion="identity",
                    source_url=result.chunk.source_span.source_url,
                    locator=result.chunk.source_span.locator,
                )
                for result in results
            ),
        )
