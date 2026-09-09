"""Application service for the initial sparse reference retriever."""

from __future__ import annotations

import hashlib
from typing import Literal, TypeAlias

from ..tools.contracts import RetrieveReferenceArguments
from .index import InMemoryBM25Index
from .ports import LogicalIndexStrategy, RetrievalIndexPort
from .records import ScoredChunk

RetrievalAdapterFailureCode: TypeAlias = Literal[
    "incompatible_manifest",
    "invalid_result",
]


class RetrievalAdapterRejected(ValueError):
    def __init__(self, code: RetrievalAdapterFailureCode) -> None:
        self.code = code
        super().__init__(code)


class ReferenceRetrievalService:
    """Compose one read-only adapter only after compatibility preflight."""

    def __init__(
        self,
        index: RetrievalIndexPort,
        *,
        expected_corpus_fingerprint: str,
        expected_strategy: LogicalIndexStrategy,
    ) -> None:
        descriptor = index.descriptor
        if (
            descriptor.corpus_fingerprint != expected_corpus_fingerprint
            or descriptor.logical_strategy != expected_strategy
        ):
            raise RetrievalAdapterRejected("incompatible_manifest")
        self._index = index
        self._descriptor = descriptor

    @property
    def index_fingerprint(self) -> str:
        return self._descriptor.index_fingerprint

    async def retrieve_reference(
        self,
        arguments: RetrieveReferenceArguments,
    ) -> tuple[ScoredChunk, ...]:
        results = self._index.search(arguments.query, top_k=arguments.top_k)
        expected_query_sha256 = hashlib.sha256(arguments.query.encode("utf-8")).hexdigest()
        if (
            not isinstance(results, tuple)
            or len(results) > arguments.top_k
            or any(not isinstance(result, ScoredChunk) for result in results)
            or any(result.index_fingerprint != self.index_fingerprint for result in results)
            or any(result.query_sha256 != expected_query_sha256 for result in results)
            or any(result.score_kind != self._descriptor.score_kind for result in results)
            or any(
                result.score_conversion != self._descriptor.score_conversion for result in results
            )
            or len({result.chunk.chunk_id for result in results}) != len(results)
            or [result.rank for result in results] != list(range(1, len(results) + 1))
        ):
            raise RetrievalAdapterRejected("invalid_result")
        return results


class LexicalReferenceService(ReferenceRetrievalService):
    """Expose the in-memory index through the bounded retrieval-tool port."""

    def __init__(self, index: InMemoryBM25Index) -> None:
        super().__init__(
            index,
            expected_corpus_fingerprint=index.manifest.corpus_fingerprint,
            expected_strategy="sparse",
        )
