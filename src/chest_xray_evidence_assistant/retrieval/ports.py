"""Provider-neutral, read-only retrieval adapter contracts."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import model_validator

from ..models import ContractModel, Identifier, Sha256Digest
from ._canonical import canonical_sha256
from .records import ScoreConversion, ScoredChunk, ScoreKind

LogicalIndexStrategy: TypeAlias = Literal["sparse", "dense", "hybrid"]
ProviderEvidenceLevel: TypeAlias = Literal[
    "pure_fake",
    "local_in_process",
    "disposable_service",
    "opt_in_live",
]


def retrieval_adapter_fingerprint(
    *,
    adapter_name: str,
    adapter_version: str,
    document_family: str,
    chunking_strategy: str,
    logical_strategy: str,
    corpus_fingerprint: str,
    index_fingerprint: str,
    score_kind: str,
    score_conversion: str,
    evidence_level: str,
) -> str:
    return canonical_sha256(
        {
            "adapter_name": adapter_name,
            "adapter_version": adapter_version,
            "chunking_strategy": chunking_strategy,
            "corpus_fingerprint": corpus_fingerprint,
            "document_family": document_family,
            "evidence_level": evidence_level,
            "index_fingerprint": index_fingerprint,
            "logical_strategy": logical_strategy,
            "score_conversion": score_conversion,
            "score_kind": score_kind,
        },
    )


class RetrievalAdapterDescriptor(ContractModel):
    """Compatibility and evidence claims for one composed retrieval adapter."""

    adapter_name: Identifier
    adapter_version: Identifier
    document_family: Literal["text"]
    chunking_strategy: Literal["source_section"]
    logical_strategy: LogicalIndexStrategy
    corpus_fingerprint: Sha256Digest
    index_fingerprint: Sha256Digest
    score_kind: ScoreKind
    score_conversion: ScoreConversion
    evidence_level: ProviderEvidenceLevel
    descriptor_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_fingerprint(self) -> RetrievalAdapterDescriptor:
        expected = retrieval_adapter_fingerprint(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            document_family=self.document_family,
            chunking_strategy=self.chunking_strategy,
            logical_strategy=self.logical_strategy,
            corpus_fingerprint=self.corpus_fingerprint,
            index_fingerprint=self.index_fingerprint,
            score_kind=self.score_kind,
            score_conversion=self.score_conversion,
            evidence_level=self.evidence_level,
        )
        if self.descriptor_fingerprint != expected:
            raise ValueError("retrieval descriptor fingerprint does not match behavior")
        return self


@runtime_checkable
class RetrievalIndexPort(Protocol):
    """Read-only candidate search after normalization and composition preflight."""

    @property
    def descriptor(self) -> RetrievalAdapterDescriptor: ...

    def search(self, query: str, *, top_k: int = 5) -> tuple[ScoredChunk, ...]: ...
