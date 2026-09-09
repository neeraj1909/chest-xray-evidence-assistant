from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.retrieval import (
    InMemoryBM25Index,
    ReferenceRetrievalService,
    RetrievalAdapterDescriptor,
    RetrievalAdapterRejected,
    RetrievalIndexPort,
    load_reference_corpus,
    retrieval_adapter_fingerprint,
)
from chest_xray_evidence_assistant.tools import RetrieveReferenceArguments


class FakeDenseIndex:
    def __init__(
        self,
        descriptor: RetrievalAdapterDescriptor,
        *,
        results: object = (),
    ) -> None:
        self.descriptor = descriptor
        self.results = results
        self.search_calls = 0

    def search(self, query: str, *, top_k: int = 5) -> object:
        del query, top_k
        self.search_calls += 1
        return self.results


def descriptor(
    *,
    corpus_fingerprint: str = "a" * 64,
    index_fingerprint: str = "b" * 64,
    strategy: str = "dense",
) -> RetrievalAdapterDescriptor:
    values = {
        "adapter_name": "fake-dense-index",
        "adapter_version": "v1",
        "document_family": "text",
        "chunking_strategy": "source_section",
        "logical_strategy": strategy,
        "corpus_fingerprint": corpus_fingerprint,
        "index_fingerprint": index_fingerprint,
        "score_kind": "cosine_similarity",
        "score_conversion": "identity",
        "evidence_level": "pure_fake",
    }
    return RetrievalAdapterDescriptor(
        **values,
        descriptor_fingerprint=retrieval_adapter_fingerprint(**values),
    )


def test_bm25_index_satisfies_the_provider_neutral_read_only_port() -> None:
    corpus = load_reference_corpus()
    lexical_index = InMemoryBM25Index.from_corpus(corpus)

    assert isinstance(lexical_index, RetrievalIndexPort)
    assert lexical_index.descriptor.logical_strategy == "sparse"
    assert lexical_index.descriptor.corpus_fingerprint == corpus.manifest.corpus_fingerprint
    assert lexical_index.descriptor.score_kind == "bm25"
    assert lexical_index.descriptor.evidence_level == "local_in_process"


def test_fake_dense_adapter_can_compose_without_a_vector_dependency() -> None:
    adapter = FakeDenseIndex(descriptor())
    service = ReferenceRetrievalService(
        adapter,
        expected_corpus_fingerprint="a" * 64,
        expected_strategy="dense",
    )

    results = asyncio.run(
        service.retrieve_reference(RetrieveReferenceArguments(query="projection", top_k=2))
    )

    assert results == ()
    assert adapter.search_calls == 1


@pytest.mark.parametrize(
    ("expected_corpus", "expected_strategy"),
    [("c" * 64, "dense"), ("a" * 64, "hybrid")],
)
def test_incompatible_adapter_fails_before_search(
    expected_corpus: str,
    expected_strategy: str,
) -> None:
    adapter = FakeDenseIndex(descriptor())

    with pytest.raises(RetrievalAdapterRejected, match="^incompatible_manifest$"):
        ReferenceRetrievalService(
            adapter,
            expected_corpus_fingerprint=expected_corpus,
            expected_strategy=expected_strategy,
        )

    assert adapter.search_calls == 0


def test_descriptor_rejects_a_stale_behavior_fingerprint() -> None:
    payload = descriptor().model_dump()
    payload["adapter_version"] = "v2"

    with pytest.raises(ValidationError, match="descriptor fingerprint"):
        RetrievalAdapterDescriptor.model_validate(payload)


def test_adapter_results_fail_closed_when_the_provenance_contract_breaks() -> None:
    corpus = load_reference_corpus()
    lexical_index = InMemoryBM25Index.from_corpus(corpus)
    query = "radiography image"
    valid = lexical_index.search(query, top_k=1)[0]
    adapter_descriptor = descriptor(
        corpus_fingerprint=corpus.manifest.corpus_fingerprint,
        index_fingerprint=lexical_index.fingerprint,
    )
    invalid_results = (
        (valid,),
        [valid],
        (object(),),
        (valid, valid.model_copy(update={"rank": 2})),
        (valid.model_copy(update={"rank": 2}),),
        (valid.model_copy(update={"query_sha256": "c" * 64}),),
        (valid.model_copy(update={"index_fingerprint": "d" * 64}),),
        (
            valid.model_copy(
                update={
                    "score_kind": "cosine_similarity",
                    "score_conversion": "negated_distance",
                }
            ),
        ),
    )

    for results in invalid_results:
        adapter = FakeDenseIndex(adapter_descriptor, results=results)
        service = ReferenceRetrievalService(
            adapter,
            expected_corpus_fingerprint=corpus.manifest.corpus_fingerprint,
            expected_strategy="dense",
        )

        with pytest.raises(RetrievalAdapterRejected, match="^invalid_result$"):
            asyncio.run(
                service.retrieve_reference(
                    RetrieveReferenceArguments(query=query, top_k=1),
                )
            )

        assert adapter.search_calls == 1
