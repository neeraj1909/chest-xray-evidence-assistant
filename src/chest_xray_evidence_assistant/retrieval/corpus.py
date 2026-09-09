"""Manifest-first loading for the small, inspectable reference corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, StringConstraints, field_validator, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from ._canonical import canonical_sha256
from .records import (
    ReferenceAsset,
    ReferenceChunk,
    ReferenceDocument,
    TextSourceSpan,
    chunk_id_for,
)

CorpusPath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
CorpusFailureCode: TypeAlias = Literal[
    "content_mismatch",
    "content_too_large",
    "invalid_content",
    "invalid_manifest",
    "untracked_source",
]


class CorpusRejected(ValueError):
    """Reject corpus input with a stable code and no file-content echo."""

    def __init__(self, code: CorpusFailureCode) -> None:
        self.code = code
        super().__init__(code)


def extractor_fingerprint_for(*, name: str, version: str, method: str) -> str:
    return canonical_sha256({"method": method, "name": name, "version": version})


def chunking_fingerprint_for(
    *,
    strategy: str,
    locator_type: str,
    max_chars: int,
) -> str:
    return canonical_sha256(
        {
            "locator_type": locator_type,
            "max_chars": max_chars,
            "strategy": strategy,
        }
    )


class ExtractorConfiguration(ContractModel):
    name: Identifier
    version: Identifier
    method: Literal["manual_curated_markdown"]
    fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_fingerprint(self) -> ExtractorConfiguration:
        expected = extractor_fingerprint_for(
            name=self.name,
            version=self.version,
            method=self.method,
        )
        if self.fingerprint != expected:
            raise ValueError("extractor fingerprint does not match configuration")
        return self


class ChunkingConfiguration(ContractModel):
    strategy: Literal["source_section"]
    locator_type: Literal["character_span"]
    max_chars: int = Field(gt=0, le=16_000)
    fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_fingerprint(self) -> ChunkingConfiguration:
        expected = chunking_fingerprint_for(
            strategy=self.strategy,
            locator_type=self.locator_type,
            max_chars=self.max_chars,
        )
        if self.fingerprint != expected:
            raise ValueError("chunking fingerprint does not match configuration")
        return self


class CorpusSource(ContractModel):
    path: CorpusPath
    section: ShortText
    source_locator: ShortText
    transformation: Literal["manual_paraphrase"]
    asset: ReferenceAsset
    document: ReferenceDocument

    @field_validator("path")
    @classmethod
    def require_document_path(cls, value: str) -> str:
        path = Path(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] != "documents"
            or path.suffix not in {".md", ".txt"}
        ):
            raise ValueError("corpus paths must name one relative document file")
        return value

    @model_validator(mode="after")
    def validate_relationships(self) -> CorpusSource:
        if self.document.asset_id != self.asset.asset_id:
            raise ValueError("document must reference its source asset")
        if self.document.title != self.asset.title:
            raise ValueError("document and asset titles must match")
        return self


class ReferenceCorpusManifest(ContractModel):
    schema_version: Literal[1]
    corpus_id: Identifier
    corpus_version: Identifier
    document_family: Literal["text"]
    logical_index: Literal["sparse"]
    extractor: ExtractorConfiguration
    chunking: ChunkingConfiguration
    sources: tuple[CorpusSource, ...] = Field(min_length=1, max_length=16)
    corpus_fingerprint: Sha256Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> ReferenceCorpusManifest:
        paths = [source.path for source in self.sources]
        asset_ids = [source.asset.asset_id for source in self.sources]
        document_ids = [source.document.document_id for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("corpus source paths must be unique")
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("corpus asset identities must be unique")
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("corpus document identities must be unique")
        if any(
            source.document.extractor_name != self.extractor.name
            or source.document.extractor_version != self.extractor.version
            or source.document.extractor_fingerprint != self.extractor.fingerprint
            for source in self.sources
        ):
            raise ValueError("source document uses an incompatible extractor")
        if self.corpus_fingerprint != corpus_fingerprint_for(self):
            raise ValueError("corpus fingerprint does not match manifest")
        return self


def corpus_fingerprint_for(manifest: ReferenceCorpusManifest | Mapping[str, object]) -> str:
    """Fingerprint every behavior- and content-affecting manifest field."""

    if isinstance(manifest, ReferenceCorpusManifest):
        payload = manifest.model_dump(mode="json", exclude={"corpus_fingerprint"})
    else:
        payload = dict(manifest)
        payload.pop("corpus_fingerprint", None)
    return canonical_sha256(payload)


class LoadedReferenceCorpus(ContractModel):
    manifest: ReferenceCorpusManifest
    chunks: tuple[ReferenceChunk, ...] = Field(min_length=1, max_length=16)


def _read_manifest(path: Path) -> ReferenceCorpusManifest:
    try:
        if path.stat().st_size > 256_000:
            raise CorpusRejected("invalid_manifest")
        with path.open("rb") as manifest_file:
            content = manifest_file.read(256_001)
        raw = json.loads(content.decode("utf-8"))
        return ReferenceCorpusManifest.model_validate(raw)
    except CorpusRejected:
        raise
    except (OSError, UnicodeError, ValueError):
        raise CorpusRejected("invalid_manifest") from None


def _verify_tracked_files(root: Path, manifest: ReferenceCorpusManifest) -> None:
    document_root = root / "documents"
    try:
        actual_paths: set[str] = set()
        for path in document_root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise CorpusRejected("untracked_source")
            actual_paths.add(path.relative_to(root).as_posix())
    except CorpusRejected:
        raise
    except OSError:
        raise CorpusRejected("invalid_manifest") from None
    tracked_paths = {source.path for source in manifest.sources}
    if actual_paths != tracked_paths:
        raise CorpusRejected("untracked_source")


def _read_source(root: Path, source: CorpusSource) -> str:
    candidate = (root / source.path).resolve()
    try:
        candidate.relative_to(root)
        if candidate.stat().st_size != source.asset.byte_size:
            raise CorpusRejected("content_mismatch")
        with candidate.open("rb") as source_file:
            content = source_file.read(source.asset.byte_size + 1)
    except CorpusRejected:
        raise
    except (OSError, ValueError):
        raise CorpusRejected("content_mismatch") from None
    if (
        len(content) != source.asset.byte_size
        or hashlib.sha256(content).hexdigest() != source.asset.content_sha256
        or hashlib.sha256(content).hexdigest() != source.document.rendition_sha256
    ):
        raise CorpusRejected("content_mismatch")
    try:
        text = content.decode("utf-8")
    except UnicodeError:
        raise CorpusRejected("invalid_content") from None
    if not text.strip():
        raise CorpusRejected("invalid_content")
    return text


def load_reference_corpus(path: Path | None = None) -> LoadedReferenceCorpus:
    """Preflight a complete manifest, then materialize one chunk per source section."""

    manifest_path = path or (
        Path(__file__).resolve().parents[3] / "data" / "references" / "manifest.json"
    )
    manifest = _read_manifest(manifest_path)
    root = manifest_path.parent.resolve()
    _verify_tracked_files(root, manifest)

    chunks: list[ReferenceChunk] = []
    for source in manifest.sources:
        text = _read_source(root, source)
        if len(text) > manifest.chunking.max_chars:
            raise CorpusRejected("content_too_large")
        source_span = TextSourceSpan(
            document_id=source.document.document_id,
            section=source.section,
            locator=source.source_locator,
            start_char=0,
            end_char=len(text),
            source_url=source.asset.source_url,
        )
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunk_payload = {
            "document_id": source.document.document_id,
            "asset_id": source.asset.asset_id,
            "ordinal": 0,
            "text": text,
            "text_sha256": text_sha256,
            "rendition_sha256": source.document.rendition_sha256,
            "extractor_fingerprint": source.document.extractor_fingerprint,
            "source_span": source_span,
        }
        chunks.append(
            ReferenceChunk(
                chunk_id=chunk_id_for(**chunk_payload),
                **chunk_payload,
            )
        )
    return LoadedReferenceCorpus(manifest=manifest, chunks=tuple(chunks))
