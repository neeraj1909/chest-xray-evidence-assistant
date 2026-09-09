"""Self-validating, non-secret identity manifests for evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from ..models import ContractModel, Identifier, Sha256Digest, ShortText
from .grading import canonical_sha256
from .offline import CONFIGURATION_IDS, AblationConfiguration, ConfigurationId


def _provider_fingerprint(*, provider: str, runner_mode: str) -> str:
    return canonical_sha256({"provider": provider, "runner_mode": runner_mode})


def _model_fingerprint(*, model: str, policy_version: str) -> str:
    return canonical_sha256({"model": model, "policy_version": policy_version})


class EvaluationArtifactManifest(ContractModel):
    """Hashes of every behavior-affecting identity needed to compare one run."""

    schema_version: Literal[1] = 1
    configuration_id: ConfigurationId
    runner_mode: Literal["offline_scripted"]
    provider: ShortText
    model: ShortText
    policy_version: Identifier
    dataset_id: Identifier
    dataset_version: Identifier
    provider_sha256: Sha256Digest
    model_sha256: Sha256Digest
    prompt_sha256: Sha256Digest
    corpus_sha256: Sha256Digest
    configuration_sha256: Sha256Digest
    dataset_sha256: Sha256Digest
    case_set_sha256: Sha256Digest
    seed_set_sha256: Sha256Digest
    limits_sha256: Sha256Digest
    rubric_sha256: Sha256Digest
    response_schema_sha256: Sha256Digest
    retrieval_index_sha256: Sha256Digest
    framework_versions: dict[Identifier, ShortText] = Field(min_length=2, max_length=8)
    manifest_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_internal_fingerprints(self) -> EvaluationArtifactManifest:
        if self.provider_sha256 != _provider_fingerprint(
            provider=self.provider,
            runner_mode=self.runner_mode,
        ):
            raise ValueError("provider fingerprint does not match manifest identity")
        if self.model_sha256 != _model_fingerprint(
            model=self.model,
            policy_version=self.policy_version,
        ):
            raise ValueError("model fingerprint does not match manifest identity")
        if set(self.framework_versions) != {"pydantic-evals", "ragas"}:
            raise ValueError("evaluation framework versions are incomplete")
        if self.manifest_sha256 != evaluation_manifest_fingerprint(self):
            raise ValueError("evaluation manifest fingerprint does not match")
        return self

    def matches_configuration(self, configuration: AblationConfiguration) -> bool:
        expected = build_evaluation_manifest(
            configuration,
            framework_versions=self.framework_versions,
        )
        return self == expected


def evaluation_manifest_fingerprint(
    manifest: EvaluationArtifactManifest | Mapping[str, object],
) -> str:
    if isinstance(manifest, EvaluationArtifactManifest):
        payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    else:
        payload = dict(manifest)
        payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


def build_evaluation_manifest(
    configuration: AblationConfiguration,
    *,
    framework_versions: Mapping[str, str],
) -> EvaluationArtifactManifest:
    payload = {
        "schema_version": 1,
        "configuration_id": configuration.configuration_id,
        "runner_mode": configuration.runner_mode,
        "provider": configuration.provider,
        "model": configuration.model,
        "policy_version": configuration.policy_version,
        "dataset_id": configuration.dataset_id,
        "dataset_version": configuration.dataset_version,
        "provider_sha256": _provider_fingerprint(
            provider=configuration.provider,
            runner_mode=configuration.runner_mode,
        ),
        "model_sha256": _model_fingerprint(
            model=configuration.model,
            policy_version=configuration.policy_version,
        ),
        "prompt_sha256": configuration.prompt_set_sha256,
        "corpus_sha256": configuration.corpus_sha256,
        "configuration_sha256": configuration.configuration_sha256,
        "dataset_sha256": configuration.dataset_sha256,
        "case_set_sha256": configuration.case_set_sha256,
        "seed_set_sha256": configuration.seed_set_sha256,
        "limits_sha256": configuration.limits_sha256,
        "rubric_sha256": configuration.rubric_sha256,
        "response_schema_sha256": configuration.response_schema_sha256,
        "retrieval_index_sha256": configuration.retrieval_index_sha256,
        "framework_versions": dict(framework_versions),
    }
    return EvaluationArtifactManifest(
        **payload,
        manifest_sha256=evaluation_manifest_fingerprint(payload),
    )


class EvaluationManifestIndex(ContractModel):
    """Fixed five-configuration manifest written beside the report bundle."""

    schema_version: Literal[1] = 1
    dataset_id: Identifier
    dataset_version: Identifier
    manifests: tuple[EvaluationArtifactManifest, ...]
    index_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_index(self) -> EvaluationManifestIndex:
        if tuple(manifest.configuration_id for manifest in self.manifests) != CONFIGURATION_IDS:
            raise ValueError("evaluation manifests must use the fixed configuration order")
        if any(
            manifest.dataset_id != self.dataset_id
            or manifest.dataset_version != self.dataset_version
            for manifest in self.manifests
        ):
            raise ValueError("evaluation manifest index mixes datasets")
        framework_sets = {
            canonical_sha256(manifest.framework_versions) for manifest in self.manifests
        }
        if len(framework_sets) != 1:
            raise ValueError("evaluation manifest index mixes framework versions")
        if self.index_sha256 != evaluation_manifest_index_fingerprint(self):
            raise ValueError("evaluation manifest index fingerprint does not match")
        return self

    @classmethod
    def build(
        cls,
        *,
        dataset_id: str,
        dataset_version: str,
        manifests: tuple[EvaluationArtifactManifest, ...],
    ) -> EvaluationManifestIndex:
        payload = {
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "manifests": manifests,
            "schema_version": 1,
        }
        return cls(
            **payload,
            index_sha256=evaluation_manifest_index_fingerprint(payload),
        )


def evaluation_manifest_index_fingerprint(
    index: EvaluationManifestIndex | Mapping[str, object],
) -> str:
    if isinstance(index, EvaluationManifestIndex):
        payload = index.model_dump(mode="json", exclude={"index_sha256"})
    else:
        payload = dict(index)
        payload.pop("index_sha256", None)
    return canonical_sha256(payload)
