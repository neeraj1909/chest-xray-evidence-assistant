from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.manifest import (
    EvaluationManifestIndex,
    build_evaluation_manifest,
)
from chest_xray_evidence_assistant.evals.offline import build_ablation_configurations
from chest_xray_evidence_assistant.evals.reporting import (
    ConfigurationReport,
    run_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")


def test_manifest_binds_non_secret_provider_model_prompt_corpus_and_config() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    manifest = build_evaluation_manifest(
        configuration,
        framework_versions={"pydantic-evals": "2.36.0", "ragas": "0.3.9"},
    )

    assert manifest.configuration_id == "full_agent"
    assert manifest.provider == "offline-scripted"
    assert manifest.model == "deterministic-capability-policy"
    assert manifest.prompt_sha256 == configuration.prompt_set_sha256
    assert manifest.corpus_sha256 == configuration.corpus_sha256
    assert manifest.configuration_sha256 == configuration.configuration_sha256
    assert len(manifest.provider_sha256) == 64
    assert len(manifest.model_sha256) == 64
    assert len(manifest.manifest_sha256) == 64
    serialized = manifest.model_dump_json().casefold()
    assert "api_key" not in serialized
    assert "authorization" not in serialized
    assert "raw_prompt" not in serialized


def test_configuration_report_rejects_a_mixed_prompt_manifest() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    payload = run_configuration(BENCHMARK, configuration, case_limit=1).model_dump(mode="json")
    payload["artifact_manifest"]["prompt_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="manifest fingerprint does not match"):
        ConfigurationReport.model_validate(payload)


def test_manifest_index_rejects_duplicate_or_mixed_configurations() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[0]
    manifest = build_evaluation_manifest(
        configuration,
        framework_versions={"pydantic-evals": "2.36.0", "ragas": "0.3.9"},
    )

    with pytest.raises(ValidationError, match="fixed configuration order"):
        EvaluationManifestIndex.build(
            dataset_id=BENCHMARK.manifest.dataset_id,
            dataset_version=BENCHMARK.manifest.dataset_version,
            manifests=(manifest, manifest),
        )
