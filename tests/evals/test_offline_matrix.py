from __future__ import annotations

from pathlib import Path

from chest_xray_evidence_assistant.evals.datasets import load_benchmark
from chest_xray_evidence_assistant.evals.grading import grade_run
from chest_xray_evidence_assistant.evals.offline import (
    CONFIGURATION_IDS,
    OfflineAblationTask,
    build_ablation_configurations,
)
from chest_xray_evidence_assistant.evals.pydantic_adapter import build_pydantic_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = load_benchmark(REPO_ROOT / "data" / "benchmark" / "manifest.json")


def _evaluation_case(category: str):
    dataset = build_pydantic_dataset(BENCHMARK)
    return next(case for case in dataset.cases if case.metadata.category == category)


def test_matrix_has_the_five_fixed_capability_configurations() -> None:
    configurations = build_ablation_configurations(BENCHMARK)

    assert tuple(configuration.configuration_id for configuration in configurations) == (
        "text_only",
        "one_shot_vlm",
        "vlm_tools",
        "vlm_rag",
        "full_agent",
    )
    assert tuple(CONFIGURATION_IDS) == tuple(
        configuration.configuration_id for configuration in configurations
    )
    assert [configuration.image_access for configuration in configurations] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert [configuration.metadata_tool for configuration in configurations] == [
        False,
        False,
        True,
        False,
        True,
    ]
    assert [configuration.retrieval_tool for configuration in configurations] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_full_policy_succeeds_without_receiving_evaluator_labels() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    task = OfflineAblationTask(BENCHMARK, configuration)

    for category in (
        "observation",
        "tool_use",
        "retrieval",
        "contradiction",
        "abstention",
    ):
        evaluation_case = _evaluation_case(category)
        assert not hasattr(evaluation_case.inputs, "expected")
        assert not hasattr(evaluation_case.inputs, "category")
        observation = task(evaluation_case.inputs)
        grade = grade_run(
            evaluation_case.metadata.benchmark_case(evaluation_case.inputs),
            observation,
        )
        assert grade.task_success is True


def test_one_shot_lacks_tool_and_retrieval_capabilities_but_falls_back_safely() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[1]
    task = OfflineAblationTask(BENCHMARK, configuration)

    results = {}
    for category in ("observation", "tool_use", "retrieval", "abstention"):
        evaluation_case = _evaluation_case(category)
        observation = task(evaluation_case.inputs)
        results[category] = grade_run(
            evaluation_case.metadata.benchmark_case(evaluation_case.inputs),
            observation,
        )

    assert results["observation"].task_success is True
    assert results["tool_use"].task_success is False
    assert results["retrieval"].task_success is False
    assert results["abstention"].task_success is True
    assert all(result.safety_passed for result in results.values())
    assert all(result.unauthorized_tool_calls == 0 for result in results.values())


def test_retrieval_policy_uses_the_real_frozen_index_and_expected_provenance() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    task = OfflineAblationTask(BENCHMARK, configuration)
    evaluation_case = _evaluation_case("retrieval")

    observation = task(evaluation_case.inputs)
    grade = grade_run(
        evaluation_case.metadata.benchmark_case(evaluation_case.inputs),
        observation,
    )

    assert grade.citations_correct is True
    assert grade.tool_selection_correct is True
    assert grade.tool_arguments_correct is True
    assert observation.tool_calls[0].name == "retrieve_reference"
    assert observation.response_payload["source_evidence"][0]["document_id"] in {
        source.document.document_id for source in task.corpus.manifest.sources
    }


def test_offline_policy_is_byte_repeatable_for_the_same_case_and_seed() -> None:
    configuration = build_ablation_configurations(BENCHMARK)[-1]
    task = OfflineAblationTask(BENCHMARK, configuration)
    evaluation_case = _evaluation_case("tool_use")

    first = task(evaluation_case.inputs).model_dump_json()
    second = task(evaluation_case.inputs).model_dump_json()

    assert first == second
