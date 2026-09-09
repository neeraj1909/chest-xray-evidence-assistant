from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from chest_xray_evidence_assistant.baselines import load_baseline_case
from chest_xray_evidence_assistant.models import VisualResponse
from chest_xray_evidence_assistant.providers import LiveModelConfig

RESPONSE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "responses"
LIVE_SMOKE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "live_smoke.py"
LIVE_SMOKE_SPEC = spec_from_file_location("cxr_live_smoke", LIVE_SMOKE_PATH)
assert LIVE_SMOKE_SPEC is not None and LIVE_SMOKE_SPEC.loader is not None
live_smoke = module_from_spec(LIVE_SMOKE_SPEC)
LIVE_SMOKE_SPEC.loader.exec_module(live_smoke)


@pytest.fixture(autouse=True)
def clear_live_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CXR_LIVE_SMOKE",
        "CXR_MODEL",
        "CXR_PROVIDER",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def output_payload(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_smoke_is_disabled_by_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert live_smoke.main() == 2
    assert output_payload(capsys) == {
        "status": "disabled",
        "error_code": "live_smoke_not_enabled",
    }


def test_live_smoke_uses_the_same_request_as_the_one_shot_baseline() -> None:
    baseline_request, baseline_image = load_baseline_case()
    smoke_request, smoke_image = live_smoke.load_smoke_case()

    assert smoke_request == baseline_request
    assert smoke_image == baseline_image


def test_enabled_smoke_with_missing_credentials_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CXR_LIVE_SMOKE", "1")
    monkeypatch.setenv("CXR_PROVIDER", "openai")
    monkeypatch.setenv("CXR_MODEL", "gpt-4.1-mini")

    assert live_smoke.main() == 2
    assert output_payload(capsys) == {
        "status": "failed",
        "error_code": "live_config_invalid",
    }


def test_smoke_orchestrates_one_image_request_without_leaking_payloads(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = LiveModelConfig(
        provider="ollama",
        model="fixture-model",
        base_url="http://localhost:11434/v1",
    )
    response = VisualResponse.model_validate_json((RESPONSE_FIXTURES / "answered.json").read_text())
    seen: dict[str, object] = {}

    monkeypatch.setenv("CXR_LIVE_SMOKE", "1")
    monkeypatch.setattr(
        live_smoke.LiveModelConfig,
        "from_env",
        classmethod(lambda cls: config),
    )
    monkeypatch.setattr(live_smoke, "build_live_model", lambda value: "model")

    async def fake_run(request: object, image_bytes: bytes, *, model: object) -> VisualResponse:
        seen.update(request=request, image_bytes=image_bytes, model=model)
        return response

    monkeypatch.setattr(live_smoke, "run_evidence_request", fake_run)

    assert live_smoke.main() == 0
    assert seen["model"] == "model"
    assert isinstance(seen["image_bytes"], bytes)
    assert output_payload(capsys) == {
        "confidence": 0.8,
        "model": "fixture-model",
        "provider": "ollama",
        "source_evidence_count": 0,
        "status": "answered",
        "trace_events": 1,
        "visual_evidence_count": 1,
    }


def test_smoke_runtime_errors_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = LiveModelConfig(
        provider="ollama",
        model="fixture-model",
        base_url="http://localhost:11434/v1",
    )
    monkeypatch.setenv("CXR_LIVE_SMOKE", "1")
    monkeypatch.setattr(
        live_smoke.LiveModelConfig,
        "from_env",
        classmethod(lambda cls: config),
    )
    monkeypatch.setattr(live_smoke, "build_live_model", lambda value: "model")

    async def fail(*args: object, **kwargs: object) -> VisualResponse:
        raise RuntimeError("secret payload must not be printed")

    monkeypatch.setattr(live_smoke, "run_evidence_request", fail)

    assert live_smoke.main() == 1
    payload = output_payload(capsys)
    assert payload == {"status": "failed", "error_code": "live_RuntimeError"}
    assert "secret payload" not in json.dumps(payload)
