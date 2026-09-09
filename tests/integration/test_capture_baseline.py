from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from chest_xray_evidence_assistant.baselines import BaselineArtifact

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capture_baseline.py"


def sanitized_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CXR_LIVE_SMOKE",
        "CXR_MODEL",
        "CXR_PROVIDER",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_capture(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=sanitized_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_offline_capture_writes_a_repeatable_valid_artifact(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = run_capture("--output", str(first_path))
    second = run_capture("--output", str(second_path))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_summary = json.loads(first.stdout)
    assert first_summary["status"] == "succeeded"
    assert first_summary["mode"] == "offline_fake"
    assert first_summary["output"] == str(first_path)
    artifact = BaselineArtifact.model_validate_json(first_path.read_text())
    assert first_summary["configuration_sha256"] == artifact.configuration_sha256
    assert first_path.read_bytes() == second_path.read_bytes()


def test_live_capture_is_disabled_before_any_output_or_model_call(tmp_path: Path) -> None:
    output_path = tmp_path / "must-not-exist.json"

    result = run_capture("--live", "--output", str(output_path))

    assert result.returncode == 2
    assert json.loads(result.stdout) == {
        "status": "disabled",
        "error_code": "live_baseline_not_enabled",
    }
    assert not output_path.exists()


def test_output_path_failures_are_redacted(tmp_path: Path) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied")

    result = run_capture("--output", str(parent_file / "baseline.json"))

    assert result.returncode == 1
    assert json.loads(result.stdout) == {
        "status": "failed",
        "error_code": "baseline_output_error",
    }
    assert "Traceback" not in result.stderr
