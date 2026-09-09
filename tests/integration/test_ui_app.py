from __future__ import annotations

import json

import pytest

pytest.importorskip("gradio")

from chest_xray_evidence_assistant.ui import DEMO_QUESTION, build_app


def test_gradio_app_exposes_only_the_bounded_offline_journey() -> None:
    config = build_app().get_config_file()
    serialized = json.dumps(config, sort_keys=True)
    components = config["components"]
    labels = {
        component.get("props", {}).get("label"): component
        for component in components
        if component.get("props", {}).get("label")
    }

    upload = labels["Synthetic PNG"]
    assert upload["type"] == "file"
    assert upload["props"]["type"] == "binary"
    assert upload["props"]["file_count"] == "single"
    assert upload["props"]["file_types"] == [".png"]
    assert labels["Question"]["props"]["value"] == DEMO_QUESTION
    assert labels["Question"]["props"]["max_length"] == 4_000
    assert labels["Additional context (optional)"]["props"]["max_length"] == 8_000
    assert "Not for diagnosis, treatment, triage, or emergencies" in serialized
    assert "Do not upload patient" in serialized

    assert len(config["dependencies"]) == 1
    submission = config["dependencies"][0]
    assert submission["api_visibility"] == "private"
    assert submission["trigger_mode"] == "once"
    assert submission["show_progress"] == "minimal"
    assert len(submission["inputs"]) == 4
    assert len(submission["outputs"]) == 10
    assert "OPENAI_API_KEY" not in serialized
    assert "CXR_MODEL" not in serialized
    assert "provider configuration" not in serialized.lower()
