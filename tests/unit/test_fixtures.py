import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.fixtures import (
    FixtureManifest,
    load_fixture_manifest,
)
from chest_xray_evidence_assistant.models import VisualResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "fixtures" / "manifest.json"
RESPONSE_ROOT = REPO_ROOT / "tests" / "fixtures" / "responses"


def _manifest_data() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_matches_fixture_files() -> None:
    manifest = load_fixture_manifest()

    assert len(manifest.fixtures) == 3
    assert {"full_frame", "crop", "abstention"} <= {
        use for fixture in manifest.fixtures for use in fixture.intended_use
    }
    assert all(fixture.asset.contains_phi is False for fixture in manifest.fixtures)


def test_manifest_requires_asset_provenance() -> None:
    raw = _manifest_data()
    del raw["fixtures"][0]["asset"]["license_status"]  # type: ignore[index]

    with pytest.raises(ValidationError):
        FixtureManifest.model_validate(raw)


def test_manifest_rejects_oversized_assets() -> None:
    raw = _manifest_data()
    raw["fixtures"][0]["asset"]["byte_size"] = 20_000_001  # type: ignore[index]

    with pytest.raises(ValidationError):
        FixtureManifest.model_validate(raw)


@pytest.mark.parametrize(
    "filename",
    ["answered.json", "needs-clarification.json", "abstain.json"],
)
def test_deterministic_response_fixture_is_valid(filename: str) -> None:
    path = RESPONSE_ROOT / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = VisualResponse.model_validate(payload)

    assert response.model_dump(mode="json") == VisualResponse.model_validate(
        response.model_dump(mode="json")
    ).model_dump(mode="json")


def test_malformed_response_fixture_is_rejected() -> None:
    path = RESPONSE_ROOT / "malformed.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        VisualResponse.model_validate(payload)
