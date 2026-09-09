import json
import shutil
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


def test_size_mismatch_is_rejected_before_fixture_content_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(MANIFEST_PATH.parent, fixture_root)
    candidate = fixture_root / "images" / "synthetic-full-frame.png"
    candidate.write_bytes(candidate.read_bytes() + b"unexpected")
    original_read_bytes = Path.read_bytes

    def reject_candidate_read(path: Path) -> bytes:
        if path == candidate:
            raise AssertionError("mismatched fixture content must not be read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_candidate_read)

    with pytest.raises(ValueError, match="byte size mismatch"):
        load_fixture_manifest(fixture_root / "manifest.json")


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
