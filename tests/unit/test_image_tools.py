from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.tools import (
    CropImageArguments,
    FixtureImageTools,
    GetImageMetadataArguments,
    ImageToolRejected,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data" / "fixtures" / "manifest.json"


def test_metadata_returns_verified_hashes_without_image_bytes() -> None:
    tools = FixtureImageTools.from_manifest(MANIFEST_PATH)

    result = asyncio.run(
        tools.get_image_metadata(GetImageMetadataArguments(image_id="synthetic-full-frame"))
    )

    assert result.image.image_id == "synthetic-full-frame"
    assert result.input_sha256 == result.image.sha256
    assert len(result.output_sha256) == 64
    assert "bytes" not in result.model_dump()
    assert "89504e47" not in result.model_dump_json()


def test_crop_is_deterministic_and_registers_a_server_owned_result() -> None:
    arguments = CropImageArguments(
        image_id="synthetic-crop-target",
        box={"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
    )
    first_tools = FixtureImageTools.from_manifest(MANIFEST_PATH)
    second_tools = FixtureImageTools.from_manifest(MANIFEST_PATH)

    first = asyncio.run(first_tools.crop_image(arguments))
    second = asyncio.run(second_tools.crop_image(arguments))

    assert first == second
    assert first.input_sha256 == (
        "014cdad389db2164809bf4decb620f52c1016a889c1614e12849168312ffa1c7"
    )
    assert first.output_sha256 == first.image.sha256
    assert (first.image.width_px, first.image.height_px) == (32, 32)
    assert first.image.image_id.startswith("crop-")

    metadata = asyncio.run(
        first_tools.get_image_metadata(GetImageMetadataArguments(image_id=first.image.image_id))
    )
    assert metadata.image == first.image
    assert metadata.input_sha256 == first.output_sha256


def test_unknown_server_owned_image_id_is_rejected_without_echoing_it() -> None:
    tools = FixtureImageTools.from_manifest(MANIFEST_PATH)

    with pytest.raises(ImageToolRejected) as caught:
        asyncio.run(
            tools.get_image_metadata(GetImageMetadataArguments(image_id="secret-path-name"))
        )

    assert caught.value.code == "image_not_found"
    assert str(caught.value) == "image_not_found"
    assert "secret-path-name" not in str(caught.value)


def test_invalid_crop_box_is_rejected_by_the_typed_boundary() -> None:
    with pytest.raises(ValidationError):
        CropImageArguments.model_validate(
            {
                "image_id": "synthetic-crop-target",
                "box": {"x_min": 0.8, "y_min": 0.2, "x_max": 0.2, "y_max": 0.9},
            }
        )


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("path_traversal", "invalid_manifest"),
        ("oversized", "invalid_manifest"),
    ],
)
def test_unsafe_manifest_entries_fail_closed(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(MANIFEST_PATH.parent, fixture_root)
    manifest_path = fixture_root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = raw["fixtures"][0]
    if mutation == "path_traversal":
        first["path"] = "../outside.png"
    else:
        first["asset"]["byte_size"] = 20_000_001
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ImageToolRejected) as caught:
        FixtureImageTools.from_manifest(manifest_path)

    assert caught.value.code == expected_code
    assert str(caught.value) == expected_code
    assert "outside.png" not in str(caught.value)


def test_unsupported_media_type_is_rejected_before_decode(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(MANIFEST_PATH.parent, fixture_root)
    manifest_path = fixture_root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["fixtures"][0]["asset"]["media_type"] = "image/jpeg"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    tools = FixtureImageTools.from_manifest(manifest_path)

    with pytest.raises(ImageToolRejected) as caught:
        asyncio.run(
            tools.crop_image(
                CropImageArguments(
                    image_id="synthetic-full-frame",
                    box={"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
                )
            )
        )

    assert caught.value.code == "unsupported_media_type"
    assert str(caught.value) == "unsupported_media_type"
