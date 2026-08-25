"""Load and verify the repository's deterministic offline fixtures."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator

from .models import ContractModel, ImageAsset, ShortText

FixturePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
FixtureUse = Literal["full_frame", "crop", "abstention"]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class FixtureRecord(ContractModel):
    path: FixturePath
    description: ShortText
    intended_use: list[FixtureUse] = Field(min_length=1, max_length=8)
    asset: ImageAsset

    @field_validator("path")
    @classmethod
    def require_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("fixture paths must remain relative to the manifest")
        return value


class FixtureManifest(ContractModel):
    schema_version: Literal[1]
    fixtures: list[FixtureRecord] = Field(min_length=3, max_length=32)

    @model_validator(mode="after")
    def validate_manifest(self) -> FixtureManifest:
        image_ids = [fixture.asset.image_id for fixture in self.fixtures]
        paths = [fixture.path for fixture in self.fixtures]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("fixture image IDs must be unique")
        if len(set(paths)) != len(paths):
            raise ValueError("fixture paths must be unique")

        declared_uses = {use for fixture in self.fixtures for use in fixture.intended_use}
        required_uses = {"full_frame", "crop", "abstention"}
        missing_uses = required_uses - declared_uses
        if missing_uses:
            missing = ", ".join(sorted(missing_uses))
            raise ValueError(f"manifest is missing intended uses: {missing}")
        return self


def _png_dimensions(content: bytes) -> tuple[int, int]:
    if not content.startswith(PNG_SIGNATURE) or content[12:16] != b"IHDR":
        raise ValueError("fixture is not a supported PNG")
    return struct.unpack(">II", content[16:24])


def _verify_file(manifest_path: Path, fixture: FixtureRecord) -> None:
    root = manifest_path.parent.resolve()
    candidate = (root / fixture.path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("fixture path escapes the manifest directory") from exc

    if not candidate.is_file():
        raise ValueError(f"fixture file does not exist: {fixture.path}")

    content = candidate.read_bytes()
    if len(content) != fixture.asset.byte_size:
        raise ValueError(f"fixture byte size mismatch: {fixture.path}")
    if hashlib.sha256(content).hexdigest() != fixture.asset.sha256:
        raise ValueError(f"fixture SHA-256 mismatch: {fixture.path}")

    if fixture.asset.media_type == "image/png":
        width, height = _png_dimensions(content)
        if (width, height) != (fixture.asset.width_px, fixture.asset.height_px):
            raise ValueError(f"fixture dimensions mismatch: {fixture.path}")


def load_fixture_manifest(path: Path | None = None) -> FixtureManifest:
    """Parse the manifest and verify every referenced file's bytes and dimensions."""

    manifest_path = path or (
        Path(__file__).resolve().parents[2] / "data" / "fixtures" / "manifest.json"
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = FixtureManifest.model_validate(raw)
    for fixture in manifest.fixtures:
        _verify_file(manifest_path, fixture)
    return manifest
