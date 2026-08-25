"""Generate the small, synthetic image fixtures used by offline tests.

The generator uses only the Python standard library so fixture bytes, hashes,
dimensions, and manifest metadata are reproducible without image dependencies.
The resulting images are abstract test patterns, not clinical radiographs.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

WIDTH = 64
HEIGHT = 64
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "data" / "fixtures"
IMAGE_ROOT = FIXTURE_ROOT / "images"

Pixel = Callable[[int, int], int]


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _write_grayscale_png(path: Path, pixel: Pixel) -> None:
    rows = []
    for y in range(HEIGHT):
        row = bytes(max(0, min(255, pixel(x, y))) for x in range(WIDTH))
        rows.append(b"\x00" + row)

    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", header)
    png += _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _full_frame(x: int, y: int) -> int:
    background = 24 + (y * 32) // (HEIGHT - 1)
    left_blob = ((x - 20) / 15) ** 2 + ((y - 32) / 23) ** 2 < 1
    right_blob = ((x - 44) / 15) ** 2 + ((y - 32) / 23) ** 2 < 1
    central_bar = 28 <= x <= 35 and 13 <= y <= 53
    if central_bar:
        return 142
    if left_blob or right_blob:
        return 92 + ((x * 7 + y * 11) % 40)
    return background


def _crop_target(x: int, y: int) -> int:
    background = 32 + (x + y) % 24
    inside_target = 16 <= x < 48 and 16 <= y < 48
    return 188 if inside_target else background


def _blank(_: int, __: int) -> int:
    return 0


def _asset_record(
    image_id: str,
    filename: str,
    description: str,
    intended_use: list[str],
) -> dict[str, object]:
    image_path = IMAGE_ROOT / filename
    content = image_path.read_bytes()
    return {
        "path": f"images/{filename}",
        "description": description,
        "intended_use": intended_use,
        "asset": {
            "image_id": image_id,
            "sha256": hashlib.sha256(content).hexdigest(),
            "media_type": "image/png",
            "byte_size": len(content),
            "width_px": WIDTH,
            "height_px": HEIGHT,
            "origin": "Generated locally by scripts/generate_fixtures.py",
            "license_status": "synthetic",
            "contains_phi": False,
        },
    }


def main() -> None:
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_grayscale_png(IMAGE_ROOT / "synthetic-full-frame.png", _full_frame)
    _write_grayscale_png(IMAGE_ROOT / "synthetic-crop-target.png", _crop_target)
    _write_grayscale_png(IMAGE_ROOT / "synthetic-abstention.png", _blank)

    manifest = {
        "schema_version": 1,
        "fixtures": [
            _asset_record(
                "synthetic-full-frame",
                "synthetic-full-frame.png",
                "Abstract grayscale pattern for full-frame evidence tests.",
                ["full_frame"],
            ),
            _asset_record(
                "synthetic-crop-target",
                "synthetic-crop-target.png",
                "Abstract grayscale pattern with a deterministic crop target.",
                ["full_frame", "crop"],
            ),
            _asset_record(
                "synthetic-abstention",
                "synthetic-abstention.png",
                "Blank synthetic image for insufficient-evidence abstention tests.",
                ["abstention"],
            ),
        ],
    }
    (FIXTURE_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
