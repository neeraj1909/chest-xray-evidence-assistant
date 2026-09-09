"""Deterministic image tools for verified, server-owned grayscale PNG assets."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from ..fixtures import PNG_SIGNATURE, load_fixture_manifest
from ..models import ImageAsset, NormalizedBoundingBox
from .contracts import (
    CropImageArguments,
    CropImageResult,
    GetImageMetadataArguments,
    ImageMetadataResult,
    ImageToolRejected,
)

MAX_DECODED_PIXELS = 20_000_000


@dataclass(frozen=True, slots=True)
class _StoredImage:
    asset: ImageAsset
    content: bytes


def _digest_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _encode_grayscale_png(rows: tuple[bytes, ...]) -> bytes:
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if width == 0 or height == 0 or any(len(row) != width for row in rows):
        raise ImageToolRejected("invalid_image")

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row for row in rows)
    return b"".join(
        (
            PNG_SIGNATURE,
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    if above_distance <= upper_left_distance:
        return above
    return upper_left


def _restore_scanline(filtered: bytes, previous: bytes, filter_type: int) -> bytes:
    restored = bytearray(len(filtered))
    for index, value in enumerate(filtered):
        left = restored[index - 1] if index else 0
        above = previous[index] if previous else 0
        upper_left = previous[index - 1] if previous and index else 0
        if filter_type == 0:
            predictor = 0
        elif filter_type == 1:
            predictor = left
        elif filter_type == 2:
            predictor = above
        elif filter_type == 3:
            predictor = (left + above) // 2
        elif filter_type == 4:
            predictor = _paeth(left, above, upper_left)
        else:
            raise ImageToolRejected("invalid_image")
        restored[index] = (value + predictor) & 0xFF
    return bytes(restored)


def _decode_grayscale_png(content: bytes) -> tuple[bytes, ...]:
    if not content.startswith(PNG_SIGNATURE):
        raise ImageToolRejected("invalid_image")

    position = len(PNG_SIGNATURE)
    width = height = 0
    idat_parts: list[bytes] = []
    saw_header = False
    saw_end = False
    while position < len(content):
        if position + 12 > len(content):
            raise ImageToolRejected("invalid_image")
        length = struct.unpack(">I", content[position : position + 4])[0]
        kind = content[position + 4 : position + 8]
        payload_start = position + 8
        payload_end = payload_start + length
        crc_end = payload_end + 4
        if crc_end > len(content):
            raise ImageToolRejected("invalid_image")
        payload = content[payload_start:payload_end]
        expected_crc = struct.unpack(">I", content[payload_end:crc_end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ImageToolRejected("invalid_image")

        if kind == b"IHDR":
            if saw_header or length != 13:
                raise ImageToolRejected("invalid_image")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (
                width == 0
                or height == 0
                or width * height > MAX_DECODED_PIXELS
                or (bit_depth, color_type, compression, filtering, interlace) != (8, 0, 0, 0, 0)
            ):
                raise ImageToolRejected("unsupported_media_type")
            saw_header = True
        elif kind == b"IDAT":
            if not saw_header or saw_end:
                raise ImageToolRejected("invalid_image")
            idat_parts.append(payload)
        elif kind == b"IEND":
            if length != 0 or not saw_header:
                raise ImageToolRejected("invalid_image")
            saw_end = True
            position = crc_end
            break
        position = crc_end

    if not saw_header or not saw_end or not idat_parts or position != len(content):
        raise ImageToolRejected("invalid_image")

    expected_size = height * (width + 1)
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(b"".join(idat_parts), expected_size + 1)
        if decompressor.unconsumed_tail or len(raw) > expected_size:
            raise ImageToolRejected("invalid_image")
        remaining = expected_size - len(raw)
        if remaining:
            raw += decompressor.flush(remaining)
        else:
            raw += decompressor.flush()
    except zlib.error:
        raise ImageToolRejected("invalid_image") from None
    if len(raw) != expected_size or not decompressor.eof:
        raise ImageToolRejected("invalid_image")

    rows: list[bytes] = []
    previous = b""
    stride = width + 1
    for offset in range(0, len(raw), stride):
        scanline = raw[offset : offset + stride]
        restored = _restore_scanline(scanline[1:], previous, scanline[0])
        rows.append(restored)
        previous = restored
    return tuple(rows)


def _pixel_box(
    box: NormalizedBoundingBox,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, math.floor(box.x_min * width)))
    top = max(0, min(height - 1, math.floor(box.y_min * height)))
    right = max(left + 1, min(width, math.ceil(box.x_max * width)))
    bottom = max(top + 1, min(height, math.ceil(box.y_max * height)))
    return left, top, right, bottom


class FixtureImageTools:
    """Crop and inspect only assets loaded from a verified fixture manifest."""

    def __init__(self, images: dict[str, _StoredImage]) -> None:
        self._images = dict(images)

    @classmethod
    def from_manifest(cls, path: Path | None = None) -> FixtureImageTools:
        manifest_path = path or (
            Path(__file__).resolve().parents[3] / "data" / "fixtures" / "manifest.json"
        )
        try:
            manifest = load_fixture_manifest(manifest_path)
            root = manifest_path.parent.resolve()
            images: dict[str, _StoredImage] = {}
            for fixture in manifest.fixtures:
                candidate = (root / fixture.path).resolve()
                candidate.relative_to(root)
                if candidate.stat().st_size != fixture.asset.byte_size:
                    raise ValueError("fixture changed during verification")
                with candidate.open("rb") as fixture_file:
                    content = fixture_file.read(fixture.asset.byte_size + 1)
                if (
                    len(content) != fixture.asset.byte_size
                    or hashlib.sha256(content).hexdigest() != fixture.asset.sha256
                ):
                    raise ValueError("fixture changed during verification")
                images[fixture.asset.image_id] = _StoredImage(fixture.asset, content)
        except (OSError, ValueError):
            raise ImageToolRejected("invalid_manifest") from None
        return cls(images)

    def _get(self, image_id: str) -> _StoredImage:
        try:
            return self._images[image_id]
        except KeyError:
            raise ImageToolRejected("image_not_found") from None

    async def get_image_metadata(
        self,
        arguments: GetImageMetadataArguments,
    ) -> ImageMetadataResult:
        stored = self._get(arguments.image_id)
        output_sha256 = _digest_json(stored.asset.model_dump(mode="json"))
        return ImageMetadataResult(
            image=stored.asset,
            input_sha256=stored.asset.sha256,
            output_sha256=output_sha256,
        )

    async def crop_image(self, arguments: CropImageArguments) -> CropImageResult:
        source = self._get(arguments.image_id)
        if source.asset.byte_size > 20_000_000:
            raise ImageToolRejected("image_too_large")
        if source.asset.media_type != "image/png":
            raise ImageToolRejected("unsupported_media_type")

        rows = _decode_grayscale_png(source.content)
        if (len(rows[0]), len(rows)) != (source.asset.width_px, source.asset.height_px):
            raise ImageToolRejected("invalid_image")
        left, top, right, bottom = _pixel_box(
            arguments.box,
            width=source.asset.width_px,
            height=source.asset.height_px,
        )
        cropped_rows = tuple(row[left:right] for row in rows[top:bottom])
        content = _encode_grayscale_png(cropped_rows)
        output_sha256 = hashlib.sha256(content).hexdigest()
        image = ImageAsset(
            image_id=f"crop-{output_sha256[:32]}",
            sha256=output_sha256,
            media_type="image/png",
            byte_size=len(content),
            width_px=right - left,
            height_px=bottom - top,
            origin=f"Deterministic server-owned crop of {source.asset.image_id}",
            origin_url=source.asset.origin_url,
            license_status=source.asset.license_status,
            contains_phi=False,
        )
        self._images[image.image_id] = _StoredImage(image, content)
        return CropImageResult(
            source_image_id=source.asset.image_id,
            box=arguments.box,
            image=image,
            input_sha256=source.asset.sha256,
            output_sha256=output_sha256,
        )
