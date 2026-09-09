"""HTTP smoke for the real Gradio upload and queued-submit boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from .fixtures import load_fixture_manifest

SMOKE_IMAGE_ID = "synthetic-full-frame"
SMOKE_QUESTION = "What can be observed in this synthetic fixture?"


def _local_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("UI smoke URL must be a plain local HTTP origin")
    try:
        if parsed.port is None:
            raise ValueError("UI smoke URL must include a port")
    except ValueError as error:
        raise ValueError("UI smoke URL must include a valid port") from error
    return value.rstrip("/")


def _read(request: str | urllib.request.Request, *, timeout_seconds: float) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ValueError("UI smoke request returned a non-success status")
            return response.read()
    except (TimeoutError, urllib.error.URLError) as error:
        raise ValueError("UI smoke request failed") from error


def wait_until_ready(base_url: str, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            _read(base_url + "/", timeout_seconds=min(2.0, timeout_seconds))
            return
        except ValueError:
            if time.monotonic() >= deadline:
                raise ValueError("UI smoke readiness deadline expired") from None
            time.sleep(0.25)


def _post_json(base_url: str, path: str, payload: object, *, timeout_seconds: float) -> object:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(_read(request, timeout_seconds=timeout_seconds))


def _upload_fixture(base_url: str, content: bytes, *, timeout_seconds: float) -> str:
    digest = hashlib.sha256(content).hexdigest()
    boundary = f"cxr-smoke-{digest[:24]}"
    filename = "synthetic-full-frame.png"
    body = b"".join(
        (
            f"--{boundary}\r\n".encode("ascii"),
            (f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n').encode(
                "ascii"
            ),
            b"Content-Type: image/png\r\n\r\n",
            content,
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        )
    )
    request = urllib.request.Request(
        base_url + "/gradio_api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    payload = json.loads(_read(request, timeout_seconds=timeout_seconds))
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], str)
        or not payload[0].startswith("/tmp/gradio/")
        or len(payload[0]) > 1_024
        or any(character in payload[0] for character in "\r\n")
    ):
        raise ValueError("UI smoke upload returned an invalid file reference")
    return payload[0]


def parse_complete_event(event_stream: str) -> list[object]:
    """Extract one JSON list from a successful Gradio SSE completion event."""

    current_event: str | None = None
    for line in event_stream.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        elif current_event == "complete" and line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: "))
            if isinstance(payload, list):
                return cast(list[object], payload)
            break
    raise ValueError("UI smoke response has no valid complete event")


def validate_result(result: list[object]) -> dict[str, object]:
    """Fail closed unless the UI returned its exact successful demo contract."""

    if len(result) != 10:
        raise ValueError("UI smoke result has an invalid shape")
    outcome, title, _, answer, _, _, _, evidence, trace, latency_ms = result
    if (
        outcome != "Answered"
        or title != "Evidence-linked observation"
        or answer != "The verified synthetic fixture contains one abstract grayscale image."
        or not isinstance(evidence, dict)
        or not isinstance(trace, list)
        or isinstance(latency_ms, bool)
        or not isinstance(latency_ms, int | float)
        or latency_ms < 0
    ):
        raise ValueError("UI smoke result did not satisfy the success contract")
    visual_evidence = evidence.get("visual")
    if not isinstance(visual_evidence, list) or not any(
        isinstance(item, dict) and item.get("locator") == "full frame: synthetic-full-frame"
        for item in visual_evidence
    ):
        raise ValueError("UI smoke result omitted the expected visual evidence")
    tool_succeeded = any(
        isinstance(event, dict)
        and event.get("kind") == "tool_call"
        and event.get("name") == "get_image_metadata"
        and event.get("status") == "succeeded"
        for event in trace
    )
    terminal_succeeded = any(
        isinstance(event, dict)
        and event.get("kind") == "final_status"
        and event.get("name") == "answered"
        and event.get("status") == "succeeded"
        for event in trace
    )
    if not tool_succeeded or not terminal_succeeded:
        raise ValueError("UI smoke result omitted successful tool or terminal evidence")
    return {
        "latency_ms": latency_ms,
        "outcome": outcome,
        "status": "succeeded",
        "tool": "get_image_metadata",
    }


def run_smoke(base_url: str, *, timeout_seconds: float) -> dict[str, object]:
    base_url = _local_base_url(base_url)
    if timeout_seconds <= 0 or timeout_seconds > 120:
        raise ValueError("UI smoke timeout must be between 0 and 120 seconds")
    wait_until_ready(base_url, timeout_seconds=timeout_seconds)

    manifest = load_fixture_manifest()
    fixture = next(
        (item for item in manifest.fixtures if item.asset.image_id == SMOKE_IMAGE_ID),
        None,
    )
    if fixture is None:
        raise ValueError("UI smoke fixture is unavailable")
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / fixture.path
    uploaded_path = _upload_fixture(
        base_url,
        fixture_path.read_bytes(),
        timeout_seconds=timeout_seconds,
    )
    submission = _post_json(
        base_url,
        "/gradio_api/call/offline_demo",
        {
            "data": [
                {
                    "path": uploaded_path,
                    "orig_name": fixture_path.name,
                    "mime_type": "image/png",
                    "meta": {"_type": "gradio.FileData"},
                },
                SMOKE_QUESTION,
                "",
                True,
            ]
        },
        timeout_seconds=timeout_seconds,
    )
    if not isinstance(submission, dict) or not isinstance(submission.get("event_id"), str):
        raise ValueError("UI smoke submission returned no event identity")
    event_id = submission["event_id"]
    if len(event_id) != 32 or any(character not in "0123456789abcdef" for character in event_id):
        raise ValueError("UI smoke submission returned an invalid event identity")
    event_stream = _read(
        base_url + f"/gradio_api/call/offline_demo/{event_id}",
        timeout_seconds=timeout_seconds,
    ).decode("utf-8")
    return validate_result(parse_complete_event(event_stream))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke the local offline Gradio journey.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--timeout", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    try:
        result = run_smoke(arguments.base_url, timeout_seconds=arguments.timeout)
    except (OSError, ValueError):
        result = {"error_code": "ui_smoke_failed", "status": "failed"}
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
