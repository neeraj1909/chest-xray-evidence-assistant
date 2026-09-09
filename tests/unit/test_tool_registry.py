from __future__ import annotations

from collections.abc import Mapping

import pytest

from chest_xray_evidence_assistant.tools import (
    INITIAL_TOOL_REGISTRY,
    CropImageCall,
    CropImagePort,
    GetImageMetadataCall,
    ImageMetadataPort,
    ReferenceRetrievalPort,
    RetrieveReferenceCall,
    ToolCallRejected,
)

EXPECTED_TOOL_NAMES = (
    "crop_image",
    "get_image_metadata",
    "retrieve_reference",
)


def test_registry_snapshot_exposes_exactly_three_bounded_tools() -> None:
    snapshot = INITIAL_TOOL_REGISTRY.snapshot()

    assert INITIAL_TOOL_REGISTRY.names == EXPECTED_TOOL_NAMES
    assert tuple(item["name"] for item in snapshot) == EXPECTED_TOOL_NAMES

    properties = {item["name"]: set(item["parameters"]["properties"]) for item in snapshot}
    assert properties == {
        "crop_image": {"box", "image_id"},
        "get_image_metadata": {"image_id"},
        "retrieve_reference": {"query", "top_k"},
    }
    assert all(item["parameters"]["additionalProperties"] is False for item in snapshot)


@pytest.mark.parametrize(
    ("name", "arguments", "expected_type"),
    [
        (
            "crop_image",
            {
                "image_id": "synthetic-crop-target",
                "box": {
                    "x_min": 0.1,
                    "y_min": 0.2,
                    "x_max": 0.8,
                    "y_max": 0.9,
                },
            },
            CropImageCall,
        ),
        (
            "get_image_metadata",
            {"image_id": "synthetic-full-frame"},
            GetImageMetadataCall,
        ),
        (
            "retrieve_reference",
            {"query": "  portable chest radiograph positioning  ", "top_k": 3},
            RetrieveReferenceCall,
        ),
    ],
)
def test_registry_returns_a_typed_call_for_valid_arguments(
    name: str,
    arguments: Mapping[str, object],
    expected_type: type[object],
) -> None:
    call = INITIAL_TOOL_REGISTRY.validate_call(name, arguments)

    assert isinstance(call, expected_type)
    assert call.name == name
    if isinstance(call, RetrieveReferenceCall):
        assert call.arguments.query == "portable chest radiograph positioning"


@pytest.mark.parametrize("name", ["read_file", "run_shell", "open_browser"])
def test_unknown_tool_names_fail_closed_without_echoing_input(name: str) -> None:
    with pytest.raises(ToolCallRejected) as caught:
        INITIAL_TOOL_REGISTRY.validate_call(name, {"api_key": "must-not-leak"})

    assert caught.value.code == "unknown_tool"
    assert str(caught.value) == "unknown_tool"
    assert name not in str(caught.value)
    assert "must-not-leak" not in str(caught.value)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (
            "crop_image",
            {
                "image_id": "synthetic-crop-target",
                "box": {"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
                "path": "/etc/passwd",
            },
        ),
        (
            "get_image_metadata",
            {"image_id": "synthetic-full-frame", "command": "cat /etc/passwd"},
        ),
        (
            "retrieve_reference",
            {"query": "positioning", "top_k": 2, "source_url": "https://example.test"},
        ),
    ],
)
def test_unexpected_arguments_fail_closed_without_echoing_payload(
    name: str,
    arguments: Mapping[str, object],
) -> None:
    with pytest.raises(ToolCallRejected) as caught:
        INITIAL_TOOL_REGISTRY.validate_call(name, arguments)

    assert caught.value.code == "invalid_arguments"
    assert str(caught.value) == "invalid_arguments"
    assert "/etc/passwd" not in str(caught.value)
    assert "example.test" not in str(caught.value)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("crop_image", {"image_id": "synthetic-crop-target"}),
        ("get_image_metadata", {"image_id": "../outside"}),
        ("retrieve_reference", {"query": "positioning", "top_k": 6}),
    ],
)
def test_invalid_typed_arguments_fail_closed(
    name: str,
    arguments: Mapping[str, object],
) -> None:
    with pytest.raises(ToolCallRejected, match="^invalid_arguments$"):
        INITIAL_TOOL_REGISTRY.validate_call(name, arguments)


def test_each_port_exposes_only_its_named_tool_operation() -> None:
    assert _public_protocol_methods(CropImagePort) == {"crop_image"}
    assert _public_protocol_methods(ImageMetadataPort) == {"get_image_metadata"}
    assert _public_protocol_methods(ReferenceRetrievalPort) == {"retrieve_reference"}


def _public_protocol_methods(protocol: type[object]) -> set[str]:
    return {
        name
        for name, value in vars(protocol).items()
        if callable(value) and not name.startswith("_")
    }
