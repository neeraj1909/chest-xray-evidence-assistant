from __future__ import annotations

import json

import pytest

from chest_xray_evidence_assistant.ui_smoke import parse_complete_event, validate_result


def test_complete_event_parser_and_result_gate_accept_real_contract() -> None:
    result = [
        "Answered",
        "Evidence-linked observation",
        "Completed in bounded offline-demo mode. This is not a diagnosis.",
        "The verified synthetic fixture contains one abstract grayscale image.",
        1.0,
        "• One registered synthetic image is available for inspection.",
        "• This deterministic fake run does not inspect anatomy.",
        {"visual": [{"locator": "full frame: synthetic-full-frame"}], "sources": []},
        [
            {
                "sequence": 1,
                "kind": "tool_call",
                "name": "get_image_metadata",
                "status": "succeeded",
                "duration_ms": 1,
            },
            {
                "sequence": 5,
                "kind": "final_status",
                "name": "answered",
                "status": "succeeded",
                "duration_ms": None,
            },
        ],
        12,
    ]
    event_stream = f"event: complete\ndata: {json.dumps(result)}\n\n"

    parsed = parse_complete_event(event_stream)

    assert parsed == result
    assert validate_result(parsed) == {
        "latency_ms": 12,
        "outcome": "Answered",
        "status": "succeeded",
        "tool": "get_image_metadata",
    }


@pytest.mark.parametrize(
    "event_stream",
    [
        "event: error\ndata: null\n\n",
        "event: complete\ndata: {}\n\n",
    ],
)
def test_complete_event_parser_rejects_failed_or_non_list_payloads(event_stream: str) -> None:
    with pytest.raises(ValueError, match="complete event"):
        parse_complete_event(event_stream)


def test_smoke_contract_rejects_an_incomplete_result() -> None:
    with pytest.raises(ValueError, match="UI smoke result"):
        validate_result(["Not run"])
