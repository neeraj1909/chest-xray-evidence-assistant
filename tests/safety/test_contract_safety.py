from __future__ import annotations

import json
import socket
from pathlib import Path 

import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.models import (
    ImageAsset,
    ImageLocator,
    SourceEvidence,
    VisualEvidence,
    VisualResponse,
)

RESPONSE_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "responses"


def valid_visual_evidence() -> VisualEvidence:
    return VisualEvidence(
        locator=ImageLocator(
            image_id="synthetic-001",
            kind="full_frame",
        ),
        description="Synthetic radiograph with no patient-identifying content.",
        confidence=0.8,
    )
    
    
def test_image_asset_rejects_phi_marker() -> None:
    with pytest.raises(ValidationError):
        ImageAsset(
            image_id="synthetic-001",
            sha256="0" * 64,
            media_type="image/png",
            byte_size=1024,
            width=64,
            height=64,
            origin="tests",
            license_status="synthetic",
            contains_phi=True,
        )
        

@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_response_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="abstain",
            confidence=confidence,
            uncertainty=["The available evidence is insufficient."],
            abstention_reason="Insufficient evidence.",
        )
        
        
def test_answered_response_requires_evidence_provenance() -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="answered",
            answer="The image is unremarkable.",
            confidence=0.8,
            uncertainty=["Limited to the supplied image."],
        )
        
        
def test_visual_evidence_requires_a_valid_locator() -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="answered",
            answer="A finding is visible.",
            confidence=0.6,
            uncertainty=["Synthetic fixture response."],
            visual_evidence=[
                {
                    "locator": {
                        "image_id": "synthetic-001",
                        "kind": "bounding_box",
                    },
                    "description": "Missing bounding-box coordinates.",
                    "confidence": 0.6,
                }
            ],
        )
        
        
def test_source_evidence_requires_document_provenance() -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="answered",
            answer="The cited document supports this answer.",
            confidence=0.7,
            uncertainty=["Synthetic fixture response."],
            source_evidence=[
                {
                    "document_id": "doc-001",
                    "section": "Findings",
                    "source_url": "https://example.org/reference",
                    "excerpt": "Supporting excerpt",
                    "relevance_score": 0.9,
                    # Required locator intentionally omitted.
                }
            ],
        )
        

@pytest.mark.parametrize(
    ("status", "extra_fields"),
    [
        (
            "needs_clarification",
            {
                "answer": "This answer must not be emitted.",
                "clarification_question": "Please provide another image.",
            },
        ),
        (
            "abstain",
            {
                "answer": "This answer must not be emitted.",
                "abstention_reason": "Evidence is insufficient",
            },
        ),
    ],
)
def test_non_answer_statuses_cannot_contain_answers(
    status: str,
    extra_fields: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status=status,
            confidence=0.0,
            uncertainty=["The fixture is intentionally incomplete."],
            **extra_fields,
        )
        
        
@pytest.mark.parametrize(
    ("filename", "expected_status"),
    [
        ("answered.json", "answered"),
        ("needs-clarification.json", "needs_clarification"),
        ("abstain.json", "abstain"),
    ],
)
def test_approved_fake_responses_validate(
    filename: str,
    expected_status: str,
) -> None:
    payload = json.loads((RESPONSE_FIXTURES / filename).read_text())
    response = VisualResponse.model_validate(payload)
    
    assert response.status == expected_status
    
    
def test_malformed_fake_response_is_rejected() -> None:
    payload = json.loads((RESPONSE_FIXTURES / "malformed.json").read_text())
    
    with pytest.raises(ValidationError):
        VisualResponse.model_validate(payload)
        
        
def test_test_suite_blocks_network_access() -> None:
    with pytest.raises(AssertionError, match="network access"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01) 
