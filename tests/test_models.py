import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant import (
    EvidenceRequest,
    ImageAsset,
    ImageLocator,
    QuestionContext,
    VisualEvidence,
    VisualResponse,
)


def test_response_contract_accepts_grounded_observation() -> None:
    response = VisualResponse(
        status="answered",
        answer="A synthetic frontal-view fixture is provided.",
        confidence=0.8,
        observations=["A single image is present."],
        visual_evidence=[
            VisualEvidence(
                locator=ImageLocator(image_id="synthetic-001", kind="full_frame"),
                description="The complete synthetic image.",
                confidence=0.8,
            )
        ],
        uncertainty=["This fixture is synthetic and non-diagnostic."],
    )

    assert response.status == "answered"
    assert response.visual_evidence[0].locator.image_id == "synthetic-001"


def test_request_requires_a_valid_image_and_question() -> None:
    with pytest.raises(ValidationError):
        EvidenceRequest(
            request_id="request-001",
            image=ImageAsset(
                image_id="synthetic-001",
                sha256="not-a-digest",
                media_type="image/png",
                byte_size=1024,
                width_px=64,
                height_px=64,
                origin="Synthetic test asset",
                license_status="synthetic",
            ),
            question=QuestionContext(question=""),
        )
