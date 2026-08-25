import pytest
from pydantic import ValidationError

from chest_xray_evidence_assistant.models import (
    EvidenceRequest,
    ImageAsset,
    ImageLocator,
    QuestionContext,
    RunLimits,
    VisualEvidence,
    VisualResponse,
)


@pytest.fixture
def image_asset() -> ImageAsset:
    return ImageAsset(
        image_id="synthetic-001",
        sha256="0" * 64,
        media_type="image/png",
        byte_size=1024,
        width_px=64,
        height_px=64,
        origin="Locally generated test pattern",
        license_status="synthetic",
    )


def test_request_round_trip(image_asset: ImageAsset) -> None:
    request = EvidenceRequest(
        request_id="request-001",
        image=image_asset,
        question=QuestionContext(question="Is this a frontal image?"),
    )

    serialized = request.model_dump(mode="json")
    assert EvidenceRequest.model_validate(serialized) == request


def test_unknown_fields_are_rejected(image_asset: ImageAsset) -> None:
    with pytest.raises(ValidationError):
        ImageAsset.model_validate(
            {
                **image_asset.model_dump(),
                "patient_name": "must-not-be-present",
            }
        )


def test_tool_limit_cannot_exceed_three() -> None:
    with pytest.raises(ValidationError):
        RunLimits(max_tool_calls=4)


def test_invalid_bounding_box_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ImageLocator.model_validate(
            {
                "image_id": "synthetic-001",
                "kind": "bounding_box",
                "box": {
                    "x_min": 0.8,
                    "y_min": 0.2,
                    "x_max": 0.4,
                    "y_max": 0.9,
                },
            }
        )


def test_answered_response_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="answered",
            answer="Unsupported answer",
            confidence=0.9,
            uncertainty=["No supporting evidence was supplied."],
        )


def test_grounded_answer_is_accepted() -> None:
    response = VisualResponse(
        status="answered",
        answer="The fixture contains one frontal-view test image.",
        confidence=0.8,
        observations=["A single synthetic image is present."],
        visual_evidence=[
            VisualEvidence(
                locator=ImageLocator(
                    image_id="synthetic-001",
                    kind="full_frame",
                ),
                description="The entire synthetic fixture.",
                confidence=0.8,
            )
        ],
        uncertainty=["This fixture is synthetic and non-diagnostic."],
    )

    assert response.status == "answered"


def test_abstention_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        VisualResponse(
            status="abstain",
            confidence=1,
            uncertainty=["No image was supplied."],
        )


def test_valid_abstention() -> None:
    response = VisualResponse(
        status="abstain",
        confidence=1,
        abstention_reason="No usable image was supplied.",
        uncertainty=["Visual evidence is unavailable."],
    )

    assert response.answer is None
