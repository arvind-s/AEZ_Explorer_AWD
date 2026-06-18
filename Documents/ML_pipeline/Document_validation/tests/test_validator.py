from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from document_validation import (
    ValidationConfig,
    ensemble_result_to_rows,
    validate_document_file,
    validate_document_file_ensemble,
    validate_document_image,
)
from document_validation.ensemble import (
    DetectorVote,
    EnsemblePageResult,
    _decision_columns_from_votes,
    recompute_page_from_votes,
)
import dataclasses


def _document_image() -> np.ndarray:
    image = np.full((720, 960, 3), 55, dtype=np.uint8)
    cv2.rectangle(image, (180, 70), (780, 650), (245, 245, 245), -1)
    cv2.rectangle(image, (180, 70), (780, 650), (220, 220, 220), 3)

    # Dense lines give enough gradient for Tenengrad to pass the sharpness threshold
    for idx in range(36):
        y = 120 + idx * 14
        cv2.line(image, (210, y), (750, y), (20, 20, 20), 2)
    for idx in range(10):
        x = 220 + idx * 55
        cv2.line(image, (x, 120), (x, 620), (30, 30, 30), 1)

    cv2.putText(
        image,
        "DOCUMENT",
        (300, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return image


def _cut_document_image() -> np.ndarray:
    image = np.full((720, 960, 3), 55, dtype=np.uint8)
    cv2.rectangle(image, (-30, 70), (740, 650), (245, 245, 245), -1)
    cv2.rectangle(image, (-30, 70), (740, 650), (220, 220, 220), 3)

    for idx in range(18):
        y = 145 + idx * 24
        cv2.line(image, (0, y), (680, y), (15, 15, 15), 3)

    cv2.putText(
        image,
        "CUT DOC",
        (2, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return image


def _random_photo_like_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, size=(720, 960, 3), dtype=np.uint8)
    image[:, :, 1] = np.maximum(image[:, :, 1], 150)
    image[:, :, 2] = np.minimum(image[:, :, 2], 90)
    cv2.circle(image, (480, 350), 180, (30, 220, 40), -1)
    return cv2.GaussianBlur(image, (5, 5), 0)


def _full_frame_document_with_edge_artifacts() -> np.ndarray:
    image = np.full((720, 960, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (959, 719), (230, 230, 230), 4)
    cv2.rectangle(image, (300, 0), (650, 14), (20, 20, 20), -1)
    cv2.rectangle(image, (945, 420), (959, 610), (20, 20, 20), -1)

    for idx in range(18):
        y = 120 + idx * 25
        cv2.line(image, (160, y), (790, y), (20, 20, 20), 2)

    cv2.putText(
        image,
        "SAFE CONTENT",
        (310, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    return image


def _low_contrast_camera_document() -> np.ndarray:
    """Moderate low-contrast: readability_contrast ~45 — passes calibrated threshold (40)."""
    image = np.full((720, 960, 3), (178, 168, 156), dtype=np.uint8)
    cv2.rectangle(image, (0, 25), (950, 700), (190, 180, 168), -1)

    for idx in range(20):
        y = 90 + idx * 28
        cv2.line(image, (70, y), (880, y), (145, 135, 125), 2)

    for x in range(80, 900, 120):
        cv2.line(image, (x, 70), (x, 650), (145, 135, 125), 2)

    cv2.putText(
        image,
        "LOW CONTRAST FORM",
        (185, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (145, 135, 125),
        2,
        cv2.LINE_AA,
    )
    return image


def _extreme_low_contrast_document() -> np.ndarray:
    """Extreme low-contrast: ink nearly same shade as background — unreadable."""
    image = np.full((720, 960, 3), (180, 175, 170), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (959, 719), (185, 180, 175), -1)

    for idx in range(20):
        y = 90 + idx * 28
        cv2.line(image, (70, y), (880, y), (172, 168, 165), 2)

    cv2.putText(
        image,
        "INVISIBLE TEXT",
        (185, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (175, 172, 169),
        2,
        cv2.LINE_AA,
    )
    return image


def test_accepts_sharp_document() -> None:
    result = validate_document_image(_document_image())

    assert result.accepted
    assert not result.blur.is_blurry
    assert result.clear.is_clear
    assert result.document.is_document
    assert not result.cut.is_cut
    assert result.to_decision_row() == {
        "is_blur": False,
        "is_cut": False,
        "is_clear": True,
        "is_image": False,
        "Decision": "accept",
    }


def test_detects_blurry_document() -> None:
    image = cv2.GaussianBlur(_document_image(), (51, 51), 0)
    result = validate_document_image(image)

    assert not result.accepted
    assert "blur" in result.issues
    assert result.blur.is_blurry
    assert not result.clear.is_clear


def test_detects_cut_document() -> None:
    result = validate_document_image(_cut_document_image())

    assert not result.accepted
    assert "cut" in result.issues
    assert result.cut.is_cut


def test_does_not_flag_full_frame_document_when_content_is_safe() -> None:
    result = validate_document_image(_full_frame_document_with_edge_artifacts())

    assert result.accepted
    assert not result.cut.is_cut
    assert result.cut.touches_border
    assert not result.cut.content_touches_border


def test_rejects_random_photo() -> None:
    result = validate_document_image(_random_photo_like_image())

    assert not result.accepted
    assert "not_document" in result.issues
    assert not result.document.is_document


def test_moderate_low_contrast_document_accepted_after_calibration() -> None:
    # readability_contrast ~45 — above the calibrated threshold (40), so now accepted
    result = validate_document_image(_low_contrast_camera_document())

    assert result.document.is_document
    assert "not_document" not in result.issues
    assert not result.blur.is_blurry
    assert result.accepted


def test_extreme_low_contrast_document_rejected() -> None:
    # ink nearly indistinguishable from background — rejected (as not_document or not_clear)
    result = validate_document_image(_extreme_low_contrast_document())
    assert not result.accepted


def test_accepts_pdf_document(tmp_path) -> None:
    success, encoded = cv2.imencode(".png", _document_image())
    assert success

    pdf_path = tmp_path / "document.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=960, height=720)
    page.insert_image(fitz.Rect(0, 0, 960, 720), stream=encoded.tobytes())
    pdf.save(pdf_path)
    pdf.close()

    result = validate_document_file(pdf_path)

    assert result.accepted
    assert result.file_type == "pdf"
    assert result.page_count == 1
    assert result.pages[0].result.document.is_document


def test_ensemble_accepts_document_with_opencv_detector(tmp_path) -> None:
    image_path = tmp_path / "document.png"
    cv2.imwrite(str(image_path), _document_image())

    result = validate_document_file_ensemble(image_path, detectors=["opencv"])

    assert result.accepted
    assert result.strategy == "weighted_max_vote"
    assert result.pages[0].label == "accepted"
    assert result.pages[0].label_votes["accepted"] == 1
    assert result.pages[0].weighted_label_votes["accepted"] == 1.0
    assert result.pages[0].votes[0].importance == 1.0
    assert result.pages[0].is_clear
    assert result.pages[0].Decision == "accept"


def test_ensemble_rejects_cut_document_with_opencv_detector(tmp_path) -> None:
    image_path = tmp_path / "cut_document.png"
    cv2.imwrite(str(image_path), _cut_document_image())

    result = validate_document_file_ensemble(image_path, detectors=["opencv"])

    assert not result.accepted
    assert "cut" in result.issues
    assert result.pages[0].label == "cut"
    assert result.pages[0].label_votes["cut"] == 1
    assert result.pages[0].weighted_label_votes["cut"] == 1.0
    assert result.pages[0].is_cut
    assert result.pages[0].Decision == "reject"


def test_ensemble_rejects_not_clear_document_with_strict_config(tmp_path) -> None:
    # Explicit strict readability threshold: any doc with rdbl < 65 is rejected.
    # With calibrated defaults (rdbl_t=35) this image (~45) is now accepted — use
    # an explicit config here to verify the low-readability path still works.
    image_path = tmp_path / "not_clear_document.png"
    cv2.imwrite(str(image_path), _low_contrast_camera_document())

    strict_config = ValidationConfig(min_readability_contrast=65.0, min_reject_confidence=0.50)
    result = validate_document_file_ensemble(image_path, config=strict_config, detectors=["opencv"])
    page = result.pages[0]

    assert not result.accepted
    assert page.label == "not_clear"
    assert not page.is_blur
    assert not page.is_cut
    assert not page.is_clear
    assert not page.is_image
    assert page.Decision == "reject"


def test_ensemble_uses_custom_detector_importance(tmp_path) -> None:
    image_path = tmp_path / "document.png"
    cv2.imwrite(str(image_path), _document_image())

    result = validate_document_file_ensemble(
        image_path,
        detectors=["opencv"],
        detector_importance={"opencv_gate": 2.5},
    )

    assert result.pages[0].votes[0].importance == 2.5
    assert result.pages[0].weighted_label_votes["accepted"] == 2.5


def test_ensemble_rows_include_detector_scores(tmp_path) -> None:
    image_path = tmp_path / "document.png"
    cv2.imwrite(str(image_path), _document_image())

    result = validate_document_file_ensemble(image_path, detectors=["opencv"])
    rows = ensemble_result_to_rows(result, include_detector_details=True)

    assert "opencv_gate_blur_confidence" in rows[0]
    assert "opencv_gate_document_confidence" in rows[0]
    assert "opencv_gate_cut_confidence" in rows[0]
    assert "opencv_gate_clear_confidence" in rows[0]


def test_ensemble_accepts_low_confidence_reject_vote() -> None:
    decision = _decision_columns_from_votes(
        [
            DetectorVote(
                detector="test",
                available=True,
                label="not_clear",
                issues=["not_clear"],
                confidence=0.2,
            )
        ],
        ValidationConfig(),
    )

    assert decision == {
        "is_blur": False,
        "is_cut": False,
        "is_clear": True,
        "is_image": False,
        "Decision": "accept",
    }


def test_ensemble_rejects_high_confidence_reject_vote() -> None:
    decision = _decision_columns_from_votes(
        [
            DetectorVote(
                detector="test",
                available=True,
                label="not_clear",
                issues=["not_clear"],
                confidence=0.95,
            )
        ],
        ValidationConfig(),
    )

    assert decision == {
        "is_blur": False,
        "is_cut": False,
        "is_clear": False,
        "is_image": False,
        "Decision": "reject",
    }


def test_ensemble_rejects_high_confidence_secondary_issue() -> None:
    decision = _decision_columns_from_votes(
        [
            DetectorVote(
                detector="opencv_gate",
                available=True,
                label="not_clear",
                issues=["not_clear", "cut"],
                confidence=0.38,
                scores={"clear_confidence": 0.62, "cut_confidence": 1.0},
            )
        ],
        ValidationConfig(),
    )

    assert decision == {
        "is_blur": False,
        "is_cut": True,
        "is_clear": True,
        "is_image": False,
        "Decision": "reject",
    }


def test_ensemble_rejects_gujarat_blur_sample() -> None:
    sample_path = Path(
        "test_data/ConcentForm/GUJRAT/is_blur/max/119030_farmer_119030_cf.pdf"
    )
    if not sample_path.exists():
        pytest.skip("local labeled sample is not available")

    result = validate_document_file_ensemble(
        sample_path,
        config=ValidationConfig(min_reject_confidence=0.70),
        detectors=["opencv"],
    )

    assert not result.accepted


def test_recompute_accepts_with_low_min_reject_confidence():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="blur",
            importance=1.0,
            issues=["blur"],
            confidence=0.55,
            scores={"blur_confidence": 0.55},
        )
    ]
    config = ValidationConfig(min_reject_confidence=0.70)
    result = recompute_page_from_votes(1, votes, config)

    # confidence 0.55 < min_reject_confidence 0.70 → should not trigger rejection
    assert result.accepted
    assert result.label == "accepted"
    assert result.Decision == "accept"


def test_recompute_rejects_with_high_min_reject_confidence():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="blur",
            importance=1.0,
            issues=["blur"],
            confidence=0.55,
            scores={"blur_confidence": 0.55},
        )
    ]
    config = ValidationConfig(min_reject_confidence=0.50)
    result = recompute_page_from_votes(1, votes, config)

    # confidence 0.55 >= min_reject_confidence 0.50 → should reject
    assert not result.accepted
    assert result.label == "blur"
    assert result.is_blur
    assert result.Decision == "reject"


def test_recompute_preserves_vote_list():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="accepted",
            importance=1.0,
            issues=[],
            confidence=0.9,
        ),
        DetectorVote(
            detector="doctr",
            available=True,
            label="accepted",
            importance=1.25,
            issues=[],
            confidence=0.88,
        ),
    ]
    config = ValidationConfig()
    result = recompute_page_from_votes(2, votes, config)

    assert result.page_number == 2
    assert result.votes == votes
    assert result.label_votes["accepted"] == 2
    assert abs(result.weighted_label_votes["accepted"] - 2.25) < 0.001


def test_recompute_aggregates_weighted_votes_correctly():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="not_document",
            importance=1.0,
            issues=["not_document"],
            confidence=0.80,
        ),
        DetectorVote(
            detector="doctr",
            available=True,
            label="accepted",
            importance=1.25,
            issues=[],
            confidence=0.85,
        ),
    ]
    config = ValidationConfig(min_reject_confidence=0.70)
    result = recompute_page_from_votes(1, votes, config)

    assert result.label_votes["not_document"] == 1
    assert result.label_votes["accepted"] == 1
    assert abs(result.weighted_label_votes["not_document"] - 1.0) < 0.001
    assert abs(result.weighted_label_votes["accepted"] - 1.25) < 0.001
