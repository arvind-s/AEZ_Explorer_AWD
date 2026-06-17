import numpy as np
import pytest
from document_validation.vlm_detector import VlmDetector, parse_label

LABELS = ["accepted", "blur", "cut", "not_document"]

def test_parse_label_exact_match():
    assert parse_label("blur") == "blur"
    assert parse_label("accepted") == "accepted"
    assert parse_label("cut") == "cut"
    assert parse_label("not_document") == "not_document"

def test_parse_label_case_insensitive():
    assert parse_label("BLUR") == "blur"
    assert parse_label("Accepted") == "accepted"

def test_parse_label_embedded_in_sentence():
    assert parse_label("The document is blur due to camera shake.") == "blur"
    assert parse_label("I classify this as not_document.") == "not_document"

def test_parse_label_fallback():
    assert parse_label("I cannot determine the quality.") == "accepted"
    assert parse_label("") == "accepted"

def test_parse_label_multiple_matches_returns_first():
    result = parse_label("this is blur but also accepted")
    assert result == "blur"

def test_detector_vote_shape():
    from pathlib import Path
    from document_validation.vlm_detector import VlmDetector
    from document_validation.validator import ValidationConfig

    detector = VlmDetector.__new__(VlmDetector)
    detector.name = "mock"
    detector._model_key = "mock"
    detector._predict_fn = lambda img: "blur"

    page = np.zeros((100, 100, 3), dtype=np.uint8)
    vote = detector.detect(page, Path("test.pdf"), 1, ValidationConfig())

    assert vote.detector == "mock"
    assert vote.available is True
    assert vote.label == "blur"
    assert vote.issues == ["blur"]
    assert vote.confidence == 1.0
