from pathlib import Path
import pandas as pd
import pytest
from document_validation.ground_truth import FOLDER_TO_LABEL, load_ground_truth, make_subset

TEST_DATA = Path(__file__).parent.parent / "test_data"

def test_folder_to_label_covers_all_variants():
    expected = {"is_blur", "is_clear", "is_proper_doc", "is_proper_concent_form",
                "is_land_record", "is_full_concent_form", "is_not_document",
                "is_not_full_document(cut)", "is_not_full_doc(cut)", "is_not_proper_doc(cut)"}
    assert expected.issubset(set(FOLDER_TO_LABEL.keys()))

def test_load_ground_truth_returns_dataframe():
    df = load_ground_truth(TEST_DATA)
    assert set(df.columns) >= {"file_path", "doc_type", "state", "expected_label"}
    assert len(df) > 319  # more than the old narrow loader

def test_load_ground_truth_labels_are_valid():
    df = load_ground_truth(TEST_DATA)
    valid = {"accepted", "blur", "cut", "not_document"}
    assert set(df["expected_label"].unique()).issubset(valid)

def test_make_subset_returns_72_files():
    df = load_ground_truth(TEST_DATA)
    subset = make_subset(df, n_per_class=18, seed=42)
    assert len(subset) == 72

def test_make_subset_is_stratified():
    df = load_ground_truth(TEST_DATA)
    subset = make_subset(df, n_per_class=18, seed=42)
    counts = subset["expected_label"].value_counts()
    assert all(counts == 18)

def test_make_subset_is_reproducible():
    df = load_ground_truth(TEST_DATA)
    a = make_subset(df, n_per_class=18, seed=42)
    b = make_subset(df, n_per_class=18, seed=42)
    assert list(a["file_path"]) == list(b["file_path"])
