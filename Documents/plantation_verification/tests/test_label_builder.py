import numpy as np
import pandas as pd
import pytest
from src.label_builder import mahalanobis_filter, agreement_filter

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_df(class_ids, embeddings, source="test", year=2020):
    rows = []
    for cid, emb in zip(class_ids, embeddings):
        row = {"source": source, "year": year, "class_id": cid,
               "class_name": str(cid), "lon": 0.0, "lat": 0.0}
        row.update(dict(zip(BAND_COLS, emb)))
        rows.append(row)
    return pd.DataFrame(rows)

def test_mahalanobis_filter_removes_outlier():
    rng = np.random.default_rng(0)
    # 99 normal points + 1 extreme outlier for class 1
    normal = rng.normal(0, 0.1, (99, 64))
    outlier = rng.normal(10, 0.1, (1, 64))  # far from cluster
    embs = np.vstack([normal, outlier])
    class_ids = [1] * 100
    df = _make_df(class_ids, embs)
    filtered = mahalanobis_filter(df, threshold=3.0)
    assert len(filtered) < len(df)
    assert len(filtered) >= 98  # outlier removed

def test_mahalanobis_filter_keeps_inliers():
    rng = np.random.default_rng(1)
    embs = rng.normal(0, 0.1, (50, 64))
    df = _make_df([1] * 50, embs)
    filtered = mahalanobis_filter(df, threshold=3.0)
    assert len(filtered) >= 45  # most kept

def test_agreement_filter_keeps_matching_pixels():
    embs = np.zeros((3, 64))
    df1 = _make_df([1, 1, 5], embs, source="esa_worldcover")
    df1["lon"] = [10.0, 11.0, 12.0]
    df1["lat"] = [20.0, 21.0, 22.0]
    df2 = _make_df([1, 0, 5], embs, source="dynamic_world")
    df2["lon"] = [10.0, 11.0, 12.0]
    df2["lat"] = [20.0, 21.0, 22.0]
    # lon=10/lat=20 → both say class 1 → keep
    # lon=11/lat=21 → disagree (1 vs 0) → drop
    # lon=12/lat=22 → both say class 5 → keep
    result = agreement_filter([df1, df2], tolerance_deg=0.0001)
    assert len(result) == 4  # 2 pixels × 2 sources

def test_agreement_filter_single_source_passes_through():
    embs = np.zeros((5, 64))
    df = _make_df([1, 1, 3, 3, 5], embs)
    result = agreement_filter([df], tolerance_deg=0.0001)
    assert len(result) == 5
