import numpy as np
import pytest
import json
import rasterio
from pathlib import Path
from src.verdict import detect_plantation_pixels, generate_verdict

def _make_tree_prob_stack(pattern: list[list[float]]) -> np.ndarray:
    """pattern: list of [H*W probs] per year → [Y, H, W]"""
    return np.array(pattern, dtype=np.float32).reshape(len(pattern), 2, 2)

def test_detect_plantation_two_consecutive_years():
    # Pixel (0,0): 0.7, 0.8 in years 1 and 2 → plantation
    # Pixel (0,1): only year 1 high → not plantation (only 1 year)
    stack = np.array([
        [[0.7, 0.4], [0.2, 0.3]],  # year 0
        [[0.8, 0.3], [0.1, 0.2]],  # year 1
        [[0.2, 0.2], [0.1, 0.1]],  # year 2
    ], dtype=np.float32)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == True   # 2 consecutive years above threshold
    assert result[0, 1] == False  # only 1 year above threshold

def test_detect_plantation_requires_consecutive():
    # High in year 0 and year 2, but not year 1 → not consecutive
    stack = np.array([
        [[0.8], [0.1]],
        [[0.2], [0.1]],
        [[0.9], [0.1]],
    ], dtype=np.float32).reshape(3, 1, 2)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == False  # not consecutive

def test_detect_plantation_three_consecutive_counts():
    stack = np.array([
        [[0.8]], [[0.9]], [[0.7]], [[0.2]]
    ], dtype=np.float32).reshape(4, 1, 1)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == True

def test_generate_verdict_structure(tmp_path):
    # Create minimal tree_prob GeoTIFFs
    years = [2020, 2021, 2022]
    H, W = 4, 4
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": H, "width": W, "crs": "EPSG:4326",
               "transform": rasterio.transform.from_bounds(10, 20, 10.01, 20.01, W, H)}

    inference_paths = {}
    for i, year in enumerate(years):
        prob = 0.8 if i >= 1 else 0.1
        data = np.full((1, H, W), prob, dtype=np.float32)
        tp = tmp_path / f"tree_prob_{year}.tif"
        lulc = tmp_path / f"lulc_{year}.tif"
        with rasterio.open(tp, "w", **profile) as dst:
            dst.write(data)
        lulc_profile = {**profile, "dtype": "uint8"}
        label = 1 if i >= 1 else 5
        with rasterio.open(lulc, "w", **lulc_profile) as dst:
            dst.write(np.full((1, H, W), label, dtype=np.uint8))
        inference_paths[year] = {"lulc": lulc, "tree_prob": tp}

    parcel_geojson = {
        "type": "Polygon",
        "coordinates": [[[10, 20], [10.01, 20], [10.01, 20.01], [10, 20.01], [10, 20]]]
    }
    verdict = generate_verdict("parcel_001", inference_paths, parcel_geojson, tmp_path)

    assert "plantation_present" in verdict
    assert "confidence" in verdict
    assert "establishment_year" in verdict
    assert "tree_pixel_fraction_latest" in verdict
    assert "prior_dominant_class" in verdict
    assert "lulc_timeseries" in verdict
    assert isinstance(verdict["plantation_present"], bool)
    assert 0.0 <= verdict["confidence"] <= 1.0

def test_generate_verdict_no_plantation(tmp_path):
    years = [2020, 2021]
    H, W = 4, 4
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": H, "width": W, "crs": "EPSG:4326",
               "transform": rasterio.transform.from_bounds(10, 20, 10.01, 20.01, W, H)}
    inference_paths = {}
    for year in years:
        tp = tmp_path / f"tree_prob_{year}.tif"
        lulc = tmp_path / f"lulc_{year}.tif"
        with rasterio.open(tp, "w", **profile) as dst:
            dst.write(np.full((1, H, W), 0.05, dtype=np.float32))
        lulc_profile = {**profile, "dtype": "uint8"}
        with rasterio.open(lulc, "w", **lulc_profile) as dst:
            dst.write(np.full((1, H, W), 5, dtype=np.uint8))
        inference_paths[year] = {"lulc": lulc, "tree_prob": tp}

    parcel_geojson = {
        "type": "Polygon",
        "coordinates": [[[10, 20], [10.01, 20], [10.01, 20.01], [10, 20.01], [10, 20]]]
    }
    verdict = generate_verdict("parcel_002", inference_paths, parcel_geojson, tmp_path)
    assert verdict["plantation_present"] == False
