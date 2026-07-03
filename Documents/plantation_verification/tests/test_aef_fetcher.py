import numpy as np
import pytest
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from unittest.mock import patch, MagicMock
from src.aef_fetcher import dequantize, _parcel_hash

def test_dequantize_positive():
    arr = np.array([127], dtype=np.int8)
    result = dequantize(arr)
    # (127/127.5)**2 * sign(127) ≈ 0.9922
    assert result.dtype == np.float32
    assert abs(result[0] - (127 / 127.5) ** 2) < 0.01

def test_dequantize_negative():
    arr = np.array([-127], dtype=np.int8)
    result = dequantize(arr)
    # negative: sign=-1
    assert result[0] < 0
    assert abs(result[0] - (-(127 / 127.5) ** 2)) < 0.01

def test_dequantize_zero():
    arr = np.array([0], dtype=np.int8)
    result = dequantize(arr)
    assert result[0] == 0.0

def test_dequantize_preserves_shape():
    arr = np.zeros((4, 8, 64), dtype=np.int8)
    result = dequantize(arr)
    assert result.shape == (4, 8, 64)

def test_parcel_hash_is_deterministic():
    geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    h1 = _parcel_hash(geojson, [2020, 2021])
    h2 = _parcel_hash(geojson, [2020, 2021])
    assert h1 == h2

def test_parcel_hash_differs_by_year():
    geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    h1 = _parcel_hash(geojson, [2020])
    h2 = _parcel_hash(geojson, [2021])
    assert h1 != h2
