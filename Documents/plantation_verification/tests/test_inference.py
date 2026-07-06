import numpy as np
import pytest
import xarray as xr
import rasterio
from pathlib import Path
from unittest.mock import MagicMock
from src.inference import run_inference

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_ds(years=(2020, 2021), h=8, w=8):
    rng = np.random.default_rng(0)
    data = rng.uniform(-1, 1, (len(years), h, w, 64)).astype(np.float32)
    return xr.Dataset(
        {"embeddings": (["year", "y", "x", "band"], data)},
        coords={"year": list(years), "band": BAND_COLS},
        attrs={"crs": "EPSG:4326", "transform": [0.001, 0, 10.0, 0, -0.001, 20.0]},
    )

def _make_model():
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.ones((len(X), 7)) / 7
    return model

def test_run_inference_returns_paths_per_year(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    assert set(result.keys()) == {2020, 2021}
    for year in (2020, 2021):
        assert "lulc" in result[year]
        assert "tree_prob" in result[year]

def test_lulc_geotiff_is_uint8(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["lulc"]) as src:
        assert src.dtypes[0] == "uint8"
        assert src.count == 1

def test_tree_prob_geotiff_is_float32(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["tree_prob"]) as src:
        assert src.dtypes[0] == "float32"

def test_lulc_values_in_valid_range(tmp_path):
    ds = _make_ds()
    # Model always predicts class 1 (trees)
    model = MagicMock()
    probs = np.zeros((64, 7), dtype=np.float32)
    probs[:, 1] = 1.0
    model.predict_proba.return_value = probs
    class_map = {i: str(i) for i in range(7)}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["lulc"]) as src:
        data = src.read(1)
        assert set(np.unique(data)).issubset(set(range(7)))
