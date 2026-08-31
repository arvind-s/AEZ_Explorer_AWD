"""Offline unit tests — no STAC / CDS network required."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenology_pipeline.config import DEFAULT_GDD_STAGES, PhenologyConfig
from phenology_pipeline.farm_signal import apply_named_season, default_config, peak_search_mask
from phenology_pipeline.gdd import compute_daily_gdd
from phenology_pipeline.heterogeneity import assess_pixel_heterogeneity
from phenology_pipeline.phenology_engine import run_phenology_from_curves
from phenology_pipeline.signal_merge import merge_s2_s1_curves
from phenology_pipeline.stages import classify_gdd_stage
from phenology_pipeline.stac_download import find_cached_timeseries
from phenology_pipeline.timeseries_adapter import csv_wide_from_stac_csv


def test_kharif_peak_search_may_to_sep():
    cfg = default_config("NDVI")
    apply_named_season(cfg, "Kharif")
    dates = pd.date_range("2025-05-01", "2025-11-15", freq="12D")
    pm = peak_search_mask(dates, cfg)
    assert pm[pd.DatetimeIndex(dates) <= "2025-09-30"].all()
    assert not pm[pd.DatetimeIndex(dates) > "2025-10-01"].any()


def test_gdd_formula():
    gdd = compute_daily_gdd(np.array([22.0, 18.0]), np.array([32.0, 28.0]), t_base=10.0)
    assert gdd[0] == 17.0
    assert gdd[1] == 13.0


def test_gdd_stage_reproductive():
    stage, _, margin = classify_gdd_stage(900.0, DEFAULT_GDD_STAGES)
    assert stage == "reproductive"
    assert margin > 0


def test_merge_prefers_s2():
    dates = pd.date_range("2025-06-01", periods=4, freq="12D")
    ndvi = np.array([0.2, 0.4, 0.6, 0.5])
    vh = np.array([-18.0, -14.0, -10.0, -11.0])
    ndvi[1] = np.nan
    unified, _, _, meta = merge_s2_s1_curves(ndvi, vh, dates)
    assert meta["S2"] == 3
    assert meta["S1"] == 1
    assert np.isfinite(unified[1])


def test_csv_wide_iso_columns():
    df = pd.DataFrame(
        {
            "farm_id": ["F1"],
            "2025-06-01": [0.25],
            "2025-06-13": [0.45],
        }
    )
    wide = csv_wide_from_stac_csv(df, "farm_id", "NDVI", uid="F1")
    assert "NDVI_20250601" in wide.columns
    assert float(wide["NDVI_20250613"].iloc[0]) == 0.45


def test_pixel_heterogeneity_homogeneous():
    cols = {
        "farm_id": ["F1", "F1"],
        "NDVI_20250601": [0.2, 0.21],
        "NDVI_20250615": [0.35, 0.34],
        "NDVI_20250701": [0.55, 0.56],
        "NDVI_20250715": [0.72, 0.71],
        "NDVI_20250801": [0.68, 0.69],
    }
    df = pd.DataFrame(cols)
    out = assess_pixel_heterogeneity(df, "farm_id", farm_id="F1", sowing_gap_threshold_days=20)
    assert out["n_pixels"] == 2
    assert out["heterogeneous"] is False


def test_run_phenology_from_curves_trusted():
    dates = pd.date_range("2025-05-01", "2025-10-15", freq="12D")
    n = len(dates)
    t = np.arange(n)
    ndvi = 0.15 + 0.6 * (1 - np.exp(-t / 5.0))
    vh = -18.0 + 8.0 * (1 - np.exp(-t / 4.0))

    transplant = pd.Timestamp("2025-06-15")
    weather_idx = pd.date_range(transplant, "2025-10-20", freq="D")
    weather = pd.DataFrame(
        {"tmin_c": 22.0, "tmax_c": 32.0, "tmean_c": 27.0},
        index=weather_idx,
    )

    with tempfile.TemporaryDirectory() as tmp:
        poly = Path(tmp) / "farm.geojson"
        poly.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"farm_id": "T1"},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [79.69, 11.74],
                                        [79.691, 11.74],
                                        [79.691, 11.741],
                                        [79.69, 11.741],
                                        [79.69, 11.74],
                                    ]
                                ],
                            },
                        }
                    ],
                }
            )
        )
        cfg = PhenologyConfig(
            polygon_path=str(poly),
            farm_id="T1",
            transplant_date="2025-06-15",
            date_confidence="trusted",
            download_data=False,
            output_dir=str(Path(tmp) / "out"),
            download_start="2025-05-01",
            download_end="2025-11-30",
        )
        result = run_phenology_from_curves(
            cfg,
            ndvi=ndvi,
            vh=vh,
            obs_dates=dates,
            weather_df=weather,
            assessment_date="2025-09-01",
        )
        assert result["current_stage"] in {
            "establishment",
            "vegetative",
            "panicle_initiation",
            "reproductive",
            "ripening",
            "maturity",
        }
        assert result["cumulative_gdd"] > 200
        assert Path(result["output_json"]).exists()


def test_find_cached_timeseries_missing():
    with tempfile.TemporaryDirectory() as tmp:
        assert find_cached_timeseries(Path(tmp), "2025-05-01", "2025-11-30") is None
