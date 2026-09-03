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
from phenology_pipeline.gdd import (
    _monthly_request_parts,
    _retrieve_era5_land_daily,
    build_era5_cds_request,
    compute_daily_gdd,
    fetch_openmeteo_era5_land_tmin_tmax,
)
from phenology_pipeline.heterogeneity import assess_pixel_heterogeneity
from phenology_pipeline.phenology_engine import run_phenology_from_curves
from phenology_pipeline.signal_merge import merge_s2_s1_curves
from phenology_pipeline.stages import classify_gdd_stage
from phenology_pipeline.stac_download import find_cached_timeseries
from phenology_pipeline.timeseries_adapter import (
    csv_wide_from_stac_csv,
    extract_curve_from_wide,
    load_farm_pixel_timeseries,
    netcdf_to_wide,
)


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


def test_era5_cds_request_uses_hourly_frequency():
    req = build_era5_cds_request(
        11.74,
        79.69,
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-06-30"),
        "daily_minimum",
        "utc+05:30",
    )
    assert req["frequency"] == "1_hourly"
    assert req["variable"] == ["2m_temperature"]
    assert req["time_zone"] == "utc+05:30"
    assert req["month"] == "06"
    assert "01" in req["day"]
    assert "30" in req["day"]
    assert "31" not in req["day"]


def test_era5_cds_request_spans_multiple_months():
    parts = list(
        _monthly_request_parts(
            pd.Timestamp("2024-06-15"),
            pd.Timestamp("2024-07-10"),
        )
    )
    assert parts == [
        ("2024", "06", [f"{d:02d}" for d in range(15, 31)]),
        ("2024", "07", [f"{d:02d}" for d in range(1, 11)]),
    ]


def test_era5_rejects_inverted_date_range():
    with pytest.raises(ValueError, match="before start"):
        _retrieve_era5_land_daily(
            11.74,
            79.69,
            pd.Timestamp("2024-10-15"),
            pd.Timestamp("2024-10-01"),
            "daily_minimum",
            Path("."),
            "utc+05:30",
        )


def test_openmeteo_parse_response(tmp_path, monkeypatch):
    payload = {
        "daily": {
            "time": ["2024-06-01", "2024-06-02"],
            "temperature_2m_min": [22.0, 21.5],
            "temperature_2m_max": [32.0, 31.0],
        }
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            import json

            return json.dumps(payload).encode()

    monkeypatch.setattr(
        "phenology_pipeline.gdd.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )

    df = fetch_openmeteo_era5_land_tmin_tmax(
        11.74,
        79.69,
        pd.Timestamp("2024-06-01"),
        pd.Timestamp("2024-06-02"),
        cache_dir=tmp_path,
    )
    assert len(df) == 2
    assert float(df["tmin_c"].iloc[0]) == 22.0
    assert float(df["tmax_c"].iloc[1]) == 31.0


def test_prepare_batch_weather_aoi_single_fetch(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import Polygon

    from phenology_pipeline.phenology_engine import prepare_batch_weather

    gdf = gpd.GeoDataFrame(
        {
            "farm_id": ["A", "B"],
            "geometry": [
                Polygon([(79.69, 11.74), (79.70, 11.74), (79.70, 11.75), (79.69, 11.75)]),
                Polygon([(79.71, 11.74), (79.72, 11.74), (79.72, 11.75), (79.71, 11.75)]),
            ],
        },
        crs="EPSG:4326",
    )
    calls = {"n": 0}

    def fake_fetch(*args, **kwargs):
        calls["n"] += 1
        idx = pd.date_range("2024-06-01", periods=3, freq="D")
        return pd.DataFrame({"tmin_c": 22.0, "tmax_c": 32.0, "tmean_c": 27.0}, index=idx)

    monkeypatch.setattr("phenology_pipeline.phenology_engine.fetch_weather_tmin_tmax", fake_fetch)

    cfg = PhenologyConfig(
        polygon_path=str(tmp_path / "x.geojson"),
        farm_id_col="farm_id",
        download_start="2024-06-01",
        download_end="2024-06-30",
        era5_spatial_mode="aoi",
        era5_cache_dir=str(tmp_path),
    )
    out = prepare_batch_weather(cfg, gdf, pd.Timestamp("2024-06-15"))
    assert calls["n"] == 1
    assert set(out.keys()) == {"A", "B"}


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


def test_pre_transplant_assessment():
    dates = pd.date_range("2026-05-01", "2026-11-30", freq="12D")
    n = len(dates)
    ndvi = np.full(n, 0.12)
    vh = np.full(n, -17.0)
    # Late-season green-up / flood signal drives transplant estimate to July.
    late = dates >= "2026-07-01"
    ndvi[late] = 0.15 + 0.55 * (1 - np.exp(-np.arange(late.sum()) / 3.0))
    vh[late] = -17.0 + 6.0 * (1 - np.exp(-np.arange(late.sum()) / 3.0))

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
            date_confidence="uncertain",
            download_data=False,
            output_dir=str(Path(tmp) / "out"),
            download_start="2026-05-01",
            download_end="2026-11-30",
        )
        result = run_phenology_from_curves(
            cfg,
            ndvi=ndvi,
            vh=vh,
            obs_dates=dates,
            assessment_date="2026-05-01",
        )
        assert result["current_stage"] == "pre_transplant"
        assert result["cumulative_gdd"] is None
        assert result["gdd_enabled"] is False
        assert result["days_after_transplant"] < 0
        assert "assessment_before_transplant" in result["anomaly_flags"]
        assert "gdd_not_computed" in result["anomaly_flags"]


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
            compute_gdd=True,
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
        assert result["gdd_enabled"] is True
        assert Path(result["output_json"]).exists()


def test_satellite_only_skips_gdd():
    dates = pd.date_range("2025-05-01", "2025-10-15", freq="12D")
    n = len(dates)
    t = np.arange(n)
    ndvi = 0.15 + 0.6 * (1 - np.exp(-t / 5.0))
    vh = -18.0 + 8.0 * (1 - np.exp(-t / 4.0))

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
            compute_gdd=False,
        )
        result = run_phenology_from_curves(
            cfg,
            ndvi=ndvi,
            vh=vh,
            obs_dates=dates,
            assessment_date="2025-09-01",
        )
        assert result["gdd_enabled"] is False
        assert result["cumulative_gdd"] is None
        assert "gdd_not_computed" in result["anomaly_flags"]
        assert result["current_stage"] in {
            "establishment",
            "vegetative",
            "reproductive",
            "ripening",
            "maturity",
        }


def test_load_farm_pixel_timeseries(tmp_path):
    csv_path = tmp_path / "s2_pixels.csv"
    csv_path.write_text(
        "farm_id,20240601,20240613\n"
        "F1,0.2,0.4\n"
        "F1,0.25,0.45\n"
        "F2,0.1,0.2\n"
    )
    dates, values = load_farm_pixel_timeseries(csv_path, "farm_id", "F1")
    assert len(dates) == 2
    assert values.shape == (2, 2)
    assert float(values[0, 0]) == 0.2
    assert float(values[1, 1]) == 0.45


def test_find_cached_timeseries_missing():
    with tempfile.TemporaryDirectory() as tmp:
        assert find_cached_timeseries(Path(tmp), "2025-05-01", "2025-11-30") is None


def test_netcdf_uid_str_int_match():
    import xarray as xr

    times = pd.date_range("2024-06-01", periods=2, freq="12D")
    ds = xr.Dataset(
        {"NDVI": (["uid", "time"], np.array([[0.3, 0.5]], dtype=np.float32))},
        coords={"uid": np.array([1], dtype=np.int32), "time": times},
    )
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "s2.nc"
        ds.to_netcdf(nc)
        out = netcdf_to_wide(nc, uid="1")
        assert "NDVI_20240601" in out.columns
        assert float(out["NDVI_20240601"].iloc[0]) == pytest.approx(0.3)


def test_netcdf_value_var_uses_indices_attr():
    import xarray as xr

    times = pd.date_range("2024-06-01", periods=2, freq="12D")
    ds = xr.Dataset(
        {"value": (["uid", "time"], np.array([[0.3, 0.5]], dtype=np.float32))},
        coords={"uid": np.array([1], dtype=np.int32), "time": times},
        attrs={"indices": '["NDVI"]', "sensor": "Sentinel-2 L2A"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "s2.nc"
        ds.to_netcdf(nc)
        out = netcdf_to_wide(nc, uid="1")
        ndvi, dates = extract_curve_from_wide(out, "NDVI")
        assert "NDVI_20240601" in out.columns
        assert float(ndvi[0]) == pytest.approx(0.3)
        assert len(dates) == 2


def test_netcdf_value_var_uses_bands_attr():
    import xarray as xr

    times = pd.date_range("2024-06-01", periods=2, freq="6D")
    ds = xr.Dataset(
        {"value": (["uid", "time"], np.array([[-12.0, -10.5]], dtype=np.float32))},
        coords={"uid": np.array([1], dtype=np.int32), "time": times},
        attrs={"bands": '["VH"]', "sensor": "Sentinel-1 GRD"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        nc = Path(tmp) / "s1.nc"
        ds.to_netcdf(nc)
        out = netcdf_to_wide(nc, uid="1")
        vh, dates = extract_curve_from_wide(out, "VH")
        assert "VH_20240601" in out.columns
        assert float(vh[0]) == pytest.approx(-12.0)
        assert len(dates) == 2
