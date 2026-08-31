"""Orchestrate STAC download, signal processing, GDD, and stage classification."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import PhenologyConfig
from .farm_signal import (
    apply_named_season,
    default_config,
    estimate_greenup_date,
    estimate_sowing_date_from_greenup,
    estimate_transplant_from_vh,
    phenology_features,
)
from .gdd import (
    cumulative_gdd_on_date,
    estimate_harvest_date,
    fetch_era5_land_tmin_tmax,
    gdd_table_from_transplant,
)
from .heterogeneity import assess_pixel_heterogeneity
from .signal_merge import clean_farm_curves, curve_stage_hint, merge_s2_s1_curves
from .stages import build_stage_result
from .stac_download import download_timeseries
from .timeseries_adapter import (
    csv_wide_from_stac_csv,
    extract_curve_from_wide,
    netcdf_to_wide,
)


def polygon_centroid(config: PhenologyConfig) -> tuple[float, float]:
    gdf = gpd.read_file(config.polygon_path)
    if config.farm_id and config.farm_id_col in gdf.columns:
        gdf = gdf[gdf[config.farm_id_col].astype(str) == str(config.farm_id)]
    geom = gdf.to_crs(4326).geometry.unary_union
    c = geom.centroid
    return float(c.y), float(c.x)


def _parse_optional_date(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT
    return pd.Timestamp(value)


def _signal_configs(config: PhenologyConfig) -> tuple:
    ndvi_cfg = default_config("NDVI")
    vh_cfg = default_config("VH")
    apply_named_season(ndvi_cfg, config.season_name)
    apply_named_season(vh_cfg, config.season_name)
    peak_end = pd.Timestamp(config.peak_search_end)
    ndvi_cfg.peak_search_end_month = peak_end.month
    ndvi_cfg.peak_search_end_day = peak_end.day
    vh_cfg.peak_search_end_month = peak_end.month
    vh_cfg.peak_search_end_day = peak_end.day
    return ndvi_cfg, vh_cfg


def _estimate_transplant(
    ndvi_clean: np.ndarray,
    vh_clean: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    ndvi_cfg,
    vh_cfg,
) -> dict:
    vh_est = estimate_transplant_from_vh(vh_clean, obs_dates, vh_cfg)
    ndvi_est = estimate_greenup_date(ndvi_clean, obs_dates, ndvi_cfg)
    greenup = ndvi_est["greenup_date"]
    ndvi_transplant = greenup - pd.Timedelta(days=ndvi_cfg.emergence_lag_days)

    candidates = []
    if pd.notna(vh_est.get("transplant_date")):
        candidates.append(("vh_flood_dip", vh_est["transplant_date"], 0.55))
    if pd.notna(ndvi_transplant):
        conf = 0.7 if ndvi_est.get("confidence") == "high" else 0.5
        candidates.append(("ndvi_greenup", ndvi_transplant, conf))

    if not candidates:
        return {
            "transplant_date": pd.NaT,
            "method": "no_estimate",
            "confidence": 0.0,
            "vh_estimate": vh_est,
            "ndvi_estimate": ndvi_est,
        }

    best = max(candidates, key=lambda x: x[2])
    if len(candidates) == 2:
        d0, d1 = candidates[0][1], candidates[1][1]
        if abs((d0 - d1).days) <= 14:
            best = ("fused_vh_ndvi", min(d0, d1), 0.85)

    return {
        "transplant_date": best[1],
        "method": best[0],
        "confidence": best[2],
        "vh_estimate": vh_est,
        "ndvi_estimate": ndvi_est,
    }


def _resolve_transplant(config: PhenologyConfig, estimate: dict) -> dict:
    provided = _parse_optional_date(config.transplant_date)
    est = estimate.get("transplant_date", pd.NaT)
    flags: list[str] = []

    if config.date_confidence == "trusted" and pd.notna(provided):
        used = provided
        source = "provided"
        if pd.notna(est) and abs((provided - est).days) > config.date_mismatch_days:
            flags.append("provided_vs_estimated_transplant_mismatch")
            flags.append(f"delta_days_{abs((provided - est).days)}")
    elif pd.notna(est):
        used = est
        source = estimate.get("method", "estimated")
    elif pd.notna(provided):
        used = provided
        source = "provided_low_confidence"
        flags.append("used_provided_without_validation")
    else:
        used = pd.NaT
        source = "none"

    sowing_provided = _parse_optional_date(config.sowing_date)
    sowing_est = pd.NaT
    if pd.notna(used):
        sowing_est = used - pd.Timedelta(days=config.nursery_to_transplant_days)
    elif pd.notna(estimate.get("ndvi_estimate", {}).get("greenup_date")):
        sowing_est = estimate_sowing_date_from_greenup(
            estimate["ndvi_estimate"]["greenup_date"],
            default_config("NDVI"),
        )

    return {
        "provided_transplant": provided if pd.notna(provided) else None,
        "estimated_transplant": est if pd.notna(est) else None,
        "used_transplant": used if pd.notna(used) else None,
        "transplant_source": source,
        "provided_sowing": sowing_provided if pd.notna(sowing_provided) else None,
        "estimated_sowing": sowing_est if pd.notna(sowing_est) else None,
        "anomaly_flags": flags,
        "estimate_detail": estimate,
    }


def load_timeseries_from_download(
    download_info: dict,
    uid: str,
    farm_id_col: str = "farm_id",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    nan_fill = -9999.0

    if download_info.get("s2_nc"):
        s2_wide = netcdf_to_wide(download_info["s2_nc"], uid=uid)
        ndvi, ndvi_dates = extract_curve_from_wide(s2_wide, "NDVI", nan_fill)
    elif download_info.get("s2_csv"):
        s2_wide = csv_wide_from_stac_csv(download_info["s2_csv"], farm_id_col, "NDVI", uid)
        ndvi, ndvi_dates = extract_curve_from_wide(s2_wide, "NDVI", nan_fill)
    else:
        raise FileNotFoundError("No S2 NetCDF or CSV in download output")

    if download_info.get("s1_nc"):
        s1_wide = netcdf_to_wide(download_info["s1_nc"], uid=uid)
        vh, vh_dates = extract_curve_from_wide(s1_wide, "VH", nan_fill)
    elif download_info.get("s1_csv"):
        s1_wide = csv_wide_from_stac_csv(download_info["s1_csv"], farm_id_col, "VH", uid)
        vh, vh_dates = extract_curve_from_wide(s1_wide, "VH", nan_fill)
    else:
        vh = np.full(len(ndvi_dates), np.nan)
        vh_dates = ndvi_dates

    all_dates = pd.DatetimeIndex(sorted(set(ndvi_dates) | set(vh_dates)))
    ndvi_map = {d: v for d, v in zip(ndvi_dates, ndvi)}
    vh_map = {d: v for d, v in zip(vh_dates, vh)}
    ndvi_aligned = np.array([ndvi_map.get(d, np.nan) for d in all_dates], dtype=float)
    vh_aligned = np.array([vh_map.get(d, np.nan) for d in all_dates], dtype=float)
    return ndvi_aligned, vh_aligned, all_dates


def _weather_for_run(
    config: PhenologyConfig,
    transplant: pd.Timestamp,
    assessment: pd.Timestamp,
    weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if weather_df is not None:
        df = weather_df.copy()
        df.index = pd.to_datetime(df.index)
        return df
    lat, lon = polygon_centroid(config)
    return fetch_era5_land_tmin_tmax(
        lat,
        lon,
        transplant,
        assessment + pd.Timedelta(days=5),
        cache_dir=config.era5_cache_dir,
        time_zone=config.era5_time_zone,
    )


def run_phenology_core(
    config: PhenologyConfig,
    uid: str,
    ndvi_raw: np.ndarray,
    vh_raw: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    assessment: pd.Timestamp,
    weather_df: pd.DataFrame | None = None,
    download_info: dict | None = None,
) -> dict:
    ndvi_cfg, vh_cfg = _signal_configs(config)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unified, ndvi_raw, vh_raw, source_meta = merge_s2_s1_curves(ndvi_raw, vh_raw, obs_dates)
    ndvi_clean, vh_clean = clean_farm_curves(ndvi_raw, vh_raw, obs_dates)

    transplant_est = _estimate_transplant(ndvi_clean, vh_clean, obs_dates, ndvi_cfg, vh_cfg)
    date_info = _resolve_transplant(config, transplant_est)
    transplant = date_info["used_transplant"]
    if transplant is None or pd.isna(transplant):
        raise ValueError("Could not determine transplant date — provide transplant_date or check satellite coverage")

    transplant = pd.Timestamp(transplant)
    weather = _weather_for_run(config, transplant, assessment, weather_df)
    gdd_df = gdd_table_from_transplant(weather, transplant, config.t_base_c)
    cum_gdd = cumulative_gdd_on_date(gdd_df, assessment)
    harvest_est = estimate_harvest_date(
        gdd_df, config.target_total_gdd, pd.Timestamp(config.download_end)
    )

    sat_hint = curve_stage_hint(ndvi_clean, vh_clean, obs_dates, transplant)
    stage = build_stage_result(
        cum_gdd,
        transplant,
        assessment,
        config.gdd_stages,
        sat_hint,
        source_meta,
    )

    heterogeneity: dict = {}
    if download_info and download_info.get("s2_csv"):
        try:
            heterogeneity = assess_pixel_heterogeneity(
                download_info["s2_csv"],
                config.farm_id_col,
                farm_id=uid,
            )
        except Exception as exc:
            heterogeneity = {"error": str(exc)}

    all_flags = list(date_info.get("anomaly_flags", [])) + stage.anomaly_flags
    if heterogeneity.get("heterogeneous"):
        all_flags.append("pixel_heterogeneity_detected")

    ndvi_features = phenology_features(ndvi_clean, obs_dates, ndvi_cfg)
    vh_features = phenology_features(vh_clean, obs_dates, vh_cfg)
    lat, lon = polygon_centroid(config)

    result = {
        "farm_id": uid,
        "assessment_date": assessment.strftime("%Y-%m-%d"),
        "mode": config.date_confidence,
        "provided_transplant": date_info.get("provided_transplant"),
        "estimated_transplant": (
            date_info["estimated_transplant"].strftime("%Y-%m-%d")
            if date_info.get("estimated_transplant")
            else None
        ),
        "used_transplant": transplant.strftime("%Y-%m-%d"),
        "transplant_source": date_info["transplant_source"],
        "provided_sowing": date_info.get("provided_sowing"),
        "estimated_sowing": (
            date_info["estimated_sowing"].strftime("%Y-%m-%d")
            if date_info.get("estimated_sowing")
            else None
        ),
        "cumulative_gdd": round(cum_gdd, 1),
        "days_after_transplant": stage.days_after_transplant,
        "current_stage": stage.stage_name,
        "stage_confidence": round(stage.confidence, 3),
        "gdd_stage_margin": round(stage.gdd_stage_margin, 1),
        "satellite_stage_hint": stage.satellite_stage_hint,
        "estimated_harvest_date": harvest_est.strftime("%Y-%m-%d"),
        "data_source_mix": source_meta,
        "pixel_heterogeneity": heterogeneity,
        "anomaly_flags": all_flags,
        "phenology_features": {"ndvi": ndvi_features, "vh": vh_features},
        "era5_centroid": {"lat": lat, "lon": lon},
        "download_run_dir": str(download_info.get("run_dir", "")) if download_info else "",
    }

    json_path = out_dir / f"phenology_{uid}.json"
    csv_path = out_dir / f"phenology_timeseries_{uid}.csv"

    ts_df = pd.DataFrame(
        {
            "date": obs_dates,
            "ndvi": ndvi_clean,
            "vh": vh_clean,
            "unified": unified,
            "cumulative_gdd": [
                cumulative_gdd_on_date(gdd_df, d) if d >= transplant else 0.0 for d in obs_dates
            ],
        }
    )
    ts_df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    result["output_json"] = str(json_path)
    result["output_timeseries_csv"] = str(csv_path)
    return result


def run_phenology_from_curves(
    config: PhenologyConfig,
    ndvi: np.ndarray,
    vh: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    weather_df: pd.DataFrame | None = None,
    assessment_date: str | None = None,
) -> dict:
    """Run phenology on pre-built curves (tests / custom inputs). Skips STAC and ERA5 if weather_df given."""
    config.validate()
    uid = config.farm_id or "farm"
    assessment = pd.Timestamp(assessment_date or config.download_end)
    return run_phenology_core(
        config,
        uid,
        np.asarray(ndvi, dtype=float),
        np.asarray(vh, dtype=float),
        pd.DatetimeIndex(obs_dates),
        assessment,
        weather_df=weather_df,
    )


def run_phenology(
    config: PhenologyConfig,
    assessment_date: str | None = None,
    force_download: bool = False,
) -> dict:
    config.validate()
    assessment = pd.Timestamp(assessment_date or config.download_end)

    download_info: dict | None = None
    uid = config.farm_id or "farm"

    if config.download_data:
        download_info = download_timeseries(config, force=force_download)
        uid = download_info["uid"]
        ndvi_raw, vh_raw, obs_dates = load_timeseries_from_download(
            download_info, uid, config.farm_id_col
        )
    else:
        cache_root = Path(config.timeseries_cache_dir or Path(config.output_dir) / "stac_download")
        from .stac_download import find_cached_timeseries

        cached = find_cached_timeseries(cache_root, config.download_start, config.download_end)
        if cached is None:
            raise FileNotFoundError(
                f"No cached STAC data under {cache_root} for "
                f"{config.download_start}_{config.download_end}. Run with download_data=True first."
            )
        download_info = {
            "run_dir": cached["run_dir"],
            "uid": uid,
            "s1_nc": cached.get("s1_nc"),
            "s2_nc": cached.get("s2_nc"),
            "s1_csv": cached.get("s1_csv"),
            "s2_csv": cached.get("s2_csv"),
            "from_cache": True,
        }
        ndvi_raw, vh_raw, obs_dates = load_timeseries_from_download(
            download_info, uid, config.farm_id_col
        )

    return run_phenology_core(
        config,
        uid,
        ndvi_raw,
        vh_raw,
        obs_dates,
        assessment,
        download_info=download_info,
    )
