"""Per-pixel heterogeneity within a farm polygon (farm_heterogeneity-style)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .farm_signal import (
    DATE_COL_RE,
    apply_named_season,
    default_config,
    estimate_greenup_date,
    estimate_sowing_date_from_greenup,
    parse_observation_date,
)

NAN_SENTINEL = -9999.0


def _value_columns(df: pd.DataFrame, uid_col: str) -> list[str]:
    cols = [c for c in df.columns if c != uid_col and DATE_COL_RE.search(str(c))]
    return sorted(cols, key=parse_observation_date)


def assess_pixel_heterogeneity(
    pixel_csv: str | pd.DataFrame,
    uid_col: str,
    farm_id: str | None = None,
    signal_type: str = "NDVI",
    sowing_gap_threshold_days: int = 20,
) -> dict:
    """
    Estimate green-up per pixel; flag heterogeneous farms when sowing dates diverge.

    Expects STAC pixel-level CSV: uid_col + wide date columns.
    """
    df = pd.read_csv(pixel_csv) if isinstance(pixel_csv, str) else pixel_csv.copy()
    if uid_col not in df.columns:
        raise ValueError(f"uid column '{uid_col}' missing from pixel CSV")

    if farm_id is not None:
        df = df[df[uid_col].astype(str) == str(farm_id)]
    if df.empty:
        return {
            "n_pixels": 0,
            "heterogeneous": False,
            "reason": "no_pixels",
        }

    value_cols = _value_columns(df, uid_col)
    if not value_cols:
        return {
            "n_pixels": len(df),
            "heterogeneous": False,
            "reason": "no_date_columns",
        }

    obs_dates = pd.DatetimeIndex([parse_observation_date(c) for c in value_cols])
    cfg = default_config("NDVI" if signal_type.upper() == "NDVI" else "VH")
    apply_named_season(cfg, "Kharif")

    sowing_dates: list[pd.Timestamp] = []
    methods: list[str] = []
    for _, row in df.iterrows():
        values = row[value_cols].to_numpy(dtype=float)
        values = np.where(values == NAN_SENTINEL, np.nan, values)
        greenup = estimate_greenup_date(values, obs_dates, cfg)
        sow = estimate_sowing_date_from_greenup(greenup["greenup_date"], cfg)
        if pd.notna(sow):
            sowing_dates.append(sow)
            methods.append(greenup.get("method", ""))

    if len(sowing_dates) < 2:
        return {
            "n_pixels": len(df),
            "n_valid_sowing_estimates": len(sowing_dates),
            "heterogeneous": False,
            "reason": "insufficient_pixel_estimates",
            "sowing_gap_days": None,
        }

    sowing_series = pd.Series(sowing_dates)
    gap = int((sowing_series.max() - sowing_series.min()).days)
    heterogeneous = gap >= sowing_gap_threshold_days

    return {
        "n_pixels": len(df),
        "n_valid_sowing_estimates": len(sowing_dates),
        "heterogeneous": heterogeneous,
        "sowing_gap_days": gap,
        "median_sowing_date": sowing_series.median().strftime("%Y-%m-%d"),
        "earliest_sowing_date": sowing_series.min().strftime("%Y-%m-%d"),
        "latest_sowing_date": sowing_series.max().strftime("%Y-%m-%d"),
        "reason": "sowing_aligned_gap" if heterogeneous else "homogeneous_sowing_window",
        "threshold_days": sowing_gap_threshold_days,
    }
