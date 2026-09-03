"""Cloud-aware merge of S2 NDVI and S1 VH time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .farm_signal import clean_signal_matrix, default_config


def merge_s2_s1_curves(
    ndvi: np.ndarray,
    vh: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    nan_sentinel: float = -9999.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Per date: prefer S2 NDVI when valid; else S1 VH.

    Returns unified normalized curve (for stage shape checks), raw NDVI-filled,
    raw VH-filled, and per-date source tags.
    """
    ndvi = np.asarray(ndvi, dtype=float)
    vh = np.asarray(vh, dtype=float)
    if nan_sentinel is not None:
        ndvi = np.where(ndvi == nan_sentinel, np.nan, ndvi)
        vh = np.where(vh == nan_sentinel, np.nan, vh)

    n = len(obs_dates)
    source = np.full(n, "missing", dtype=object)
    unified = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if np.isfinite(ndvi[i]):
            unified[i] = ndvi[i]
            source[i] = "S2"
        elif np.isfinite(vh[i]):
            unified[i] = vh[i]
            source[i] = "S1"
        else:
            source[i] = "missing"

    # Light gap fill on unified for plotting / derivative features
    s = pd.Series(unified).interpolate(limit_direction="both")
    unified_filled = s.to_numpy(dtype=float)

    meta = {
        "S2": int(np.sum(source == "S2")),
        "S1": int(np.sum(source == "S1")),
        "missing": int(np.sum(source == "missing")),
    }
    return unified_filled, ndvi, vh, meta


def clean_farm_curves(
    ndvi: np.ndarray,
    vh: np.ndarray,
    obs_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply farm_heterogeneity cleaning to 1D curves via synthetic single-row matrices."""
    ndvi_cfg = default_config("NDVI")
    vh_cfg = default_config("VH")

    ndvi_df = pd.DataFrame(
        [ndvi],
        columns=[f"NDVI_{d:%Y%m%d}" for d in obs_dates],
    )
    vh_df = pd.DataFrame(
        [vh],
        columns=[f"VH_{d:%Y%m%d}" for d in obs_dates],
    )

    ndvi_cols = list(ndvi_df.columns)
    vh_cols = list(vh_df.columns)
    ndvi_clean = clean_signal_matrix(ndvi_df, ndvi_cols, ndvi_cfg)[0]
    vh_clean = clean_signal_matrix(vh_df, vh_cols, vh_cfg)[0]
    return ndvi_clean, vh_clean


def curve_stage_hint(
    ndvi_clean: np.ndarray,
    vh_clean: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    transplant_date: pd.Timestamp,
    assessment_date: pd.Timestamp | None = None,
) -> str:
    """Rough satellite-only stage hint for validation."""
    if pd.isna(transplant_date):
        return "unknown"

    dates = pd.DatetimeIndex(obs_dates)
    if assessment_date is not None:
        assessment_date = pd.Timestamp(assessment_date)
        if assessment_date < transplant_date:
            return "pre_transplant"
        eligible = dates <= assessment_date
        if not eligible.any():
            return "unknown"
        ref_idx = int(np.where(eligible)[0][-1])
        dat = int((dates[ref_idx] - transplant_date).days)
    else:
        dat = (dates - transplant_date).days
        ref_idx = int(np.nanargmax(dat))
        dat = int(dat[ref_idx])

    ndvi = ndvi_clean[ref_idx] if np.isfinite(ndvi_clean[ref_idx]) else np.nanmean(ndvi_clean)
    if dat < 15:
        return "establishment"
    if dat < 55 and (np.isnan(ndvi) or ndvi < 0.55):
        return "vegetative"
    if dat < 85 and np.isfinite(ndvi) and ndvi >= 0.55:
        return "reproductive"
    if dat < 120:
        return "ripening"
    return "maturity"
