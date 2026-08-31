"""
Signal processing and phenological event detection (from farm_heterogeneity_app).

Kharif defaults: season May 1 – Nov 30; peak search limited to May 1 – Sep 30.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DATE_COL_RE = re.compile(r"(20\d{6})")


@dataclass
class FarmSignalConfig:
    signal_name: str
    output_prefix: str
    valid_min: Optional[float]
    valid_max: Optional[float]
    min_greenup_amplitude: float
    aligned_rmse_threshold: float
    vigor_gap: float
    lower_value_cap: Optional[float] = None
    min_crop_duration_days: int = 60
    emergence_lag_days: int = 10
    greenup_absolute_threshold: Optional[float] = 0.25
    greenup_fraction_of_amplitude: float = 0.25
    pre_peak_drop_threshold: Optional[float] = 0.3
    season_name: str = "Kharif"
    season_start_month: int = 5
    season_start_day: int = 1
    season_end_month: int = 11
    season_end_day: int = 30
    peak_search_end_month: int = 9
    peak_search_end_day: int = 30
    random_state: int = 42


def default_config(signal_type: str) -> FarmSignalConfig:
    if signal_type == "VH":
        return FarmSignalConfig(
            signal_name="VH",
            output_prefix="vh",
            valid_min=-35.0,
            valid_max=5.0,
            min_greenup_amplitude=1.0,
            aligned_rmse_threshold=0.5,
            vigor_gap=1.5,
            lower_value_cap=-22.0,
            greenup_absolute_threshold=None,
            pre_peak_drop_threshold=-20.0,
        )
    return FarmSignalConfig(
        signal_name="NDVI",
        output_prefix="ndvi",
        valid_min=-0.2,
        valid_max=1.0,
        min_greenup_amplitude=0.12,
        aligned_rmse_threshold=0.24,
        vigor_gap=0.16,
        lower_value_cap=None,
        greenup_absolute_threshold=0.25,
        pre_peak_drop_threshold=0.3,
    )


def parse_observation_date(col: str) -> pd.Timestamp:
    match = DATE_COL_RE.search(str(col))
    if not match:
        raise ValueError(f"Column does not contain YYYYMMDD date: {col}")
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="raise")


def date_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if DATE_COL_RE.search(str(col))]
    return sorted(cols, key=parse_observation_date)


def auto_signal_columns(df: pd.DataFrame, signal_type: str) -> list[str]:
    cols = date_columns(df)
    if signal_type == "VH":
        vh_cols = [col for col in cols if "VH" in str(col) or "vh" in str(col)]
        return vh_cols if vh_cols else cols
    ndvi_cols = [col for col in cols if "NDVI" in str(col).upper()]
    return ndvi_cols if ndvi_cols else cols


def to_ordinal_days(dates: pd.DatetimeIndex | pd.Series | list[pd.Timestamp]) -> np.ndarray:
    return pd.to_datetime(dates).map(pd.Timestamp.toordinal).to_numpy(dtype=float)


def apply_named_season(cfg: FarmSignalConfig, season_name: str) -> None:
    seasons = {
        "Full season": None,
        "Kharif": (5, 1, 11, 30),
        "Rabi": (10, 1, 3, 31),
        "Zaid": (3, 1, 6, 30),
    }
    cfg.season_name = season_name
    if seasons[season_name] is None:
        return
    (
        cfg.season_start_month,
        cfg.season_start_day,
        cfg.season_end_month,
        cfg.season_end_day,
    ) = seasons[season_name]


def season_date_window(obs_dates: pd.DatetimeIndex, cfg: FarmSignalConfig) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if cfg.season_name == "Full season":
        return None

    dates = pd.DatetimeIndex(obs_dates)
    candidates = []
    for year in range(int(dates.min().year) - 1, int(dates.max().year) + 2):
        start = pd.Timestamp(year=year, month=cfg.season_start_month, day=cfg.season_start_day)
        end_year = (
            year
            if (cfg.season_end_month, cfg.season_end_day) >= (cfg.season_start_month, cfg.season_start_day)
            else year + 1
        )
        end = pd.Timestamp(year=end_year, month=cfg.season_end_month, day=cfg.season_end_day)
        overlap = int(((dates >= start) & (dates <= end)).sum())
        candidates.append((overlap, start, end))

    overlap, start, end = max(candidates, key=lambda item: item[0])
    return (start, end) if overlap > 0 else None


def season_mask(obs_dates: pd.DatetimeIndex, cfg: FarmSignalConfig) -> np.ndarray:
    window = season_date_window(obs_dates, cfg)
    if window is None:
        return np.ones(len(obs_dates), dtype=bool)
    start, end = window
    dates = pd.DatetimeIndex(obs_dates)
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def peak_search_window(obs_dates: pd.DatetimeIndex, cfg: FarmSignalConfig) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """May 1 through peak_search_end (default Sep 30) within the active season year."""
    season_window = season_date_window(obs_dates, cfg)
    if season_window is None:
        return None
    season_start, season_end = season_window
    dates = pd.DatetimeIndex(obs_dates)
    candidates = []
    for year in range(int(dates.min().year) - 1, int(dates.max().year) + 2):
        start = pd.Timestamp(year=year, month=cfg.season_start_month, day=cfg.season_start_day)
        peak_end = pd.Timestamp(year=year, month=cfg.peak_search_end_month, day=cfg.peak_search_end_day)
        if peak_end < start:
            peak_end = pd.Timestamp(year=year + 1, month=cfg.peak_search_end_month, day=cfg.peak_search_end_day)
        peak_end = min(peak_end, season_end)
        overlap = int(((dates >= start) & (dates <= peak_end)).sum())
        candidates.append((overlap, start, peak_end))

    overlap, start, end = max(candidates, key=lambda item: item[0])
    return (start, end) if overlap > 0 else None


def peak_search_mask(obs_dates: pd.DatetimeIndex, cfg: FarmSignalConfig) -> np.ndarray:
    """Mask for peak detection: season start through Sep 30 (not full Nov ripening window)."""
    window = peak_search_window(obs_dates, cfg)
    if window is None:
        return season_mask(obs_dates, cfg)
    start, end = window
    dates = pd.DatetimeIndex(obs_dates)
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def clean_signal_matrix(df: pd.DataFrame, signal_cols: list[str], cfg: FarmSignalConfig) -> np.ndarray:
    raw = df[signal_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    if cfg.valid_min is not None:
        raw[raw < cfg.valid_min] = np.nan
    if cfg.valid_max is not None:
        raw[raw > cfg.valid_max] = np.nan
    if cfg.lower_value_cap is not None:
        raw[raw < cfg.lower_value_cap] = cfg.lower_value_cap

    cleaned = np.empty_like(raw, dtype=float)
    for i, row in enumerate(raw):
        s = pd.Series(row, dtype="float64")

        rolling_med = s.rolling(window=5, center=True, min_periods=1).median()
        residual = (s - rolling_med).abs()
        local_mad = residual.rolling(window=5, center=True, min_periods=1).median()
        spike_mask = residual > np.maximum(0.22, 4.0 * local_mad.fillna(0.0))
        s[spike_mask] = np.nan

        s = s.interpolate(limit_direction="both")
        if s.isna().all():
            cleaned[i] = np.nan
            continue
        s = s.fillna(float(np.nanmedian(s)))

        values = s.to_numpy(dtype=float)
        if values.size >= 5:
            window = min(values.size if values.size % 2 else values.size - 1, 7)
            if window >= 5:
                values = savgol_filter(values, window_length=window, polyorder=2, mode="nearest")

        if cfg.valid_min is not None or cfg.valid_max is not None:
            lo = cfg.valid_min if cfg.valid_min is not None else -np.inf
            hi = cfg.valid_max if cfg.valid_max is not None else np.inf
            values = np.clip(values, lo, hi)
        if cfg.lower_value_cap is not None:
            values = np.maximum(values, cfg.lower_value_cap)
        cleaned[i] = values

    return cleaned


def greenup_threshold_candidates(
    values: np.ndarray,
    baseline_idx: int,
    cfg: FarmSignalConfig,
    baseline: float,
    amplitude: float,
) -> tuple[np.ndarray, str]:
    search = np.arange(len(values)) >= baseline_idx
    if cfg.greenup_absolute_threshold is not None:
        absolute_candidates = np.flatnonzero(search & (values >= cfg.greenup_absolute_threshold))
        if absolute_candidates.size:
            return absolute_candidates, f"absolute_{cfg.greenup_absolute_threshold:g}"

    dynamic_threshold = baseline + cfg.greenup_fraction_of_amplitude * amplitude
    dynamic_candidates = np.flatnonzero(search & (values >= dynamic_threshold))
    if dynamic_candidates.size:
        return dynamic_candidates, f"fraction_{cfg.greenup_fraction_of_amplitude:g}"

    return np.array([], dtype=int), "no_threshold_crossing"


def select_season_greenup_index(
    candidates: np.ndarray,
    dates: pd.DatetimeIndex,
    peak_idx: int,
    in_season: np.ndarray,
    cfg: FarmSignalConfig,
) -> int | None:
    if candidates.size == 0:
        return None

    candidate_dates = dates[candidates]
    duration_days = (dates[peak_idx] - candidate_dates).days
    season_candidates = candidates[in_season[candidates] & (duration_days >= cfg.min_crop_duration_days)]
    if season_candidates.size:
        return int(season_candidates[0])

    season_candidates = candidates[in_season[candidates]]
    if season_candidates.size:
        return int(season_candidates[0])

    return int(candidates[0])


def has_pre_peak_drop(values: np.ndarray, peak_idx: int, cfg: FarmSignalConfig) -> bool:
    if cfg.pre_peak_drop_threshold is None:
        return True
    pre_peak = values[:peak_idx]
    return bool(np.any(np.isfinite(pre_peak) & (pre_peak < cfg.pre_peak_drop_threshold)))


def local_peak_indices(values: np.ndarray, candidate_indices: np.ndarray) -> np.ndarray:
    peaks = []
    n = len(values)
    for idx in candidate_indices:
        if not np.isfinite(values[idx]):
            continue
        left = values[idx - 1] if idx > 0 and np.isfinite(values[idx - 1]) else -np.inf
        right = values[idx + 1] if idx < n - 1 and np.isfinite(values[idx + 1]) else -np.inf
        if values[idx] >= left and values[idx] >= right:
            peaks.append(int(idx))
    return np.array(peaks, dtype=int)


def select_peak_index(values: np.ndarray, in_season: np.ndarray, cfg: FarmSignalConfig) -> tuple[int, str]:
    season_indices = np.flatnonzero(in_season)
    if season_indices.size and np.isfinite(values[season_indices]).any():
        search_indices = season_indices[np.isfinite(values[season_indices])]
        peak_method = "season_peak"
    else:
        search_indices = np.flatnonzero(np.isfinite(values))
        peak_method = "full_series_peak"

    if search_indices.size == 0:
        return 0, f"{peak_method}_fallback"

    first_peak = int(search_indices[np.nanargmax(values[search_indices])])
    if has_pre_peak_drop(values, first_peak, cfg):
        return first_peak, peak_method

    later_indices = np.flatnonzero(np.isfinite(values) & (np.arange(len(values)) > first_peak))
    later_peaks = local_peak_indices(values, later_indices)
    fallback_peak = int(later_peaks[0]) if later_peaks.size else first_peak
    for peak_idx in later_peaks:
        if has_pre_peak_drop(values, int(peak_idx), cfg):
            return int(peak_idx), f"{peak_method}_next_peak_pre_drop"

    if fallback_peak != first_peak:
        return fallback_peak, f"{peak_method}_next_peak_no_pre_drop"
    return first_peak, f"{peak_method}_no_pre_drop"


def phenology_features(
    curve: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    cfg: FarmSignalConfig,
) -> dict:
    """Phenology metrics for a single farm-level curve (1D)."""
    values = np.asarray(curve, dtype=float)
    day_axis = to_ordinal_days(obs_dates)
    start_day = day_axis[0]
    dates = pd.DatetimeIndex(obs_dates)
    in_season = season_mask(dates, cfg)
    peak_mask = peak_search_mask(dates, cfg)
    season_indices = np.flatnonzero(in_season)

    peak_idx, peak_method = select_peak_index(values, peak_mask, cfg)
    if season_indices.size:
        area_values = values[season_indices]
        area_days = day_axis[season_indices] - start_day
    else:
        area_values = values
        area_days = day_axis - start_day

    peak_day = day_axis[peak_idx] - start_day
    peak_value = float(values[peak_idx])
    baseline = float(np.nanpercentile(values, 20))
    amplitude = float(peak_value - baseline)
    pre_peak = values[: peak_idx + 1]
    baseline_idx = int(np.nanargmin(pre_peak)) if len(pre_peak) else 0
    candidates, _ = greenup_threshold_candidates(pre_peak, baseline_idx, cfg, baseline, amplitude)
    greenup_idx = select_season_greenup_index(candidates, dates[: peak_idx + 1], peak_idx, in_season, cfg)
    if greenup_idx is None:
        greenup_idx = peak_idx
    greenup_day = float(day_axis[greenup_idx] - start_day)
    area = float(np.trapezoid(np.nan_to_num(area_values, nan=np.nanmedian(values)), area_days))
    slope = np.gradient(values, day_axis)

    return {
        "greenup_day_abs": greenup_day,
        "peak_day_abs": float(peak_day),
        "season_peak_used": bool(peak_mask[peak_idx]),
        "peak_method": peak_method,
        f"peak_{cfg.output_prefix}": peak_value,
        f"baseline_{cfg.output_prefix}": baseline,
        f"amplitude_{cfg.output_prefix}": amplitude,
        "season_integral": area,
        f"mean_{cfg.output_prefix}": float(np.nanmean(values)),
        "max_greenup_slope": float(np.nanmax(slope)),
        "min_senescence_slope": float(np.nanmin(slope)),
    }


def estimate_greenup_date(curve: np.ndarray, obs_dates: pd.DatetimeIndex, cfg: FarmSignalConfig) -> dict:
    """Green-up date from NDVI-style curve (emergence proxy)."""
    values = np.asarray(curve, dtype=float)
    if np.isfinite(values).sum() < 4:
        return {
            "greenup_date": pd.NaT,
            "confidence": "low",
            "method": "insufficient_curve",
        }

    if len(values) >= 5:
        window = min(len(values) if len(values) % 2 else len(values) - 1, 7)
        values = savgol_filter(values, window_length=window, polyorder=2, mode="nearest")

    dates = pd.DatetimeIndex(obs_dates)
    in_season = season_mask(dates, cfg)
    peak_mask = peak_search_mask(dates, cfg)
    peak_idx, peak_method = select_peak_index(values, peak_mask, cfg)
    pre_peak = values[: peak_idx + 1]
    pre_dates = dates[: peak_idx + 1]

    baseline_idx = int(np.nanargmin(pre_peak)) if len(pre_peak) else 0
    baseline = float(pre_peak[baseline_idx])
    peak = float(values[peak_idx])
    amplitude = peak - baseline

    if amplitude < cfg.min_greenup_amplitude:
        greenup_idx = baseline_idx
        confidence = "low"
        method = "low_amplitude_curve"
    else:
        candidates, threshold_method = greenup_threshold_candidates(pre_peak, baseline_idx, cfg, baseline, amplitude)
        selected = select_season_greenup_index(candidates, pre_dates, peak_idx, in_season, cfg)
        if selected is not None:
            greenup_idx = selected
            method = f"{peak_method}_{threshold_method}"
        else:
            greenup_idx = baseline_idx
            method = f"{peak_method}_baseline_fallback"
        confidence = "high" if greenup_idx > baseline_idx and peak_idx > greenup_idx else "medium"

    greenup_date = pre_dates[greenup_idx]
    if greenup_idx == 0:
        confidence = "low"
        method = "greenup_before_first_observation"

    return {
        "greenup_date": greenup_date,
        "confidence": confidence,
        "method": method,
        f"baseline_{cfg.output_prefix}": baseline,
        f"peak_{cfg.output_prefix}": peak,
        f"amplitude_{cfg.output_prefix}": float(amplitude),
    }


def estimate_transplant_from_vh(
    vh_curve: np.ndarray,
    obs_dates: pd.DatetimeIndex,
    cfg: FarmSignalConfig,
    search_days: int = 75,
) -> dict:
    """
  Transplant proxy from S1 VH minimum (flooded transplant dip) within early season window.
    """
    values = np.asarray(vh_curve, dtype=float)
    dates = pd.DatetimeIndex(obs_dates)
    if np.isfinite(values).sum() < 3:
        return {"transplant_date": pd.NaT, "confidence": "low", "method": "insufficient_vh"}

    window = season_date_window(dates, cfg)
    if window is None:
        search_start = dates.min()
    else:
        search_start = window[0]
    search_end = search_start + pd.Timedelta(days=search_days)

    mask = (dates >= search_start) & (dates <= search_end) & np.isfinite(values)
    if not mask.any():
        return {"transplant_date": pd.NaT, "confidence": "low", "method": "no_vh_in_search_window"}

    idx = int(np.nanargmin(values[mask]))
    valid_indices = np.flatnonzero(mask)
    dip_idx = valid_indices[idx]
    return {
        "transplant_date": dates[dip_idx],
        "confidence": "medium",
        "method": "vh_flood_dip",
        f"vh_dip_{cfg.output_prefix}": float(values[dip_idx]),
    }


def estimate_sowing_date_from_greenup(greenup_date: pd.Timestamp, cfg: FarmSignalConfig) -> pd.Timestamp:
    if pd.isna(greenup_date):
        return pd.NaT
    return greenup_date - pd.Timedelta(days=cfg.emergence_lag_days)

