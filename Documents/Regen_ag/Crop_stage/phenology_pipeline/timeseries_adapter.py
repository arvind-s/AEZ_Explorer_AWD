"""Convert STAC pipeline outputs to farm_heterogeneity-compatible wide tables."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DATE_IN_NAME_RE = re.compile(r"(20\d{6})")
DATE_DASH_RE = re.compile(r"(20\d{2})-(20\d{2})-(20\d{2})")


def _normalize_date_token(token: str) -> str:
    """Return YYYYMMDD from column/band name."""
    m = DATE_IN_NAME_RE.search(token)
    if m:
        return m.group(1)
    m2 = re.search(r"(20\d{2})-(\d{2})-(\d{2})", token)
    if m2:
        return f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
    raise ValueError(f"Cannot parse date from: {token}")


def _parse_nc_attr_list(val) -> list[str]:
    """Parse NetCDF global attrs that may be a list or JSON-encoded list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        return [str(x) for x in val]
    if isinstance(val, str):
        text = val.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        return [text] if text else []
    return [str(val)]


def _prefix_for_var(var_name: str, nc_attrs: dict | None = None) -> str:
    """
    Map NetCDF variable name → wide-table column prefix.

    STAC pipeline_runner uses plain date columns when only one index/band is
    requested, storing the variable as ``value``. Signal name comes from global
    attrs ``indices`` (S2) or ``bands`` (S1).
    """
    name = var_name.lower()
    if name == "value" and nc_attrs:
        for key in ("indices", "bands"):
            items = _parse_nc_attr_list(nc_attrs.get(key))
            if items:
                return items[0].upper()
        sensor = str(nc_attrs.get("sensor", "")).lower()
        if "sentinel-2" in sensor:
            return "NDVI"
        if "sentinel-1" in sensor:
            return "VH"
    if "ndvi" in name:
        return "NDVI"
    if "evi" in name:
        return "EVI"
    if name in {"vh", "vv"} or name.startswith("vh_") or name.startswith("vv_"):
        return name.split("_")[0].upper()
    return var_name.upper()


def _resolve_netcdf_uid(ds: xr.Dataset, uid) -> object:
    """Return native uid coordinate value (handles str vs int/np.int32 mismatch)."""
    target = str(uid)
    for u in ds["uid"].values:
        if str(u) == target:
            return u.item() if hasattr(u, "item") else u
    available = [str(u) for u in ds["uid"].values]
    raise ValueError(f"uid {uid!r} not in NetCDF (have {available})")


def netcdf_to_wide(nc_path: str | Path, uid: str | None = None) -> pd.DataFrame:
    """
    Read polygon-aggregated NetCDF from pipeline_runner → one row per uid.

    Expected dims: uid × time; variables like vh, NDVI, etc.
    """
    ds = xr.open_dataset(nc_path)
    if "uid" not in ds.dims:
        raise ValueError(f"NetCDF missing 'uid' dimension: {nc_path}")

    nc_attrs = dict(ds.attrs)
    uids_raw = list(ds["uid"].values)
    if uid is not None:
        uid_key = _resolve_netcdf_uid(ds, uid)
        uids = [uid_key]
    else:
        uids = uids_raw

    rows = []
    for u in uids:
        row = {"farm_id": str(u)}
        for var in ds.data_vars:
            da = ds[var].sel(uid=u)
            if "time" not in da.dims:
                continue
            prefix = _prefix_for_var(var, nc_attrs)
            for t, val in zip(da["time"].values, da.values):
                date_str = pd.Timestamp(str(t)).strftime("%Y%m%d")
                row[f"{prefix}_{date_str}"] = float(val) if np.isfinite(val) else np.nan
        rows.append(row)

    ds.close()
    return pd.DataFrame(rows)


def csv_wide_from_stac_csv(
    csv_path: str | Path,
    uid_col: str,
    signal_prefix: str,
    uid: str | None = None,
) -> pd.DataFrame:
    """
    Convert STAC per-pixel CSV (uid | date columns or band_date columns) to wide YYYYMMDD format.
    """
    df = pd.read_csv(csv_path) if isinstance(csv_path, (str, Path)) else csv_path.copy()
    if uid_col not in df.columns:
        raise ValueError(f"Column {uid_col} not in {csv_path}")

    if uid is not None:
        df = df[df[uid_col].astype(str) == str(uid)]
        if df.empty:
            raise ValueError(f"No rows for uid={uid} in {csv_path}")

    # Already wide with YYYYMMDD in names
    date_cols = [c for c in df.columns if DATE_IN_NAME_RE.search(str(c))]
    if date_cols:
        out = df.copy()
        out["farm_id"] = out[uid_col].astype(str)
        rename = {}
        for c in date_cols:
            if not str(c).upper().startswith(signal_prefix.upper()):
                d = _normalize_date_token(str(c))
                rename[c] = f"{signal_prefix.upper()}_{d}"
        out = out.rename(columns=rename)
        return out

    # Long format: date column + value column
    value_cols = [c for c in df.columns if c != uid_col and not c.lower().endswith("date")]
    if len(value_cols) == 1 and any("date" in c.lower() for c in df.columns):
        date_col = next(c for c in df.columns if "date" in c.lower())
        val_col = value_cols[0]
        row = {"farm_id": str(df[uid_col].iloc[0])}
        for _, r in df.iterrows():
            d = pd.Timestamp(r[date_col]).strftime("%Y%m%d")
            row[f"{signal_prefix.upper()}_{d}"] = float(r[val_col])
        return pd.DataFrame([row])

    # Columns are ISO dates YYYY-MM-DD
    iso_cols = [c for c in df.columns if re.match(r"20\d{2}-\d{2}-\d{2}", str(c))]
    if iso_cols:
        row = {"farm_id": str(df[uid_col].iloc[0])}
        for c in iso_cols:
            d = _normalize_date_token(str(c))
            row[f"{signal_prefix.upper()}_{d}"] = float(df[c].iloc[0])
        return pd.DataFrame([row])

    raise ValueError(f"Unrecognized STAC CSV layout: {csv_path}")


def extract_curve_from_wide(
    wide_df: pd.DataFrame,
    signal_prefix: str,
    nan_sentinel: float = -9999.0,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    prefix = signal_prefix.upper()
    cols = sorted(
        [c for c in wide_df.columns if str(c).upper().startswith(prefix + "_")],
        key=lambda c: _normalize_date_token(str(c)),
    )
    if not cols:
        # Legacy STAC NetCDF: single index stored as variable "value" → VALUE_YYYYMMDD
        value_cols = sorted(
            [c for c in wide_df.columns if str(c).upper().startswith("VALUE_")],
            key=lambda c: _normalize_date_token(str(c)),
        )
        if value_cols:
            cols = value_cols
    if not cols:
        available = [c for c in wide_df.columns if c != "farm_id"]
        raise ValueError(
            f"No columns for signal {signal_prefix} in dataframe "
            f"(available: {available[:8]}{'...' if len(available) > 8 else ''})"
        )

    values = wide_df[cols].iloc[0].to_numpy(dtype=float)
    if nan_sentinel is not None:
        values = np.where(values == nan_sentinel, np.nan, values)
    dates = pd.DatetimeIndex(
        [pd.to_datetime(_normalize_date_token(str(c)), format="%Y%m%d") for c in cols]
    )
    return values, dates


def align_curves_on_dates(
    dates_a: pd.DatetimeIndex,
    values_a: np.ndarray,
    dates_b: pd.DatetimeIndex,
    values_b: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Union of observation dates with NaN where a sensor has no value."""
    all_dates = pd.DatetimeIndex(sorted(set(dates_a) | set(dates_b)))
    a_map = {d: v for d, v in zip(dates_a, values_a)}
    b_map = {d: v for d, v in zip(dates_b, values_b)}
    va = np.array([a_map.get(d, np.nan) for d in all_dates], dtype=float)
    vb = np.array([b_map.get(d, np.nan) for d in all_dates], dtype=float)
    return all_dates, va, vb


def _date_value_columns(df: pd.DataFrame, uid_col: str) -> list[str]:
    from .farm_signal import DATE_COL_RE, parse_observation_date

    cols = [c for c in df.columns if c != uid_col and DATE_COL_RE.search(str(c))]
    return sorted(cols, key=parse_observation_date)


def load_farm_pixel_timeseries(
    csv_path: str | Path,
    uid_col: str,
    farm_id: str,
    nan_sentinel: float = -9999.0,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Load raw per-pixel time series for one farm from STAC pixel CSV.

    Returns (obs_dates, values) with values shaped (n_pixels, n_dates). No smoothing applied.
    """
    from .farm_signal import parse_observation_date

    df = pd.read_csv(csv_path)
    if uid_col not in df.columns:
        raise ValueError(f"Column {uid_col!r} not in {csv_path}")

    sub = df[df[uid_col].astype(str) == str(farm_id)]
    if sub.empty:
        return pd.DatetimeIndex([]), np.empty((0, 0), dtype=float)

    value_cols = _date_value_columns(sub, uid_col)
    if not value_cols:
        return pd.DatetimeIndex([]), np.empty((len(sub), 0), dtype=float)

    obs_dates = pd.DatetimeIndex([parse_observation_date(c) for c in value_cols])
    values = sub[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if nan_sentinel is not None:
        values = np.where(values == nan_sentinel, np.nan, values)
    return obs_dates, values
