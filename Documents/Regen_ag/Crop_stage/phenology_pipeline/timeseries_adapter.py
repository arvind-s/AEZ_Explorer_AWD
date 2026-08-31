"""Convert STAC pipeline outputs to farm_heterogeneity-compatible wide tables."""

from __future__ import annotations

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


def _prefix_for_var(var_name: str) -> str:
    name = var_name.lower()
    if "ndvi" in name:
        return "NDVI"
    if "evi" in name:
        return "EVI"
    if name in {"vh", "vv"} or name.startswith("vh_") or name.startswith("vv_"):
        return name.split("_")[0].upper()
    return var_name.upper()


def netcdf_to_wide(nc_path: str | Path, uid: str | None = None) -> pd.DataFrame:
    """
    Read polygon-aggregated NetCDF from pipeline_runner → one row per uid.

    Expected dims: uid × time; variables like vh, NDVI, etc.
    """
    ds = xr.open_dataset(nc_path)
    if "uid" not in ds.dims:
        raise ValueError(f"NetCDF missing 'uid' dimension: {nc_path}")

    uids = list(ds["uid"].values)
    if uid is not None:
        if uid not in uids:
            raise ValueError(f"uid {uid} not in NetCDF (have {uids})")
        uids = [uid]

    rows = []
    for u in uids:
        row = { "farm_id": str(u) }
        for var in ds.data_vars:
            da = ds[var].sel(uid=u)
            if "time" not in da.dims:
                continue
            prefix = _prefix_for_var(var)
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
        raise ValueError(f"No columns for signal {signal_prefix} in dataframe")

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
