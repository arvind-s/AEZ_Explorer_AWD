"""Growing degree days from ERA5-Land daily min/max temperature (CDS API)."""

from __future__ import annotations


import numpy as np
import pandas as pd

ERA5_LAND_DAILY_DATASET = "derived-era5-land-daily-statistics"
ERA5_VARIABLE = "2m_temperature"


def _month_day_lists(start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[str], list[str], list[str]]:
    years = sorted({start.year, end.year})
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    return [str(y) for y in years], months, days


def _area_box(lat: float, lon: float, delta_deg: float = 0.05) -> list[float]:
    """CDS area: [North, West, South, East]."""
    return [lat + delta_deg, lon - delta_deg, lat - delta_deg, lon + delta_deg]


def _cache_path(cache_dir: Path, start: str, end: str, lat: float, lon: float, stat: str) -> Path:
    key = f"era5land_{stat}_{start}_{end}_{lat:.3f}_{lon:.3f}.nc"
    return cache_dir / key


def _retrieve_era5_land_daily(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    daily_statistic: str,
    cache_dir: Path,
    time_zone: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = _cache_path(cache_dir, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), lat, lon, daily_statistic)
    if out.exists() and out.stat().st_size > 0:
        return out

    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError("cdsapi is required for ERA5-Land GDD — pip install cdsapi") from exc

    years, months, days = _month_day_lists(start, end)
    area = _area_box(lat, lon)

    client = cdsapi.Client()
    client.retrieve(
        ERA5_LAND_DAILY_DATASET,
        {
            "variable": ERA5_VARIABLE,
            "year": years,
            "month": months,
            "day": days,
            "daily_statistic": daily_statistic,
            "time_zone": time_zone,
            "frequency": "daily",
            "area": area,
        },
        str(out),
    )
    return out


def _read_daily_temp_series(nc_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    # Variable name varies: t2m or 2m_temperature with min/max suffix
    var_candidates = [v for v in ds.data_vars if "t2m" in v.lower() or "temperature" in v.lower()]
    if not var_candidates:
        var_candidates = list(ds.data_vars)
    da = ds[var_candidates[0]]
    if "time" in da.dims:
        times = pd.to_datetime(da["time"].values)
        vals = np.asarray(da.values, dtype=float)
        # Kelvin → Celsius if needed
        if np.nanmean(vals) > 150:
            vals = vals - 273.15
        series = pd.Series(vals, index=times).sort_index()
    else:
        raise ValueError(f"No time dimension in {nc_path}")
    ds.close()
    series = series[(series.index >= start) & (series.index <= end)]
    return series


def fetch_era5_land_tmin_tmax(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = ".cache/phenology/era5",
    time_zone: str = "utc+05:30",
) -> pd.DataFrame:
    """
    Fetch ERA5-Land daily Tmin and Tmax at a point (small area mean).

    Requires CDS API credentials (~/.cdsapirc or CDS_API_URL / CDS_API_KEY env vars).
    """
    cache = Path(cache_dir)
    tmin_path = _retrieve_era5_land_daily(lat, lon, start, end, "daily_minimum", cache, time_zone)
    tmax_path = _retrieve_era5_land_daily(lat, lon, start, end, "daily_maximum", cache, time_zone)

    tmin = _read_daily_temp_series(tmin_path, start, end)
    tmax = _read_daily_temp_series(tmax_path, start, end)

    df = pd.DataFrame({"tmin_c": tmin, "tmax_c": tmax})
    df = df.dropna(how="all")
    df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0
    return df


def compute_daily_gdd(tmin: np.ndarray, tmax: np.ndarray, t_base: float = 10.0) -> np.ndarray:
    tmean = (tmin + tmax) / 2.0
    return np.maximum(0.0, tmean - t_base)


def gdd_table_from_transplant(
    weather: pd.DataFrame,
    transplant_date: pd.Timestamp,
    t_base: float = 10.0,
) -> pd.DataFrame:
    if pd.isna(transplant_date):
        raise ValueError("transplant_date is required for GDD")

    df = weather.copy()
    df.index = pd.to_datetime(df.index)
    df = df[df.index >= transplant_date].copy()
    if df.empty:
        raise ValueError("No ERA5-Land weather on or after transplant_date")

    df["daily_gdd"] = compute_daily_gdd(df["tmin_c"].values, df["tmax_c"].values, t_base)
    df["cumulative_gdd"] = df["daily_gdd"].cumsum()
    df["days_after_transplant"] = (df.index - transplant_date).days
    return df


def cumulative_gdd_on_date(gdd_df: pd.DataFrame, on_date: pd.Timestamp) -> float:
    on_date = pd.Timestamp(on_date)
    sub = gdd_df[gdd_df.index <= on_date]
    if sub.empty:
        return 0.0
    return float(sub["cumulative_gdd"].iloc[-1])


def estimate_harvest_date(
    gdd_df: pd.DataFrame,
    target_total_gdd: float,
    max_date: pd.Timestamp | None = None,
) -> pd.Timestamp:
    remaining = gdd_df[gdd_df["cumulative_gdd"] < target_total_gdd]
    if remaining.empty:
        last = gdd_df.index[-1]
        return pd.Timestamp(last)

    # Linear extrapolation from last week mean daily GDD
    tail = gdd_df.tail(14)
    mean_daily = float(tail["daily_gdd"].mean())
    if mean_daily <= 0:
        return pd.Timestamp(gdd_df.index[-1])

    last_idx = gdd_df.index[-1]
    last_row = gdd_df.iloc[-1]
    gap = target_total_gdd - float(last_row["cumulative_gdd"])
    extra_days = int(np.ceil(gap / mean_daily))
    est = pd.Timestamp(last_idx) + pd.Timedelta(days=extra_days)
    if max_date is not None:
        est = min(est, pd.Timestamp(max_date))
    return est
