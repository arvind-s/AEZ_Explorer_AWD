"""Growing degree days from ERA5-Land daily min/max temperature."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Era5Source = Literal["openmeteo", "cds"]

ERA5_LAND_DAILY_DATASET = "derived-era5-land-daily-statistics"
ERA5_VARIABLE = "2m_temperature"
ERA5_FREQUENCY = "1_hourly"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# In-process cache: avoids duplicate HTTP/CDS calls within one Python session.
_WEATHER_MEM_CACHE: dict[str, pd.DataFrame] = {}


def _normalize_cds_time_zone(time_zone: str) -> str:
    """Normalize user time zone to CDS format, e.g. utc+05:30."""
    tz = time_zone.strip().lower()
    if not tz.startswith("utc"):
        raise ValueError(f"ERA5 time_zone must start with 'utc', got {time_zone!r}")
    offset = tz[3:]
    if not offset:
        return "utc+00:00"
    sign = offset[0]
    if sign not in "+-":
        raise ValueError(f"Invalid ERA5 time_zone: {time_zone!r}")
    body = offset[1:]
    if ":" in body:
        return f"utc{sign}{body}"
    if len(body) == 4:  # e.g. 0530
        return f"utc{sign}{body[:2]}:{body[2:]}"
    if len(body) == 2:  # e.g. 05
        return f"utc{sign}{body}:00"
    return tz


def _monthly_request_parts(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (year, month, days) tuples covering [start, end] without invalid day/month pairs."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        return
    cur = start
    while cur <= end:
        month_end = min(end, cur + pd.offsets.MonthEnd(0))
        days = [f"{d.day:02d}" for d in pd.date_range(cur, month_end, freq="D")]
        yield str(cur.year), f"{cur.month:02d}", days
        cur = (month_end + pd.Timedelta(days=1)).normalize()


def build_era5_cds_request(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    daily_statistic: str,
    time_zone: str,
    *,
    year: str | None = None,
    month: str | None = None,
    days: list[str] | None = None,
) -> dict:
    """Build a CDS API request dict for one month (or full span when parts omitted)."""
    if year is None or month is None or days is None:
        year, month, days = next(_monthly_request_parts(start, end))
    return {
        "variable": [ERA5_VARIABLE],
        "year": year,
        "month": month,
        "day": days,
        "daily_statistic": daily_statistic,
        "time_zone": _normalize_cds_time_zone(time_zone),
        "frequency": ERA5_FREQUENCY,
        "area": _area_box(lat, lon),
    }


def _area_box(lat: float, lon: float, delta_deg: float = 0.05) -> list[float]:
    """CDS area: [North, West, South, East]."""
    return [lat + delta_deg, lon - delta_deg, lat - delta_deg, lon + delta_deg]


def _cache_path(cache_dir: Path, start: str, end: str, lat: float, lon: float, stat: str) -> Path:
    key = f"era5land_{stat}_{start}_{end}_{lat:.3f}_{lon:.3f}.nc"
    return cache_dir / key


def _merge_temp_nc_files(paths: list[Path], out_path: Path) -> None:
    if not paths:
        raise ValueError("No ERA5 NetCDF files to merge")
    import xarray as xr

    datasets = [xr.open_dataset(p) for p in paths]
    try:
        merged = xr.concat(datasets, dim="time")
        _, unique_idx = np.unique(merged["time"].values, return_index=True)
        merged = merged.isel(time=sorted(unique_idx))
        merged.to_netcdf(out_path)
    finally:
        for ds in datasets:
            ds.close()


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
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError(
            f"ERA5 weather end ({end.date()}) is before start ({start.date()}). "
            "Transplant date must be on or before the assessment date."
        )

    out = _cache_path(cache_dir, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), lat, lon, daily_statistic)
    if out.exists() and out.stat().st_size > 0:
        return out

    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError("cdsapi is required for ERA5-Land GDD — pip install cdsapi") from exc

    client = cdsapi.Client()
    month_parts = list(_monthly_request_parts(start, end))
    if not month_parts:
        raise ValueError(
            f"No ERA5 months to request between {start.date()} and {end.date()}"
        )
    temp_paths: list[Path] = []

    try:
        for year, month, days in month_parts:
            part = cache_dir / f".tmp_{out.stem}_{year}{month}.nc"
            request = build_era5_cds_request(
                lat,
                lon,
                start,
                end,
                daily_statistic,
                time_zone,
                year=year,
                month=month,
                days=days,
            )
            try:
                client.retrieve(ERA5_LAND_DAILY_DATASET, request, str(part))
            except Exception as exc:
                raise RuntimeError(
                    f"CDS ERA5-Land request failed for {year}-{month} "
                    f"({daily_statistic}). Check ~/.cdsapirc credentials and "
                    f"request parameters. Underlying error: {exc}"
                ) from exc
            if not part.exists() or part.stat().st_size == 0:
                raise RuntimeError(
                    f"CDS ERA5-Land returned an empty file for {year}-{month} ({daily_statistic})"
                )
            temp_paths.append(part)

        if len(temp_paths) == 1:
            temp_paths[0].replace(out)
        elif len(temp_paths) > 1:
            _merge_temp_nc_files(temp_paths, out)
        else:
            raise RuntimeError("No ERA5-Land monthly files were retrieved")
    finally:
        for part in temp_paths:
            if part.exists() and part != out:
                part.unlink(missing_ok=True)

    return out


def _read_daily_temp_series(
    nc_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    daily_statistic: str | None = None,
) -> pd.Series:
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
        # Some CDS responses still return sub-daily steps; collapse to daily values.
        if len(series) > max((end - start).days + 2, 1) * 2:
            agg = "mean"
            if daily_statistic == "daily_minimum":
                agg = "min"
            elif daily_statistic == "daily_maximum":
                agg = "max"
            series = getattr(series.resample("D"), agg)()
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
    Fetch ERA5-Land daily Tmin and Tmax via Copernicus CDS (slow — queued jobs).

    Requires CDS API credentials (~/.cdsapirc or CDS_API_URL / CDS_API_KEY env vars).
    """
    cache = Path(cache_dir)
    tmin_path = _retrieve_era5_land_daily(lat, lon, start, end, "daily_minimum", cache, time_zone)
    tmax_path = _retrieve_era5_land_daily(lat, lon, start, end, "daily_maximum", cache, time_zone)

    tmin = _read_daily_temp_series(tmin_path, start, end, "daily_minimum")
    tmax = _read_daily_temp_series(tmax_path, start, end, "daily_maximum")

    df = pd.DataFrame({"tmin_c": tmin, "tmax_c": tmax})
    df = df.dropna(how="all")
    df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0
    return df


def _weather_mem_key(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    source: str,
    time_zone: str,
) -> str:
    return (
        f"{source}|{lat:.3f}|{lon:.3f}|"
        f"{pd.Timestamp(start).strftime('%Y%m%d')}|{pd.Timestamp(end).strftime('%Y%m%d')}|{time_zone}"
    )


def _openmeteo_timezone(time_zone: str) -> str:
    tz = time_zone.strip().lower()
    if "05:30" in tz or tz in {"utc+05:30", "utc+0530", "ist", "asia/kolkata"}:
        return "Asia/Kolkata"
    if "+00" in tz or tz == "utc":
        return "UTC"
    return "Asia/Kolkata"


def _openmeteo_cache_path(
    cache_dir: Path,
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz_safe: str,
) -> Path:
    name = (
        f"openmeteo_{lat:.3f}_{lon:.3f}_"
        f"{pd.Timestamp(start).strftime('%Y%m%d')}_{pd.Timestamp(end).strftime('%Y%m%d')}_{tz_safe}.csv"
    )
    return cache_dir / name


def fetch_openmeteo_era5_land_tmin_tmax(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = ".cache/phenology/era5",
    time_zone: str = "utc+05:30",
) -> pd.DataFrame:
    """
    Fast ERA5-Land daily Tmin/Tmax via Open-Meteo archive API (no CDS queue).

    Uses the same ERA5-Land reanalysis as CDS; suitable for GDD phenology staging.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError(f"ERA5 end ({end.date()}) is before start ({start.date()})")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tz_name = _openmeteo_timezone(time_zone)
    tz_safe = tz_name.replace("/", "_")
    disk_path = _openmeteo_cache_path(cache, lat, lon, start, end, tz_safe)

    if disk_path.exists() and disk_path.stat().st_size > 0:
        df = pd.read_csv(disk_path, index_col=0, parse_dates=True)
        return df

    params = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.6f}",
            "longitude": f"{lon:.6f}",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": tz_name,
        }
    )
    url = f"{OPENMETEO_ARCHIVE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Open-Meteo ERA5 request failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Open-Meteo ERA5 network error: {exc}") from exc

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    if not dates:
        raise RuntimeError(f"Open-Meteo returned no daily data for {lat:.3f}, {lon:.3f}")

    df = pd.DataFrame(
        {
            "tmin_c": tmin,
            "tmax_c": tmax,
        },
        index=pd.to_datetime(dates),
    )
    df["tmean_c"] = (df["tmin_c"] + df["tmax_c"]) / 2.0
    df.index.name = "date"
    df.to_csv(disk_path)
    return df


def fetch_weather_tmin_tmax(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = ".cache/phenology/era5",
    time_zone: str = "utc+05:30",
    source: Era5Source = "openmeteo",
) -> pd.DataFrame:
    """Fetch daily Tmin/Tmax using Open-Meteo (fast) or Copernicus CDS (slow, queued)."""
    mem_key = _weather_mem_key(lat, lon, start, end, source, time_zone)
    if mem_key in _WEATHER_MEM_CACHE:
        return _WEATHER_MEM_CACHE[mem_key].copy()

    if source == "openmeteo":
        df = fetch_openmeteo_era5_land_tmin_tmax(lat, lon, start, end, cache_dir, time_zone)
    elif source == "cds":
        df = fetch_era5_land_tmin_tmax(lat, lon, start, end, cache_dir, time_zone)
    else:
        raise ValueError(f"Unknown era5_source: {source}")

    _WEATHER_MEM_CACHE[mem_key] = df.copy()
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
