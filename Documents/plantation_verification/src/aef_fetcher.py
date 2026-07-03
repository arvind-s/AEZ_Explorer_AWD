from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from shapely.geometry import shape, box

# AEF index GeoParquet on source.coop
# After downloading, inspect columns with: print(index_gdf.columns.tolist())
# Expected columns: geometry, crs_wkt (or epsg), year, href (or url or path)
INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/index.parquet"
YEARS = list(range(2017, 2026))
N_BANDS = 64
BAND_NAMES = [f"A{i:02d}" for i in range(N_BANDS)]
DEFAULT_CACHE = Path.home() / ".cache" / "aef_embeddings"


def dequantize(arr: np.ndarray) -> np.ndarray:
    """Dequantize int8 AEF values to float32 in [-1, 1]."""
    f = arr.astype(np.float32)
    return (f / 127.5) ** 2 * np.sign(f)


def _parcel_hash(geojson: dict, years: list[int]) -> str:
    key = json.dumps(geojson, sort_keys=True) + str(sorted(years))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_index(cache_dir: Path = DEFAULT_CACHE) -> gpd.GeoDataFrame:
    """Download and cache the AEF index GeoParquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "aef_index.parquet"
    if not cache_path.exists():
        import urllib.request
        print(f"Downloading AEF index to {cache_path} …")
        urllib.request.urlretrieve(INDEX_URL, cache_path)
    return gpd.read_parquet(cache_path)


def _url_column(gdf: gpd.GeoDataFrame) -> str:
    """Return whichever column holds the COG file URL/path."""
    for col in ("href", "url", "path", "filename"):
        if col in gdf.columns:
            return col
    raise KeyError(
        f"Cannot find URL column in AEF index. Available columns: {gdf.columns.tolist()}"
    )


def _year_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Return year column if present, else None (year embedded in URL)."""
    return "year" if "year" in gdf.columns else None


def _read_tile_window(
    url: str, minx: float, miny: float, maxx: float, maxy: float
) -> np.ndarray:
    """Open a COG via HTTPS and read the bbox window. Returns [64, H, W] int8."""
    # rasterio uses GDAL /vsicurl/ under the hood for HTTPS URLs
    with rasterio.open(url) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(window=window)  # [64, H, W] int8
    return data


def fetch_embeddings(
    parcel_geojson: dict,
    years: list[int] | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> xr.Dataset:
    """
    Fetch AEF embeddings for a parcel polygon across years.

    parcel_geojson: GeoJSON Feature or Geometry dict in EPSG:4326
    years: list of years to fetch (default 2017-2025)
    Returns: xr.Dataset with data_var "embeddings" dims (year, y, x, band)
    """
    if years is None:
        years = YEARS

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _parcel_hash(parcel_geojson, years)
    cache_path = cache_dir / f"{cache_key}.nc"

    if cache_path.exists():
        return xr.open_dataset(cache_path)

    geom = parcel_geojson.get("geometry", parcel_geojson)
    parcel = shape(geom)
    minx, miny, maxx, maxy = parcel.bounds

    index_gdf = load_index(cache_dir)
    url_col = _url_column(index_gdf)
    year_col = _year_column(index_gdf)

    # Spatial filter
    parcel_box = box(minx, miny, maxx, maxy)
    overlapping = index_gdf[index_gdf.geometry.intersects(parcel_box)].copy()

    if overlapping.empty:
        raise ValueError(f"No AEF tiles overlap parcel bounds {parcel.bounds}")

    year_arrays: dict[int, np.ndarray] = {}
    ref_transform = None
    ref_crs = None

    for year in sorted(years):
        if year_col:
            tiles = overlapping[overlapping[year_col] == year]
        else:
            # Year embedded in URL — filter by year string in path
            tiles = overlapping[overlapping[url_col].str.contains(str(year))]

        if tiles.empty:
            print(f"  Warning: no AEF tiles found for year {year}, skipping")
            continue

        # If multiple tiles overlap (tile boundary crosses parcel), merge them.
        # Simple case: take first tile (most parcels fit inside one UTM tile).
        # TODO: implement mosaic for parcels spanning tile boundaries.
        url = tiles.iloc[0][url_col]
        raw = _read_tile_window(url, minx, miny, maxx, maxy)  # [64, H, W]

        if ref_transform is None:
            with rasterio.open(url) as src:
                from rasterio.windows import from_bounds as fb
                w = fb(minx, miny, maxx, maxy, src.transform)
                ref_transform = src.window_transform(w)
                ref_crs = src.crs

        deq = dequantize(raw)  # [64, H, W] float32
        year_arrays[year] = deq.transpose(1, 2, 0)  # [H, W, 64]

    if not year_arrays:
        raise RuntimeError("No embeddings fetched for any requested year")

    fetched_years = sorted(year_arrays.keys())
    H, W, _ = next(iter(year_arrays.values())).shape
    stack = np.stack([year_arrays[y] for y in fetched_years], axis=0)  # [Y, H, W, 64]

    ds = xr.Dataset(
        {"embeddings": (["year", "y", "x", "band"], stack)},
        coords={
            "year": fetched_years,
            "band": BAND_NAMES,
        },
        attrs={
            "crs": str(ref_crs),
            "transform": list(ref_transform)[:6] if ref_transform else [],
            "parcel_hash": cache_key,
        },
    )
    ds.to_netcdf(cache_path)
    return ds
