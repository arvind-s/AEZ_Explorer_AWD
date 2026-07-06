from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm

from src.harmonize import CLASS_NAMES, map_label

BAND_COLS = [f"A{i:02d}" for i in range(64)]
ESA_S3_TEMPLATE = (
    "s3://esa-worldcover/v200/{year}/map/"
    "ESA_WorldCover_10m_{year}_v200_{tile}_Map.tif"
)
# ESA WorldCover is also accessible via HTTPS:
ESA_HTTPS_TEMPLATE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/{year}/map/"
    "ESA_WorldCover_10m_{year}_v200_{tile}_Map.tif"
)


def _sample_raster_stratified(
    raster_path: str,
    class_map_fn,
    n_per_class: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, int]]:
    """
    Read a classification raster and return stratified (lon, lat, class_id) samples.
    class_map_fn: callable(native_value) -> int | None
    Returns list of (lon, lat, class_id).
    """
    samples = []
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # [H, W]
        transform = src.transform
        unique_vals = np.unique(data)
        for val in unique_vals:
            mapped = class_map_fn(val)
            if mapped is None:
                continue
            ys, xs = np.where(data == val)
            if len(ys) == 0:
                continue
            idx = rng.choice(len(ys), size=min(n_per_class, len(ys)), replace=False)
            for i in idx:
                lon, lat = rasterio.transform.xy(transform, ys[i], xs[i])
                samples.append((float(lon), float(lat), int(mapped)))
    return samples


def sample_esa_worldcover(
    bbox: tuple[float, float, float, float],
    year: int,
    n_per_class: int,
    embedding_ds,  # xr.Dataset from fetch_embeddings
    cache_dir: Path,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Sample ESA WorldCover labels within bbox for given year,
    then look up AEF embeddings at those locations.

    bbox: (minx, miny, maxx, maxy) in EPSG:4326
    embedding_ds: xr.Dataset from aef_fetcher.fetch_embeddings for this bbox
    Returns DataFrame with schema [source, year, class_id, class_name, lon, lat, A00…A63]
    """
    if rng is None:
        rng = np.random.default_rng()

    # ESA WorldCover tiles are named by 3°×3° grid cells (e.g. N00E006)
    # For simplicity, download the global mosaic VRT or iterate tiles.
    # Use HTTPS (no credentials needed — ESA bucket is public read).
    # The tile naming: N{lat:02d}{E|W}{lon:03d} for SW corner of each 3° tile.
    minx, miny, maxx, maxy = bbox
    esa_year = year if year in (2020, 2021) else 2021  # only 2020/2021 available

    # Build list of overlapping ESA tiles
    tile_lats = range(int(np.floor(miny / 3)) * 3, int(np.ceil(maxy / 3)) * 3, 3)
    tile_lons = range(int(np.floor(minx / 3)) * 3, int(np.ceil(maxx / 3)) * 3, 3)

    all_samples: list[tuple[float, float, int]] = []
    for tlat in tile_lats:
        for tlon in tile_lons:
            ns = "N" if tlat >= 0 else "S"
            ew = "E" if tlon >= 0 else "W"
            tile = f"{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}"
            url = ESA_HTTPS_TEMPLATE.format(year=esa_year, tile=tile)
            try:
                samples = _sample_raster_stratified(
                    url,
                    lambda v: map_label("esa_worldcover", int(v)),
                    n_per_class,
                    rng,
                )
                all_samples.extend(samples)
            except Exception as e:
                print(f"  Skipping ESA tile {tile}: {e}")

    if not all_samples:
        return pd.DataFrame(columns=["source", "year", "class_id", "class_name", "lon", "lat"] + BAND_COLS)

    # Look up AEF embeddings at sample locations
    rows = []
    emb_arr = embedding_ds["embeddings"].sel(year=year, method="nearest").values  # [H, W, 64]
    h, w, _ = emb_arr.shape

    # embedding_ds coords for y/x spatial lookup
    # Use nearest pixel lookup via xarray
    for lon, lat, class_id in all_samples:
        try:
            emb = embedding_ds["embeddings"].sel(
                year=year, method="nearest"
            )
            # Find nearest pixel by indexing: use .sel with nearest on spatial dims
            # Note: y/x dims are pixel indices, not coordinates — use linear interp
            # Simple approach: find closest pixel by bbox fraction
            ds_attrs = embedding_ds.attrs
            transform_coeffs = ds_attrs.get("transform", [])
            if len(transform_coeffs) >= 6:
                from affine import Affine
                t = Affine(*transform_coeffs[:6])
                col, row = ~t * (lon, lat)
                col, row = int(col), int(row)
                if 0 <= row < h and 0 <= col < w:
                    pixel_emb = emb_arr[row, col, :]  # [64]
                else:
                    continue
            else:
                continue
        except Exception:
            continue

        row_dict = {
            "source": "esa_worldcover",
            "year": year,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "lon": lon,
            "lat": lat,
        }
        row_dict.update(dict(zip(BAND_COLS, pixel_emb.tolist())))
        rows.append(row_dict)

    return pd.DataFrame(rows)


def mahalanobis_filter(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove per-class embedding outliers using Mahalanobis distance.
    Drops rows where distance from class centroid exceeds threshold standard deviations.
    """
    kept = []
    for class_id, group in df.groupby("class_id"):
        embs = group[BAND_COLS].values.astype(np.float64)
        if len(embs) < 10:
            kept.append(group)
            continue
        cov = EmpiricalCovariance().fit(embs)
        dists = cov.mahalanobis(embs) ** 0.5
        # Remove outliers beyond threshold standard deviations
        # Use mean + threshold * std to identify outliers
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        if std_dist > 0:
            mask = dists < mean_dist + threshold * std_dist
        else:
            mask = np.ones(len(dists), dtype=bool)
        kept.append(group[mask])
    return pd.concat(kept, ignore_index=True)


def agreement_filter(
    dfs: list[pd.DataFrame],
    tolerance_deg: float = 0.0001,
) -> pd.DataFrame:
    """
    For pixels covered by ≥2 sources, keep only where all sources agree on class_id.
    Pixels unique to one source pass through unchanged.
    tolerance_deg: spatial tolerance for matching lon/lat across sources.
    """
    if len(dfs) <= 1:
        return dfs[0] if dfs else pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Round lon/lat to tolerance grid
    scale = 1 / tolerance_deg
    combined["_lon_r"] = (combined["lon"] * scale).round().astype(int)
    combined["_lat_r"] = (combined["lat"] * scale).round().astype(int)

    # Find pixels covered by multiple sources
    pixel_sources = combined.groupby(["_lon_r", "_lat_r"])["source"].nunique()
    multi = pixel_sources[pixel_sources > 1].index

    # For multi-source pixels, keep only where all sources agree on class_id
    multi_mask = combined.set_index(["_lon_r", "_lat_r"]).index.isin(multi)
    single_rows = combined[~multi_mask].drop(columns=["_lon_r", "_lat_r"])

    multi_rows = combined[multi_mask].copy()
    # Group by pixel; keep only groups where all class_ids are identical
    agreed = []
    for (lon_r, lat_r), grp in multi_rows.groupby(["_lon_r", "_lat_r"]):
        if grp["class_id"].nunique() == 1:
            agreed.append(grp.drop(columns=["_lon_r", "_lat_r"]))

    if agreed:
        return pd.concat([single_rows] + agreed, ignore_index=True)
    return single_rows


def build_training_set(
    esa_bbox: tuple[float, float, float, float],
    embedding_ds,
    output_path: Path,
    n_per_class: int = 5000,
    dw_geotiff_dir: Path | None = None,
    oem_tile_dir: Path | None = None,
    seed: int = 42,
) -> Path:
    """
    Orchestrate all label sources, harmonize, filter, and save training Parquet.

    esa_bbox: (minx, miny, maxx, maxy) EPSG:4326 covering training region
    embedding_ds: AEF embeddings for that region
    dw_geotiff_dir: directory of DW GeoTIFFs exported from GEE (optional)
    oem_tile_dir: directory of OpenEarthMap tiles (optional)
    """
    rng = np.random.default_rng(seed)
    source_dfs = []

    # ESA WorldCover (2020, 2021)
    for year in (2020, 2021):
        print(f"Sampling ESA WorldCover {year}…")
        df = sample_esa_worldcover(esa_bbox, year, n_per_class, embedding_ds, output_path.parent, rng)
        if not df.empty:
            source_dfs.append(df)

    # Dynamic World (pre-exported GeoTIFFs from scripts/export_dw.py)
    if dw_geotiff_dir and dw_geotiff_dir.exists():
        print("Sampling Dynamic World GeoTIFFs…")
        for tif in tqdm(list(dw_geotiff_dir.glob("*.tif"))):
            stem_parts = tif.stem.split("_")
            year = int(stem_parts[-1]) if stem_parts[-1].isdigit() else 2021
            raw_samples = _sample_raster_stratified(
                str(tif),
                lambda v: map_label("dynamic_world", _DW_VALUE_MAP.get(int(v))),
                n_per_class,
                rng,
            )
            emb_year = min(year, 2025)
            if emb_year not in embedding_ds["year"].values:
                emb_year = int(embedding_ds["year"].values[-1])
            emb_arr = embedding_ds["embeddings"].sel(year=emb_year).values
            h, w, _ = emb_arr.shape
            transform_coeffs = embedding_ds.attrs.get("transform", [])
            if len(transform_coeffs) < 6:
                continue
            from affine import Affine
            t = Affine(*transform_coeffs[:6])
            rows_dw = []
            for lon, lat, class_id in raw_samples:
                col_px, row_px = ~t * (lon, lat)
                col_px, row_px = int(col_px), int(row_px)
                if not (0 <= row_px < h and 0 <= col_px < w):
                    continue
                pixel_emb = emb_arr[row_px, col_px, :]
                row_dict = {
                    "source": "dynamic_world",
                    "year": year,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "lon": lon,
                    "lat": lat,
                }
                row_dict.update(dict(zip(BAND_COLS, pixel_emb.tolist())))
                rows_dw.append(row_dict)
            if rows_dw:
                source_dfs.append(pd.DataFrame(rows_dw))

    # OpenEarthMap (pre-downloaded tiles from Zenodo)
    if oem_tile_dir and oem_tile_dir.exists():
        print("Sampling OpenEarthMap tiles…")
        img_dir = oem_tile_dir / "images"
        lbl_dir = oem_tile_dir / "labels"
        for lbl_path in tqdm(list(lbl_dir.glob("*.tif"))):
            raw_samples = _sample_raster_stratified(
                str(lbl_path),
                lambda v: map_label("openearthmap", int(v)),
                n_per_class,
                rng,
            )
            emb_arr = embedding_ds["embeddings"].sel(year=2021, method="nearest").values
            h, w, _ = emb_arr.shape
            transform_coeffs = embedding_ds.attrs.get("transform", [])
            if len(transform_coeffs) < 6:
                continue
            from affine import Affine
            t = Affine(*transform_coeffs[:6])
            rows_oem = []
            for lon, lat, class_id in raw_samples:
                col_px, row_px = ~t * (lon, lat)
                col_px, row_px = int(col_px), int(row_px)
                if not (0 <= row_px < h and 0 <= col_px < w):
                    continue
                pixel_emb = emb_arr[row_px, col_px, :]
                row_dict = {
                    "source": "openearthmap",
                    "year": 2021,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "lon": lon,
                    "lat": lat,
                }
                row_dict.update(dict(zip(BAND_COLS, pixel_emb.tolist())))
                rows_oem.append(row_dict)
            if rows_oem:
                source_dfs.append(pd.DataFrame(rows_oem))

    # Multi-source agreement filter
    combined = agreement_filter(source_dfs)

    # Mahalanobis outlier filter
    combined = mahalanobis_filter(combined, threshold=3.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"Training set saved: {len(combined):,} rows → {output_path}")
    return output_path


# Dynamic World integer → string label (used when reading exported GeoTIFFs)
_DW_VALUE_MAP: dict[int, str] = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice",
}
