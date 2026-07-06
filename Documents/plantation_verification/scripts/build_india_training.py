"""
Build India-wide AEF training dataset for the plantation verification pipeline.

Strategy:
  1. List all AEF tile VRTs for UTM zones 42N–47N (years 2020, 2021).
  2. Parse each VRT GeoTransform; filter tiles intersecting India's WGS84 bbox.
  3. For each India tile, read N_WINDOWS_PER_TILE random 512×512 sub-windows.
  4. Match each window with ESA WorldCover 2020/2021 labels via rasterio windowed read.
  5. Stratified-sample N_SAMPLES_PER_CLASS pixels per class per window.
  6. Aggregate all windows, apply Mahalanobis filter, save training Parquet.

Usage:
    cd Documents/plantation_verification
    python scripts/build_india_training.py [--workers 8] [--windows 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
import os
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from pyproj import Transformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.harmonize import CLASS_NAMES, map_label
from src.aef_fetcher import dequantize, BAND_NAMES
from src.label_builder import mahalanobis_filter

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = "https://data.source.coop/tge-labs"
YEARS = [2020, 2021]
ZONES = ["42N", "43N", "44N", "45N", "46N", "47N"]

INDIA_BBOX_WGS84 = (68.0, 8.0, 97.5, 37.5)  # minx, miny, maxx, maxy

TILE_PX = 8192           # pixels per tile side
WINDOW_PX = 512          # sub-window size (5.12 km at 10 m)
N_WINDOWS_PER_TILE = 3   # random sub-windows sampled per India tile
N_SAMPLES_PER_CLASS = 150  # stratified samples per class per window
NODATA = -128            # AEF int8 nodata value

BAND_COLS = [f"A{i:02d}" for i in range(64)]

ESA_HTTPS = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/{year}/map/"
    "ESA_WorldCover_10m_{year}_v200_{tile}_Map.tif"
)

OUTPUT_DIR = Path("data/training")
CACHE_DIR = Path(".cache/aef_tile_index")
INDEX_CACHE = CACHE_DIR / "india_tiles.json"

# ── UTM helpers ────────────────────────────────────────────────────────────────

def _zone_epsg(zone: str) -> int:
    return 32600 + int(zone[:-1])  # northern hemisphere


def _india_utm_bbox(zone: str) -> tuple[float, float, float, float]:
    """India's WGS84 bbox reprojected into the given UTM zone (minx,miny,maxx,maxy)."""
    epsg = _zone_epsg(zone)
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    minx_w, miny_w, maxx_w, maxy_w = INDIA_BBOX_WGS84
    xs, ys = t.transform(
        [minx_w, minx_w, maxx_w, maxx_w],
        [miny_w, maxy_w, miny_w, maxy_w],
    )
    return min(xs), min(ys), max(xs), max(ys)


def _utm_to_wgs84(zone: str, x: float, y: float) -> tuple[float, float]:
    t = Transformer.from_crs(f"EPSG:{_zone_epsg(zone)}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return lon, lat


# ── VRT parsing ────────────────────────────────────────────────────────────────

def _parse_vrt_geotransform(content: str) -> list[float] | None:
    m = re.search(r"<GeoTransform>\s*([^<]+)</GeoTransform>", content)
    if not m:
        return None
    return [float(v.strip()) for v in m.group(1).split(",")]


def _tile_utm_bounds(gt: list[float]) -> tuple[float, float, float, float]:
    ox, dx, _, oy, _, dy = gt
    return ox, oy + dy * TILE_PX, ox + dx * TILE_PX, oy  # minx,miny,maxx,maxy


def _intersects(a: tuple, b: tuple) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _fetch_vrt(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read().decode()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1)
    return None


# ── Tile index ─────────────────────────────────────────────────────────────────

def _list_zone_vrts(year: int, zone: str) -> list[str]:
    """Return all VRT keys for a zone/year (handles S3 pagination)."""
    prefix = f"aef/v1/annual/{year}/{zone}/"
    keys: list[str] = []
    continuation = ""
    while True:
        suffix = f"&continuation-token={urllib.request.quote(continuation)}" if continuation else ""
        url = f"{BASE_URL}/?list-type=2&prefix={prefix}&max-keys=1000{suffix}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                content = r.read().decode()
        except Exception:
            break
        keys += re.findall(r"<Key>([^<]*\.vrt)</Key>", content)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", content)
        if not m:
            break
        continuation = m.group(1)
    return keys


def build_tile_index(workers: int = 8) -> list[dict]:
    """
    Download VRT headers for all India zones/years, return list of tile dicts:
      {year, zone, tiff_url, gt, utm_bounds}
    Caches result to INDEX_CACHE.
    """
    if INDEX_CACHE.exists():
        print(f"Loading tile index from cache: {INDEX_CACHE}")
        with open(INDEX_CACHE) as f:
            return json.load(f)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Building tile index (downloading VRT headers)…")

    # Collect all VRT keys
    all_vrt_keys: list[tuple[int, str, str]] = []  # (year, zone, key)
    for year in YEARS:
        for zone in ZONES:
            keys = _list_zone_vrts(year, zone)
            all_vrt_keys.extend((year, zone, k) for k in keys)
    print(f"  Found {len(all_vrt_keys)} VRT files across {len(YEARS)} years × {len(ZONES)} zones")

    # Precompute India UTM bboxes per zone
    india_utm = {z: _india_utm_bbox(z) for z in ZONES}

    india_tiles: list[dict] = []

    def _process_vrt(args):
        year, zone, key = args
        url = f"{BASE_URL}/{key}"
        content = _fetch_vrt(url)
        if not content:
            return None
        gt = _parse_vrt_geotransform(content)
        if not gt:
            return None
        bounds = _tile_utm_bounds(gt)
        if not _intersects(bounds, india_utm[zone]):
            return None
        tiff_url = f"{BASE_URL}/{key.replace('.vrt', '.tiff')}"
        return {"year": year, "zone": zone, "tiff_url": tiff_url, "gt": gt, "utm_bounds": bounds}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_vrt, args): args for args in all_vrt_keys}
        for fut in tqdm(as_completed(futures), total=len(all_vrt_keys), desc="Scanning VRTs"):
            result = fut.result()
            if result:
                india_tiles.append(result)

    print(f"  {len(india_tiles)} tiles intersect India bbox")
    with open(INDEX_CACHE, "w") as f:
        json.dump(india_tiles, f)
    return india_tiles


# ── ESA WorldCover helpers ─────────────────────────────────────────────────────

def _esa_tile_name(lat_sw: int, lon_sw: int) -> str:
    ns = "N" if lat_sw >= 0 else "S"
    ew = "E" if lon_sw >= 0 else "W"
    return f"{ns}{abs(lat_sw):02d}{ew}{abs(lon_sw):03d}"


def _esa_urls_for_wgs84_bbox(minx: float, miny: float, maxx: float, maxy: float, year: int) -> list[str]:
    esa_year = 2021  # ESA WorldCover available for 2020 and 2021; use 2021 for both
    urls = []
    for lat in range(int(np.floor(miny / 3)) * 3, int(np.ceil(maxy / 3)) * 3, 3):
        for lon in range(int(np.floor(minx / 3)) * 3, int(np.ceil(maxx / 3)) * 3, 3):
            tile = _esa_tile_name(lat, lon)
            urls.append(ESA_HTTPS.format(year=esa_year, tile=tile))
    return urls


# ── Per-window sampling ────────────────────────────────────────────────────────

def _sample_window(
    tiff_url: str,
    zone: str,
    gt: list[float],
    row_off: int,
    col_off: int,
    rng: np.random.Generator,
    year: int,
) -> pd.DataFrame | None:
    """
    Read a WINDOW_PX×WINDOW_PX sub-window from the AEF tile and match with
    ESA WorldCover labels. Return a DataFrame with training samples.
    """
    ox, dx, _, oy, _, dy = gt
    # Window origin in UTM
    win_x = ox + col_off * dx
    win_y = oy + row_off * dy   # dy is negative
    win_w = WINDOW_PX * dx
    win_h = WINDOW_PX * dy      # negative

    # Convert window corners to WGS84 for ESA lookup
    t_to_wgs = Transformer.from_crs(f"EPSG:{_zone_epsg(zone)}", "EPSG:4326", always_xy=True)
    corners_utm_x = [win_x, win_x + win_w, win_x, win_x + win_w]
    corners_utm_y = [win_y, win_y, win_y + win_h, win_y + win_h]
    corners_lon, corners_lat = t_to_wgs.transform(corners_utm_x, corners_utm_y)
    wgs_minx, wgs_maxx = min(corners_lon), max(corners_lon)
    wgs_miny, wgs_maxy = min(corners_lat), max(corners_lat)

    # ── Read AEF embeddings ──
    try:
        with rasterio.open(f"/vsicurl/{tiff_url}") as src:
            window = rasterio.windows.from_bounds(
                win_x, win_y + win_h, win_x + win_w, win_y, src.transform
            )
            raw = src.read(window=window)  # [64, H, W] int8
    except Exception as e:
        return None

    if raw.shape[0] != 64:
        return None

    H, W = raw.shape[1], raw.shape[2]
    if H == 0 or W == 0:
        return None

    # Dequantize: [64, H, W] → float32
    emb = dequantize(raw)  # [64, H, W]
    nodata_mask = (raw == NODATA).all(axis=0)  # [H, W] — all-nodata pixels

    # ── Read ESA WorldCover labels ──
    esa_urls = _esa_urls_for_wgs84_bbox(wgs_minx, wgs_miny, wgs_maxx, wgs_maxy, year)

    # Build a label grid matching AEF pixel grid
    label_grid = np.full((H, W), -1, dtype=np.int8)

    # Transform from UTM pixel (row, col) → UTM coords
    aef_transform = rasterio.transform.from_origin(win_x, win_y, dx, -dy)

    for esa_url in esa_urls:
        try:
            with rasterio.open(f"/vsicurl/{esa_url}") as esa_src:
                # Reproject AEF pixel centres to ESA CRS and read
                esa_window = rasterio.windows.from_bounds(
                    wgs_minx, wgs_miny, wgs_maxx, wgs_maxy, esa_src.transform
                )
                esa_data = esa_src.read(1, window=esa_window)  # [Hesa, Wesa]
                if esa_data.size == 0:
                    continue
                esa_trans = esa_src.window_transform(esa_window)
        except Exception:
            continue

        # Map AEF pixel centres to ESA rows/cols
        for r in range(H):
            for c in range(W):
                if nodata_mask[r, c] or label_grid[r, c] >= 0:
                    continue
                # UTM coords of AEF pixel centre
                utm_x, utm_y = rasterio.transform.xy(aef_transform, r, c)
                # WGS84 coords
                lon, lat = t_to_wgs.transform(utm_x, utm_y)
                # ESA row/col
                esa_col = int((lon - esa_trans.c) / esa_trans.a)
                esa_row = int((lat - esa_trans.f) / esa_trans.e)
                if 0 <= esa_row < esa_data.shape[0] and 0 <= esa_col < esa_data.shape[1]:
                    native = int(esa_data[esa_row, esa_col])
                    mapped = map_label("esa_worldcover", native)
                    if mapped is not None:
                        label_grid[r, c] = mapped

    # ── Stratified sampling ──
    rows_list = []
    for class_id in range(7):
        ys, xs = np.where(label_grid == class_id)
        if len(ys) == 0:
            continue
        idx = rng.choice(len(ys), size=min(N_SAMPLES_PER_CLASS, len(ys)), replace=False)
        for i in idx:
            r, c = int(ys[i]), int(xs[i])
            pixel_emb = emb[:, r, c]
            if (pixel_emb == NODATA).all():
                continue
            lon, lat = t_to_wgs.transform(
                *rasterio.transform.xy(aef_transform, r, c)
            )
            row = {
                "source": "esa_worldcover",
                "year": year,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "lon": float(lon),
                "lat": float(lat),
            }
            row.update({b: float(pixel_emb[i]) for i, b in enumerate(BAND_COLS)})
            rows_list.append(row)

    return pd.DataFrame(rows_list) if rows_list else None


# Per-pixel loop above is slow for large windows — vectorised replacement below
def _sample_window_fast(
    tiff_url: str,
    zone: str,
    gt: list[float],
    row_off: int,
    col_off: int,
    rng: np.random.Generator,
    year: int,
) -> pd.DataFrame | None:
    """
    Vectorised version: read AEF window + ESA window, resample ESA to AEF grid,
    stratified sample.
    """
    ox, dx, _, oy, _, dy = gt
    win_x = ox + col_off * dx
    win_y = oy + row_off * dy

    # Convert window bounds to WGS84
    t_to_wgs = Transformer.from_crs(f"EPSG:{_zone_epsg(zone)}", "EPSG:4326", always_xy=True)
    win_x2 = win_x + WINDOW_PX * dx
    win_y2 = win_y + WINDOW_PX * dy  # dy negative → win_y2 < win_y

    corners_lon, corners_lat = t_to_wgs.transform(
        [win_x, win_x2, win_x, win_x2],
        [win_y, win_y, win_y2, win_y2],
    )
    wgs_minx = min(corners_lon); wgs_maxx = max(corners_lon)
    wgs_miny = min(corners_lat); wgs_maxy = max(corners_lat)

    # ── AEF read ──
    try:
        with rasterio.open(f"/vsicurl/{tiff_url}") as src:
            # from_bounds expects (left, bottom, right, top)
            window = rasterio.windows.from_bounds(
                win_x, min(win_y, win_y2), win_x2, max(win_y, win_y2), src.transform
            )
            raw = src.read(window=window)  # [64, H, W] int8
            aef_transform = src.window_transform(window)
    except Exception:
        return None

    if raw.shape[0] != 64 or raw.shape[1] == 0 or raw.shape[2] == 0:
        return None

    H, W = raw.shape[1], raw.shape[2]
    emb = dequantize(raw)                         # [64, H, W] float32
    nodata_mask = (raw == NODATA).all(axis=0)     # [H, W]

    # ── ESA read + nearest-neighbour resample to AEF grid ──
    label_grid = np.full((H, W), -1, dtype=np.int8)

    # Build arrays of (lon, lat) for every valid AEF pixel
    rows_arr, cols_arr = np.where(~nodata_mask)
    if len(rows_arr) == 0:
        return None

    utm_xs, utm_ys = rasterio.transform.xy(aef_transform, rows_arr, cols_arr)
    lons, lats = t_to_wgs.transform(utm_xs, utm_ys)
    lons = np.array(lons); lats = np.array(lats)

    for esa_url in _esa_urls_for_wgs84_bbox(wgs_minx, wgs_miny, wgs_maxx, wgs_maxy, year):
        try:
            with rasterio.open(f"/vsicurl/{esa_url}") as esa_src:
                esa_win = rasterio.windows.from_bounds(
                    wgs_minx, wgs_miny, wgs_maxx, wgs_maxy, esa_src.transform
                )
                esa_data = esa_src.read(1, window=esa_win)
                if esa_data.size == 0:
                    continue
                et = esa_src.window_transform(esa_win)
        except Exception:
            continue

        # Vectorised pixel lookup
        esa_cols = np.floor((lons - et.c) / et.a).astype(int)
        esa_rows = np.floor((lats - et.f) / et.e).astype(int)
        valid = (
            (esa_rows >= 0) & (esa_rows < esa_data.shape[0]) &
            (esa_cols >= 0) & (esa_cols < esa_data.shape[1])
        )
        for idx in np.where(valid)[0]:
            r, c = rows_arr[idx], cols_arr[idx]
            if label_grid[r, c] >= 0:
                continue
            native = int(esa_data[esa_rows[idx], esa_cols[idx]])
            mapped = map_label("esa_worldcover", native)
            if mapped is not None:
                label_grid[r, c] = mapped

    # ── Stratified sampling ──
    records = []
    for class_id in range(7):
        ys, xs = np.where(label_grid == class_id)
        if len(ys) == 0:
            continue
        chosen = rng.choice(len(ys), size=min(N_SAMPLES_PER_CLASS, len(ys)), replace=False)
        for i in chosen:
            r, c = int(ys[i]), int(xs[i])
            pixel_emb = emb[:, r, c].tolist()
            lon_px, lat_px = t_to_wgs.transform(
                *rasterio.transform.xy(aef_transform, r, c)
            )
            rec = {
                "source": "esa_worldcover",
                "year": year,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "lon": float(lon_px),
                "lat": float(lat_px),
            }
            rec.update(dict(zip(BAND_COLS, pixel_emb)))
            records.append(rec)

    return pd.DataFrame(records) if records else None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6, help="parallel fetch workers")
    parser.add_argument("--windows", type=int, default=N_WINDOWS_PER_TILE,
                        help="random sub-windows per tile")
    parser.add_argument("--dry-run", action="store_true",
                        help="build tile index only, don't fetch data")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Step 1: build / load tile index
    tiles = build_tile_index(workers=args.workers)
    print(f"\nIndia tiles: {len(tiles)}")
    if args.dry_run:
        print("Dry run — stopping after tile index.")
        return

    # Step 2: sample windows from each tile
    # Check for already-processed intermediate parquets
    interim_dir = OUTPUT_DIR / "india_interim"
    interim_dir.mkdir(exist_ok=True)

    tasks = []
    for tile in tiles:
        for w in range(args.windows):
            # Random offset within the tile (leave 512 px margin)
            margin = WINDOW_PX
            max_offset = TILE_PX - WINDOW_PX - margin
            if max_offset <= 0:
                row_off = col_off = 0
            else:
                row_off = int(rng.integers(margin, max_offset))
                col_off = int(rng.integers(margin, max_offset))
            task_id = f"{tile['year']}_{tile['zone']}_{Path(tile['tiff_url']).stem}_w{w}"
            out_path = interim_dir / f"{task_id}.parquet"
            if out_path.exists():
                continue  # already done — resumable
            tasks.append((tile, row_off, col_off, task_id, out_path))

    print(f"\nTotal windows to fetch: {len(tasks):,}  ({args.windows} per tile)")

    def _run_task(args_t):
        tile, row_off, col_off, task_id, out_path = args_t
        df = _sample_window_fast(
            tile["tiff_url"], tile["zone"], tile["gt"],
            row_off, col_off,
            np.random.default_rng(abs(hash(task_id)) % (2**31)),
            tile["year"],
        )
        if df is not None and len(df) > 0:
            df.to_parquet(out_path, index=False)
            return len(df)
        return 0

    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        with tqdm(as_completed(futures), total=len(tasks), desc="Sampling windows") as pbar:
            for fut in pbar:
                n = fut.result()
                total_rows += n
                pbar.set_postfix(rows=f"{total_rows:,}")

    # Step 3: combine all interim parquets
    print("\nCombining interim parquets…")
    parts = list(interim_dir.glob("*.parquet"))
    if not parts:
        print("No data collected. Check connectivity / tile index.")
        return

    combined = pd.concat([pd.read_parquet(p) for p in tqdm(parts, desc="Reading parts")])
    print(f"  Combined: {len(combined):,} rows")

    # Step 4: Mahalanobis filter
    print("Applying Mahalanobis filter…")
    filtered = mahalanobis_filter(combined, threshold=3.0)
    print(f"  After filter: {len(filtered):,} rows")

    out_path = OUTPUT_DIR / "india_training.parquet"
    filtered.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(filtered.groupby("class_name")["class_id"].count().rename("n_samples").to_string())


if __name__ == "__main__":
    main()
