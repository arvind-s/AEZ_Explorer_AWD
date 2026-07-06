from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape

from src.harmonize import CLASS_NAMES

TREES_CLASS_ID = 1
TREE_PROB_THRESHOLD = 0.6
MIN_CONSECUTIVE_YEARS = 2
TREE_FRACTION_THRESHOLD = 0.30
CONFIDENCE_THRESHOLD = 0.50
ESTABLISHMENT_FRACTION = 0.20


def detect_plantation_pixels(
    tree_prob_stack: np.ndarray,
    threshold: float = TREE_PROB_THRESHOLD,
    min_consecutive: int = MIN_CONSECUTIVE_YEARS,
) -> np.ndarray:
    """
    Detect pixels with plantation presence.

    tree_prob_stack: [Y, H, W] float32 — Trees-class probabilities per year
    Returns: [H, W] bool — True where plantation confirmed
    """
    Y, H, W = tree_prob_stack.shape
    above = tree_prob_stack > threshold  # [Y, H, W] bool

    # Count max consecutive years above threshold per pixel
    max_consecutive = np.zeros((H, W), dtype=np.int32)
    current_run = np.zeros((H, W), dtype=np.int32)

    for y in range(Y):
        current_run = np.where(above[y], current_run + 1, 0)
        max_consecutive = np.maximum(max_consecutive, current_run)

    return max_consecutive >= min_consecutive


def _first_consecutive_year(
    above: np.ndarray,
    years: list[int],
    min_consecutive: int,
) -> np.ndarray:
    """
    Return year of first confirmed plantation per pixel.
    above: [Y, H, W] bool
    Returns [H, W] int — year value (0 if never plantation)
    """
    Y, H, W = above.shape
    result = np.zeros((H, W), dtype=np.int32)
    current_run = np.zeros((H, W), dtype=np.int32)
    confirmed = np.zeros((H, W), dtype=bool)
    start_year = np.zeros((H, W), dtype=np.int32)

    for i, year in enumerate(years):
        newly_started = (current_run == 0) & above[i]
        start_year = np.where(newly_started, year, start_year)
        current_run = np.where(above[i], current_run + 1, 0)
        just_confirmed = (current_run == min_consecutive) & ~confirmed
        result = np.where(just_confirmed, start_year, result)
        confirmed = confirmed | just_confirmed

    return result


def _load_tree_prob_stack(
    inference_paths: dict[int, dict[str, Path]],
) -> tuple[np.ndarray, np.ndarray, list[int], Any, Any]:
    """Load tree_prob GeoTIFFs. Returns (stack, lulc_stack, years, transform, crs)."""
    years = sorted(inference_paths.keys())
    arrays, lulc_arrays = [], []
    transform, crs = None, None

    for year in years:
        with rasterio.open(inference_paths[year]["tree_prob"]) as src:
            arrays.append(src.read(1))
            if transform is None:
                transform = src.transform
                crs = src.crs
        with rasterio.open(inference_paths[year]["lulc"]) as src:
            lulc_arrays.append(src.read(1))

    return (
        np.stack(arrays, axis=0).astype(np.float32),
        np.stack(lulc_arrays, axis=0).astype(np.int32),
        years,
        transform,
        crs,
    )


def generate_verdict(
    parcel_id: str,
    inference_paths: dict[int, dict[str, Path]],
    parcel_geojson: dict,
    out_dir: Path,
) -> dict:
    """
    Generate plantation verification verdict for a parcel.

    parcel_geojson: GeoJSON Geometry (Polygon) in same CRS as GeoTIFFs
    Returns structured verdict dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_stack, lulc_stack, years, transform, crs = _load_tree_prob_stack(inference_paths)
    Y, H, W = tree_stack.shape

    # Parcel mask (True = inside parcel)
    parcel_geom = shape(parcel_geojson)
    mask = geometry_mask(
        [parcel_geom.__geo_interface__],
        out_shape=(H, W),
        transform=transform,
        invert=True,
    )

    # Per-pixel plantation detection
    plantation_px = detect_plantation_pixels(tree_stack, TREE_PROB_THRESHOLD, MIN_CONSECUTIVE_YEARS)

    # Establishment year raster
    above = tree_stack > TREE_PROB_THRESHOLD
    est_year_map = _first_consecutive_year(above, years, MIN_CONSECUTIVE_YEARS)

    # Per-parcel metrics
    parcel_pixels = mask.sum()
    if parcel_pixels == 0:
        parcel_pixels = H * W  # fallback: no mask applied

    latest_year = years[-1]
    latest_tree_prob = tree_stack[-1]
    latest_lulc = lulc_stack[-1]

    # tree_pixel_fraction in latest year (plantation pixels inside parcel)
    latest_plantation_mask = plantation_px & mask
    tree_pixel_fraction = float(latest_plantation_mask.sum()) / parcel_pixels

    # Confidence: mean Trees prob over plantation pixels in latest year
    plantation_probs = latest_tree_prob[latest_plantation_mask]
    if len(plantation_probs) > 0:
        persistence_fraction = float(plantation_px[mask].mean()) if mask.any() else 0.0
        confidence = float(plantation_probs.mean()) * persistence_fraction
    else:
        confidence = 0.0

    # Establishment year: first year where ≥20% of parcel pixels are plantation
    establishment_year = None
    for i, year in enumerate(years):
        above_year = tree_stack[i] > TREE_PROB_THRESHOLD
        frac = float((above_year & mask).sum()) / parcel_pixels
        if frac >= ESTABLISHMENT_FRACTION:
            establishment_year = year
            break

    # Prior dominant class (years before establishment)
    if establishment_year is not None:
        est_idx = years.index(establishment_year)
        if est_idx > 0:
            prior_lulc = lulc_stack[:est_idx, mask].ravel()
        else:
            prior_lulc = lulc_stack[0, mask].ravel()
        if prior_lulc.size > 0:
            vals, cnts = np.unique(prior_lulc, return_counts=True)
            prior_class_id = int(vals[cnts.argmax()])
        else:
            prior_class_id = int(lulc_stack[0][mask].ravel().mean().round())
    else:
        prior_class_id = int(np.bincount(lulc_stack[-1][mask].ravel()).argmax())

    prior_class_name = CLASS_NAMES.get(prior_class_id, "unknown")

    # LULC timeseries (class fractions per year)
    lulc_timeseries: dict[str, dict[str, float]] = {}
    for i, year in enumerate(years):
        px = lulc_stack[i][mask].ravel()
        counts = np.bincount(px, minlength=7)
        total = counts.sum() or 1
        lulc_timeseries[str(year)] = {
            CLASS_NAMES[c]: float(counts[c] / total)
            for c in range(7)
            if counts[c] > 0
        }

    plantation_present = bool(
        tree_pixel_fraction > TREE_FRACTION_THRESHOLD and confidence > CONFIDENCE_THRESHOLD
    )

    verdict = {
        "parcel_id": parcel_id,
        "plantation_present": plantation_present,
        "confidence": round(confidence, 4),
        "establishment_year": establishment_year,
        "tree_pixel_fraction_latest": round(tree_pixel_fraction, 4),
        "prior_dominant_class": prior_class_name,
        "lulc_timeseries": lulc_timeseries,
        "outputs": {
            "lulc_maps": f"lulc_{{year}}.tif",
            "tree_prob_maps": f"tree_prob_{{year}}.tif",
            "establishment_raster": f"establishment_year_{parcel_id}.tif",
        },
    }

    # Write establishment-year raster
    est_path = out_dir / f"establishment_year_{parcel_id}.tif"
    with rasterio.open(
        est_path, "w",
        driver="GTiff", dtype="uint16", count=1, height=H, width=W,
        crs=crs, transform=transform, compress="lzw",
    ) as dst:
        dst.write(est_year_map.astype(np.uint16)[np.newaxis, :, :])

    # Write JSON verdict
    verdict_path = out_dir / f"verdict_{parcel_id}.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)

    return verdict
