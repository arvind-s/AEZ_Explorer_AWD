from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

TREES_CLASS_ID = 1
BAND_COLS = [f"A{i:02d}" for i in range(64)]


def run_inference(
    ds: xr.Dataset,
    model,
    class_map: dict[int, str],
    parcel_dir: Path,
) -> dict[int, dict[str, Path]]:
    """
    Run per-year LULC classification on the AEF embedding dataset.

    ds: xr.Dataset from fetch_embeddings — dims (year, y, x, band)
    model: fitted XGBClassifier (or any sklearn-compatible predict_proba)
    class_map: {class_id: class_name}
    parcel_dir: output directory for GeoTIFFs

    Returns: {year: {"lulc": Path, "tree_prob": Path}}
    """
    parcel_dir.mkdir(parents=True, exist_ok=True)
    years = ds["year"].values.tolist()
    transform_coeffs = ds.attrs.get("transform", [])
    crs_str = ds.attrs.get("crs", "EPSG:4326")

    if len(transform_coeffs) >= 6:
        transform = Affine(*transform_coeffs[:6])
    else:
        transform = Affine.identity()

    try:
        crs = CRS.from_string(crs_str)
    except Exception:
        crs = CRS.from_epsg(4326)

    results: dict[int, dict[str, Path]] = {}

    for year in years:
        emb = ds["embeddings"].sel(year=year).values  # [H, W, 64]
        H, W, _ = emb.shape
        X = emb.reshape(-1, 64).astype(np.float32)

        probs = model.predict_proba(X)  # [N, 7]
        n_classes = probs.shape[1]

        lulc_flat = probs.argmax(axis=1).astype(np.uint8)
        tree_prob_flat = probs[:, TREES_CLASS_ID].astype(np.float32)

        lulc = lulc_flat.reshape(H, W)
        tree_prob = tree_prob_flat.reshape(H, W)

        profile = {
            "driver": "GTiff",
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }

        lulc_path = parcel_dir / f"lulc_{year}.tif"
        with rasterio.open(lulc_path, "w", **{**profile, "dtype": "uint8", "count": 1, "height": H, "width": W}) as dst:
            dst.write(lulc[np.newaxis, :, :])

        tree_path = parcel_dir / f"tree_prob_{year}.tif"
        with rasterio.open(tree_path, "w", **{**profile, "dtype": "float32", "count": 1, "height": H, "width": W}) as dst:
            dst.write(tree_prob[np.newaxis, :, :])

        results[int(year)] = {"lulc": lulc_path, "tree_prob": tree_path}

    return results
