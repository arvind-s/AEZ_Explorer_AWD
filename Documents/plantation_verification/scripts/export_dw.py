"""
Export Dynamic World annual composite labels from Google Earth Engine.

Usage:
    pip install earthengine-api
    earthengine authenticate
    python scripts/export_dw.py \
        --bbox "minx miny maxx maxy" \
        --years 2020 2021 2022 \
        --out_dir data/training/dw_exports

This writes one GeoTIFF per year to out_dir. Each pixel value is the
modal DW class (0–8) for that year. The export runs on GEE and downloads
to local disk.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee


DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
DW_LABEL_BAND = "label"


def _annual_modal(bbox: list[float], year: int) -> ee.Image:
    region = ee.Geometry.BBox(*bbox)
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    modal = (
        ee.ImageCollection(DW_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
        .select(DW_LABEL_BAND)
        .reduce(ee.Reducer.mode())
        .rename(DW_LABEL_BAND)
        .clip(region)
    )
    return modal


def export_year(
    bbox: list[float],
    year: int,
    out_dir: Path,
    scale: int = 10,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dw_{year}.tif"
    if out_path.exists():
        print(f"  {out_path} already exists, skipping")
        return

    region = ee.Geometry.BBox(*bbox)
    image = _annual_modal(bbox, year)

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=f"dw_{year}",
        folder="aef_dw_exports",
        fileNamePrefix=f"dw_{year}",
        region=region,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    print(f"  GEE export started for {year}: task id = {task.id}")

    # Poll until complete
    while task.active():
        time.sleep(30)
        status = task.status()
        print(f"    {year} → {status['state']}")

    if task.status()["state"] != "COMPLETED":
        raise RuntimeError(f"GEE export failed for {year}: {task.status()}")

    print(f"  {year} export complete — download from Google Drive: aef_dw_exports/dw_{year}.tif")
    print(f"  Then move to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True,
                        help="'minx miny maxx maxy' in EPSG:4326")
    parser.add_argument("--years", nargs="+", type=int,
                        default=list(range(2017, 2024)))
    parser.add_argument("--out_dir", default="data/training/dw_exports")
    parser.add_argument("--scale", type=int, default=10)
    args = parser.parse_args()

    ee.Initialize()
    bbox = [float(x) for x in args.bbox.split()]
    out_dir = Path(args.out_dir)

    for year in args.years:
        print(f"Exporting DW {year}…")
        export_year(bbox, year, out_dir, args.scale)


if __name__ == "__main__":
    main()
