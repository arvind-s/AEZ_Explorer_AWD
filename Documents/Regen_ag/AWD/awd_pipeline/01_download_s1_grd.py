#!/usr/bin/env python3
"""
Download Sentinel-1 imagery for an AOI + date range from Microsoft Planetary
Computer (MPC), clipped to the AOI so we don't pull full ~250x170 km scenes.

Supports two MPC collections:
  --collection sentinel-1-rtc   (recommended, default)
      Already radiometrically terrain corrected + calibrated to gamma-naught
      by Microsoft/Catalyst. Simpler and more robust -- no manual calibration
      LUT parsing needed. Requires a free Planetary Computer account/
      subscription key (msft:requires_account=true for this collection):
      request one at https://planetarycomputer.microsoft.com/account/request
      and set env var PC_SDK_SUBSCRIPTION_KEY.

  --collection sentinel-1-grd    (raw GRD, matches literature's "S1 GRD")
      No account needed, but pixel values are amplitude DN and need manual
      calibration (see utils.amplitude_dn_to_beta_naught_db /
      parse_calibration_lut) before they're usable as backscatter. Use this
      if you specifically need to replicate Lovell (2019)'s beta-nought
      methodology, or don't have/want a PC account yet.

IMPORTANT: this script could not be executed against real MPC data in the
build sandbox because outbound network access there is restricted to an
allowlist that excludes every satellite-data host (Planetary Computer, Google
Earth Engine, AWS Open Data, Copernicus Data Space were all tested and
blocked alike -- this is a sandbox-wide restriction, not specific to any one
provider). Run this on a machine/VM with normal internet access. The logic
itself follows the standard pystac-client + planetary-computer + rasterio
pattern documented at https://planetarycomputer.microsoft.com/docs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pystac_client
import planetary_computer
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT

from utils import load_aoi_geojson

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_scenes(aoi_bbox, start_date: str, end_date: str, collection: str):
    catalog = pystac_client.Client.open(
        MPC_STAC_URL, modifier=planetary_computer.sign_inplace
    )
    search = catalog.search(
        collections=[collection],
        bbox=aoi_bbox,
        datetime=f"{start_date}/{end_date}",
        query={"sar:instrument_mode": {"eq": "IW"}},
    )
    items = list(search.items())
    items.sort(key=lambda it: it.datetime)
    return items


def clip_asset_to_aoi(asset_href: str, aoi_geom: dict, out_path: Path):
    """Windowed-read + mask an MPC COG asset to the AOI, writing a small
    local GeoTIFF instead of pulling the full scene."""
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(asset_href) as src:
            with WarpedVRT(src, crs="EPSG:4326") as vrt:
                out_image, out_transform = rio_mask(vrt, [aoi_geom], crop=True)
                out_meta = vrt.meta.copy()
                out_meta.update(
                    {
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "driver": "GTiff",
                        "compress": "deflate",
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", default="aoi_sathupalle_wgs84.geojson", help="AOI GeoJSON (WGS84)")
    ap.add_argument("--start", default="2025-06-01", help="Start date (kharif 2025 default)")
    ap.add_argument("--end", default="2025-11-30", help="End date")
    ap.add_argument("--collection", default="sentinel-1-rtc", choices=["sentinel-1-rtc", "sentinel-1-grd"])
    ap.add_argument("--out", default="raw_scenes", help="Output directory")
    ap.add_argument("--polarizations", nargs="+", default=["vv", "vh"])
    args = ap.parse_args()

    aoi_geom, aoi_bbox = load_aoi_geojson(args.aoi)
    print(f"AOI bbox (WGS84): {aoi_bbox}")

    items = search_scenes(aoi_bbox, args.start, args.end, args.collection)
    print(f"Found {len(items)} {args.collection} items between {args.start} and {args.end}")

    manifest = []
    out_dir = Path(args.out)
    for item in items:
        date_str = item.datetime.strftime("%Y%m%dT%H%M%S")
        orbit = item.properties.get("sat:orbit_state", "unknown")
        record = {"id": item.id, "datetime": item.datetime.isoformat(), "orbit_state": orbit, "assets": {}}
        for pol in args.polarizations:
            if pol not in item.assets:
                continue
            href = item.assets[pol].href  # already signed via sign_inplace
            out_path = out_dir / f"{date_str}_{orbit}_{pol}.tif"
            print(f"  Clipping {item.id} [{pol}] -> {out_path}")
            try:
                clip_asset_to_aoi(href, aoi_geom, out_path)
                record["assets"][pol] = str(out_path)
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {e}")
        manifest.append(record)

    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
