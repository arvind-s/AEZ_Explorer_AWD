#!/usr/bin/env python3
"""AWD shapefile -> stats + GeoTIFF pipeline (PALSAR-2 L-band, via Earth Engine).

Shapefile in -> per-pixel Alternate-Wetting-and-Drying classification out, as a
GeoTIFF plus aggregate acreage stats (JSON + CSV). Faithful port of
06_gee_awd_full_pixel_model.js to the Earth Engine Python API, with a shapefile
AOI and all model thresholds exposed as CLI flags.

    python awd_pipeline.py --shapefile fields.shp \
        --start 2020-06-01 --end 2020-11-30 --out outputs/

CAVEATS (carried over from 06_gee_awd_full_pixel_model.js -- read before
trusting output):
  * PALSAR-2 ScanSAR HH only (dual-pol not guaranteed on every scene).
  * Per-pixel drying-cycle count is a shifted-array simplification of the
    sequential wet->dry->rewet state machine in utils.detect_drying_cycles;
    validate acreage against known AWD/non-AWD fields before trusting it.
  * Speckle filtering is a focal-median smooth, not the adaptive Refined Lee
    filter in utils.py.
  * No rice-crop mask -- "ever flooded" is used as a rough paddy proxy, so other
    seasonally-flooded land (wetlands, ponds) can be misclassified as paddy.
  * Fixed WI thresholds are a starting guess, not yet calibrated to ground truth.
  * Start with a SMALL AOI (a village/block) for the first run -- per-pixel array
    ops over a large area are expensive and hit the direct-download ceiling.
"""
import argparse
import csv
import json
import os

import geopandas as gpd

PALSAR2_COLLECTION = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"


def parse_args(argv=None):
    """Parse CLI args. Threshold defaults mirror 06_gee_awd_full_pixel_model.js."""
    p = argparse.ArgumentParser(
        description="AWD detection from a shapefile AOI -> GeoTIFF + acreage stats "
                    "(PALSAR-2 L-band, via Earth Engine).")
    p.add_argument("--shapefile", required=True, help="Input AOI shapefile (.shp).")
    p.add_argument("--start", required=True, help="Season start date, YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="Season end date, YYYY-MM-DD.")
    p.add_argument("--out", default="outputs", help="Output directory (default: outputs).")
    p.add_argument("--project", default=None,
                   help="Earth Engine Cloud project id (for ee.Initialize).")
    p.add_argument("--collection", default=PALSAR2_COLLECTION,
                   help="Earth Engine SAR ImageCollection id.")
    p.add_argument("--wet", type=float, default=0.3,
                   help="Wetness index >= this counts as flooded (default 0.3).")
    p.add_argument("--dry", type=float, default=-0.1,
                   help="Wetness index <= this counts as dried down (default -0.1).")
    p.add_argument("--ref-cycles", type=int, default=5, dest="ref_cycles",
                   help="Drying cycles/season considered fully AWD -> likelihood 1.0 "
                        "(default 5).")
    p.add_argument("--awd-thresh", type=float, default=0.4, dest="awd_thresh",
                   help="AWD likelihood >= this is classified AWD (default 0.4).")
    p.add_argument("--scale", type=int, default=25,
                   help="Export/reduce scale in metres (default 25, PALSAR-2 native).")
    return p.parse_args(argv)


def load_aoi(shapefile_path):
    """Read a shapefile, reproject to WGS84, dissolve all polygons into one AOI.

    Returns a GeoJSON geometry dict (the dissolved union in EPSG:4326), ready to
    wrap in ee.Geometry(...). All features are merged: the whole shapefile is
    treated as a single area of interest (aggregate-stats design).
    """
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError(
            f"{shapefile_path} has no CRS; cannot reproject to WGS84. "
            "Assign one (e.g. with a .prj file) and retry."
        )
    gdf = gdf.to_crs("EPSG:4326")
    dissolved = gdf.geometry.union_all()
    return dissolved.__geo_interface__


def _ee():
    """Lazy import of earthengine-api so the pure units/tests never touch EE."""
    import ee
    return ee


def init_ee(project=None):
    """Initialise Earth Engine. Assumes a one-time `earthengine authenticate`.

    Raises a friendly error pointing at the auth step if credentials are missing.
    """
    ee = _ee()
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # ee.EEException and auth errors both land here
        raise RuntimeError(
            "Earth Engine init failed. Run `earthengine authenticate` once, and "
            "pass --project <your-ee-cloud-project> if your account needs it. "
            f"Underlying error: {exc}"
        ) from exc


def build_awd_image(geometry, start, end, params, collection=PALSAR2_COLLECTION):
    """Per-pixel AWD classification image. Direct port of section 1-4 of
    06_gee_awd_full_pixel_model.js (PALSAR-2 ScanSAR, HH only).

    Returns (ee.Image, n_images). The image has bands: awd_likelihood, is_awd,
    is_continuous_flood, n_cycles, ever_flooded.
    """
    ee = _ee()
    col = (ee.ImageCollection(collection)
           .filterBounds(geometry)
           .filterDate(start, end)
           .sort("system:time_start"))
    n_images = col.size().getInfo()

    def calibrate_hh(image):
        valid = image.select("MSK").eq(1)
        hh_dn = image.select("HH").updateMask(image.select("HH").gt(0))
        hh_db = hh_dn.pow(2).log10().multiply(10).subtract(83.0).rename("HH_dB")
        return hh_db.updateMask(valid).copyProperties(image, ["system:time_start"])

    def despeckle(image):
        return (image.focal_median(radius=1.5, kernelType="square", units="pixels")
                .rename(image.bandNames())
                .copyProperties(image, ["system:time_start"]))

    hh = col.map(calibrate_hh).map(despeckle)

    # Per-pixel seasonal min/max -> Wetness Index rescaled to -1..+1.
    min_img = hh.min()
    max_img = hh.max()
    rng = max_img.subtract(min_img)
    valid_range = rng.gt(0)  # guard divide-by-zero on perfectly flat pixels

    def wetness_index(img):
        wi = (ee.Image(1)
              .subtract(ee.Image(2).multiply(img.subtract(min_img)).divide(rng))
              .updateMask(valid_range).rename("WI"))
        return wi.copyProperties(img, ["system:time_start"])

    wi_col = hh.map(wetness_index)

    # Drying-cycle count via shifted-array "dry@t -> wet@t+1" (caveat 2 in #06).
    wi_array = wi_col.toArray().arrayProject([0])
    wet_bool = wi_array.gte(params["wet"])
    dry_bool = wi_array.lte(params["dry"])
    dry_at_t = dry_bool.arraySlice(0, 0, -1)
    wet_at_t1 = wet_bool.arraySlice(0, 1)
    reflood = dry_at_t.multiply(wet_at_t1)
    n_cycles = (reflood.arrayReduce(reducer=ee.Reducer.sum(), axes=[0])
                .arrayGet([0]).rename("n_cycles"))

    awd_likelihood = (n_cycles.divide(params["ref_cycles"]).min(1).max(0)
                      .rename("awd_likelihood"))
    ever_flooded = wi_col.max().gte(params["wet"]).rename("ever_flooded")

    is_awd = awd_likelihood.gte(params["awd_thresh"]).And(ever_flooded).rename("is_awd")
    is_continuous_flood = ever_flooded.And(is_awd.Not()).rename("is_continuous_flood")

    image = (awd_likelihood
             .addBands(is_awd)
             .addBands(is_continuous_flood)
             .addBands(n_cycles)
             .addBands(ever_flooded))
    return image, n_images


def compute_stats(image, geometry, args, n_images):
    """Aggregate acreage stats over the whole AOI (reduceRegion sum of per-class
    pixel-area). Port of section 5 of #06, plus paddy total and %AWD."""
    ee = _ee()
    area_ha = ee.Image.pixelArea().divide(10000)

    def _sum(mask_band):
        return area_ha.updateMask(image.select(mask_band)).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geometry, scale=args.scale,
            maxPixels=1e13, bestEffort=True).get("area")

    raw = ee.Dictionary({
        "awd_area_ha": _sum("is_awd"),
        "continuous_flood_area_ha": _sum("is_continuous_flood"),
        "paddy_area_ha": _sum("ever_flooded"),
    }).getInfo()

    awd = raw.get("awd_area_ha") or 0.0
    flood = raw.get("continuous_flood_area_ha") or 0.0
    paddy = raw.get("paddy_area_ha") or 0.0
    pct_awd = (awd / paddy * 100.0) if paddy > 0 else 0.0

    return {
        "awd_area_ha": round(awd, 4),
        "continuous_flood_area_ha": round(flood, 4),
        "paddy_area_ha": round(paddy, 4),
        "pct_awd": round(pct_awd, 2),
        "n_images": n_images,
        "season_start": args.start,
        "season_end": args.end,
        "wet_threshold": args.wet,
        "dry_threshold": args.dry,
        "ref_cycles": args.ref_cycles,
        "awd_likelihood_threshold": args.awd_thresh,
        "scale_m": args.scale,
        "collection": args.collection,
    }


def download_geotiff(image, geometry, scale, out_path):
    """Download the classification image as a local GeoTIFF via getDownloadURL.

    Raises a clear, actionable error (tile the AOI / raise --scale) if GEE
    rejects the request for exceeding the direct-download size ceiling, instead
    of surfacing a raw traceback or writing a corrupt file.
    """
    import urllib.request

    ee = _ee()
    try:
        url = image.clip(geometry).getDownloadURL({
            "scale": scale,
            "region": geometry,
            "format": "GEO_TIFF",
            "filePerBand": False,
        })
    except Exception as exc:
        raise RuntimeError(_size_hint(exc)) from exc

    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    # A too-large or failed request comes back as a JSON/text error body, not a
    # GeoTIFF. Guard against writing that error out as a corrupt .tif.
    if not _looks_like_geotiff(data):
        raise RuntimeError(_size_hint(data[:800].decode("utf-8", "replace")))

    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def _looks_like_geotiff(data):
    """True if bytes begin with little-/big-endian TIFF magic; False otherwise
    (e.g. a GEE JSON error body returned when a download is rejected)."""
    return data[:4] in (b"II*\x00", b"MM\x00*")


def _size_hint(detail):
    return (
        "GeoTIFF direct-download failed (most often the AOI exceeds Earth "
        "Engine's ~50 MB direct-download ceiling). Options: run on a smaller "
        "AOI (a single village/block), raise --scale (coarser pixels = smaller "
        "file), or tile the shapefile. "
        f"Details: {detail}"
    )


def write_stats(stats, out_dir):
    """Write the aggregate stats dict as awd_stats.json + awd_stats.csv.

    Returns (json_path, csv_path). The CSV is a single header row + one data row
    (whole-shapefile aggregate design).
    """
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "awd_stats.json")
    csv_path = os.path.join(out_dir, "awd_stats.csv")

    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats))
        writer.writeheader()
        writer.writerow(stats)

    return json_path, csv_path


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    print(f"[1/5] Loading AOI from {args.shapefile} ...")
    geojson = load_aoi(args.shapefile)

    print("[2/5] Initialising Earth Engine ...")
    init_ee(args.project)
    ee = _ee()
    geometry = ee.Geometry(geojson)

    print(f"[3/5] Building AWD model for {args.start}..{args.end} "
          f"({args.collection}) ...")
    params = {
        "wet": args.wet, "dry": args.dry, "ref_cycles": args.ref_cycles,
        "awd_thresh": args.awd_thresh,
    }
    image, n_images = build_awd_image(geometry, args.start, args.end, params,
                                      args.collection)
    print(f"      scenes in season: {n_images}")
    if n_images < 5:
        print("      WARNING: fewer than 5 scenes -- PALSAR-2 ScanSAR revisit is "
              "too sparse here for reliable per-pixel drying-cycle counts. Treat "
              "the output as a snapshot, not a validated AWD map.")

    print("[4/5] Computing acreage stats ...")
    stats = compute_stats(image, geometry, args, n_images)
    json_path, csv_path = write_stats(stats, args.out)
    print(f"      AWD: {stats['awd_area_ha']} ha | "
          f"continuous flood: {stats['continuous_flood_area_ha']} ha | "
          f"paddy total: {stats['paddy_area_ha']} ha | "
          f"%AWD: {stats['pct_awd']}")

    print("[5/5] Downloading classification GeoTIFF ...")
    export_image = image.select(
        ["awd_likelihood", "is_awd", "is_continuous_flood", "n_cycles"])
    tif_path = download_geotiff(export_image, geometry, args.scale,
                                os.path.join(args.out, "awd_classification.tif"))

    print("\nDone. Outputs:")
    print(f"  GeoTIFF : {tif_path}")
    print(f"  Stats   : {json_path}")
    print(f"            {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
