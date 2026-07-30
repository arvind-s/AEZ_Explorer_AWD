"""Tests for the pure-Python units of awd_pipeline.

These run without live Earth Engine (the sandbox has no EE auth/network).
The EE model functions are validated separately against a small live AOI.
"""
import json

import geopandas as gpd
from shapely.geometry import Polygon

import awd_pipeline


def _write_two_square_shapefile(path):
    """Two edge-sharing 1km squares in UTM 43N (EPSG:32643), near ~18N/73E."""
    sq1 = Polygon([(200000, 2000000), (201000, 2000000),
                   (201000, 2001000), (200000, 2001000)])
    sq2 = Polygon([(201000, 2000000), (202000, 2000000),
                   (202000, 2001000), (201000, 2001000)])
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[sq1, sq2], crs="EPSG:32643")
    gdf.to_file(path)
    return gdf


def test_looks_like_geotiff_accepts_tiff_magic_rejects_error_body():
    # Real GeoTIFFs start with little- or big-endian TIFF magic bytes.
    assert awd_pipeline._looks_like_geotiff(b"II*\x00" + b"\x00" * 40) is True
    assert awd_pipeline._looks_like_geotiff(b"MM\x00*" + b"\x00" * 40) is True
    # GEE returns a JSON/text error body (not a TIFF) when a request is rejected.
    assert awd_pipeline._looks_like_geotiff(
        b'{"error":{"message":"Total request size ... too large"}}') is False
    assert awd_pipeline._looks_like_geotiff(b"") is False


def test_parse_args_defaults_match_script06():
    args = awd_pipeline.parse_args(
        ["--shapefile", "f.shp", "--start", "2020-06-01", "--end", "2020-11-30"]
    )
    assert args.shapefile == "f.shp"
    assert args.start == "2020-06-01"
    assert args.end == "2020-11-30"
    assert args.out == "outputs"
    # Model-threshold defaults mirror 06_gee_awd_full_pixel_model.js.
    assert args.wet == 0.3
    assert args.dry == -0.1
    assert args.ref_cycles == 5
    assert args.awd_thresh == 0.4
    assert args.scale == 25


def test_parse_args_overrides_thresholds():
    args = awd_pipeline.parse_args(
        ["--shapefile", "f.shp", "--start", "2020-06-01", "--end", "2020-11-30",
         "--wet", "0.4", "--dry", "-0.2", "--ref-cycles", "3",
         "--awd-thresh", "0.5", "--scale", "50", "--out", "myout"]
    )
    assert args.wet == 0.4
    assert args.dry == -0.2
    assert args.ref_cycles == 3
    assert args.awd_thresh == 0.5
    assert args.scale == 50
    assert args.out == "myout"


def test_write_stats_emits_matching_json_and_csv(tmp_path):
    stats = {
        "awd_area_ha": 123.45,
        "continuous_flood_area_ha": 67.8,
        "paddy_area_ha": 191.25,
        "pct_awd": 64.55,
        "n_images": 7,
        "season_start": "2020-06-01",
        "season_end": "2020-11-30",
    }

    json_path, csv_path = awd_pipeline.write_stats(stats, str(tmp_path))

    # JSON round-trips exactly.
    with open(json_path) as f:
        assert json.load(f) == stats

    # CSV has a header row + exactly one data row, same keys and values.
    import csv
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0]) == set(stats)
    assert rows[0]["n_images"] == "7"
    assert float(rows[0]["awd_area_ha"]) == 123.45


def test_load_aoi_dissolves_and_reprojects_to_wgs84(tmp_path):
    shp = tmp_path / "fields.shp"
    _write_two_square_shapefile(shp)

    geom = awd_pipeline.load_aoi(str(shp))

    # Dissolved: two edge-sharing squares become a single Polygon feature.
    assert geom["type"] == "Polygon"
    # Reprojected to WGS84: every coordinate is a lon/lat in degrees, not the
    # ~200000 m UTM eastings of the input. If reprojection were skipped, these
    # would be six-figure numbers.
    coords = geom["coordinates"][0]
    assert all(abs(x) < 180 and abs(y) < 90 for x, y in coords)
    # Sanity: lands in western-India lon/lat, not (0,0).
    lons = [x for x, y in coords]
    lats = [y for x, y in coords]
    assert 70 < min(lons) < 80
    assert 15 < min(lats) < 20
