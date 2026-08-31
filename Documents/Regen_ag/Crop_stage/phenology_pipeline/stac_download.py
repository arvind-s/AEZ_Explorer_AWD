"""Locate or download STAC S1/S2 time series."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

from .config import PhenologyConfig


def _ensure_stac_on_path(stac_root: Path) -> Path:
    root = Path(stac_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"STAC pipeline root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resolve_farm_row(
    gdf: gpd.GeoDataFrame,
    farm_id_col: str,
    farm_id: str | None,
) -> tuple[gpd.GeoDataFrame, str]:
    if farm_id_col not in gdf.columns:
        raise ValueError(f"Column '{farm_id_col}' not in polygon file")
    if farm_id is not None:
        subset = gdf[gdf[farm_id_col].astype(str) == str(farm_id)].copy()
        if subset.empty:
            raise ValueError(f"farm_id={farm_id} not found in {farm_id_col}")
        return subset, str(farm_id)
    if len(gdf) != 1:
        raise ValueError("polygon file has multiple features — set farm_id or use single polygon")
    uid = str(gdf[farm_id_col].iloc[0])
    return gdf.copy(), uid


def _artifact_paths(run_dir: Path, tag: str) -> dict:
    s1_dir = run_dir / "s1"
    s2_dir = run_dir / "s2"
    s1_nc = s1_dir / f"s1_timeseries_{tag}.nc"
    s2_nc = s2_dir / f"s2_timeseries_{tag}.nc"
    s1_csv = s1_dir / f"s1_timeseries_{tag}.csv"
    s2_csv = s2_dir / f"s2_timeseries_{tag}.csv"
    return {
        "s1_nc": s1_nc if s1_nc.exists() else None,
        "s2_nc": s2_nc if s2_nc.exists() else None,
        "s1_csv": s1_csv if s1_csv.exists() else None,
        "s2_csv": s2_csv if s2_csv.exists() else None,
    }


def find_cached_timeseries(
    cache_root: Path,
    start_date: str,
    end_date: str,
) -> dict | None:
    """Find an existing pipeline_runner output with S2 NetCDF for the date tag."""
    tag = f"{start_date}_{end_date}"
    root = Path(cache_root)
    if not root.exists():
        return None

    best: tuple[Path, dict] | None = None
    for s2_nc in root.rglob(f"s2_timeseries_{tag}.nc"):
        run_dir = s2_nc.parent.parent
        artifacts = _artifact_paths(run_dir, tag)
        if artifacts["s2_nc"] is None:
            continue
        if best is None or run_dir.stat().st_mtime > best[0].stat().st_mtime:
            best = (run_dir, artifacts)

    if best is None:
        return None

    run_dir, artifacts = best
    return {"run_dir": run_dir, "tag": tag, **artifacts}


def download_timeseries(config: PhenologyConfig, force: bool = False) -> dict:
    """
    Run STAC pipeline_runner for one farm polygon, or reuse cached NetCDF.

    Returns dict with paths: run_dir, uid, s1_nc, s2_nc, s1_csv, s2_csv
    """
    out_base = Path(config.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(config.polygon_path)
    farm_gdf, uid = _resolve_farm_row(gdf, config.farm_id_col, config.farm_id)

    cache_root = Path(config.timeseries_cache_dir or (out_base / "stac_download"))
    if not force:
        cached = find_cached_timeseries(cache_root, config.download_start, config.download_end)
        if cached is not None:
            print(f"Using cached STAC output: {cached['run_dir']}")
            return {
                "run_dir": cached["run_dir"],
                "uid": uid,
                "s1_nc": cached.get("s1_nc"),
                "s2_nc": cached.get("s2_nc"),
                "s1_csv": cached.get("s1_csv"),
                "s2_csv": cached.get("s2_csv"),
                "from_cache": True,
            }

    stac_root = _ensure_stac_on_path(config.stac_root)
    from pipeline_runner import (
        OutputConfig,
        PipelineConfig,
        S1Config,
        S2CloudConfig,
        S2Config,
        run_pipeline,
    )

    aoi_path = out_base / f"aoi_{uid}.geojson"
    farm_gdf.to_file(aoi_path, driver="GeoJSON")

    pc = PipelineConfig(
        input_path=str(aoi_path),
        uid_col=config.farm_id_col,
        start_date=config.download_start,
        end_date=config.download_end,
        interval=config.stac_interval,
        resolution=config.stac_resolution_m,
        max_workers=config.stac_max_workers,
        s1=S1Config(enabled=True, bands=["vh"]),
        s2=S2Config(
            enabled=True,
            indices=["NDVI"],
            bands=[],
            cloud=S2CloudConfig(use_scl=True, max_scene_cloud=90),
        ),
        output=OutputConfig(
            directory=str(cache_root),
            formats=["netcdf", "csv"],
            nan_fill=-9999.0,
            netcdf_agg="mean",
        ),
    )

    result = run_pipeline(pc)
    if not result.success:
        raise RuntimeError(f"STAC download failed: {result.errors}")

    run_dir = result.output_dir
    tag = f"{config.download_start}_{config.download_end}"
    artifacts = _artifact_paths(run_dir, tag)

    return {
        "run_dir": run_dir,
        "uid": uid,
        "aoi_path": aoi_path,
        "from_cache": False,
        **artifacts,
    }
