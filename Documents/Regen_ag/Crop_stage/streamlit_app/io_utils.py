"""Upload helpers for shapefile / GeoJSON inputs."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd


def _find_shapefile(root: Path) -> Path | None:
    shps = sorted(root.rglob("*.shp"))
    return shps[0] if shps else None


def save_uploaded_vector(uploaded_bytes: bytes, filename: str, work_dir: Path) -> Path:
    """Persist an uploaded zip shapefile or GeoJSON to disk and return vector path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    name = filename.lower()

    if name.endswith(".zip"):
        zip_path = work_dir / filename
        zip_path.write_bytes(uploaded_bytes)
        extract_dir = work_dir / "shp_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        shp = _find_shapefile(extract_dir)
        if shp is None:
            raise ValueError("Zip archive does not contain a .shp file")
        return shp

    if name.endswith(".geojson") or name.endswith(".json"):
        out = work_dir / filename
        out.write_bytes(uploaded_bytes)
        return out

    if name.endswith(".shp"):
        out = work_dir / filename
        out.write_bytes(uploaded_bytes)
        return out

    raise ValueError("Upload a .zip shapefile, .shp, or .geojson")


def read_uploaded_gdf(uploaded_bytes: bytes, filename: str) -> tuple[gpd.GeoDataFrame, Path]:
    with tempfile.TemporaryDirectory() as tmp:
        vector_path = save_uploaded_vector(uploaded_bytes, filename, Path(tmp))
        gdf = gpd.read_file(vector_path)
        # Copy to a stable path inside tmp for the duration of the caller's session
        stable = Path(tmp) / "input.geojson"
        gdf.to_file(stable, driver="GeoJSON")
        gdf = gpd.read_file(stable)
        return gdf, stable
