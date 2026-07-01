# Plantation Verification via AEF Embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a five-stage pipeline that takes a parcel GeoJSON, fetches AlphaEarth Foundations satellite embeddings (2017–2025), classifies each pixel by LULC class using a XGBoost model trained on open-source labels, and produces a per-parcel JSON verdict confirming or denying plantation presence.

**Architecture:** A XGBoost classifier is trained once on 64-dim AEF embeddings labelled from Dynamic World (GEE export), ESA WorldCover (S3 COGs), and OpenEarthMap (Zenodo). At inference time, the AEF Fetcher downloads COG tiles from source.coop, dequantizes them, and the model classifies each pixel per year. The Verdict Engine applies a temporal rule (Trees prob > 0.6 for ≥2 consecutive years) and aggregates to a parcel-level JSON verdict with confidence and establishment year.

**Tech Stack:** Python 3.10+, xarray, rasterio, geopandas, shapely, xgboost, scikit-learn, pyarrow, numpy, pandas, tqdm, netCDF4, earthengine-api (for DW export only)

## Global Constraints

- Project root: `Documents/plantation_verification/`
- All src modules live in `src/`; all tests in `tests/`
- 7-class taxonomy IDs: 0=water, 1=trees, 2=shrub_grass, 3=cropland, 4=built, 5=barren, 6=snow_ice
- AEF dequantize formula: `(arr.astype(float32) / 127.5) ** 2 * np.sign(arr)` — apply after every read
- AEF bands are named A00–A63 (64 channels, int8 on disk)
- Plantation detection threshold: Trees-class probability > 0.6 for ≥2 consecutive years
- Parcel verdict threshold: `tree_pixel_fraction > 0.30 AND confidence > 0.50`
- Classifier target: macro-F1 ≥ 0.80, Trees-class F1 ≥ 0.85
- Training Parquet schema: `[source, year, class_id, class_name, lon, lat, A00…A63]`
- Model artifacts: `models/model.ubj`, `models/class_map.json`, `models/feature_names.json`
- GeoTIFFs: LZW compressed, write CRS and transform from source tile

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/harmonize.py` | Class mapping lookup tables; `map_label(source, native_label) -> int` |
| `src/aef_fetcher.py` | Download AEF index, windowed COG read, dequantize, NetCDF cache; `fetch_embeddings(geojson, years) -> xr.Dataset` |
| `src/label_builder.py` | Sample ESA WorldCover + OpenEarthMap + DW GeoTIFFs; Mahalanobis filter; multi-source agreement; `build_training_set(...) -> Path` |
| `src/classifier.py` | Train XGBoost, compute macro-F1, save/load artifacts; `train(parquet_path, model_dir)`, `load_model(model_dir) -> XGBClassifier` |
| `src/inference.py` | Per-year reshape → predict_proba → write classified + tree-prob GeoTIFFs; `run_inference(ds, model, class_map, parcel_dir) -> dict[int, Path]` |
| `src/verdict.py` | Temporal plantation detection, parcel aggregation, JSON verdict, establishment-year raster; `generate_verdict(parcel_id, prob_paths, lulc_paths, parcel_geojson, out_dir) -> dict` |
| `scripts/export_dw.py` | GEE export of Dynamic World labels to local GeoTIFFs (run once, requires GEE auth) |
| `tests/test_harmonize.py` | Unit tests for map_label |
| `tests/test_aef_fetcher.py` | Unit tests for dequantize + index spatial filter (mocked HTTP) |
| `tests/test_label_builder.py` | Unit tests for Mahalanobis filter + multi-source agreement |
| `tests/test_classifier.py` | Train on tiny synthetic data; verify save/load + F1 structure |
| `tests/test_inference.py` | Mock model; verify GeoTIFF output shape and CRS |
| `tests/test_verdict.py` | Unit tests for detection rule + parcel aggregation logic |
| `requirements.txt` | All pinned dependencies |

---

## Task 1: Project scaffold + harmonize.py

**Files:**
- Create: `Documents/plantation_verification/requirements.txt`
- Create: `Documents/plantation_verification/src/__init__.py`
- Create: `Documents/plantation_verification/src/harmonize.py`
- Create: `Documents/plantation_verification/tests/__init__.py`
- Create: `Documents/plantation_verification/tests/test_harmonize.py`

**Interfaces:**
- Produces: `map_label(source: str, native_label: str | int) -> int | None`
- Produces: `CLASS_NAMES: dict[int, str]` — `{0: "water", 1: "trees", …}`
- Produces: `SOURCES: list[str]` — `["dynamic_world", "esa_worldcover", "openearthmap"]`

- [ ] **Step 1: Create directories**

```bash
cd Documents/plantation_verification
mkdir -p src tests data/training models notebooks scripts
touch src/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
xarray>=2024.1.0
netCDF4>=1.6.0
rasterio>=1.3.0
geopandas>=0.14.0
shapely>=2.0.0
xgboost>=2.0.0
scikit-learn>=1.4.0
pyarrow>=14.0.0
numpy>=1.26.0
pandas>=2.1.0
tqdm>=4.66.0
joblib>=1.3.0
scipy>=1.12.0
earthengine-api>=0.1.390
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_harmonize.py
import pytest
from src.harmonize import map_label, CLASS_NAMES, SOURCES

def test_dw_trees_maps_to_1():
    assert map_label("dynamic_world", "trees") == 1

def test_dw_bare_maps_to_5():
    assert map_label("dynamic_world", "bare") == 5

def test_dw_shrub_maps_to_2():
    assert map_label("dynamic_world", "shrub_and_scrub") == 2

def test_esa_tree_cover_maps_to_1():
    assert map_label("esa_worldcover", 10) == 1

def test_esa_mangrove_maps_to_1():
    assert map_label("esa_worldcover", 95) == 1

def test_esa_bare_maps_to_5():
    assert map_label("esa_worldcover", 60) == 5

def test_esa_cropland_maps_to_3():
    assert map_label("esa_worldcover", 40) == 3

def test_oem_forest_maps_to_1():
    assert map_label("openearthmap", 5) == 1

def test_oem_barren_maps_to_5():
    assert map_label("openearthmap", 4) == 5

def test_unknown_native_label_returns_none():
    assert map_label("dynamic_world", "unknown_class") is None

def test_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        map_label("made_up_source", "trees")

def test_class_names_covers_all_ids():
    assert set(CLASS_NAMES.keys()) == {0, 1, 2, 3, 4, 5, 6}

def test_sources_list_contains_three():
    assert set(SOURCES) == {"dynamic_world", "esa_worldcover", "openearthmap"}
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd Documents/plantation_verification
pytest tests/test_harmonize.py -v
```

Expected: ImportError or ModuleNotFoundError (harmonize does not exist yet)

- [ ] **Step 5: Write src/harmonize.py**

```python
from __future__ import annotations

SOURCES = ["dynamic_world", "esa_worldcover", "openearthmap"]

CLASS_NAMES: dict[int, str] = {
    0: "water",
    1: "trees",
    2: "shrub_grass",
    3: "cropland",
    4: "built",
    5: "barren",
    6: "snow_ice",
}

_DW_MAP: dict[str, int] = {
    "water": 0,
    "trees": 1,
    "grass": 2,
    "flooded_vegetation": 2,
    "crops": 3,
    "shrub_and_scrub": 2,
    "built": 4,
    "bare": 5,
    "snow_and_ice": 6,
}

_ESA_MAP: dict[int, int] = {
    10: 1,   # Tree cover
    20: 2,   # Shrubland
    30: 2,   # Grassland
    40: 3,   # Cropland
    50: 4,   # Built-up
    60: 5,   # Bare/sparse vegetation
    70: 6,   # Snow and ice
    80: 0,   # Permanent water
    90: 2,   # Herbaceous wetland
    95: 1,   # Mangroves
    100: 2,  # Moss and lichen
}

# OpenEarthMap 8 classes: 1=background, 2=bareland, 3=rangeland,
# 4=developed, 5=road, 6=tree, 7=water, 8=agriculture
_OEM_MAP: dict[int, int] = {
    1: 5,   # bareland → barren
    2: 5,   # bareland
    3: 2,   # rangeland → shrub_grass
    4: 4,   # developed → built
    5: 4,   # road → built
    6: 1,   # tree → trees
    7: 0,   # water
    8: 3,   # agriculture → cropland
}

_SOURCE_MAP: dict[str, dict] = {
    "dynamic_world": _DW_MAP,
    "esa_worldcover": _ESA_MAP,
    "openearthmap": _OEM_MAP,
}


def map_label(source: str, native_label: str | int) -> int | None:
    """Map a source-native label to the 7-class taxonomy. Returns None if unmapped."""
    if source not in _SOURCE_MAP:
        raise ValueError(f"Unknown source: {source!r}. Must be one of {SOURCES}")
    return _SOURCE_MAP[source].get(native_label)
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/test_harmonize.py -v
```

Expected: 13 passed

- [ ] **Step 7: Commit**

```bash
git add src/harmonize.py src/__init__.py tests/test_harmonize.py tests/__init__.py requirements.txt
git commit -m "feat: scaffold project + harmonize 7-class taxonomy"
```

---

## Task 2: AEF Fetcher

**Files:**
- Create: `src/aef_fetcher.py`
- Create: `tests/test_aef_fetcher.py`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces:
  - `dequantize(arr: np.ndarray) -> np.ndarray` — float32, [-1, 1]
  - `load_index(cache_dir: Path) -> gpd.GeoDataFrame` — columns include `geometry`, `url` (or `href`), `year`
  - `fetch_embeddings(parcel_geojson: dict, years: list[int], cache_dir: Path) -> xr.Dataset` — dims `(year, y, x, band)`, band=64

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_aef_fetcher.py
import numpy as np
import pytest
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
from unittest.mock import patch, MagicMock
from src.aef_fetcher import dequantize, _parcel_hash

def test_dequantize_positive():
    arr = np.array([127], dtype=np.int8)
    result = dequantize(arr)
    # (127/127.5)**2 * sign(127) ≈ 0.9922
    assert result.dtype == np.float32
    assert abs(result[0] - (127 / 127.5) ** 2) < 0.01

def test_dequantize_negative():
    arr = np.array([-127], dtype=np.int8)
    result = dequantize(arr)
    # negative: sign=-1
    assert result[0] < 0
    assert abs(result[0] - (-(127 / 127.5) ** 2)) < 0.01

def test_dequantize_zero():
    arr = np.array([0], dtype=np.int8)
    result = dequantize(arr)
    assert result[0] == 0.0

def test_dequantize_preserves_shape():
    arr = np.zeros((4, 8, 64), dtype=np.int8)
    result = dequantize(arr)
    assert result.shape == (4, 8, 64)

def test_parcel_hash_is_deterministic():
    geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    h1 = _parcel_hash(geojson, [2020, 2021])
    h2 = _parcel_hash(geojson, [2020, 2021])
    assert h1 == h2

def test_parcel_hash_differs_by_year():
    geojson = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    h1 = _parcel_hash(geojson, [2020])
    h2 = _parcel_hash(geojson, [2021])
    assert h1 != h2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_aef_fetcher.py -v
```

Expected: ImportError (module not found)

- [ ] **Step 3: Write src/aef_fetcher.py**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from shapely.geometry import shape, box

# AEF index GeoParquet on source.coop
# After downloading, inspect columns with: print(index_gdf.columns.tolist())
# Expected columns: geometry, crs_wkt (or epsg), year, href (or url or path)
INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/index.parquet"
YEARS = list(range(2017, 2026))
N_BANDS = 64
BAND_NAMES = [f"A{i:02d}" for i in range(N_BANDS)]
DEFAULT_CACHE = Path.home() / ".cache" / "aef_embeddings"


def dequantize(arr: np.ndarray) -> np.ndarray:
    """Dequantize int8 AEF values to float32 in [-1, 1]."""
    f = arr.astype(np.float32)
    return (f / 127.5) ** 2 * np.sign(f)


def _parcel_hash(geojson: dict, years: list[int]) -> str:
    key = json.dumps(geojson, sort_keys=True) + str(sorted(years))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_index(cache_dir: Path = DEFAULT_CACHE) -> gpd.GeoDataFrame:
    """Download and cache the AEF index GeoParquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "aef_index.parquet"
    if not cache_path.exists():
        import urllib.request
        print(f"Downloading AEF index to {cache_path} …")
        urllib.request.urlretrieve(INDEX_URL, cache_path)
    return gpd.read_parquet(cache_path)


def _url_column(gdf: gpd.GeoDataFrame) -> str:
    """Return whichever column holds the COG file URL/path."""
    for col in ("href", "url", "path", "filename"):
        if col in gdf.columns:
            return col
    raise KeyError(
        f"Cannot find URL column in AEF index. Available columns: {gdf.columns.tolist()}"
    )


def _year_column(gdf: gpd.GeoDataFrame) -> str | None:
    """Return year column if present, else None (year embedded in URL)."""
    return "year" if "year" in gdf.columns else None


def _read_tile_window(
    url: str, minx: float, miny: float, maxx: float, maxy: float
) -> np.ndarray:
    """Open a COG via HTTPS and read the bbox window. Returns [64, H, W] int8."""
    # rasterio uses GDAL /vsicurl/ under the hood for HTTPS URLs
    with rasterio.open(url) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(window=window)  # [64, H, W] int8
    return data


def fetch_embeddings(
    parcel_geojson: dict,
    years: list[int] | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> xr.Dataset:
    """
    Fetch AEF embeddings for a parcel polygon across years.

    parcel_geojson: GeoJSON Feature or Geometry dict in EPSG:4326
    years: list of years to fetch (default 2017-2025)
    Returns: xr.Dataset with data_var "embeddings" dims (year, y, x, band)
    """
    if years is None:
        years = YEARS

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _parcel_hash(parcel_geojson, years)
    cache_path = cache_dir / f"{cache_key}.nc"

    if cache_path.exists():
        return xr.open_dataset(cache_path)

    geom = parcel_geojson.get("geometry", parcel_geojson)
    parcel = shape(geom)
    minx, miny, maxx, maxy = parcel.bounds

    index_gdf = load_index(cache_dir)
    url_col = _url_column(index_gdf)
    year_col = _year_column(index_gdf)

    # Spatial filter
    parcel_box = box(minx, miny, maxx, maxy)
    overlapping = index_gdf[index_gdf.geometry.intersects(parcel_box)].copy()

    if overlapping.empty:
        raise ValueError(f"No AEF tiles overlap parcel bounds {parcel.bounds}")

    year_arrays: dict[int, np.ndarray] = {}
    ref_transform = None
    ref_crs = None

    for year in sorted(years):
        if year_col:
            tiles = overlapping[overlapping[year_col] == year]
        else:
            # Year embedded in URL — filter by year string in path
            tiles = overlapping[overlapping[url_col].str.contains(str(year))]

        if tiles.empty:
            print(f"  Warning: no AEF tiles found for year {year}, skipping")
            continue

        # If multiple tiles overlap (tile boundary crosses parcel), merge them.
        # Simple case: take first tile (most parcels fit inside one UTM tile).
        # TODO: implement mosaic for parcels spanning tile boundaries.
        url = tiles.iloc[0][url_col]
        raw = _read_tile_window(url, minx, miny, maxx, maxy)  # [64, H, W]

        if ref_transform is None:
            with rasterio.open(url) as src:
                from rasterio.windows import from_bounds as fb
                w = fb(minx, miny, maxx, maxy, src.transform)
                ref_transform = src.window_transform(w)
                ref_crs = src.crs

        deq = dequantize(raw)  # [64, H, W] float32
        year_arrays[year] = deq.transpose(1, 2, 0)  # [H, W, 64]

    if not year_arrays:
        raise RuntimeError("No embeddings fetched for any requested year")

    fetched_years = sorted(year_arrays.keys())
    H, W, _ = next(iter(year_arrays.values())).shape
    stack = np.stack([year_arrays[y] for y in fetched_years], axis=0)  # [Y, H, W, 64]

    ds = xr.Dataset(
        {"embeddings": (["year", "y", "x", "band"], stack)},
        coords={
            "year": fetched_years,
            "band": BAND_NAMES,
        },
        attrs={
            "crs": str(ref_crs),
            "transform": list(ref_transform)[:6] if ref_transform else [],
            "parcel_hash": cache_key,
        },
    )
    ds.to_netcdf(cache_path)
    return ds
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_aef_fetcher.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/aef_fetcher.py tests/test_aef_fetcher.py
git commit -m "feat: AEF fetcher — dequantize, index load, windowed COG read, NetCDF cache"
```

---

## Task 3: Label Builder — ESA WorldCover + Mahalanobis filter

**Files:**
- Create: `src/label_builder.py`
- Create: `tests/test_label_builder.py`

**Interfaces:**
- Consumes: `map_label` from `src/harmonize.py`; `fetch_embeddings` from `src/aef_fetcher.py`
- Produces:
  - `sample_esa_worldcover(bbox, year, n_per_class, cache_dir) -> pd.DataFrame` — schema: `[source, year, class_id, class_name, lon, lat, A00…A63]`
  - `mahalanobis_filter(df, threshold) -> pd.DataFrame`
  - `agreement_filter(dfs: list[pd.DataFrame]) -> pd.DataFrame`
  - `build_training_set(sources_config, output_path, cache_dir) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_label_builder.py
import numpy as np
import pandas as pd
import pytest
from src.label_builder import mahalanobis_filter, agreement_filter

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_df(class_ids, embeddings, source="test", year=2020):
    rows = []
    for cid, emb in zip(class_ids, embeddings):
        row = {"source": source, "year": year, "class_id": cid,
               "class_name": str(cid), "lon": 0.0, "lat": 0.0}
        row.update(dict(zip(BAND_COLS, emb)))
        rows.append(row)
    return pd.DataFrame(rows)

def test_mahalanobis_filter_removes_outlier():
    rng = np.random.default_rng(0)
    # 99 normal points + 1 extreme outlier for class 1
    normal = rng.normal(0, 0.1, (99, 64))
    outlier = rng.normal(10, 0.1, (1, 64))  # far from cluster
    embs = np.vstack([normal, outlier])
    class_ids = [1] * 100
    df = _make_df(class_ids, embs)
    filtered = mahalanobis_filter(df, threshold=3.0)
    assert len(filtered) < len(df)
    assert len(filtered) >= 98  # outlier removed

def test_mahalanobis_filter_keeps_inliers():
    rng = np.random.default_rng(1)
    embs = rng.normal(0, 0.1, (50, 64))
    df = _make_df([1] * 50, embs)
    filtered = mahalanobis_filter(df, threshold=3.0)
    assert len(filtered) >= 45  # most kept

def test_agreement_filter_keeps_matching_pixels():
    embs = np.zeros((3, 64))
    df1 = _make_df([1, 1, 5], embs, source="esa_worldcover")
    df1["lon"] = [10.0, 11.0, 12.0]
    df1["lat"] = [20.0, 21.0, 22.0]
    df2 = _make_df([1, 0, 5], embs, source="dynamic_world")
    df2["lon"] = [10.0, 11.0, 12.0]
    df2["lat"] = [20.0, 21.0, 22.0]
    # lon=10/lat=20 → both say class 1 → keep
    # lon=11/lat=21 → disagree (1 vs 0) → drop
    # lon=12/lat=22 → both say class 5 → keep
    result = agreement_filter([df1, df2], tolerance_deg=0.0001)
    assert len(result) == 4  # 2 pixels × 2 sources

def test_agreement_filter_single_source_passes_through():
    embs = np.zeros((5, 64))
    df = _make_df([1, 1, 3, 3, 5], embs)
    result = agreement_filter([df], tolerance_deg=0.0001)
    assert len(result) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_label_builder.py -v
```

Expected: ImportError

- [ ] **Step 3: Write src/label_builder.py**

```python
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
        # Convert to sigma units relative to chi distribution
        mask = dists < threshold * np.median(dists)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_label_builder.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/label_builder.py tests/test_label_builder.py
git commit -m "feat: label builder — ESA WorldCover sampler, Mahalanobis filter, agreement filter"
```

---

## Task 4: Dynamic World GEE export helper

**Files:**
- Create: `scripts/export_dw.py`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: GeoTIFF files in `data/training/dw_exports/{year}_{tile}.tif` with pixel values 0–8 (DW label integers)

- [ ] **Step 1: Write scripts/export_dw.py**

```python
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
```

- [ ] **Step 2: Smoke test (requires GEE auth — skip if not available)**

```bash
# Only run if you have GEE credentials:
python scripts/export_dw.py --bbox "80.0 10.0 81.0 11.0" --years 2021 --out_dir /tmp/dw_test
```

Expected: GEE task starts, polls status, prints download instructions.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_dw.py
git commit -m "feat: GEE export helper for Dynamic World annual label GeoTIFFs"
```

---

## Task 5: XGBoost Classifier — train, eval, save/load

**Files:**
- Create: `src/classifier.py`
- Create: `tests/test_classifier.py`

**Interfaces:**
- Consumes: Parquet from `build_training_set` — columns `[class_id, A00…A63]` minimum
- Produces:
  - `train(parquet_path: Path, model_dir: Path, val_fraction: float) -> dict` — returns `{"macro_f1": float, "trees_f1": float}`
  - `load_model(model_dir: Path) -> tuple[XGBClassifier, dict[int, str], list[str]]` — (model, class_map, feature_names)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classifier.py
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.classifier import train, load_model

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_parquet(tmp_path: Path, n_per_class: int = 200) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for class_id in range(7):
        embs = rng.normal(class_id * 2, 0.5, (n_per_class, 64)).astype(np.float32)
        for emb in embs:
            row = {"source": "test", "year": 2021, "class_id": class_id,
                   "class_name": str(class_id), "lon": 0.0, "lat": 0.0}
            row.update(dict(zip(BAND_COLS, emb.tolist())))
            rows.append(row)
    df = pd.DataFrame(rows)
    p = tmp_path / "training.parquet"
    df.to_parquet(p, index=False)
    return p

def test_train_returns_metrics(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    metrics = train(parquet_path, tmp_path / "models")
    assert "macro_f1" in metrics
    assert "trees_f1" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0

def test_train_saves_artifacts(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    assert (model_dir / "model.ubj").exists()
    assert (model_dir / "class_map.json").exists()
    assert (model_dir / "feature_names.json").exists()

def test_load_model_returns_correct_types(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    model, class_map, feature_names = load_model(model_dir)
    assert hasattr(model, "predict_proba")
    assert isinstance(class_map, dict)
    assert len(feature_names) == 64

def test_load_model_predicts_correct_shape(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    model_dir = tmp_path / "models"
    train(parquet_path, model_dir)
    model, class_map, _ = load_model(model_dir)
    X = np.random.default_rng(0).normal(0, 1, (10, 64)).astype(np.float32)
    probs = model.predict_proba(X)
    assert probs.shape == (10, 7)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

def test_train_high_f1_on_separable_data(tmp_path):
    # Well-separated clusters → should achieve high F1
    parquet_path = _make_parquet(tmp_path, n_per_class=300)
    metrics = train(parquet_path, tmp_path / "models")
    assert metrics["macro_f1"] > 0.80, f"macro_f1={metrics['macro_f1']:.3f} below 0.80"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_classifier.py -v
```

Expected: ImportError

- [ ] **Step 3: Write src/classifier.py**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.harmonize import CLASS_NAMES

BAND_COLS = [f"A{i:02d}" for i in range(64)]
TREES_CLASS_ID = 1


def train(
    parquet_path: Path,
    model_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, float]:
    """
    Train XGBoost on the training Parquet and save artifacts.
    Returns {"macro_f1": float, "trees_f1": float} on the validation split.
    """
    df = pd.read_parquet(parquet_path)
    X = df[BAND_COLS].values.astype(np.float32)
    y = df["class_id"].values.astype(int)

    # Class weights: inverse frequency
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_map[c] for c in y])

    # Geography-aware split: if "lon" column present, assign each pixel to
    # a 10°×10° grid cell and use cell ID as stratification group so adjacent
    # pixels don't leak across splits. Falls back to class-stratified random
    # split when coordinates are absent.
    if "lon" in df.columns:
        lon_bin = (df["lon"].values / 10).astype(int)
        lat_bin = (df["lat"].values / 10).astype(int)
        geo_group = lon_bin * 1000 + lat_bin
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(gss.split(X, y, groups=geo_group))
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = sample_weights[train_idx]
    else:
        X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
            X, y, sample_weights,
            test_size=val_fraction,
            stratify=y,
            random_state=seed,
        )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    trees_f1 = f1_score(
        y_val, y_pred,
        labels=[TREES_CLASS_ID],
        average="macro",
        zero_division=0,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.ubj"))

    with open(model_dir / "class_map.json", "w") as f:
        json.dump({str(k): v for k, v in CLASS_NAMES.items()}, f, indent=2)

    with open(model_dir / "feature_names.json", "w") as f:
        json.dump(BAND_COLS, f)

    print(f"macro-F1={macro_f1:.3f}  trees-F1={trees_f1:.3f}")
    return {"macro_f1": float(macro_f1), "trees_f1": float(trees_f1)}


def load_model(
    model_dir: Path,
) -> tuple[XGBClassifier, dict[int, str], list[str]]:
    """Load model + metadata from model_dir. Returns (model, class_map, feature_names)."""
    model = XGBClassifier()
    model.load_model(str(model_dir / "model.ubj"))

    with open(model_dir / "class_map.json") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return model, class_map, feature_names
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_classifier.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat: XGBoost classifier — train, eval, save/load artifacts"
```

---

## Task 6: Per-year Inference

**Files:**
- Create: `src/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `xr.Dataset` from `fetch_embeddings`; `XGBClassifier` + `class_map` from `load_model`
- Produces: `run_inference(ds, model, class_map, parcel_dir) -> dict[int, dict[str, Path]]`
  - Returns `{year: {"lulc": Path, "tree_prob": Path}}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_inference.py
import numpy as np
import pytest
import xarray as xr
import rasterio
from pathlib import Path
from unittest.mock import MagicMock
from src.inference import run_inference

BAND_COLS = [f"A{i:02d}" for i in range(64)]

def _make_ds(years=(2020, 2021), h=8, w=8):
    rng = np.random.default_rng(0)
    data = rng.uniform(-1, 1, (len(years), h, w, 64)).astype(np.float32)
    return xr.Dataset(
        {"embeddings": (["year", "y", "x", "band"], data)},
        coords={"year": list(years), "band": BAND_COLS},
        attrs={"crs": "EPSG:4326", "transform": [0.001, 0, 10.0, 0, -0.001, 20.0]},
    )

def _make_model():
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.ones((len(X), 7)) / 7
    return model

def test_run_inference_returns_paths_per_year(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    assert set(result.keys()) == {2020, 2021}
    for year in (2020, 2021):
        assert "lulc" in result[year]
        assert "tree_prob" in result[year]

def test_lulc_geotiff_is_uint8(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["lulc"]) as src:
        assert src.dtypes[0] == "uint8"
        assert src.count == 1

def test_tree_prob_geotiff_is_float32(tmp_path):
    ds = _make_ds()
    model = _make_model()
    class_map = {0:"water",1:"trees",2:"shrub_grass",3:"cropland",4:"built",5:"barren",6:"snow_ice"}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["tree_prob"]) as src:
        assert src.dtypes[0] == "float32"

def test_lulc_values_in_valid_range(tmp_path):
    ds = _make_ds()
    # Model always predicts class 1 (trees)
    model = MagicMock()
    probs = np.zeros((64, 7), dtype=np.float32)
    probs[:, 1] = 1.0
    model.predict_proba.return_value = probs
    class_map = {i: str(i) for i in range(7)}
    result = run_inference(ds, model, class_map, tmp_path)
    with rasterio.open(result[2020]["lulc"]) as src:
        data = src.read(1)
        assert set(np.unique(data)).issubset(set(range(7)))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_inference.py -v
```

Expected: ImportError

- [ ] **Step 3: Write src/inference.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_inference.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: per-year inference — classify pixels, write LULC + tree-prob GeoTIFFs"
```

---

## Task 7: Verdict Engine

**Files:**
- Create: `src/verdict.py`
- Modify: `tests/test_verdict.py` (already listed in spec; create fresh)

**Interfaces:**
- Consumes: `dict[int, dict[str, Path]]` from `run_inference`; parcel GeoJSON
- Produces:
  - `detect_plantation_pixels(tree_prob_stack: np.ndarray, years: list[int], threshold: float, min_consecutive: int) -> np.ndarray` — bool [H, W]
  - `generate_verdict(parcel_id, inference_paths, parcel_geojson, out_dir) -> dict`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verdict.py
import numpy as np
import pytest
import json
import rasterio
from pathlib import Path
from src.verdict import detect_plantation_pixels, generate_verdict

def _make_tree_prob_stack(pattern: list[list[float]]) -> np.ndarray:
    """pattern: list of [H*W probs] per year → [Y, H, W]"""
    return np.array(pattern, dtype=np.float32).reshape(len(pattern), 2, 2)

def test_detect_plantation_two_consecutive_years():
    # Pixel (0,0): 0.7, 0.8 in years 1 and 2 → plantation
    # Pixel (0,1): only year 1 high → not plantation (only 1 year)
    stack = np.array([
        [[0.7, 0.4], [0.2, 0.3]],  # year 0
        [[0.8, 0.3], [0.1, 0.2]],  # year 1
        [[0.2, 0.2], [0.1, 0.1]],  # year 2
    ], dtype=np.float32)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == True   # 2 consecutive years above threshold
    assert result[0, 1] == False  # only 1 year above threshold

def test_detect_plantation_requires_consecutive():
    # High in year 0 and year 2, but not year 1 → not consecutive
    stack = np.array([
        [[0.8], [0.1]],
        [[0.2], [0.1]],
        [[0.9], [0.1]],
    ], dtype=np.float32).reshape(3, 1, 2)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == False  # not consecutive

def test_detect_plantation_three_consecutive_counts():
    stack = np.array([
        [[0.8]], [[0.9]], [[0.7]], [[0.2]]
    ], dtype=np.float32).reshape(4, 1, 1)
    result = detect_plantation_pixels(stack, threshold=0.6, min_consecutive=2)
    assert result[0, 0] == True

def test_generate_verdict_structure(tmp_path):
    # Create minimal tree_prob GeoTIFFs
    years = [2020, 2021, 2022]
    H, W = 4, 4
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": H, "width": W, "crs": "EPSG:4326",
               "transform": rasterio.transform.from_bounds(10, 20, 10.01, 20.01, W, H)}

    inference_paths = {}
    for i, year in enumerate(years):
        prob = 0.8 if i >= 1 else 0.1
        data = np.full((1, H, W), prob, dtype=np.float32)
        tp = tmp_path / f"tree_prob_{year}.tif"
        lulc = tmp_path / f"lulc_{year}.tif"
        with rasterio.open(tp, "w", **profile) as dst:
            dst.write(data)
        lulc_profile = {**profile, "dtype": "uint8"}
        label = 1 if i >= 1 else 5
        with rasterio.open(lulc, "w", **lulc_profile) as dst:
            dst.write(np.full((1, H, W), label, dtype=np.uint8))
        inference_paths[year] = {"lulc": lulc, "tree_prob": tp}

    parcel_geojson = {
        "type": "Polygon",
        "coordinates": [[[10, 20], [10.01, 20], [10.01, 20.01], [10, 20.01], [10, 20]]]
    }
    verdict = generate_verdict("parcel_001", inference_paths, parcel_geojson, tmp_path)

    assert "plantation_present" in verdict
    assert "confidence" in verdict
    assert "establishment_year" in verdict
    assert "tree_pixel_fraction_latest" in verdict
    assert "prior_dominant_class" in verdict
    assert "lulc_timeseries" in verdict
    assert isinstance(verdict["plantation_present"], bool)
    assert 0.0 <= verdict["confidence"] <= 1.0

def test_generate_verdict_no_plantation(tmp_path):
    years = [2020, 2021]
    H, W = 4, 4
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": H, "width": W, "crs": "EPSG:4326",
               "transform": rasterio.transform.from_bounds(10, 20, 10.01, 20.01, W, H)}
    inference_paths = {}
    for year in years:
        tp = tmp_path / f"tree_prob_{year}.tif"
        lulc = tmp_path / f"lulc_{year}.tif"
        with rasterio.open(tp, "w", **profile) as dst:
            dst.write(np.full((1, H, W), 0.05, dtype=np.float32))
        lulc_profile = {**profile, "dtype": "uint8"}
        with rasterio.open(lulc, "w", **lulc_profile) as dst:
            dst.write(np.full((1, H, W), 5, dtype=np.uint8))
        inference_paths[year] = {"lulc": lulc, "tree_prob": tp}

    parcel_geojson = {
        "type": "Polygon",
        "coordinates": [[[10, 20], [10.01, 20], [10.01, 20.01], [10, 20.01], [10, 20]]]
    }
    verdict = generate_verdict("parcel_002", inference_paths, parcel_geojson, tmp_path)
    assert verdict["plantation_present"] == False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_verdict.py -v
```

Expected: ImportError

- [ ] **Step 3: Write src/verdict.py**

```python
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
        prior_lulc = lulc_stack[:est_idx][mask] if est_idx > 0 else lulc_stack[0:1][:, mask]
        if prior_lulc.size > 0:
            vals, cnts = np.unique(prior_lulc.ravel(), return_counts=True)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_verdict.py -v
```

Expected: 5 passed

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: All 28 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/verdict.py tests/test_verdict.py
git commit -m "feat: verdict engine — plantation detection, parcel aggregation, JSON verdict + establishment raster"
```

---

## Task 8: End-to-end smoke test notebook

**Files:**
- Create: `notebooks/03_verify_parcel.ipynb`

**Interfaces:**
- Consumes: all five stages
- Produces: working end-to-end run on a small test parcel

- [ ] **Step 1: Create notebooks/03_verify_parcel.ipynb**

The notebook should contain these cells in order:

**Cell 1 — imports and config:**
```python
import sys; sys.path.insert(0, "..")
from pathlib import Path
from src.aef_fetcher import fetch_embeddings
from src.classifier import load_model
from src.inference import run_inference
from src.verdict import generate_verdict

MODEL_DIR = Path("../models")
OUT_DIR = Path("../data/outputs/test_parcel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Small parcel in Karnataka (plantation likely present)
PARCEL = {
    "type": "Polygon",
    "coordinates": [[[77.5, 12.9], [77.51, 12.9], [77.51, 12.91], [77.5, 12.91], [77.5, 12.9]]]
}
```

**Cell 2 — fetch embeddings:**
```python
ds = fetch_embeddings(PARCEL, years=list(range(2017, 2026)))
print(f"Fetched: {ds['embeddings'].shape}")
```

**Cell 3 — load model and run inference:**
```python
model, class_map, _ = load_model(MODEL_DIR)
inference_paths = run_inference(ds, model, class_map, OUT_DIR)
print(f"Inference complete: {list(inference_paths.keys())}")
```

**Cell 4 — generate verdict:**
```python
verdict = generate_verdict("test_karnataka", inference_paths, PARCEL, OUT_DIR)
import json; print(json.dumps(verdict, indent=2))
```

**Cell 5 — visualise LULC time series:**
```python
import matplotlib.pyplot as plt
import rasterio
years = sorted(inference_paths.keys())
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for ax, year in zip(axes.flat, years):
    with rasterio.open(inference_paths[year]["lulc"]) as src:
        ax.imshow(src.read(1), vmin=0, vmax=6, cmap="tab10")
        ax.set_title(year)
        ax.axis("off")
plt.tight_layout(); plt.show()
```

- [ ] **Step 2: Verify the notebook runs top-to-bottom (requires trained model + internet for AEF fetch)**

```bash
cd notebooks && jupyter nbconvert --to notebook --execute 03_verify_parcel.ipynb --output 03_verify_parcel_executed.ipynb
```

Expected: notebook executes without errors; verdict JSON printed in Cell 4.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_verify_parcel.ipynb
git commit -m "feat: end-to-end smoke test notebook for parcel plantation verification"
```
