# Plantation Verification via AlphaEarth Foundations Embeddings

**Date:** 2026-07-01  
**Status:** Approved  
**Objective:** Given a parcel where a landowner claims plantation, confirm or deny canopy/plantation presence using AlphaEarth Foundations (AEF) satellite embeddings + open-source LULC training data.

---

## 1. Context

Carbon credit projects require independent verification that claimed land-use changes (e.g., barren → plantation) actually occurred. This pipeline uses pre-computed AEF 64-dim satellite embeddings (Taylor Geospatial, annual 2017–2025) as input features, trains a lightweight XGBoost classifier on open-source LULC training labels, and produces both a time-series LULC map and a per-parcel structured verdict.

**Dataset:** AlphaEarth Foundations v1 annual  
`https://source.coop/tge-labs/aef/v1/annual`  
- 64 channels per pixel (A00–A63), int8, dequantize: `(val / 127.5) ** 2 * np.sign(val)` → float32 in [-1, 1]  
- 8192×8192 px tiles organised by 120 UTM zones  
- Global coverage, annual 2017–2025, served as COGs on Source Cooperative

---

## 2. Architecture

Five stages, each independently runnable:

```
[1] Label Builder      (one-time, offline)
    Open-source LULC products → harmonized 7-class training Parquet

[2] AEF Fetcher
    Parcel GeoJSON → COG query → clipped embedding stack
    Output: xarray [n_years × H × W × 64]

[3] XGBoost Classifier
    Trained on Stage 1 output
    Input: 64-dim embedding (one pixel, one year)
    Output: 7-class probabilities

[4] Per-year Inference
    Runs Stage 3 on Stage 2 output
    → 9 classified GeoTIFFs + Trees-probability rasters (2017–2025)

[5] Verdict Engine
    9-year class sequence per pixel
    → per-parcel JSON verdict + establishment-year raster
```

---

## 3. LULC Taxonomy

| ID | Class | Dynamic World label | ESA WorldCover code |
|----|-------|---------------------|---------------------|
| 0 | Water | water | 80 |
| 1 | Trees / Plantation | trees | 10, 95 |
| 2 | Shrub / Grass | shrub_and_scrub, grass | 20, 30 |
| 3 | Cropland | crops | 40 |
| 4 | Built-up | built | 50 |
| 5 | Barren | bare | 60 |
| 6 | Snow / Ice | snow_and_ice | 70 |

---

## 4. Stage 1 — Label Builder

### 4a. Label sources

| Source | Access | Years | Format |
|--------|--------|-------|--------|
| Dynamic World training data | Google Earth Engine (`GOOGLE/DYNAMICWORLD/V1`) — export labels via GEE Python API | 2015–2023 | GEE export → GeoTIFF |
| ESA WorldCover | `s3://esa-worldcover/v200/` (public) | 2020, 2021 | COG GeoTIFF |
| OpenEarthMap | Zenodo (public) | various | PNG tiles + label PNG |

### 4b. Sampling

For each label source × AEF tile:
- Spatially join label raster to AEF tile
- Stratified sample: 5,000 pixels per class per source (balanced)
- Extract 64-dim AEF embedding at sampled pixel — match label year to AEF year exactly where available (2017–2023); for label years outside AEF range use nearest AEF year
- Store as Parquet: `[source, year, class_id, class_name, lon, lat, A00…A63]`

### 4c. Harmonization + quality filter

- Map native class → 7-class taxonomy via lookup table
- For pixels covered by ≥2 sources: keep only where all sources agree
- Mahalanobis filter per class: drop embeddings >3σ from class centroid
- Split: 80% train / 10% val / 10% test, stratified by class and geography (held-out UTM tiles, no spatial leakage)

---

## 5. Stage 2 — AEF Fetcher

**Input:** GeoJSON parcel (any CRS) + year range (default 2017–2025)

**Steps:**
1. Reproject parcel to EPSG:4326, compute bbox
2. Query AEF index Parquet to find overlapping COG tiles per year
3. For each tile × year: windowed read via HTTPS, clipped to parcel bbox
4. Dequantize: `(arr / 127.5) ** 2 * np.sign(arr)` → float32
5. Stack → `xarray.Dataset` with dims `(year, y, x, band)` + parcel polygon mask
6. Cache to local NetCDF keyed by `sha256(parcel_geojson + year_range)` to avoid re-fetching

**Output:** `xarray.Dataset` — shape `[n_years, H, W, 64]`

---

## 6. Stage 3 — XGBoost Classifier

**Model:** `XGBClassifier`
- `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`
- `eval_metric='mlogloss'`, early stopping patience=20 on val set
- Class weights: inverse class frequency (barren and snow are globally rare)

**Artifacts saved:**
- `model.ubj` — serialized XGBoost model
- `class_map.json` — `{class_id: class_name}`
- `feature_names.json` — `["A00", …, "A63"]`

**Evaluation target:** macro-F1 ≥ 0.80 on held-out test set before deployment.

---

## 7. Stage 4 — Per-year Inference

For each year in the AOI embedding stack:
1. Reshape `[H × W × 64]` → `[N_pixels × 64]`
2. `model.predict_proba()` → `[N_pixels × 7]`
3. Reshape → `[H × W × 7]` probability cube
4. Argmax → classified GeoTIFF (uint8, one file per year, LZW compressed)
5. Write Trees-class probability GeoTIFF (float32) — primary evidence layer for plantation claims

---

## 8. Stage 5 — Verdict Engine

### 8a. Per-pixel plantation detection rule

A pixel is marked "plantation" if:
- Trees-class probability > 0.6 for ≥2 consecutive years

"Prior class" = modal class of that pixel across all years preceding first Trees detection.

### 8b. Per-parcel aggregation

| Metric | Definition |
|--------|------------|
| `tree_pixel_fraction` | Fraction of parcel pixels classified as plantation in the latest year |
| `establishment_year` | Earliest year where ≥20% of parcel pixels first exceed the Trees threshold |
| `confidence` | Mean Trees probability across plantation pixels × persistence fraction |
| `plantation_present` | `tree_pixel_fraction > 0.30 AND confidence > 0.50` |

### 8c. Output JSON

```json
{
  "parcel_id": "ABC123",
  "plantation_present": true,
  "confidence": 0.84,
  "establishment_year": 2021,
  "tree_pixel_fraction_latest": 0.73,
  "prior_dominant_class": "barren",
  "lulc_timeseries": {
    "2017": {"trees": 0.02, "barren": 0.81, "cropland": 0.14},
    "2021": {"trees": 0.61, "barren": 0.21, "cropland": 0.12},
    "2025": {"trees": 0.73, "barren": 0.10, "cropland": 0.11}
  },
  "outputs": {
    "lulc_maps": "lulc_{year}.tif",
    "tree_prob_maps": "tree_prob_{year}.tif",
    "establishment_raster": "establishment_year.tif"
  }
}
```

### 8d. Establishment-year raster

GeoTIFF (uint16) where each pixel value = year of first confirmed plantation (0 = never plantation).

---

## 9. Project Structure

```
plantation_verification/
├── data/
│   └── training/              # Parquet files from Label Builder
├── models/
│   ├── model.ubj
│   ├── class_map.json
│   └── feature_names.json
├── src/
│   ├── label_builder.py       # Stage 1
│   ├── aef_fetcher.py         # Stage 2
│   ├── classifier.py          # Stage 3 (train + eval)
│   ├── inference.py           # Stage 4
│   ├── verdict.py             # Stage 5
│   └── harmonize.py           # Class mapping lookup tables
├── notebooks/
│   ├── 01_label_builder.ipynb
│   ├── 02_train_classifier.ipynb
│   └── 03_verify_parcel.ipynb
├── tests/
│   └── test_verdict.py
└── requirements.txt
```

---

## 10. Dependencies

```
xarray, rasterio, geopandas, shapely
xgboost, scikit-learn
numpy, pandas, pyarrow
tqdm, joblib
```

---

## 11. Success Criteria

| Criterion | Target |
|-----------|--------|
| Classifier macro-F1 on held-out test | ≥ 0.80 |
| Trees-class F1 specifically | ≥ 0.85 |
| Fetch + inference time per parcel (≤100 ha) | < 5 minutes |
| Verdict JSON produced for any GeoJSON input | 100% |
| False positive rate for plantation claims | < 15% |

---

## 12. Out of Scope

- Real-time / sub-annual inference (AEF is annual)
- Species-level plantation classification (e.g. eucalyptus vs teak)
- Canopy height estimation
- UI / web frontend
- Automated retraining pipeline
