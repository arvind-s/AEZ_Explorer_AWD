# AWD shapefile → stats + GeoTIFF pipeline — design

**Date:** 2026-07-30
**Status:** Approved, building
**References:**
- EO4AWD (Space Climate Observatory): S1 C-band + ALOS-2/PALSAR-2 L-band + optical → ML classification of paddy irrigation regime → inundation-status map, methane map, MRV dashboard.
- **MDPI 2190** — Hoang-Phi, Lam-Dao, Dang-Pham-Bao, Le-Toan, Truong-Nhat-Kieu &
  Sobue (2026), *"Inundation Monitoring in Rice Fields Using ALOS-2 PALSAR-2: A
  Case Study of An Giang, the Mekong Delta in Vietnam"*, *Remote Sensing*
  18(13):2190, DOI 10.3390/rs18132190. See "Alignment with MDPI 2190" below.
- Existing repo scripts `05_gee_palsar2_awd.js`, `06_gee_awd_full_pixel_model.js`, `utils.py`.

## Goal

Turn the hand-pasted, hardcoded-AOI GEE script `06_gee_awd_full_pixel_model.js`
into a repeatable command-line pipeline: **shapefile in → AWD stats + GeoTIFF out**,
fully automatic, no Code Editor paste.

```bash
python awd_pipeline.py --shapefile fields.shp \
    --start 2020-06-01 --end 2020-11-30 --out outputs/
```

## Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Execution backend | **Python + `earthengine-api`** | Server-side compute, local download, no manual paste. One-time EE auth. |
| Stats granularity | **Whole-shapefile aggregate** | All polygons dissolved to one AOI; a single stats row. |
| SAR source | **PALSAR-2 L-band** (`JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`, HH) | L-band senses sub-canopy water; matches `05`/`06` and the L-band papers. |
| Output delivery | **Always local download** | `getDownloadURL` → files land in `--out/`. Size-ceiling caught with a clear message, not a stack trace. |
| Model | **Faithful port of `06`** | Same WI + shifted-array drying-cycle + classification; thresholds exposed as CLI flags. |

## Architecture

Single script `awd_pipeline.py`, small focused functions:

```
shapefile ─▶ load_aoi() ──▶ dissolved WGS84 ee.Geometry
                                  │
                    build_awd_image()   ← ports 06_gee_awd_full_pixel_model.js
                                  │        (per-pixel, server-side)
                                  ▼
        ee.Image { awd_likelihood, is_awd, is_continuous_flood,
                   n_cycles, ever_flooded }
                 │                                 │
        compute_stats()                    download_geotiff()
        reduceRegion(sum) → dict           getDownloadURL(GEO_TIFF) → local .tif
                 │                                 │
                 ▼                                 ▼
     awd_stats.json + awd_stats.csv       awd_classification.tif
```

- `load_aoi(path)` — geopandas read → reproject EPSG:4326 → `dissolve()` all
  polygons to one geometry → return `__geo_interface__` dict + bounds. **Pure, unit-tested.**
- `init_ee(project)` — `ee.Initialize` (auth handled once via `earthengine authenticate`).
- `build_awd_image(geometry, start, end, params)` — the model (below).
- `compute_stats(image, geometry, params)` — `reduceRegion(sum)` on per-class
  pixel-area images → aggregate dict.
- `write_stats(stats, out_dir)` — json + csv. **Pure, unit-tested.**
- `download_geotiff(image, geometry, scale, out_path)` — `getDownloadURL` + urllib.
- `main()` — CLI + orchestration.

## Model (port of `06`)

PALSAR-2 ScanSAR, HH only, over `[start, end]`:

1. **Calibrate** DN→γ⁰ dB: `10·log10(DN²) − 83.0`; mask `MSK≠1`; focal-median despeckle.
2. **Per-pixel Wetness Index**: `WI = 1 − 2·(x − seasonMin)/(seasonMax − seasonMin)`,
   masked where range = 0.
3. **Drying-cycle count**: shifted-array "dry@t (`WI ≤ DRY`) → wet@t+1 (`WI ≥ WET`)"
   events summed over the season. *(Simplification of the Python sequential state
   machine — documented in `06`; validate acreage against known fields.)*
4. **Classification**: `awd_likelihood = clamp(n_cycles / REF_CYCLES, 0..1)`;
   `is_awd = likelihood ≥ AWD_THRESH AND ever_flooded`;
   `is_continuous_flood = ever_flooded AND not is_awd`.

Thresholds → **CLI flags** with `06`'s defaults: `--wet 0.3 --dry -0.1 --ref-cycles 5
--awd-thresh 0.4 --scale 25`. When MDPI 2190's numbers are available, pass them in.

## Outputs (`--out/`)

- **`awd_classification.tif`** — 4 bands: `awd_likelihood`, `is_awd`,
  `is_continuous_flood`, `n_cycles`; clipped to the shapefile; 25 m.
- **`awd_stats.json`** + **`awd_stats.csv`** — one aggregate row:
  `awd_area_ha`, `continuous_flood_area_ha`, `paddy_area_ha`, `pct_awd`,
  `n_images`, plus season + thresholds used (provenance).

## Guardrails

- **Image-count check** — print scenes-in-season; warn loudly if `< 5`
  (drying-cycle counts unreliable — the `05`/`06` caveat).
- **Download-size ceiling** — catch GEE's direct-download rejection and print an
  AOI-tiling hint instead of a raw traceback.
- **Honest header** carrying `06`'s caveats: HH-only, no rice-crop mask
  ("ever flooded" proxy), simplified despeckle + cycle-count, not yet validated
  against ground truth.

## Testing

Runs without live GEE (sandbox has no EE auth/network):
- `load_aoi` — synthetic 2-polygon shapefile → assert single dissolved geometry,
  reprojection to 4326, correct bounds.
- `write_stats` — dict → assert json + csv content/round-trip.
- CLI parsing — defaults + overrides.

EE functions (`build_awd_image`, `compute_stats`, `download_geotiff`) require live
EE and are validated against `06`'s logic; run on a small AOI first.

## Alignment with MDPI 2190

The 2190 paper is the closest published benchmark to this pipeline, but its
method is **not a threshold swap** for the current WI model — folding it in
literally is a v2 modelling change. What the abstract establishes (the numeric
per-stage dB thresholds are in the paywalled methods section, not incorporated
here — no invented values):

- **Reported accuracy: 81% overall, Kappa 0.77** for inundated-vs-non-inundated
  classification. Use this as the **validation target** once we have labels.
- **L-band VV** penetrates dense canopy best. Our pipeline uses **HH** — the
  free GEE `JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR` collection is HH/HV, with no VV,
  so matching 2190's polarization needs a different PALSAR-2 product.
- **Phenology-conditioned**: inundation is discriminated **per rice growth
  stage**, with **Sentinel-1 time series estimating rice age**. Our model applies
  one fixed threshold pair across the whole season.
- **Absolute backscatter (dB)** thresholds, vs our **relative** min-max Wetness
  Index. The paper's dB numbers therefore cannot be dropped into `--wet/--dry`;
  a dB path would be a separate classifier.

**Three divergences to close for a 2190-aligned v2:** absolute-dB classification;
Sentinel-1 rice-age/phenology conditioning with per-stage thresholds; VV
polarization (needs a VV-capable PALSAR-2 product). Each needs the paper's
numeric threshold table (paywalled) to calibrate.

## Out of scope (v2)

S1+PALSAR-2 fusion; per-feature stats; real rice-crop mask (WorldCereal/GloRice);
Refined Lee filter parity; ascending+descending fusion; soil/AEZ covariates.
**MDPI-2190-aligned modelling** (absolute-dB, phenology-conditioned, VV) — see
"Alignment with MDPI 2190" above; blocked on the paper's numeric thresholds.
