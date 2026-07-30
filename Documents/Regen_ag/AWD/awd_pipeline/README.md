# AWD v0 Pipeline — Sathupalle Pilot (Khammam, Telangana)

Builds on `AWD_S1_GRD_literature_review.md` and `AWD_Global_Model_Roadmap.md`
in the parent folder. This is the first concrete, runnable implementation:
download → calibrate/filter → v0 wetness-index model → validation.

## What actually ran in this build session, and what didn't

**Important limitation discovered while building this:** the sandbox this
was built in has outbound network access locked to a small allowlist (pypi,
github). I tested direct access to Planetary Computer, Google Earth Engine,
AWS Open Data, and the Copernicus Data Space — **all were blocked identically**,
so this is a sandbox-wide restriction, not something specific to Planetary
Computer. Practical consequence: the actual satellite-data download (step 1)
could not be executed here and needs to run somewhere with normal internet
access (your machine, a cloud VM, Varaha's compute environment).

What I *did* verify in this sandbox, with real execution (not just written,
actually run and checked):

| Step | Status |
|---|---|
| AOI extraction & reprojection (TS_new_blocks.shp → WGS84) | **Ran successfully.** AOI = Sathupalle block, Khammam district, Telangana. BBox (WGS84): `[80.76522, 17.12462795, 80.92645283, 17.33644]`. Saved as `aoi_sathupalle_wgs84.geojson`. |
| MPC STAC query code | Written, follows documented MPC API pattern; **not executed** (network blocked in this sandbox — confirmed via `pystac_client` and raw `curl`, both got `403` at the proxy). |
| Refined Lee speckle filter | **Ran successfully** on synthetic noisy data — reduced std from 1.00 to 0.19 (`test_synthetic.py`). |
| Wetness Index + Otsu threshold + drying-cycle detection | **Ran successfully** on synthetic data mimicking a continuously-flooded field vs. an AWD field with 5 drain/re-flood cycles. Correctly scored the AWD series 5 cycles / 1.00 likelihood vs. 0 cycles / 0.00 likelihood for the continuous-flood series. |
| Calibration of raw GRD DN → beta-nought | Written, but **not validated against a real calibration XML** (couldn't download one). Flagged clearly in code comments — verify before trusting output. |
| Validation harness (`04_validate_against_open_sources.py`) | Written and ready; needs either a downloaded RiceAtlas calendar CSV or real AWD labels to actually run against. |

So: the algorithmic core is proven correct on synthetic data; the real-world
data plumbing is written and ready but unexecuted. That's the honest state —
treat `test_synthetic.py`'s PASS as "the logic works," not "the model is
validated."

## How to run this for real

1. **Get a machine with real internet access** (your laptop, a Varaha cloud VM, Colab, etc.).
2. `pip install pystac-client planetary-computer rasterio shapely pyproj scipy scikit-image geopandas matplotlib`
3. (Recommended path) Request a free Planetary Computer account for the
   `sentinel-1-rtc` collection (simpler — already calibrated + terrain
   corrected): https://planetarycomputer.microsoft.com/account/request,
   then `export PC_SDK_SUBSCRIPTION_KEY=...`.
   (Alternative, no account needed: `--collection sentinel-1-grd`, but you'll
   need to wire up real per-scene calibration constants — see
   `utils.parse_calibration_lut`.)
4. Run the pipeline:
   ```bash
   python 01_download_s1_grd.py --aoi aoi_sathupalle_wgs84.geojson \
       --start 2025-06-01 --end 2025-11-30 --collection sentinel-1-rtc
   python 02_calibrate_and_filter.py --collection sentinel-1-rtc
   python 03_wetness_index_model.py
   ```
5. Look at `awd_wetness_index.png` and the printed cycle count / likelihood
   score.
6. Sanity-check against RiceAtlas (download the calendar CSV from
   https://dx.doi.org/10.7910/DVN/JE6R2R first):
   ```bash
   python 04_validate_against_open_sources.py calendar-check \
       --date 2025-07-05 --calendar-csv riceatlas.csv --state Telangana --district Khammam
   ```
7. Once you have real labels (open regional dataset, once access is confirmed,
   or — per our agreement — Varaha's own field data as a final holdout only):
   ```bash
   python 04_validate_against_open_sources.py score \
       --labels-csv field_labels.csv --predictions-csv model_outputs.csv
   ```

## Files

- `utils.py` — calibration math, Refined Lee filter, Otsu threshold, wetness
  index, drying-cycle detection. This is the part with real unit-tested logic.
- `01_download_s1_grd.py` — MPC STAC search + AOI-clipped download.
- `02_calibrate_and_filter.py` — calibration + speckle filter + AOI zonal
  stats → time series CSV.
- `03_wetness_index_model.py` — the v0 "model": wetness index, drying-cycle
  detection, AWD-likelihood score, plot.
- `04_validate_against_open_sources.py` — RiceAtlas sanity check + label
  scoring harness (precision/recall/F1) for whenever real labels are available.
- `test_synthetic.py` — **run this first** on any new machine to confirm the
  environment/logic works before spending API quota on real downloads.
- `aoi_sathupalle_wgs84.geojson` — the pilot AOI.

## Alternative data path: PALSAR-2 via Google Earth Engine (no download step needed)

`05_gee_palsar2_awd.js` implements the same AWD approach on JAXA's PALSAR-2
ScanSAR L-band data (`JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR`), run entirely in
the Google Earth Engine Code Editor (https://code.earthengine.google.com).
This sidesteps the network problem above completely: GEE executes server-side
under your own Google/EE account in your browser, not in this sandbox, so
there's nothing to download and no proxy restriction to hit.

What it does:
1. Checks image availability for your AOI/date range first (do this before
   anything else — PALSAR-2 ScanSAR's revisit over a given area is typically
   sparser than Sentinel-1's, and if there are only 0-2 images in a season
   you don't have enough for drying-cycle detection, only a snapshot).
2. Calibrates DN → gamma-naught dB (`10*log10(DN^2) - 83.0`, JAXA's documented
   formula for this collection) and masks out layover/shadow/ocean/invalid
   pixels using the `MSK` band.
3. Visualizes the calibrated HH layer per date (adapted from the snippet
   this was built from).
4. Extracts an AOI-mean/median/p10/p90 time series per date, in the **same
   CSV schema** as `02_calibrate_and_filter.py`'s output — so once exported,
   it's a drop-in input to `03_wetness_index_model.py` with no code changes:
   `python 03_wetness_index_model.py --timeseries palsar2_ambala_timeseries.csv --polarization HH`
5. Exports the time series table to Google Drive as CSV.

Why PALSAR-2 in addition to (not just instead of) Sentinel-1: L-band
penetrates the rice canopy better than C-band, so it can sense sub-canopy
water status even after canopy closure — this is why Arai et al. (2022) and
the ALOS-2+IoT paper in the literature review use it specifically for
irrigation-status detection. The tradeoff is coarser revisit, which is
exactly what step 1 above checks before you invest further.

This script has **not been run against live GEE data** in this build session
(no GEE-authenticated session available here) — paste it into the Code
Editor yourself and check the Console output, especially the image-count
checks, before trusting the rest.

## Full pixel-level pipeline in Earth Engine only (`06_gee_awd_full_pixel_model.js`)

`05_gee_palsar2_awd.js` extracts a single AOI-averaged time series (good for
sanity-checking thresholds via a chart, then handing off to the Python
model). `06_gee_awd_full_pixel_model.js` goes further: it runs the entire
methodology **per pixel, inside Earth Engine**, and the only things you pull
out are the two deliverables you asked for:

- **GeoTIFF** (`Export.image.toDrive`) with bands: `awd_likelihood` (0-1
  score per pixel), `is_awd` / `is_continuous_flood` (binary classification),
  `n_cycles` (detected drying/re-flood events per pixel).
- **Acreage stats CSV** (`Export.table.toDrive`): total hectares classified
  AWD vs. continuously-flooded within the AOI, plus the thresholds/season
  used to produce them.

How it ports the Python model (`utils.py`) to per-pixel GEE logic:
- Wetness index: `ImageCollection.min()`/`.max()` give the per-pixel seasonal
  min/max directly (GEE's reducers are inherently per-pixel), so the
  min-max rescale is a straightforward image expression.
- Drying-cycle count: **this is a simplification**, not an exact port. The
  Python version runs a true sequential state machine per series;
  Earth Engine has no native per-pixel sequential loop, so this instead
  counts "dry at time *t*, flooded again at time *t+1*" events via
  shifted-array comparison (`arraySlice` + elementwise multiply). Same
  underlying idea as Lovell (2019)'s own consecutive-time-step change
  detection, but validate the resulting acreage numbers against known
  fields before trusting them for anything decision-grade.

Other differences from the Python path, all noted in the script's header
comment: speckle filtering is a simpler focal-median smooth (not the
adaptive Refined Lee filter in `utils.py`), there's no real rice-crop mask
yet (uses "was this pixel ever flooded" as a rough paddy proxy), and it uses
HH only (see the HH-vs-HV caveat from `05_gee_palsar2_awd.js`).

**Not run against live data** — paste into the Code Editor, expect at least
one round of fixing real errors (same pattern as the HV band issue in `05`),
and start with a small AOI (a single village/block) before scaling to a
whole district, since per-pixel array operations are much more
computationally expensive than the AOI-averaged version.

## Automated pipeline: shapefile → stats + GeoTIFF (`awd_pipeline.py`)

`06_gee_awd_full_pixel_model.js` is the same methodology but has to be pasted
into the Code Editor by hand, with a hardcoded rectangle AOI and Drive exports.
`awd_pipeline.py` turns it into a repeatable command: **shapefile in → AWD stats
+ GeoTIFF out**, run from your terminal against Earth Engine's Python API, with
the outputs downloaded straight into a local folder. Design doc:
`docs/superpowers/specs/2026-07-30-awd-shapefile-pipeline-design.md`.

```bash
pip install -r requirements.txt
earthengine authenticate          # one time; opens a browser
python awd_pipeline.py \
    --shapefile my_fields.shp \
    --start 2020-06-01 --end 2020-11-30 \
    --out outputs/ \
    --project <your-ee-cloud-project>   # if your EE account requires one
```

What it does (all model logic is a faithful port of `06`):
1. `load_aoi` — reads the shapefile, reprojects to WGS84, **dissolves all
   polygons into one AOI** (whole-shapefile aggregate).
2. `build_awd_image` — PALSAR-2 ScanSAR HH → calibrate → despeckle → per-pixel
   Wetness Index → shifted-array drying-cycle count → AWD classification.
3. `compute_stats` — `reduceRegion` sum of per-class pixel area over the AOI.
4. `download_geotiff` — pulls the classification raster locally via
   `getDownloadURL`.

Outputs in `--out/`:
- `awd_classification.tif` — 4 bands: `awd_likelihood`, `is_awd`,
  `is_continuous_flood`, `n_cycles` (25 m, clipped to the shapefile).
- `awd_stats.json` + `awd_stats.csv` — one aggregate row: `awd_area_ha`,
  `continuous_flood_area_ha`, `paddy_area_ha`, `pct_awd`, `n_images`, plus the
  season and every threshold used (provenance).

All model thresholds are CLI flags defaulting to `06`'s values
(`--wet 0.3 --dry -0.1 --ref-cycles 5 --awd-thresh 0.4 --scale 25`) — override
them to match a calibrated reference (e.g. the MDPI 18(13):2190 numbers) without
touching code.

**Tested vs. not:** the pure-Python units (`load_aoi`, `write_stats`,
`parse_args`, the download-guardrail) have unit tests — run `pytest
test_awd_pipeline.py`. The Earth Engine functions call live GEE and are **not
executed in CI**; they mirror `06`'s already-reasoned logic. **Run on a SMALL
AOI first** (a single village/block): per-pixel array ops are expensive and
large AOIs hit GEE's ~50 MB direct-download ceiling — the script catches that
and prints a tiling/`--scale` hint rather than a stack trace, but it's still a
real limit. The same caveats as `06` apply (HH-only, no rice mask, simplified
despeckle + cycle-count, thresholds not yet ground-truth-calibrated).

## Known gaps / what this is *not* yet

- Not yet using ascending+descending orbit fusion (roadmap item, addresses
  the Sentinel-1 revisit-frequency limitation from Shah et al. 2025).
- Not yet conditioning on soil type/AEZ covariates (Varaha's existing
  `aez_soil_texture_comnined.shp` / `Soil_texture.ipynb` are the natural
  source for this — not wired in yet).
- Not yet field-boundary-aware (currently aggregates over the whole block
  polygon; real deployment should run per-field, using field boundaries if
  available, to avoid mixed-pixel dilution).
- Fixed wet/dry WI thresholds (0.3 / -0.1) are a starting guess, not
  calibrated against any ground truth yet — first thing to tune once real
  labels are available.
- This is the unsupervised v0 baseline. The pooled, covariate-conditioned ML
  sequence model described in `AWD_Global_Model_Roadmap.md` is the next
  phase, once open regional AWD datasets' access terms are confirmed.
