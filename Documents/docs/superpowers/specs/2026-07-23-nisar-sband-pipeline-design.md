# NISAR SAR Data Pipeline (Bhoonidhi S-band / ASF L-band)

**Date:** 2026-07-23
**Status:** Approved
**Objective:** Given an AOI shapefile and a date range, search, download, and process NISAR L2 GSLC/GCOV granules into clipped, mosaicked GeoTIFFs — sourced from ASF (NASA L-band, active now) with a pluggable interface so ISRO's Bhoonidhi S-band source drops in later without touching downstream processing.

---

## 1. Context

The end goal is an S-band NISAR pipeline via ISRO's Bhoonidhi portal (`https://bhoonidhi.nrsc.gov.in`), which hosts S-SAR daily processed products over the Indian landmass. Bhoonidhi has a documented STAC-based search API with JWT auth, but the API host (`bhoonidhi-api.nrsc.gov.in`) is unreachable from this environment — DNS resolves it to a different IP than the main portal, but every port (443, 80, ICMP) times out, while the main portal and other `gov.in` domains are reachable fine. This matches NRSC's documented note that API access requires a separate request to `bhoonidhi@nrsc.gov.in` — likely IP-allowlisting gates the API host even though portal login itself works. The user has emailed NRSC requesting API access (sent 2026-07-22); this design proceeds without waiting on that reply.

In the meantime, NASA's Alaska Satellite Facility (ASF) hosts the same mission's L-band (L-SAR) products, with a fully public search API and download auth via the user's already-configured NASA Earthdata Login (`.netrc`). This was empirically validated end-to-end during design: searched live GSLC/GCOV granules, downloaded a real 406MB dual-pol GCOV file, and ran `polsartools.import_nisar_gcov` on it to produce valid georeferenced GeoTIFFs (UTM 39N, gamma0 values physically sensible — HH mean 0.024 > HV mean 0.003).

One real bug was found in this process: `polsartools.import_nisar_gcov` only handles dual-pol (2 channel) or quad-pol (4 channel) products — a single-pol granule hits an unhandled branch with a broken `raise('string')` (not a valid exception instance), surfacing as a confusing `TypeError` instead of a clear message. The pipeline must filter out single-pol granules before attempting conversion.

Given this, the pipeline is architected with a swappable data-source interface: `ASFSource` (L-band, working today) and `BhoonidhiSource` (S-band, stubbed pending API access) both implement the same `search()`/`download()` contract, so everything downstream — pol-mode filtering, HDF5→GeoTIFF conversion, per-date mosaicking, clipping — is identical regardless of which mission's data is flowing through it.

---

## 2. Architecture

```
[Shapefile AOI]
      │
      ▼
[0] Credential bootstrap   ensure_netrc(host="urs.earthdata.nasa.gov")
                           checks ~/.netrc for the required `machine` entry;
                           if missing, getpass.getpass() prompt (never echoed
                           to chat/logs) → appends entry, chmod 600.
                           Bhoonidhi (once active) needs an analogous check
                           for BHOONIDHI_USER_ID / BHOONIDHI_PASSWORD env vars
                           — its auth is JWT-based, not .netrc.
      │
      ▼
[1] AOI Loader             shapefile → WGS84 bbox (search) + native/UTM geometry
                           (clip), via geopandas
      │
      ▼
[2] Search                  DataSource.search(bbox, start_date, end_date, product_type)
    ├── ASFSource            api.daac.asf.alaska.edu, platform=NISAR,
    │                        processingLevel=GSLC|GCOV                [ACTIVE]
    └── BhoonidhiSource       bhoonidhi-api STAC search, collections=
                              NISAR_SSAR-Beta_GSLC|GCOV                [STUB]
      │
      ▼
[3] Pol-mode filter        keep only dual-pol / quad-pol granules (single-pol
                           breaks polsartools' importer — see bug above);
                           group remaining granules by acquisition date
      │
      ▼
[4] Download               DataSource.download(granule) → local .h5
                           ASF: .netrc-authenticated redirect chain (verified)
                           Bhoonidhi: JWT bearer token                 [STUB]
      │
      ▼
[5] HDF5 → GeoTIFF         polsartools.import_nisar_gslc(mat=C2/C3 by pol count)
                           polsartools.import_nisar_gcov (auto I2/I4 by pol count)
                           → one GeoTIFF per band per granule
      │
      ▼
[6] Per-date mosaic        merge same-date granule GeoTIFFs (rioxarray.merge_arrays),
                           reprojecting to the AOI's UTM zone first (granules near
                           zone boundaries may differ in native CRS)
      │
      ▼
[7] Clip + save            clip to shapefile geometry (force_2d, same Z-coordinate
                           fix as the S1 STAC pipeline) → final multi-band GeoTIFF
                           per pol/matrix band, one raster band per acquisition
                           date (mirrors the S1 pipeline's
                           mosaic_s1_{band}_{tag}.tif convention)
```

---

## 3. Product handling

| Product | Pol count | polsartools call | Output bands |
|---|---|---|---|
| GSLC, dual-pol | 2 | `import_nisar_gslc(mat='C2', azlks, rglks)` | C11, C12_real, C12_imag, C22 |
| GSLC, quad-pol | 4 | `import_nisar_gslc(mat='C3', azlks, rglks)` | C11, C12_real/imag, C13_real/imag, C22, C23_real/imag, C33 |
| GCOV, dual-pol | 2 | `import_nisar_gcov(azlks, rglks)` (auto → I2) | HHHH, HVHV (or whichever pol pair) |
| GCOV, quad-pol | 4 | `import_nisar_gcov(azlks, rglks)` (auto → I4) | HHHH, HVHV, VHVH, VVVV |
| single-pol (any) | 1 | **skipped at the filter stage**, logged with reason | — |

Output is raw calibrated bands only — no polarimetric decomposition, speckle filtering, or Pauli RGB baked into v1 (that can be a separate later step reading these outputs).

---

## 4. Output structure

```
out_dir/
├── granules/                          # per-granule polsartools output (intermediate)
│   └── {granule_id}/{I2|I4|C2HX|C3}/{BAND}.tif
├── {band}_{date}.tif                  # per-date mosaic, clipped to AOI, one per band
└── mosaic_{band}_{start}_{end}.tif    # final multi-band (band=date) stack, one per pol/matrix element
```

Raw `.h5` files are deleted after conversion by default (matches the user's existing practice — Downloads only retains derived `.tif` + `.aux.xml` sidecars from past runs, not the source HDF5s). A `keep_h5` flag disables this.

---

## 5. Multi-date handling

A date-range search may return multiple acquisition dates, each with one or more spatial granules covering the AOI. Per date: mosaic that date's granules, clip once. Final output stacks all dates as bands of one multi-band GeoTIFF per pol/matrix element (band descriptions = date strings) — not a single all-dates composite, and not one file per date. This matches the S1 STAC pipeline's existing `mosaic_s1_{band}_{tag}.tif` convention exactly.

---

## 6. Error handling

- No granules found for bbox/date/product → clear `RuntimeError`, not a silent empty output
- Single-pol granule → skip + log (not a hard failure)
- Bhoonidhi `Online: "N"` granule (once active) → skip + log, don't block the run
- A granule that fails download or conversion → skip that granule, continue with the rest (same per-item try/except pattern as the S1 pipeline's `process_tile`), report all failures in a summary at the end

---

## 7. Credentials

- **ASF**: NASA Earthdata Login via `~/.netrc` (`machine urs.earthdata.nasa.gov`) — already configured for the current user and verified working end-to-end. `ensure_netrc()` checks for this entry and prompts via `getpass` (append-only, never overwrites other existing `.netrc` entries) if a new user runs the notebook without one.
- **Bhoonidhi** (once active): `BHOONIDHI_USER_ID` / `BHOONIDHI_PASSWORD` environment variables → JWT bearer token exchange. Separate check/prompt analogous to `ensure_netrc()`.

---

## 8. Test AOI

A real Google S2 geometry cell (via the `s2sphere` library), converted to a polygon and saved as `test_aoi/s2_cell_test_aoi.geojson` — used only as a convenient, reproducible test input shapefile. This is unrelated to NISAR's own data organization: NISAR granules are searched by bbox/frame intersection, not any tile grid, so the S2 cell is purely a test-fixture convenience, not a pipeline concept.

---

## 9. Project structure

```
Documents/Agroforestry/GEE_Codebase/Download_pipelines/NISAR_bhoonidhi_asf/
├── nisar_pipeline.py          # AOI loader, ASFSource/BhoonidhiSource, pol filter,
│                               #   download, polsartools wrapper, mosaic, clip,
│                               #   save, ensure_netrc()
├── nisar_pipeline_run.ipynb   # notebook interface — imports the module, runs the
│                               #   pipeline step-by-step with quick-look plots
│                               #   between stages (matches existing NISAR notebook
│                               #   style: plot_images, RGB previews)
├── environment_nisar.yml      # new conda env `nisar_pipeline`
└── test_aoi/
    └── s2_cell_test_aoi.geojson
```

---

## 10. Environment

New dedicated conda env `nisar_pipeline`, built on the same package set as the existing `pst_env` (already has `geopandas`, `shapely`, `rasterio`, `gdal`, `h5py`, `netCDF4`, `requests`, `polsartools`), plus three additions:

- `rioxarray` — mosaic/clip operations
- `s2sphere` — test AOI generation only
- `asf_search` — official ASF client library, wrapping the search/download calls already hand-verified with raw `requests`/`curl` during design

---

## 11. Notebook interface

A `CONFIG` cell (shapefile path, start/end date, product type, `azlks`/`rglks` default `5`/`5` — matches the validated smoke-test run, `out_dir`, `keep_h5` flag default `False`) feeding a `run(**config)` function in `nisar_pipeline.py` — same shape as the S1 STAC pipeline's `CONFIG` constants + `run()` function, just called from a notebook cell instead of `argparse`.

---

## 12. Out of scope (v1)

- Bhoonidhi/S-band source — `BhoonidhiSource` implements the same interface but is stubbed/untested until API access is granted
- Polarimetric decomposition, speckle filtering, Pauli RGB decomposition
- Interferometric products (RIFG/RUNW/GUNW) and offset products (ROFF/GOFF) — GSLC/GCOV only
- Parallel downloads — granules are large (100MB–2GB+) and Bhoonidhi caps concurrency anyway; v1 processes sequentially
- Retry/backoff beyond a single skip-and-log per granule
- Disk-space checks beyond the `keep_h5` flag

---

## 13. Success criteria

| Criterion | Target |
|---|---|
| ASF search → download → polsartools conversion, real AOI | Runs end-to-end, produces valid georeferenced GeoTIFFs |
| Single-pol granules | Skipped with a logged reason, never crash the run |
| Multi-date range with >1 acquisition | Produces one multi-band mosaic per pol/matrix band, band-per-date |
| New user, no `.netrc` | Prompted once via `ensure_netrc()`, never asked again |
| `BhoonidhiSource` | Same interface as `ASFSource`; swapping source requires no changes to stages 3–7 |
