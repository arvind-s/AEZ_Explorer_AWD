# Crop phenology assessment (120-day kharif paddy)

Farm-level crop phenology using:

1. **STAC S1/S2 download** — Agroforestry `STAC_s1_s2` `pipeline_runner` (Sentinel-1 RTC VH + Sentinel-2 L2A NDVI)
2. **farm_heterogeneity signal logic** — green-up, peak search, curve cleaning
3. **ERA5-Land GDD** — daily min/max temperature via Copernicus CDS
4. **D-6072 GDD stage thresholds** — scaled for ~120-day variety

## Kharif calendar

| Window | Dates |
|--------|--------|
| Satellite download | **May 1** → Nov 30 |
| Peak search (NDVI/VH) | **May 1** → Sep 30 |
| Ripening / harvest GDD | through Nov |

## Quick start

### 1. CDS API (ERA5-Land)

Register at [Copernicus CDS](https://cds.climate.copernicus.eu/) and accept the ERA5-Land datasets. Then either:

```bash
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API_KEY>
```

Or set `CDS_API_URL` and `CDS_API_KEY` in the environment.

### 2. Planetary Computer (STAC)

For Sentinel-1 RTC, request a PC subscription key and export:

```bash
export PC_SDK_SUBSCRIPTION_KEY=...
```

### 3. Install

```bash
cd Crop_stage
pip install -r phenology_pipeline/requirements.txt
```

### 4. Run

**Case 1 — trusted transplant date:**

```bash
python examples/run_phenology.py \
  --polygon examples/sample_farm.geojson \
  --farm-id demo_farm_001 \
  --download-start 2025-05-01 \
  --download-end 2025-11-30 \
  --transplant-date 2025-08-09 \
  --date-confidence trusted \
  --assessment-date 2025-10-15 \
  --output-dir phenology_output
```

**Case 2 — estimate dates from satellite:**

```bash
python examples/run_phenology.py \
  --polygon examples/sample_farm.geojson \
  --farm-id demo_farm_001 \
  --download-start 2025-05-01 \
  --download-end 2025-11-30 \
  --date-confidence uncertain \
  --output-dir phenology_output
```

## Outputs

- `phenology_output/phenology_<farm_id>.json` — stage, GDD, dates, confidence, flags
- `phenology_output/phenology_timeseries_<farm_id>.csv` — NDVI, VH, unified curve, cumulative GDD
- `phenology_output/stac_download/` — raw STAC NetCDF/CSV from pipeline_runner

## STAC path

Default STAC root:

`/Users/mipl/Documents/Agroforestry/GEE_Codebase/Download_pipelines/STAC_s1_s2`

Override with `--stac-root` or `PhenologyConfig.stac_root`.

## Signal selection

Per observation date: **S2 NDVI** when valid (cloud-masked), else **S1 VH**.

Transplant estimation (uncertain mode):

- VH flood dip (primary for transplanted paddy)
- NDVI green-up minus emergence lag
- Fused when both agree within 14 days

## GDD stages (cumulative from transplant, °C)

| Stage | GDD range |
|-------|-----------|
| establishment | 0 – 80 |
| vegetative | 80 – 520 |
| panicle_initiation | 520 – 640 |
| reproductive | 640 – 1063 |
| ripening | 1063 – 1176 |
| maturity | 1176 – 1650 |

Reference: [D-6072](https://arccjournals.com/journal/agricultural-science-digest/D-6072) (scaled for 120-day variety).

## Pixel heterogeneity

When STAC pixel CSV is available, the pipeline runs per-pixel green-up analysis and flags farms where pixel-level sowing estimates diverge by ≥20 days (`pixel_heterogeneity` in JSON output).

## Cached STAC reuse

Re-run without re-downloading:

```bash
python examples/run_phenology.py \
  --polygon examples/real_paddy_farm.geojson \
  --farm-id rice_farm_1929 \
  --no-download \
  --stac-cache-dir phenology_output/real_paddy_run/stac_download \
  --output-dir phenology_output/real_paddy_run
```

## Real farm example

`examples/real_paddy_farm.geojson` — rice field extracted from `ALU_AMED/farm_data.geojson` (Tamil Nadu).

```bash
bash examples/run_real_paddy.sh
```

## Tests (offline)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r phenology_pipeline/requirements.txt pytest
pytest tests/test_phenology.py -q
```

## Visualization notebook

`notebooks/phenology_visualization.ipynb` — plots NDVI/VH curves, GDD stage bands, farm map, and summary table.

```bash
jupyter notebook notebooks/phenology_visualization.ipynb
```

Set `FARM_ID` and `OUTPUT_DIR` in the notebook config cell to match your pipeline run.

## Python API

```python
from phenology_pipeline import PhenologyConfig, run_phenology

cfg = PhenologyConfig(
    polygon_path="farm.geojson",
    farm_id="F001",
    transplant_date="2025-08-09",
    date_confidence="trusted",
    download_start="2025-05-01",
    download_end="2025-11-30",
)
result = run_phenology(cfg, assessment_date="2025-10-01")
```
