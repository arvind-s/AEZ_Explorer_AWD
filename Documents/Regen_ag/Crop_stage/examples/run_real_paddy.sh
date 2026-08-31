#!/usr/bin/env bash
# Real paddy farm (Tamil Nadu, from ALU_AMED/farm_data.geojson rice field)
# Requires: CDS API credentials + Planetary Computer subscription key

set -euo pipefail
cd "$(dirname "$0")/.."

POLYGON="examples/real_paddy_farm.geojson"
FARM_ID="rice_farm_1929"
OUT="phenology_output/real_paddy_run"

python examples/run_phenology.py \
  --polygon "$POLYGON" \
  --farm-id "$FARM_ID" \
  --download-start 2024-05-01 \
  --download-end 2024-11-30 \
  --peak-search-end 2024-09-30 \
  --date-confidence uncertain \
  --assessment-date 2024-10-01 \
  --output-dir "$OUT" \
  --stac-cache-dir "$OUT/stac_download" \
  --workers 2

echo "Results: $OUT/phenology_${FARM_ID}.json"
