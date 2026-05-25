"""
Canonical column schema and GlobalSoilMap standard depth intervals.

Every collector produces a DataFrame that exactly matches CANONICAL_COLUMNS.
The depth harmonizer converts any raw horizon table to STANDARD_DEPTHS.
"""

from dataclasses import dataclass
from typing import List, Tuple

# GlobalSoilMap standard depth intervals (cm)
STANDARD_DEPTHS: List[Tuple[int, int]] = [
    (0, 5),
    (5, 15),
    (15, 30),
    (30, 60),
    (60, 100),
    (100, 200),
]

# ---------------------------------------------------------------------------
# Canonical raw-horizon schema
# Every collector must produce exactly these columns (NaN where unavailable).
# ---------------------------------------------------------------------------
CANONICAL_COLUMNS = [
    # --- Identifiers ---
    "source_db",          # str  : 'WoSIS', 'ISCN', 'LUCAS', 'NCSCD', 'RaCA', 'AfSIS'
    "profile_id",         # str  : unique profile identifier within source
    "site_id",            # str  : optional site/plot grouping (may equal profile_id)

    # --- Location ---
    "longitude",          # float: WGS-84 decimal degrees
    "latitude",           # float: WGS-84 decimal degrees
    "location_accuracy",  # float: positional uncertainty (m), NaN if unknown
    "country_code",       # str  : ISO-3166-1 alpha-3

    # --- Temporal ---
    "observation_date",   # str  : ISO-8601 date of sampling (YYYY-MM-DD or YYYY)

    # --- Horizon geometry ---
    "upper_depth",        # float: upper boundary of horizon (cm)
    "lower_depth",        # float: lower boundary of horizon (cm)

    # --- Primary target variable ---
    "soc_g_per_kg",       # float: Soil Organic Carbon (g kg⁻¹ fine earth)

    # --- Supporting properties for SOC stock calculation ---
    "bulk_density_g_cm3", # float: fine-earth bulk density (g cm⁻³)
    "coarse_fragments_pct",# float: volumetric coarse fragment content (%)
    "ph_h2o",             # float: pH in water (1:2.5 or 1:5 suspension)
    "clay_pct",           # float: clay content (% fine earth)
    "silt_pct",           # float: silt content (% fine earth)
    "sand_pct",           # float: sand content (% fine earth)
    "cec_cmol_kg",        # float: CEC (cmol(+) kg⁻¹)
    "total_n_g_per_kg",   # float: total N (g kg⁻¹)

    # --- Metadata ---
    "soc_method",         # str  : analytical method (e.g. 'Walkley-Black', 'LOI', 'DUMAS')
    "land_use",           # str  : broad land-use class
    "biome",              # str  : biome/ecoregion label
    "horizon_designation",# str  : FAO/USDA horizon code (e.g. 'A', 'Bh')
]


@dataclass
class DepthSlice:
    """A harmonised observation at a single GlobalSoilMap depth interval."""
    source_db: str
    profile_id: str
    longitude: float
    latitude: float
    upper_depth: int
    lower_depth: int
    soc_g_per_kg: float
    bulk_density_g_cm3: float = float("nan")
    coarse_fragments_pct: float = 0.0
    soc_stock_t_ha: float = float("nan")  # derived
