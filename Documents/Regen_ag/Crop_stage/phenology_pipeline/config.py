"""Configuration for paddy crop phenology assessment (120-day kharif variety)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_STAC_ROOT = Path(
    "/Users/mipl/Documents/Agroforestry/GEE_Codebase/Download_pipelines/STAC_s1_s2"
)

DateConfidence = Literal["trusted", "uncertain"]
Era5Source = Literal["openmeteo", "cds"]
Era5SpatialMode = Literal["aoi", "per_farm"]


@dataclass
class GDDStageThreshold:
    """Cumulative GDD from transplant (°C), scaled for ~120-day paddy (D-6072)."""

    name: str
    gdd_min: float
    gdd_max: float


DEFAULT_GDD_STAGES: tuple[GDDStageThreshold, ...] = (
    GDDStageThreshold("establishment", 0, 80),
    GDDStageThreshold("vegetative", 80, 520),
    GDDStageThreshold("panicle_initiation", 520, 640),
    GDDStageThreshold("reproductive", 640, 1063),
    GDDStageThreshold("ripening", 1063, 1176),
    GDDStageThreshold("maturity", 1176, 1650),
)


@dataclass
class PhenologyConfig:
    """End-to-end phenology run configuration."""

    # Input geometry
    polygon_path: str = ""
    farm_id_col: str = "farm_id"
    farm_id: str | None = None

    # Crop calendar
    variety_duration_days: int = 120
    t_base_c: float = 10.0
    nursery_to_transplant_days: int = 28
    target_total_gdd: float = 1650.0

    # Kharif window: download May 1 → Nov 30; peak search May 1 → Sep 30
    season_name: str = "Kharif"
    download_start: str = "2025-05-01"
    download_end: str = "2025-11-30"
    peak_search_end: str = "2025-09-30"

    # Provided dates (optional)
    sowing_date: str | None = None
    transplant_date: str | None = None
    date_confidence: DateConfidence = "uncertain"
    date_mismatch_days: int = 14

    # STAC download
    stac_root: Path = DEFAULT_STAC_ROOT
    download_data: bool = True
    stac_interval: str = "12D"
    stac_resolution_m: int = 10
    stac_max_workers: int = 2
    timeseries_cache_dir: str | None = None

    # ERA5-Land weather for GDD
    era5_cache_dir: str = ".cache/phenology/era5"
    era5_time_zone: str = "utc+05:30"
    era5_source: Era5Source = "openmeteo"  # openmeteo: seconds; cds: queued CDS jobs
    era5_spatial_mode: Era5SpatialMode = "aoi"  # aoi: one fetch for whole shapefile
    compute_gdd: bool = False  # optional ERA5 weather + GDD staging (slow if CDS)

    # GDD stage thresholds
    gdd_stages: tuple[GDDStageThreshold, ...] = field(default_factory=lambda: DEFAULT_GDD_STAGES)

    # Output
    output_dir: str = "phenology_output"

    def validate(self) -> None:
        if not self.polygon_path:
            raise ValueError("polygon_path is required")
        p = Path(self.polygon_path)
        if not p.exists():
            raise ValueError(f"polygon_path not found: {p}")
        if self.date_confidence not in ("trusted", "uncertain"):
            raise ValueError("date_confidence must be 'trusted' or 'uncertain'")
        if self.date_confidence == "trusted" and not self.transplant_date and not self.sowing_date:
            raise ValueError(
                "transplant_date or sowing_date is required when date_confidence='trusted'"
            )
        if not self.download_data and not self.timeseries_cache_dir:
            # allow default stac_download under output_dir
            pass
        if self.era5_source not in ("openmeteo", "cds"):
            raise ValueError("era5_source must be 'openmeteo' or 'cds'")
        if self.era5_spatial_mode not in ("aoi", "per_farm"):
            raise ValueError("era5_spatial_mode must be 'aoi' or 'per_farm'")
