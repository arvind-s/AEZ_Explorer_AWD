"""
RaCA (Rapid Carbon Assessment) collector — USDA NRCS.

Source  : https://www.nrcs.usda.gov/resources/data-and-reports/rapid-carbon-assessment-raca
Dataset : ~6 000 sites, ~58 000 soil layers, contiguous United States.
Bulk DL : https://www.nrcs.usda.gov/sites/default/files/2022-09/raca-data.zip
License : Public Domain (US Federal Government)

RaCA used a probabilistic design — samples are nationally representative
for CONUS and include stocks to 100 cm.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

_RACA_URL = ("https://www.nrcs.usda.gov/sites/default/files/2022-09/raca-data.zip")


class RaCACollector(BaseCollector):

    SOURCE_NAME = "RaCA"

    def __init__(self, raw_dir: str | Path = "data/raw/raca"):
        super().__init__(raw_dir)
        self._zip_path = self.raw_dir / "raca-data.zip"

    # ------------------------------------------------------------------

    def download(self) -> None:
        self._download_file(_RACA_URL, self._zip_path)
        with zipfile.ZipFile(self._zip_path) as zf:
            zf.extractall(self.raw_dir)

    def _is_downloaded(self) -> bool:
        return any(self.raw_dir.glob("*.csv")) or any(self.raw_dir.glob("*.xlsx"))

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        # RaCA ships as separate CSVs: sites + layers
        site_files  = sorted(self.raw_dir.glob("*site*.csv"))
        layer_files = sorted(self.raw_dir.glob("*layer*.csv"))

        sites  = pd.concat([pd.read_csv(f, low_memory=False) for f in site_files],
                            ignore_index=True) if site_files else pd.DataFrame()
        layers = pd.concat([pd.read_csv(f, low_memory=False) for f in layer_files],
                            ignore_index=True) if layer_files else pd.DataFrame()

        if layers.empty:
            raise FileNotFoundError(f"No RaCA layer CSVs found in {self.raw_dir}")

        # ---- layer columns ----
        rename_layers = {
            "rcasiteid":    "site_id",
            "lay_seq":      "horizon_designation",
            "hztop":        "upper_depth",
            "hzbot":        "lower_depth",
            "soc":          "soc_g_per_kg",      # stored as g/kg
            "bulk_den":     "bulk_density_g_cm3",
            "cf_vol":       "coarse_fragments_pct",
            "ph_h2o":       "ph_h2o",
            "clay_tot":     "clay_pct",
            "silt_tot":     "silt_pct",
            "sand_tot":     "sand_pct",
        }
        layers = layers.rename(columns={k: v for k, v in rename_layers.items()
                                        if k in layers.columns})

        # ---- site columns ----
        rename_sites = {
            "rcasiteid": "site_id",
            "longitude": "longitude",
            "latitude":  "latitude",
            "sampledate": "observation_date",
            "landuse":   "land_use",
            "ecoregiond": "biome",
        }
        if not sites.empty:
            sites = sites.rename(columns={k: v for k, v in rename_sites.items()
                                           if k in sites.columns})
            layers = layers.merge(
                sites[["site_id", "longitude", "latitude",
                        "observation_date", "land_use", "biome"]].drop_duplicates("site_id"),
                on="site_id", how="left",
            )

        layers["profile_id"] = layers["site_id"].astype(str)
        layers["country_code"] = "USA"
        return layers
