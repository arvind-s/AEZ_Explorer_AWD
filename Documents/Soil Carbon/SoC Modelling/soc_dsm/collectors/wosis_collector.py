"""
WoSIS (World Soil Information Service) collector — ISRIC.

Source  : https://www.isric.org/explore/wosis
Bulk DL : https://zenodo.org/records/7799781  (WoSIS snapshot 2023)
License : CC BY 4.0

Provides ~196 000 geo-referenced soil profiles worldwide.
Download is two TSV files:
  wosis_latest_profiles.tsv  — profile metadata (lat/lon, date, country)
  wosis_latest_layers.tsv    — horizon data (depths, OC, BD, texture …)
"""

import logging
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

# Zenodo record for WoSIS snapshot 2023 (update DOI when a newer snapshot ships)
_ZENODO_BASE = "https://zenodo.org/records/7799781/files"
_PROFILES_URL = f"{_ZENODO_BASE}/wosis_latest_profiles.tsv?download=1"
_LAYERS_URL   = f"{_ZENODO_BASE}/wosis_latest_layers.tsv?download=1"


class WoSISCollector(BaseCollector):

    SOURCE_NAME = "WoSIS"

    def __init__(self, raw_dir: str | Path = "data/raw/wosis"):
        super().__init__(raw_dir)
        self._profiles_path = self.raw_dir / "wosis_latest_profiles.tsv"
        self._layers_path   = self.raw_dir / "wosis_latest_layers.tsv"

    # ------------------------------------------------------------------

    def download(self) -> None:
        self._download_file(_PROFILES_URL, self._profiles_path)
        self._download_file(_LAYERS_URL,   self._layers_path)

    def _is_downloaded(self) -> bool:
        return self._profiles_path.exists() and self._layers_path.exists()

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        profiles = pd.read_csv(self._profiles_path, sep="\t", low_memory=False)
        layers   = pd.read_csv(self._layers_path,   sep="\t", low_memory=False)

        # ---- normalise profile columns ----
        prof_cols = {
            "profile_id": "profile_id",
            "longitude":  "longitude",
            "latitude":   "latitude",
            "country_code": "country_code",
            "date_of_observation": "observation_date",
        }
        profiles = profiles.rename(columns={v: k for k, v in prof_cols.items()
                                            if v in profiles.columns})

        # ---- normalise layer columns ----
        # WoSIS exposes OC as 'oc' (g kg⁻¹), BD as 'bulk_density'
        layer_cols = {
            "profile_id":       "profile_id",
            "upper_depth":      "upper_depth",
            "lower_depth":      "lower_depth",
            "oc":               "soc_g_per_kg",
            "oc_value_avg":     "soc_g_per_kg",   # fallback column name
            "bulk_density":     "bulk_density_g_cm3",
            "cfvo":             "coarse_fragments_pct",
            "ph_h2o":           "ph_h2o",
            "clay":             "clay_pct",
            "silt":             "silt_pct",
            "sand":             "sand_pct",
            "cec":              "cec_cmol_kg",
            "nitrogen":         "total_n_g_per_kg",
            "oc_method":        "soc_method",
            "horizon_master":   "horizon_designation",
        }
        layers = layers.rename(columns={v: k for k, v in layer_cols.items()
                                        if v in layers.columns})

        # keep only the first 'soc_g_per_kg' if renamed twice
        if layers.columns.duplicated().any():
            layers = layers.loc[:, ~layers.columns.duplicated()]

        df = layers.merge(
            profiles[["profile_id", "longitude", "latitude",
                       "country_code", "observation_date"]],
            on="profile_id",
            how="left",
        )

        df["site_id"] = df["profile_id"].astype(str)
        df["profile_id"] = df["profile_id"].astype(str)
        return df
