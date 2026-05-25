"""
ISCN (International Soil Carbon Network) collector.

Source  : https://iscn.fluxdata.org
Dataset : ISCN3 — ~71 000 soil profiles, focus on organic carbon.
Bulk DL : https://zenodo.org/records/3372812
License : CC BY 4.0

The archive is a ZIP containing several CSVs:
  ISCN_all_data.csv   — horizon-level measurements
  ISCN_metadata.csv   — dataset-level attribution

Note: ISCN3 merges many regional databases so expect duplicates
with WoSIS. The QualityController handles spatial deduplication.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

_ISCN3_URL = "https://zenodo.org/records/3372812/files/ISCN3_data.zip?download=1"


class ISCNCollector(BaseCollector):

    SOURCE_NAME = "ISCN"

    def __init__(self, raw_dir: str | Path = "data/raw/iscn"):
        super().__init__(raw_dir)
        self._zip_path = self.raw_dir / "ISCN3_data.zip"
        self._csv_path = self.raw_dir / "ISCN_all_data.csv"

    # ------------------------------------------------------------------

    def download(self) -> None:
        self._download_file(_ISCN3_URL, self._zip_path)
        with zipfile.ZipFile(self._zip_path) as zf:
            zf.extractall(self.raw_dir)

    def _is_downloaded(self) -> bool:
        return self._csv_path.exists() or any(self.raw_dir.glob("ISCN*.csv"))

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        csv_file = self._csv_path if self._csv_path.exists() else \
                   next(self.raw_dir.glob("ISCN*.csv"))

        df = pd.read_csv(csv_file, low_memory=False)

        rename = {
            "site_name":          "site_id",
            "profile_name":       "profile_id",
            "lon (dec. deg)":     "longitude",
            "lat (dec. deg)":     "latitude",
            "country":            "country_code",
            "observation_date":   "observation_date",
            "layer_top (cm)":     "upper_depth",
            "layer_bot (cm)":     "lower_depth",
            "oc (percent)":       "_soc_pct",          # convert below
            "bd_samp (g cm-3)":   "bulk_density_g_cm3",
            "coarse_frag":        "coarse_fragments_pct",
            "ph_h2o":             "ph_h2o",
            "clay_tot_psa (percent)": "clay_pct",
            "silt_tot_psa (percent)": "silt_pct",
            "sand_tot_psa (percent)": "sand_pct",
            "cec_sum (cmolc kg-1)":   "cec_cmol_kg",
            "n_tot_ncs (percent)":    "_n_pct",
            "method":             "soc_method",
            "layer_name":         "horizon_designation",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Convert SOC % → g kg⁻¹
        if "_soc_pct" in df.columns:
            df["soc_g_per_kg"] = pd.to_numeric(df["_soc_pct"], errors="coerce") * 10.0

        if "_n_pct" in df.columns:
            df["total_n_g_per_kg"] = pd.to_numeric(df["_n_pct"], errors="coerce") * 10.0

        df["profile_id"] = df.get("profile_id", df.index).astype(str)
        df["site_id"]    = df.get("site_id",    df["profile_id"]).astype(str)
        return df
