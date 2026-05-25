"""
AfSIS (Africa Soil Information Service) collector.

Source  : https://www.africasoils.net / https://registry.opendata.aws/afsis/
Dataset : AfSIS-I  (~1 000 sentinel sites, 60 sub-plots, 0-20 & 20-50 cm)
          AfSIS-II (~2 000 additional sentinel sites)
Bulk DL : AWS Open Data Registry  s3://afsis/  (no auth needed)
          or Zenodo: https://zenodo.org/records/1041831
License : CC BY 4.0

Contains wet-chemistry + mid-IR spectroscopy data.
This collector fetches the wet-chemistry CSV from Zenodo.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

_AFSIS_URL = "https://zenodo.org/records/1041831/files/afsis1-wetchem.zip?download=1"


class AfSISCollector(BaseCollector):

    SOURCE_NAME = "AfSIS"

    def __init__(self, raw_dir: str | Path = "data/raw/afsis"):
        super().__init__(raw_dir)
        self._zip_path = self.raw_dir / "afsis1-wetchem.zip"

    # ------------------------------------------------------------------

    def download(self) -> None:
        self._download_file(_AFSIS_URL, self._zip_path)
        with zipfile.ZipFile(self._zip_path) as zf:
            zf.extractall(self.raw_dir)

    def _is_downloaded(self) -> bool:
        return any(self.raw_dir.glob("*.csv"))

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        csv_files = sorted(self.raw_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No AfSIS CSV files in {self.raw_dir}")

        frames = []
        for f in csv_files:
            df = pd.read_csv(f, low_memory=False)
            frames.append(self._normalise(df))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "SSN":         "profile_id",
            "Longitude":   "longitude",
            "Latitude":    "latitude",
            "Country":     "country_code",
            "Depth":       "_depth_code",   # '0-20' or '20-50'
            "OC":          "soc_g_per_kg",  # g/kg
            "BD":          "bulk_density_g_cm3",
            "pH.H2O":      "ph_h2o",
            "Clay":        "clay_pct",
            "Silt":        "silt_pct",
            "Sand":        "sand_pct",
            "TN":          "total_n_g_per_kg",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Parse depth code into upper/lower
        if "_depth_code" in df.columns:
            depth_map = {"0-20": (0, 20), "20-50": (20, 50),
                         "top":  (0, 20), "sub":   (20, 50)}
            df["upper_depth"] = df["_depth_code"].map(
                lambda x: depth_map.get(str(x).strip(), (None, None))[0])
            df["lower_depth"] = df["_depth_code"].map(
                lambda x: depth_map.get(str(x).strip(), (None, None))[1])

        df["profile_id"] = df.get("profile_id", df.index).astype(str)
        df["site_id"]    = df["profile_id"]
        df["observation_date"] = "2009"   # AfSIS-I sampling year
        return df
