"""
NCSCD (Northern Circumpolar Soil Carbon Database) collector.

Source  : https://bolin.su.se/data/ncscd
Dataset : NCSCD v2 — ~3 600 profiles across Arctic/boreal zone
Bulk DL : https://zenodo.org/records/1250661
License : CC BY 4.0

Critical for high-latitude permafrost SOC which is massively
underrepresented in temperate databases.
Depths extend to 300 cm where permafrost is present.
"""

import logging
import zipfile
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

_NCSCD_URL = "https://zenodo.org/records/1250661/files/NCSCD_Siberia_2015.zip?download=1"
# v2 combines North America + Eurasia as separate files; include both
_URLS = {
    "NA":  "https://zenodo.org/records/1250661/files/NCSCD_NorthAmerica_2015.zip?download=1",
    "EU":  "https://zenodo.org/records/1250661/files/NCSCD_Eurasia_2015.zip?download=1",
}


class NCSCDCollector(BaseCollector):

    SOURCE_NAME = "NCSCD"

    def __init__(self, raw_dir: str | Path = "data/raw/ncscd"):
        super().__init__(raw_dir)

    # ------------------------------------------------------------------

    def download(self) -> None:
        for region, url in _URLS.items():
            dest = self.raw_dir / f"ncscd_{region}.zip"
            self._download_file(url, dest)
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(self.raw_dir / region)

    def _is_downloaded(self) -> bool:
        return any(self.raw_dir.rglob("*.csv")) or any(self.raw_dir.rglob("*.xlsx"))

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        frames = []
        for path in sorted(self.raw_dir.rglob("*.csv")):
            df = pd.read_csv(path, low_memory=False)
            frames.append(self._normalise(df))
        for path in sorted(self.raw_dir.rglob("*.xlsx")):
            df = pd.read_excel(path)
            frames.append(self._normalise(df))

        if not frames:
            raise FileNotFoundError(f"No NCSCD files found in {self.raw_dir}")
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        rename = {
            "PROFILE_ID": "profile_id",
            "LONG":       "longitude",
            "LAT":        "latitude",
            "COUNTRY":    "country_code",
            "UPPER":      "upper_depth",
            "LOWER":      "lower_depth",
            "SOC":        "soc_g_per_kg",   # stored as g/kg in NCSCD
            "BD":         "bulk_density_g_cm3",
            "CF":         "coarse_fragments_pct",
            "PH":         "ph_h2o",
            "CLAY":       "clay_pct",
            "SILT":       "silt_pct",
            "SAND":       "sand_pct",
            "YEAR":       "observation_date",
            "LANDUSE":    "land_use",
            "HORIZON":    "horizon_designation",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["profile_id"] = df.get("profile_id", df.index).astype(str)
        df["site_id"]    = df["profile_id"]
        return df
