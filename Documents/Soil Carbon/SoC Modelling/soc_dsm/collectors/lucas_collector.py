"""
LUCAS Soil collector — EU Joint Research Centre.

Source  : https://esdac.jrc.ec.europa.eu/projects/lucas
Dataset : LUCAS Topsoil Survey (0-20 cm) 2009, 2015, 2018
License : CC BY 4.0 (after registration)

Provides ~40 000 topsoil samples across EU27+.
Variables include OC (g/kg), pH, texture, N.

Download requires a one-time registration; the collector expects the
user to place the CSV files manually in raw_dir, OR pass pre-registered
direct download links via the constructor.
"""

import logging
from pathlib import Path

import pandas as pd

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

# Public direct-download URLs for each LUCAS wave (Zenodo mirrors)
_LUCAS_URLS = {
    2009: "https://zenodo.org/records/7472144/files/LUCAS_TOPSOIL_v1.csv?download=1",
    2015: "https://zenodo.org/records/7472144/files/LUCAS2015_topsoildata_20200323.csv?download=1",
    2018: "https://zenodo.org/records/6523185/files/LUCAS_Topsoil_2018_all_data.csv?download=1",
}

_YEAR_FILE = {
    2009: "LUCAS_TOPSOIL_v1.csv",
    2015: "LUCAS2015_topsoildata.csv",
    2018: "LUCAS_Topsoil_2018_all_data.csv",
}


class LUCASCollector(BaseCollector):

    SOURCE_NAME = "LUCAS"

    def __init__(self, raw_dir: str | Path = "data/raw/lucas",
                 waves: list[int] | None = None):
        super().__init__(raw_dir)
        self.waves = waves or [2009, 2015, 2018]

    # ------------------------------------------------------------------

    def download(self) -> None:
        for year in self.waves:
            dest = self.raw_dir / _YEAR_FILE[year]
            if not dest.exists():
                url = _LUCAS_URLS[year]
                try:
                    self._download_file(url, dest)
                except Exception as exc:
                    logger.warning("[LUCAS %d] Auto-download failed (%s). "
                                   "Place the CSV manually in %s.", year, exc, self.raw_dir)

    def _is_downloaded(self) -> bool:
        return any((self.raw_dir / _YEAR_FILE[y]).exists() for y in self.waves)

    # ------------------------------------------------------------------

    def parse(self) -> pd.DataFrame:
        frames = []
        for year in self.waves:
            csv = self.raw_dir / _YEAR_FILE[year]
            if not csv.exists():
                logger.warning("[LUCAS %d] File not found, skipping.", year)
                continue
            df = pd.read_csv(csv, low_memory=False)
            df = self._normalise_year(df, year)
            df["observation_date"] = str(year)
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                f"No LUCAS CSV files found in {self.raw_dir}. "
                "Download manually from https://esdac.jrc.ec.europa.eu/projects/lucas"
            )
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------

    def _normalise_year(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        # LUCAS topsoil is always 0–20 cm
        df["upper_depth"] = 0
        df["lower_depth"] = 20

        # Column name mapping varies slightly by year
        oc_candidates  = ["OC", "oc", "OC_g_kg", "organic_carbon"]
        lon_candidates = ["GPS_LONG", "Longitude", "longitude", "lon"]
        lat_candidates = ["GPS_LAT",  "Latitude",  "latitude",  "lat"]
        id_candidates  = ["POINT_ID", "Point_ID",  "point_id",  "ID"]
        ph_candidates  = ["pH_CaCl2", "pH_H2O", "pH"]
        clay_candidates= ["clay",  "Clay",  "clay_pct"]
        silt_candidates= ["silt",  "Silt",  "silt_pct"]
        sand_candidates= ["sand",  "Sand",  "sand_pct"]
        n_candidates   = ["N",     "TN",    "total_N"]
        cc_candidates  = ["COUNTRY_CODE", "Country", "country"]

        def first(cols):
            for c in cols:
                if c in df.columns:
                    return c
            return None

        rename = {}
        for col, cands in [
            ("soc_g_per_kg",       oc_candidates),
            ("longitude",          lon_candidates),
            ("latitude",           lat_candidates),
            ("profile_id",         id_candidates),
            ("ph_h2o",             ph_candidates),
            ("clay_pct",           clay_candidates),
            ("silt_pct",           silt_candidates),
            ("sand_pct",           sand_candidates),
            ("total_n_g_per_kg",   n_candidates),
            ("country_code",       cc_candidates),
        ]:
            src = first(cands)
            if src:
                rename[src] = col

        df = df.rename(columns=rename)
        df["site_id"] = df.get("profile_id", df.index).astype(str)
        df["profile_id"] = df["site_id"]

        # LUCAS OC is g/kg — no conversion needed
        if "soc_g_per_kg" in df.columns:
            df["soc_g_per_kg"] = pd.to_numeric(df["soc_g_per_kg"], errors="coerce")

        return df
