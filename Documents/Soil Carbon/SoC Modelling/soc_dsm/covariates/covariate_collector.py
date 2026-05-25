"""
Covariate collector — attaches environmental predictors to each soil profile.

All covariates are open-source and can be fetched without authentication
(except Google Earth Engine which requires a free sign-up).

Covariate groups
----------------
A. Topography (SRTM 90m via OpenTopography REST API)
     elevation, slope, aspect, TWI, TPI, curvature, roughness

B. Climate (CHELSA v2.1 — 30-year normals 1981-2010, ~1 km)
     mean annual temperature (MAT), mean annual precipitation (MAP),
     temperature seasonality, precipitation seasonality,
     bio01…bio19 (WorldClim-style bioclimatic variables)

C. Vegetation (MODIS NDVI/EVI composites via NASA CMR / Earthdata)
     annual mean NDVI (2000–2020), EVI, LSWI

D. Parent material / Geology
     SoilGrids 250m: parent material probability classes (API)
     GLiM: Global Lithological Map (static raster — fetch once)

E. Soil legacy (SoilGrids v2 REST API)
     existing SOC prediction, clay, BD, pH at same depth slice
     (used as additional covariates, NOT as labels)

F. Land use / Land cover
     ESA CCI Land Cover 300m (annual, 1992–2020) via static GeoTIFF

Usage
-----
collector = CovariateCollector(cache_dir="data/covariates")
df_with_covs = collector.attach(df_harmonised)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _batch(iterable, size: int):
    """Yield successive chunks of *size* from *iterable*."""
    lst = list(iterable)
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


class CovariateCollector:

    def __init__(self, cache_dir: str | Path = "data/covariates",
                 batch_size: int = 100,
                 request_delay: float = 0.3):
        self.cache_dir     = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size    = batch_size
        self.request_delay = request_delay   # seconds between API calls

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def attach(self, df: pd.DataFrame,
               groups: list[str] | None = None) -> pd.DataFrame:
        """
        Attach covariates to *df* (must have longitude, latitude columns).

        Parameters
        ----------
        groups : list of group names to fetch, e.g. ['topography', 'climate'].
                 None → fetch all groups.
        """
        active = set(groups) if groups else {
            "topography", "climate", "soilgrids", "landcover"
        }
        df = df.copy()

        if "topography" in active:
            df = self._attach_topography(df)
        if "climate" in active:
            df = self._attach_chelsa(df)
        if "soilgrids" in active:
            df = self._attach_soilgrids(df)
        if "landcover" in active:
            df = self._attach_esa_lc(df)

        return df

    # ------------------------------------------------------------------
    # A. Topography — OpenTopography SRTM 90m REST API
    # ------------------------------------------------------------------

    def _attach_topography(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch SRTM elevation for each point and derive terrain indices."""
        logger.info("[Covariates] Fetching SRTM elevation …")
        cache_file = self.cache_dir / "topography.parquet"
        if cache_file.exists():
            topo = pd.read_parquet(cache_file)
            return df.merge(topo, on=["longitude", "latitude"], how="left")

        points = df[["longitude", "latitude"]].drop_duplicates().dropna()
        elevations = []
        for chunk in _batch(points.itertuples(index=False), self.batch_size):
            for pt in chunk:
                elev = self._opentopo_elevation(pt.latitude, pt.longitude)
                elevations.append({
                    "longitude": pt.longitude,
                    "latitude":  pt.latitude,
                    "elevation_m": elev,
                })
            time.sleep(self.request_delay)

        topo = pd.DataFrame(elevations)
        # Terrain indices require gridded data; here we store only elevation.
        # For full terrain analysis (TWI, slope, curvature), use a raster
        # pipeline (e.g. GDAL / richdem) on the SRTM tile grid.
        topo.to_parquet(cache_file, index=False)
        return df.merge(topo, on=["longitude", "latitude"], how="left")

    @staticmethod
    def _opentopo_elevation(lat: float, lon: float) -> float:
        """Query OpenTopography SRTM GL3 (90m) for a single point."""
        url = (
            "https://api.opentopodata.org/v1/srtm90m"
            f"?locations={lat:.6f},{lon:.6f}"
        )
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()["results"][0]["elevation"]
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    # B. Climate — CHELSA v2.1 bioclimatic variables (REST-queryable tiles)
    # ------------------------------------------------------------------

    def _attach_chelsa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach CHELSA v2.1 bioclimatic variables via the CHELSA REST endpoint.

        For bulk use, download the full GeoTIFFs once and sample locally
        with rasterio (much faster). The REST endpoint is used here for
        point-level prototyping.
        """
        logger.info("[Covariates] Fetching CHELSA climate variables …")
        cache_file = self.cache_dir / "chelsa.parquet"
        if cache_file.exists():
            chelsa = pd.read_parquet(cache_file)
            return df.merge(chelsa, on=["longitude", "latitude"], how="left")

        # CHELSA REST API (ClimateDataStore-like endpoint)
        base = "https://chelsa-climate.org/api/v2/timeseries"
        points = df[["longitude", "latitude"]].drop_duplicates().dropna()
        records = []
        for pt in points.itertuples(index=False):
            row = {"longitude": pt.longitude, "latitude": pt.latitude}
            for var in ["bio01", "bio12", "bio04", "bio15"]:
                # bio01=MAT(×0.1°C), bio12=MAP(mm), bio04=T-seasonality, bio15=P-seasonality
                try:
                    r = requests.get(
                        f"{base}/{var}",
                        params={"lon": pt.longitude, "lat": pt.latitude,
                                "start": "1981-01", "end": "2010-12"},
                        timeout=15,
                    )
                    r.raise_for_status()
                    data = r.json()
                    row[f"chelsa_{var}"] = np.mean(data.get("data", [float("nan")]))
                except Exception:
                    row[f"chelsa_{var}"] = float("nan")
                time.sleep(self.request_delay)
            records.append(row)

        chelsa = pd.DataFrame(records)
        chelsa.to_parquet(cache_file, index=False)
        return df.merge(chelsa, on=["longitude", "latitude"], how="left")

    # ------------------------------------------------------------------
    # C. SoilGrids v2 — REST API (iSDA / ISRIC)
    # ------------------------------------------------------------------

    def _attach_soilgrids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach SoilGrids v2 predictions at the matching depth slice.

        Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query
        """
        logger.info("[Covariates] Fetching SoilGrids v2 …")
        cache_file = self.cache_dir / "soilgrids.parquet"
        if cache_file.exists():
            sg = pd.read_parquet(cache_file)
            return df.merge(sg, on=["longitude", "latitude", "upper_depth", "lower_depth"],
                            how="left")

        # SoilGrids depth codes
        depth_map = {
            (0, 5):    "0-5cm",
            (5, 15):   "5-15cm",
            (15, 30):  "15-30cm",
            (30, 60):  "30-60cm",
            (60, 100): "60-100cm",
            (100, 200):"100-200cm",
        }
        props = ["oc", "clay", "bdod", "phh2o", "soc"]

        records = []
        grouped = df[["longitude", "latitude", "upper_depth", "lower_depth"]]\
                    .drop_duplicates().dropna()

        for row in grouped.itertuples(index=False):
            depth_key = (int(row.upper_depth), int(row.lower_depth))
            depth_str = depth_map.get(depth_key, "0-5cm")
            rec = {
                "longitude":   row.longitude,
                "latitude":    row.latitude,
                "upper_depth": row.upper_depth,
                "lower_depth": row.lower_depth,
            }
            try:
                r = requests.get(
                    "https://rest.isric.org/soilgrids/v2.0/properties/query",
                    params={
                        "lon":    row.longitude,
                        "lat":    row.latitude,
                        "property": props,
                        "depth":  depth_str,
                        "value":  "mean",
                    },
                    timeout=20,
                )
                r.raise_for_status()
                jdata = r.json()
                for prop_info in jdata.get("properties", {}).get("layers", []):
                    pname = prop_info["name"]
                    val   = prop_info["depths"][0]["values"].get("mean", None)
                    if val is not None:
                        # SoilGrids returns values ×10 for most properties
                        rec[f"sg_{pname}"] = val / 10.0
            except Exception as exc:
                logger.debug("SoilGrids query failed for (%.4f, %.4f): %s",
                             row.longitude, row.latitude, exc)
            records.append(rec)
            time.sleep(self.request_delay)

        sg = pd.DataFrame(records)
        sg.to_parquet(cache_file, index=False)
        return df.merge(sg, on=["longitude", "latitude", "upper_depth", "lower_depth"],
                        how="left")

    # ------------------------------------------------------------------
    # D. ESA CCI Land Cover (static GeoTIFF)
    # ------------------------------------------------------------------

    def _attach_esa_lc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample ESA CCI 300m annual land cover for the nearest year to the
        observation date.

        Requires the user to download the annual GeoTIFFs from:
        https://www.esa-landcover-cci.org/?q=node/164
        and place them in self.cache_dir/esa_lc/YYYY.tif

        If no files are found this step is silently skipped.
        """
        lc_dir = self.cache_dir / "esa_lc"
        tifs = sorted(lc_dir.glob("*.tif")) if lc_dir.exists() else []
        if not tifs:
            logger.warning("[Covariates] ESA CCI land cover TIFFs not found in %s — "
                           "skipping. Download from https://www.esa-landcover-cci.org",
                           lc_dir)
            return df

        try:
            import rasterio
            from rasterio.sample import sample_gen
        except ImportError:
            logger.warning("[Covariates] rasterio not installed — skipping ESA LC.")
            return df

        # Use the most recent available file
        tif = tifs[-1]
        year = tif.stem  # filename is e.g. "2020"

        points = df[["longitude", "latitude"]].drop_duplicates().dropna()
        coords = list(zip(points["longitude"], points["latitude"]))

        with rasterio.open(tif) as src:
            lc_values = [x[0] for x in sample_gen(src, coords)]

        points = points.copy()
        points[f"esa_lc_{year}"] = lc_values
        return df.merge(points, on=["longitude", "latitude"], how="left")
