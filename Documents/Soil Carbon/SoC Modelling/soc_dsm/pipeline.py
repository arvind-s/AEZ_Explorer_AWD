"""
End-to-end SOC DSM pipeline orchestrator.

Steps
-----
1.  Collect      — run each enabled collector → raw canonical DataFrames
2.  Pool         — concatenate all sources
3.  Convert      — normalise units (UnitConverter.apply per source)
4.  Harmonise    — standardise depths (DepthHarmonizer)
5.  QC           — range checks, outliers, spatial deduplication
6.  Covariates   — attach environmental predictors
7.  Export       — save parquet + CSV for ML

Usage
-----
from soc_dsm.pipeline import DSMPipeline

pipeline = DSMPipeline(
    raw_dir        = "data/raw",
    harmonized_dir = "data/harmonized",
    ml_dir         = "data/ml_ready",
)
df_ml = pipeline.run(sources=["WoSIS", "ISCN", "LUCAS", "RaCA"])
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from soc_dsm.collectors.wosis_collector  import WoSISCollector
from soc_dsm.collectors.iscn_collector   import ISCNCollector
from soc_dsm.collectors.lucas_collector  import LUCASCollector
from soc_dsm.collectors.ncscd_collector  import NCSCDCollector
from soc_dsm.collectors.raca_collector   import RaCACollector
from soc_dsm.collectors.afsis_collector  import AfSISCollector
from soc_dsm.harmonizer.unit_converter   import UnitConverter
from soc_dsm.harmonizer.depth_harmonizer import DepthHarmonizer
from soc_dsm.harmonizer.qc               import QualityController
from soc_dsm.covariates.covariate_collector import CovariateCollector

logger = logging.getLogger(__name__)

# Source → (CollectorClass, soc_unit, ph_unit)
SOURCE_REGISTRY = {
    "WoSIS":  (WoSISCollector,  "g_per_kg", "h2o"),
    "ISCN":   (ISCNCollector,   "pct",      "h2o"),
    "LUCAS":  (LUCASCollector,  "g_per_kg", "h2o"),
    "NCSCD":  (NCSCDCollector,  "g_per_kg", "h2o"),
    "RaCA":   (RaCACollector,   "g_per_kg", "h2o"),
    "AfSIS":  (AfSISCollector,  "g_per_kg", "h2o"),
}


class DSMPipeline:

    def __init__(self,
                 raw_dir:         str | Path = "data/raw",
                 harmonized_dir:  str | Path = "data/harmonized",
                 ml_dir:          str | Path = "data/ml_ready",
                 covariate_dir:   str | Path = "data/covariates",
                 spatial_dedup_m: float = 100.0,
                 drop_flagged:    bool  = False):

        self.raw_dir        = Path(raw_dir)
        self.harmonized_dir = Path(harmonized_dir)
        self.ml_dir         = Path(ml_dir)
        for d in [self.raw_dir, self.harmonized_dir, self.ml_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.unit_converter   = UnitConverter()
        self.depth_harmonizer = DepthHarmonizer()
        self.qc               = QualityController(
            drop_flagged=drop_flagged,
            spatial_dedup_radius_m=spatial_dedup_m,
        )
        self.covariate_collector = CovariateCollector(cache_dir=covariate_dir)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self,
            sources: list[str] | None = None,
            force_download: bool = False,
            attach_covariates: bool = True,
            covariate_groups: list[str] | None = None) -> pd.DataFrame:
        """
        Execute the full pipeline.

        Parameters
        ----------
        sources            : list of source names (default: all registered sources)
        force_download     : re-download even if raw files exist
        attach_covariates  : fetch and attach environmental predictors
        covariate_groups   : subset of covariate groups (None = all)

        Returns
        -------
        pd.DataFrame — ML-ready table saved to self.ml_dir
        """
        active_sources = sources or list(SOURCE_REGISTRY.keys())
        t0 = time.time()

        # ---- Step 1: Collect ----
        raw_frames = []
        for src_name in active_sources:
            if src_name not in SOURCE_REGISTRY:
                logger.warning("Unknown source '%s' — skipping.", src_name)
                continue
            CollectorCls, soc_unit, ph_unit = SOURCE_REGISTRY[src_name]
            collector = CollectorCls(raw_dir=self.raw_dir / src_name.lower())
            try:
                df_raw = collector.collect(force_download=force_download)
                # ---- Step 2: Unit conversion (per-source) ----
                df_raw = UnitConverter.apply(df_raw, soc_unit=soc_unit, ph_unit=ph_unit)
                raw_frames.append(df_raw)
                logger.info("[Pipeline] %s: %d horizons after unit conversion.", src_name, len(df_raw))
            except Exception as exc:
                logger.error("[Pipeline] %s failed: %s", src_name, exc, exc_info=True)

        if not raw_frames:
            raise RuntimeError("No data collected — check network access and source availability.")

        # ---- Step 3: Pool ----
        df_pooled = pd.concat(raw_frames, ignore_index=True)
        logger.info("[Pipeline] Pooled: %d total horizons from %d sources.",
                    len(df_pooled), len(raw_frames))

        pooled_path = self.harmonized_dir / "pooled_raw.parquet"
        df_pooled.to_parquet(pooled_path, index=False)
        logger.info("[Pipeline] Saved pooled raw → %s", pooled_path)

        # ---- Step 4: Depth harmonisation ----
        logger.info("[Pipeline] Harmonising depths …")
        df_harm = self.depth_harmonizer.transform(df_pooled)
        logger.info("[Pipeline] After depth harmonisation: %d depth-slice records.", len(df_harm))

        # ---- Step 5: QC ----
        logger.info("[Pipeline] Running QC …")
        df_qc = self.qc.run(df_harm)
        qc_path = self.harmonized_dir / "harmonized_qc.parquet"
        df_qc.to_parquet(qc_path, index=False)
        logger.info("[Pipeline] Saved QC'd harmonised data → %s", qc_path)

        # ---- Step 6: Covariates ----
        if attach_covariates:
            logger.info("[Pipeline] Attaching covariates …")
            df_ml = self.covariate_collector.attach(df_qc, groups=covariate_groups)
        else:
            df_ml = df_qc

        # ---- Step 7: Export ----
        self._export(df_ml)

        elapsed = time.time() - t0
        logger.info("[Pipeline] Complete in %.1f s. ML dataset: %d rows × %d cols.",
                    elapsed, len(df_ml), df_ml.shape[1])
        return df_ml

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export(self, df: pd.DataFrame) -> None:
        parquet_path = self.ml_dir / "soc_dsm_ml_ready.parquet"
        csv_path     = self.ml_dir / "soc_dsm_ml_ready.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info("[Pipeline] ML dataset exported → %s", parquet_path)
        logger.info("[Pipeline] ML dataset exported → %s", csv_path)
        self._print_summary(df)

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        print("\n" + "=" * 60)
        print("  SOC DSM — ML-Ready Dataset Summary")
        print("=" * 60)
        print(f"  Total records      : {len(df):,}")
        if "source_db" in df.columns:
            print(f"  Sources            : {df['source_db'].value_counts().to_dict()}")
        if "upper_depth" in df.columns and "lower_depth" in df.columns:
            depth_label = (df["upper_depth"].astype(str) + "-"
                           + df["lower_depth"].astype(str) + " cm")
            print(f"  Depth slices       :\n{depth_label.value_counts().to_string()}")
        if "soc_g_per_kg" in df.columns:
            soc = pd.to_numeric(df["soc_g_per_kg"], errors="coerce").dropna()
            print(f"  SOC (g/kg) range   : {soc.min():.1f} – {soc.max():.1f}")
            print(f"  SOC mean ± std     : {soc.mean():.1f} ± {soc.std():.1f}")
        if "country_code" in df.columns:
            print(f"  Countries          : {df['country_code'].nunique()}")
        if "qc_flags" in df.columns:
            flagged = (df["qc_flags"].str.len() > 0).sum()
            print(f"  QC-flagged records : {flagged:,}")
        print("=" * 60 + "\n")
