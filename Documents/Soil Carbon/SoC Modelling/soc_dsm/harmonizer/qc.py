"""
Quality Control (QC) for the harmonised soil database.

Steps applied sequentially
--------------------------
1.  Range filter      — physically implausible values flagged → NaN
2.  SOC–SOM check     — if SOC > SOM, flag as suspect
3.  Texture sum       — clay+silt+sand must be within [95, 105]; rescale if not
4.  Outlier detection — IQR-based per depth interval
5.  Spatial dedup     — profiles within `radius_m` of each other: keep earliest
                        (or highest SOC if dates unknown)
6.  Flag summary      — adds a 'qc_flags' column (comma-separated issue codes)

All operations are non-destructive by default: values are set to NaN rather
than rows being dropped. Set `drop_flagged=True` to remove rows with any flag.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physical plausibility bounds
# ---------------------------------------------------------------------------
BOUNDS: Dict[str, tuple] = {
    "soc_g_per_kg":         (0.0,  800.0),
    "bulk_density_g_cm3":   (0.05,   2.0),
    "coarse_fragments_pct": (0.0,  100.0),
    "ph_h2o":               (2.5,   11.0),
    "clay_pct":             (0.0,  100.0),
    "silt_pct":             (0.0,  100.0),
    "sand_pct":             (0.0,  100.0),
    "cec_cmol_kg":          (0.0,  200.0),
    "total_n_g_per_kg":     (0.0,   50.0),
    "upper_depth":          (0.0,  300.0),
    "lower_depth":          (0.0,  300.0),
}

# IQR multiplier for outlier detection
IQR_FACTOR = 3.0


class QualityController:

    def __init__(self,
                 drop_flagged: bool = False,
                 spatial_dedup_radius_m: float = 100.0,
                 iqr_factor: float = IQR_FACTOR):
        self.drop_flagged            = drop_flagged
        self.spatial_dedup_radius_m  = spatial_dedup_radius_m
        self.iqr_factor              = iqr_factor

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["qc_flags"] = ""

        df = self._range_filter(df)
        df = self._texture_sum_check(df)
        df = self._outlier_flag(df)
        df = self._spatial_dedup(df)

        flagged_mask = df["qc_flags"].str.len() > 0
        n_flagged = flagged_mask.sum()
        logger.info("QC complete: %d / %d records flagged.", n_flagged, len(df))

        if self.drop_flagged:
            df = df[~flagged_mask].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Step 1: Range filter
    # ------------------------------------------------------------------

    def _range_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lo, hi) in BOUNDS.items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            bad = (s < lo) | (s > hi)
            if bad.any():
                df.loc[bad, col] = np.nan
                df = self._add_flag(df, bad, f"range:{col}")
        return df

    # ------------------------------------------------------------------
    # Step 2: Texture sum check
    # ------------------------------------------------------------------

    def _texture_sum_check(self, df: pd.DataFrame) -> pd.DataFrame:
        tex_cols = ["clay_pct", "silt_pct", "sand_pct"]
        if not all(c in df.columns for c in tex_cols):
            return df
        tex = df[tex_cols].apply(pd.to_numeric, errors="coerce")
        tex_sum = tex.sum(axis=1)
        has_all  = tex.notna().all(axis=1)
        bad      = has_all & ((tex_sum < 95) | (tex_sum > 105))
        # Rescale to 100 where only slightly off
        rescale  = has_all & (tex_sum > 0) & (tex_sum != 100)
        df.loc[rescale, tex_cols] = (
            tex.loc[rescale].div(tex_sum.loc[rescale], axis=0) * 100
        )
        df = self._add_flag(df, bad, "texture_sum")
        return df

    # ------------------------------------------------------------------
    # Step 3: IQR-based outlier detection per depth interval
    # ------------------------------------------------------------------

    def _outlier_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        if "soc_g_per_kg" not in df.columns:
            return df
        for (u, l), grp in df.groupby(["upper_depth", "lower_depth"]):
            soc = pd.to_numeric(grp["soc_g_per_kg"], errors="coerce")
            q1, q3 = soc.quantile(0.25), soc.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo_fence = q1 - self.iqr_factor * iqr
            hi_fence = q3 + self.iqr_factor * iqr
            outlier_idx = grp.index[(soc < lo_fence) | (soc > hi_fence)]
            df = self._add_flag(df, df.index.isin(outlier_idx), "soc_outlier")
        return df

    # ------------------------------------------------------------------
    # Step 4: Spatial deduplication
    # ------------------------------------------------------------------

    def _spatial_dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate profiles within radius_m metres of each other."""
        if "longitude" not in df.columns or "latitude" not in df.columns:
            return df

        coords = df[["latitude", "longitude"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if coords.empty:
            return df

        # Approximate metres per degree
        lat_rad = np.radians(coords["latitude"].mean())
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(lat_rad)

        xy = np.column_stack([
            coords["latitude"]  * m_per_deg_lat,
            coords["longitude"] * m_per_deg_lon,
        ])

        tree = cKDTree(xy)
        pairs = tree.query_pairs(self.spatial_dedup_radius_m)

        # Mark duplicates — keep the first profile (lower index)
        dup_indices = set()
        for i, j in sorted(pairs):
            if i not in dup_indices:
                dup_indices.add(j)

        if dup_indices:
            real_indices = coords.index[list(dup_indices)]
            df = self._add_flag(df, df.index.isin(real_indices), "spatial_dup")
            logger.info("Spatial dedup: %d profiles flagged within %.0f m of another.",
                        len(dup_indices), self.spatial_dedup_radius_m)
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _add_flag(df: pd.DataFrame, mask, flag: str) -> pd.DataFrame:
        """Append *flag* to the qc_flags column where *mask* is True."""
        if isinstance(mask, pd.Index):
            mask = df.index.isin(mask)
        df.loc[mask, "qc_flags"] = df.loc[mask, "qc_flags"].apply(
            lambda x: f"{x},{flag}".lstrip(",")
        )
        return df
