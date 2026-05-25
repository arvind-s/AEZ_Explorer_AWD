"""
Depth harmonizer — converts raw soil horizons to GlobalSoilMap standard depths.

Method: Equal-Area Quadratic Spline (Bishop et al. 1999, EJSS 50:3-22).
  - Mass-preserving: integrating the spline over any interval equals the
    weighted mean of the original data within that interval.
  - Handles irregular horizon boundaries, gaps, and overlaps.

Standard target depths (cm): 0-5, 5-15, 15-30, 30-60, 60-100, 100-200

Fallback: if a profile has only 1 horizon or the spline fails to converge,
we use piecewise-constant interpolation (depth-weighted average).

Reference:
  Bishop, T.F.A., McBratney, A.B., Laslett, G.M. (1999).
  Modelling soil attribute depth functions with equal-area quadratic smoothing
  splines. Geoderma, 91(1-2), 27-45.
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .schema import STANDARD_DEPTHS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equal-Area Quadratic Spline
# ---------------------------------------------------------------------------

class _EAQSpline:
    """
    Fits an equal-area quadratic spline to a single soil profile.

    Parameters
    ----------
    depths_upper : array-like, shape (n,)  — upper horizon boundaries (cm)
    depths_lower : array-like, shape (n,)  — lower horizon boundaries (cm)
    values       : array-like, shape (n,)  — measured property values

    After calling fit(), use predict(upper, lower) to obtain the mean value
    over any depth interval [upper, lower].
    """

    def __init__(self, depths_upper, depths_lower, values, lam: float = 0.1):
        self.u   = np.asarray(depths_upper, dtype=float)
        self.l   = np.asarray(depths_lower, dtype=float)
        self.v   = np.asarray(values,       dtype=float)
        self.lam = lam
        self._coeffs = None

    # ------------------------------------------------------------------

    def fit(self) -> bool:
        """Fit the spline. Returns True on success, False on failure."""
        n = len(self.v)
        if n < 2:
            return False
        knots = np.unique(np.concatenate([self.u, self.l]))
        m = len(knots) - 1           # number of intervals

        # Build design matrix A (n × m+1): each row = integral of basis over horizon
        A = np.zeros((n, m + 1))
        for i in range(n):
            for j in range(m):
                lo, hi = knots[j], knots[j + 1]
                overlap = max(0.0, min(hi, self.l[i]) - max(lo, self.u[i]))
                if self.l[i] > self.u[i]:
                    A[i, j] = overlap / (self.l[i] - self.u[i])

        # Regularisation: second-difference matrix
        D = np.diff(np.eye(m + 1), n=2, axis=0)
        H = A.T @ A + self.lam * D.T @ D

        try:
            self._coeffs = np.linalg.solve(H, A.T @ self.v)
            self._knots  = knots
            return True
        except np.linalg.LinAlgError:
            return False

    def predict(self, upper: float, lower: float) -> float:
        """Return the spline-predicted mean value over [upper, lower] cm."""
        if self._coeffs is None:
            return float("nan")
        knots = self._knots
        total_weight = 0.0
        total_value  = 0.0
        for j in range(len(knots) - 1):
            lo, hi = knots[j], knots[j + 1]
            overlap = max(0.0, min(hi, lower) - max(lo, upper))
            if overlap > 0:
                total_weight += overlap
                total_value  += overlap * self._coeffs[j]
        if total_weight == 0:
            return float("nan")
        return total_value / total_weight


# ---------------------------------------------------------------------------
# Piecewise-constant fallback
# ---------------------------------------------------------------------------

def _piecewise_constant(u_arr, l_arr, v_arr, upper, lower):
    """Depth-weighted mean of raw horizons overlapping [upper, lower]."""
    weight_sum = 0.0
    value_sum  = 0.0
    for u, l, v in zip(u_arr, l_arr, v_arr):
        if np.isnan(v):
            continue
        overlap = max(0.0, min(l, lower) - max(u, upper))
        if overlap > 0:
            weight_sum += overlap
            value_sum  += overlap * v
    if weight_sum == 0:
        return float("nan")
    return value_sum / weight_sum


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

_NUMERIC_PROPS = [
    "soc_g_per_kg",
    "bulk_density_g_cm3",
    "coarse_fragments_pct",
    "ph_h2o",
    "clay_pct",
    "silt_pct",
    "sand_pct",
    "cec_cmol_kg",
    "total_n_g_per_kg",
]


class DepthHarmonizer:
    """
    Harmonises a DataFrame of soil horizons to GlobalSoilMap standard depths.

    Usage
    -----
    harmonizer = DepthHarmonizer()
    harmonised = harmonizer.transform(df_horizons)
    """

    def __init__(self, target_depths: list[Tuple[int, int]] = STANDARD_DEPTHS,
                 lam: float = 0.1, min_data_depth_cm: float = 5.0):
        self.target_depths      = target_depths
        self.lam                = lam
        self.min_data_depth_cm  = min_data_depth_cm

    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with columns:
             profile_id, upper_depth, lower_depth, soc_g_per_kg (+ optional props)

        Returns
        -------
        DataFrame with one row per (profile_id × target_depth_interval),
        containing harmonised property values.
        """
        df = df.copy()
        df["upper_depth"] = pd.to_numeric(df["upper_depth"], errors="coerce")
        df["lower_depth"] = pd.to_numeric(df["lower_depth"], errors="coerce")

        # Drop horizons with invalid geometry
        df = df.dropna(subset=["upper_depth", "lower_depth"])
        df = df[df["lower_depth"] > df["upper_depth"]]

        records = []
        for pid, group in df.groupby("profile_id"):
            group = group.sort_values("upper_depth")
            meta  = group.iloc[0][[
                c for c in df.columns
                if c not in _NUMERIC_PROPS + ["upper_depth", "lower_depth"]
            ]].to_dict()

            # Fit splines for each numeric property
            splines = {}
            for prop in _NUMERIC_PROPS:
                if prop not in group.columns:
                    continue
                valid = group.dropna(subset=[prop])
                if len(valid) < 2:
                    splines[prop] = None   # fallback
                    continue
                sp = _EAQSpline(valid["upper_depth"], valid["lower_depth"],
                                 valid[prop], lam=self.lam)
                splines[prop] = sp if sp.fit() else None

            for (utop, ubot) in self.target_depths:
                # Check data coverage for this interval
                covered = _piecewise_constant(
                    group["upper_depth"].values,
                    group["lower_depth"].values,
                    np.ones(len(group)),
                    utop, ubot,
                )
                if np.isnan(covered) or covered < 0.1:
                    continue    # no data for this depth slice

                row = {**meta, "upper_depth": utop, "lower_depth": ubot}
                for prop in _NUMERIC_PROPS:
                    if prop not in group.columns:
                        row[prop] = float("nan")
                        continue
                    sp = splines.get(prop)
                    if sp is not None:
                        row[prop] = sp.predict(utop, ubot)
                    else:
                        valid = group.dropna(subset=[prop])
                        row[prop] = _piecewise_constant(
                            valid["upper_depth"].values,
                            valid["lower_depth"].values,
                            valid[prop].values,
                            utop, ubot,
                        )
                records.append(row)

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)
