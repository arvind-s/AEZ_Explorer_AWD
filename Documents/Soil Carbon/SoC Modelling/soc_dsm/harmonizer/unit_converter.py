"""
Unit converter — normalises all SOC measurements to g kg⁻¹ fine earth.

Conversion table
----------------
Source unit          → g kg⁻¹                 Notes
--------------------   --------------------   ------------------------------------------
%  (g/100g)          × 10                     Most common non-SI unit
mg/kg                ÷ 1000                   Rare, but appears in some spectral datasets
SOM % (loss-on-ignition) × 10 ÷ 1.724        Van Bemmelen factor (Walkley-Black correction)
SOC stock (t/ha/cm)  context-dependent        Requires BD — not handled here (see pipeline)

The converter also fixes pH units (CaCl₂ → H₂O) and coarse-fragment
expressions (mass% → volume% using BD when available).
"""

import numpy as np
import pandas as pd


# Plausible SOC range for fine earth (g kg⁻¹)
SOC_MIN =   0.0
SOC_MAX = 800.0    # Histosols can exceed 500; cap at 800

# Conversion factors
VAN_BEMMELEN = 1.724    # SOM → SOC


class UnitConverter:
    """Stateless collection of unit-conversion helpers."""

    # ------------------------------------------------------------------
    # SOC
    # ------------------------------------------------------------------

    @staticmethod
    def soc_pct_to_g_per_kg(series: pd.Series) -> pd.Series:
        """Convert SOC expressed as % (g/100g) → g kg⁻¹."""
        return pd.to_numeric(series, errors="coerce") * 10.0

    @staticmethod
    def som_pct_to_soc_g_per_kg(series: pd.Series) -> pd.Series:
        """Convert SOM% (loss-on-ignition) → SOC g kg⁻¹ via Van Bemmelen."""
        return pd.to_numeric(series, errors="coerce") * 10.0 / VAN_BEMMELEN

    @staticmethod
    def mg_per_kg_to_g_per_kg(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce") / 1000.0

    # ------------------------------------------------------------------
    # Bulk density
    # ------------------------------------------------------------------

    @staticmethod
    def bd_range_check(series: pd.Series) -> pd.Series:
        """Mask BD values outside physically plausible range (0.05 – 2.0 g/cm³)."""
        s = pd.to_numeric(series, errors="coerce")
        return s.where((s >= 0.05) & (s <= 2.0))

    # ------------------------------------------------------------------
    # pH
    # ------------------------------------------------------------------

    @staticmethod
    def ph_cacl2_to_h2o(series: pd.Series) -> pd.Series:
        """Approximate CaCl₂ pH → H₂O pH (+0.6 offset, Ruehlmann 2006)."""
        return pd.to_numeric(series, errors="coerce") + 0.6

    # ------------------------------------------------------------------
    # Coarse fragments
    # ------------------------------------------------------------------

    @staticmethod
    def cf_mass_to_volume(cf_mass_pct: pd.Series, bd: pd.Series,
                          particle_density: float = 2.65) -> pd.Series:
        """Convert coarse-fragment mass% → volume%."""
        cf = pd.to_numeric(cf_mass_pct, errors="coerce") / 100.0
        bd = pd.to_numeric(bd, errors="coerce")
        vol_pct = cf * bd / particle_density * 100.0
        return vol_pct

    # ------------------------------------------------------------------
    # SOC stock calculation
    # ------------------------------------------------------------------

    @staticmethod
    def soc_stock_t_ha(soc_g_per_kg: pd.Series,
                       bulk_density: pd.Series,
                       thickness_cm: pd.Series,
                       coarse_frag_pct: pd.Series | None = None) -> pd.Series:
        """
        Calculate SOC stock (t ha⁻¹) for a single horizon.

        Formula:
          stock = SOC (g/kg) × BD (g/cm³) × thickness (cm) × (1 - CF/100) × 0.1
          (the 0.1 converts from g/m² to t/ha, i.e. × 10 000 m²/ha ÷ 1 000 000 g/t)
        """
        soc = pd.to_numeric(soc_g_per_kg,   errors="coerce")
        bd  = pd.to_numeric(bulk_density,    errors="coerce")
        h   = pd.to_numeric(thickness_cm,    errors="coerce")
        if coarse_frag_pct is not None:
            cf = pd.to_numeric(coarse_frag_pct, errors="coerce").fillna(0).clip(0, 99)
        else:
            cf = pd.Series(0.0, index=soc.index)
        return soc * bd * h * (1 - cf / 100.0) * 0.1

    # ------------------------------------------------------------------
    # Batch application
    # ------------------------------------------------------------------

    @classmethod
    def apply(cls, df: pd.DataFrame,
              soc_unit: str = "g_per_kg",
              ph_unit: str = "h2o") -> pd.DataFrame:
        """
        Apply unit conversions in-place and return the modified DataFrame.

        Parameters
        ----------
        soc_unit : str
            One of 'g_per_kg', 'pct', 'som_pct', 'mg_per_kg'.
        ph_unit : str
            One of 'h2o', 'cacl2'.
        """
        df = df.copy()

        # --- SOC ---
        if "soc_g_per_kg" in df.columns:
            if soc_unit == "pct":
                df["soc_g_per_kg"] = cls.soc_pct_to_g_per_kg(df["soc_g_per_kg"])
            elif soc_unit == "som_pct":
                df["soc_g_per_kg"] = cls.som_pct_to_soc_g_per_kg(df["soc_g_per_kg"])
            elif soc_unit == "mg_per_kg":
                df["soc_g_per_kg"] = cls.mg_per_kg_to_g_per_kg(df["soc_g_per_kg"])
            # ensure numeric
            df["soc_g_per_kg"] = pd.to_numeric(df["soc_g_per_kg"], errors="coerce")

        # --- BD ---
        if "bulk_density_g_cm3" in df.columns:
            df["bulk_density_g_cm3"] = cls.bd_range_check(df["bulk_density_g_cm3"])

        # --- pH ---
        if "ph_h2o" in df.columns and ph_unit == "cacl2":
            df["ph_h2o"] = cls.ph_cacl2_to_h2o(df["ph_h2o"])

        # --- SOC stock ---
        if {"soc_g_per_kg", "bulk_density_g_cm3", "upper_depth", "lower_depth"}.issubset(df.columns):
            thickness = (pd.to_numeric(df["lower_depth"], errors="coerce")
                         - pd.to_numeric(df["upper_depth"], errors="coerce"))
            df["soc_stock_t_ha"] = cls.soc_stock_t_ha(
                df["soc_g_per_kg"],
                df["bulk_density_g_cm3"],
                thickness,
                df.get("coarse_fragments_pct"),
            )

        return df
