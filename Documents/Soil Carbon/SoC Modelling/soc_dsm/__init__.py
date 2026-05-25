"""
SOC Digital Soil Mapping (DSM) pipeline.

Workflow:
  1. collectors/   — download raw profile data from each open-source database
  2. harmonizer/   — convert units, standardise depths, apply QC
  3. covariates/   — attach environmental predictors at each site
  4. pipeline.py   — orchestrate end-to-end into an ML-ready parquet/CSV
"""

__version__ = "0.1.0"
