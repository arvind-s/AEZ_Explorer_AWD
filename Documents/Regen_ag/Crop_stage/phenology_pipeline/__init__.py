"""Paddy crop phenology assessment — STAC S1/S2 + ERA5-Land GDD."""

from .config import DEFAULT_GDD_STAGES, PhenologyConfig

__all__ = ["PhenologyConfig", "DEFAULT_GDD_STAGES", "run_phenology", "run_phenology_from_curves"]


def run_phenology(*args, **kwargs):
    from .phenology_engine import run_phenology as _run

    return _run(*args, **kwargs)


def run_phenology_from_curves(*args, **kwargs):
    from .phenology_engine import run_phenology_from_curves as _run

    return _run(*args, **kwargs)
