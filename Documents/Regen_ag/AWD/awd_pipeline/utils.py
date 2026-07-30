"""
Shared utilities for the AWD Sentinel-1 pipeline.

- AOI loading / reprojection
- Refined Lee speckle filter
- Otsu dynamic thresholding
- dB <-> linear conversions
- Lovell-style (2019) per-series wetness index
- Drying-cycle detection -> AWD-likelihood score
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from skimage.filters import threshold_otsu
except ImportError:  # pragma: no cover
    threshold_otsu = None


# --------------------------------------------------------------------------
# dB / linear conversions
# --------------------------------------------------------------------------

def to_db(linear: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert linear power/intensity values to decibels."""
    linear = np.asarray(linear, dtype=np.float64)
    return 10.0 * np.log10(np.clip(linear, eps, None))


def from_db(db: np.ndarray) -> np.ndarray:
    """Convert decibel values back to linear power/intensity."""
    db = np.asarray(db, dtype=np.float64)
    return 10.0 ** (db / 10.0)


def amplitude_dn_to_beta_naught_db(dn: np.ndarray, beta_naught_lut_value: float,
                                    eps: float = 1e-6) -> np.ndarray:
    """
    Calibrate raw Sentinel-1 GRD amplitude digital numbers (DN) to beta-nought
    backscatter in dB, following the approach used in Lovell (2019) -- beta
    nought is chosen over sigma/gamma because, for a single GRD product, the
    beta-nought calibration LUT is effectively a single constant across the
    swath (unlike sigma/gamma which vary with incidence angle across range),
    which is what makes it tractable to calibrate without a full per-pixel
    LUT interpolation.

    dn: raw amplitude array read from the GRD 'vv'/'vh' asset.
    beta_naught_lut_value: the betaNought calibration constant extracted from
        the scene's `schema-calibration-<pol>` XML sidecar
        (<calibrationVector><betaNought>...).

    NOTE: this assumes the simplified case where the betaNought LUT is flat
    across the product (true for most IW GRD products). If you see banding
    artifacts in range, you likely have a product where the LUT does vary and
    you should interpolate the full calibrationVectorList instead of using a
    single constant -- see `parse_calibration_lut` below for the full vector.
    Verify against a real downloaded product before relying on this in
    production; it has not been tested against a live GRD calibration file in
    this sandbox because outbound access to Sentinel data hosts was blocked
    here (see README).
    """
    dn = np.asarray(dn, dtype=np.float64)
    beta0_linear = (dn ** 2) / (beta_naught_lut_value ** 2)
    return to_db(beta0_linear, eps=eps)


def parse_calibration_lut(calibration_xml_path: str) -> dict:
    """
    Parse a Sentinel-1 GRD `calibration-<pol>.xml` annotation file into
    per-line/per-pixel LUTs for sigmaNought, betaNought, gamma and dn.

    Returns a dict with numpy arrays: 'line', 'pixel' (2D per line), and one
    2D array per calibration type. Use this instead of
    `amplitude_dn_to_beta_naught_db`'s single-constant shortcut if you need
    full radiometric accuracy (recommended before this goes into anything
    used for MRV/carbon accounting, since that has real financial stakes).

    Not exercised against a live file in this session -- see README for why.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(calibration_xml_path)
    root = tree.getroot()

    lines, pixels, sigma0, beta0, gamma, dn = [], [], [], [], [], []
    for vec in root.iter("calibrationVector"):
        line = int(vec.findtext("line"))
        pixel = [int(x) for x in vec.findtext("pixel").split()]
        s0 = [float(x) for x in vec.findtext("sigmaNought").split()]
        b0 = [float(x) for x in vec.findtext("betaNought").split()]
        g0 = [float(x) for x in vec.findtext("gamma").split()]
        d0 = [float(x) for x in vec.findtext("dn").split()]
        lines.append(line)
        pixels.append(pixel)
        sigma0.append(s0)
        beta0.append(b0)
        gamma.append(g0)
        dn.append(d0)

    return {
        "line": np.array(lines),
        "pixel": np.array(pixels),
        "sigmaNought": np.array(sigma0),
        "betaNought": np.array(beta0),
        "gamma": np.array(gamma),
        "dn": np.array(dn),
    }


# --------------------------------------------------------------------------
# Speckle filtering
# --------------------------------------------------------------------------

def refined_lee_filter(img: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Refined Lee speckle filter (single-image version) on a linear-power SAR
    image. Tran et al. (2022) and related literature report ~10-25 percentage
    point accuracy gains from speckle filtering before thresholding, so this
    is treated as a mandatory step, not optional, per the project roadmap.

    img: 2D array, linear power (not dB).
    window: odd window size (5 or 7 are typical choices in the literature).
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")

    img = np.asarray(img, dtype=np.float64)
    pad = window // 2
    padded = np.pad(img, pad, mode="reflect")

    out = np.empty_like(img)
    # Overall image statistics for the noise coefficient of variation.
    img_mean = np.nanmean(img)
    img_std = np.nanstd(img)
    cu = img_std / img_mean if img_mean else 0.0

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, (window, window))
    local_mean = windows.mean(axis=(-1, -2))
    local_std = windows.std(axis=(-1, -2))

    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(local_mean != 0, local_std / local_mean, 0.0)
        w = np.where(ci > cu, 1.0 - (cu ** 2) / np.maximum(ci ** 2, 1e-12), 0.0)
        w = np.clip(w, 0.0, 1.0)

    out = local_mean + w * (img - local_mean)
    return out


# --------------------------------------------------------------------------
# Otsu dynamic threshold (per Tran, Menenti & Jia, 2022)
# --------------------------------------------------------------------------

def otsu_water_threshold(vv_db: np.ndarray) -> float:
    """
    Data-driven (per-scene) threshold on VV backscatter (dB) separating
    flooded/water pixels (low VV) from non-water pixels (high VV), following
    the Otsu-thresholding approach in Tran, Menenti & Jia (2022). This is
    used instead of a fixed global dB cutoff, per the roadmap's "relative,
    not absolute" design principle.
    """
    if threshold_otsu is None:
        raise ImportError("scikit-image is required for otsu_water_threshold")
    valid = vv_db[np.isfinite(vv_db)]
    if valid.size == 0:
        raise ValueError("no valid pixels to threshold")
    return float(threshold_otsu(valid))


# --------------------------------------------------------------------------
# Lovell (2019) style wetness index
# --------------------------------------------------------------------------

def wetness_index(series: Sequence[float]) -> np.ndarray:
    """
    Rescale a backscatter time series (dB, ideally beta-nought VV) into a
    Lovell-style Wetness Index (WI) on a -1..+1 scale, using the *series'
    own* min/max rather than a fixed global constant -- this is what makes
    the index comparable across geographies/soil types without hardcoding a
    threshold (roadmap principle: relative, not absolute, features).

    -1 == driest observation in the series, +1 == wettest (lowest backscatter).
    """
    x = np.asarray(series, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        raise ValueError("need at least 2 valid observations to compute a WI")
    x_min, x_max = finite.min(), finite.max()
    if x_max == x_min:
        return np.zeros_like(x)
    # Lower backscatter (wetter) -> +1 ; higher backscatter (drier) -> -1
    return 1.0 - 2.0 * (x - x_min) / (x_max - x_min)


@dataclass
class DryingCycleResult:
    n_cycles: int
    cycle_dates: list
    awd_likelihood: float
    wi: np.ndarray


def detect_drying_cycles(dates: Sequence, wi: Sequence[float],
                          wet_threshold: float = 0.3,
                          dry_threshold: float = -0.1) -> DryingCycleResult:
    """
    Count wet->dry->wet (re-flood) transitions in a wetness-index time
    series -- this is the core "AWD is a pattern, not a single-date class"
    logic from the roadmap, adapted from Lovell (2019)'s change-detection
    approach. A single continuously-flooded field should show ~1 wet peak and
    no re-flood cycles after transplanting; an AWD field should show multiple
    wet/dry oscillations across the season.

    wet_threshold / dry_threshold: WI must cross above wet_threshold then
        below dry_threshold then back above wet_threshold to count as one
        full drying cycle. Defaults are a reasonable starting point; tune
        against real/open ground truth once available (see 04_validate).
    """
    wi = np.asarray(wi, dtype=np.float64)
    state = "seeking_wet"
    cycles = 0
    cycle_dates = []
    last_wet_idx = None

    for i, v in enumerate(wi):
        if not np.isfinite(v):
            continue
        if state == "seeking_wet" and v >= wet_threshold:
            state = "seeking_dry"
            last_wet_idx = i
        elif state == "seeking_dry" and v <= dry_threshold:
            state = "seeking_reflood"
        elif state == "seeking_reflood" and v >= wet_threshold:
            cycles += 1
            cycle_dates.append((dates[last_wet_idx], dates[i]))
            last_wet_idx = i
            state = "seeking_dry"

    # Simple likelihood score: more completed drying/re-flood cycles per
    # season -> higher AWD likelihood. Normalize against a nominal "AWD
    # practiced well" reference of ~4-6 cycles/season (typical for a
    # 100-120 day rice crop with a ~10-15 day AWD interval); continuously
    # flooded fields should score near 0.
    reference_cycles = 5.0
    awd_likelihood = float(np.clip(cycles / reference_cycles, 0.0, 1.0))

    return DryingCycleResult(
        n_cycles=cycles,
        cycle_dates=cycle_dates,
        awd_likelihood=awd_likelihood,
        wi=wi,
    )


# --------------------------------------------------------------------------
# AOI helpers
# --------------------------------------------------------------------------

def load_aoi_geojson(path: str):
    """Load an AOI GeoJSON and return (geometry_dict, bbox) in WGS84."""
    with open(path) as f:
        gj = json.load(f)
    feature = gj["features"][0] if gj.get("type") == "FeatureCollection" else gj
    geom = feature["geometry"]

    coords = geom["coordinates"]

    def flatten(c):
        if isinstance(c[0], (float, int)):
            yield c
        else:
            for sub in c:
                yield from flatten(sub)

    pts = list(flatten(coords))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    return geom, bbox
