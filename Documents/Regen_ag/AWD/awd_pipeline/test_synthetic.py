#!/usr/bin/env python3
"""
Sanity-test the core algorithms (refined_lee_filter, wetness_index,
otsu_water_threshold, detect_drying_cycles) against SYNTHETIC backscatter
data, since the sandbox this pipeline was built in could not reach any real
satellite data host (see README.md).

Synthesizes two 120-day seasons of VV backscatter, sampled every 6 days
(Sentinel-1-like revisit):
  - "continuous flood": one deep dip at transplanting, then a monotonic rise
    as the canopy closes, then a plateau -- classic non-AWD signature.
  - "AWD field": same initial transplant dip, but with 5 additional
    wet/dry oscillations superimposed during the vegetative stage, mimicking
    repeated drain/re-flood cycles.

Expectation: the AWD series should show materially more drying cycles and a
higher awd_likelihood score than the continuous-flood series. This doesn't
validate real-world accuracy (that needs real imagery + real labels, per
04_validate_against_open_sources.py) -- it validates that the code correctly
implements the intended logic before it's ever pointed at real data.
"""
import numpy as np

from utils import wetness_index, otsu_water_threshold, detect_drying_cycles, refined_lee_filter

rng = np.random.default_rng(42)


def make_dates(n, step_days=6):
    from datetime import datetime, timedelta
    start = datetime(2025, 6, 15)
    return [start + timedelta(days=step_days * i) for i in range(n)]


def continuous_flood_series(n=20):
    t = np.arange(n)
    # dB backscatter: dips to ~-18 dB at transplant (t=0-1), rises to ~-9 dB
    # plateau by tillering, following the phenology curve documented in the
    # literature review (min ~ -18 to -22 dB flooded, max ~ -9 to -14 dB
    # vegetative peak).
    curve = -18 + 9 * (1 - np.exp(-t / 4.0))
    noise = rng.normal(0, 0.5, n)
    return curve + noise


def awd_series(n=20, n_dry_events=5):
    base = continuous_flood_series(n)
    awd = base.copy()
    # Superimpose drying spikes (backscatter rises during dry-down, drops
    # again on re-flood) across the vegetative window (roughly index 2..15).
    dry_positions = np.linspace(3, 15, n_dry_events).astype(int)
    for pos in dry_positions:
        if pos < n:
            awd[pos] += 6.0  # dries out -> backscatter rises
        if pos + 1 < n:
            awd[pos + 1] -= 5.0  # re-flood -> backscatter drops again
    awd += rng.normal(0, 0.5, n)
    return awd


def run_case(name, series):
    dates = make_dates(len(series))
    wi = wetness_index(series)
    otsu_thresh = otsu_water_threshold(series)
    result = detect_drying_cycles(dates, wi)
    print(f"\n--- {name} ---")
    print("VV dB series:", np.round(series, 1))
    print("Wetness Index:", np.round(wi, 2))
    print(f"Otsu threshold: {otsu_thresh:.2f} dB")
    print(f"Drying cycles detected: {result.n_cycles}")
    print(f"AWD-likelihood score: {result.awd_likelihood:.2f}")
    return result


def test_lee_filter_reduces_variance():
    clean = np.ones((50, 50)) * 2.0
    noisy = clean * rng.gamma(shape=4.0, scale=0.25, size=clean.shape)  # multiplicative speckle
    filtered = refined_lee_filter(noisy, window=7)
    assert filtered.std() < noisy.std(), "Refined Lee filter should reduce local variance"
    print(f"\nSpeckle filter check: noisy std={noisy.std():.3f} -> filtered std={filtered.std():.3f} (PASS)")


def main():
    flood_result = run_case("Continuous flood (non-AWD)", continuous_flood_series())
    awd_result = run_case("AWD field (5 drain/re-flood cycles)", awd_series())
    test_lee_filter_reduces_variance()

    print("\n=== Summary ===")
    print(f"Continuous flood: {flood_result.n_cycles} cycles, "
          f"awd_likelihood={flood_result.awd_likelihood:.2f}")
    print(f"AWD field:        {awd_result.n_cycles} cycles, "
          f"awd_likelihood={awd_result.awd_likelihood:.2f}")

    assert awd_result.n_cycles > flood_result.n_cycles, (
        "AWD series should show more drying cycles than the continuously "
        "flooded series -- if this fails, the cycle-detection logic (or its "
        "default thresholds) needs revisiting before pointing it at real data."
    )
    assert awd_result.awd_likelihood > flood_result.awd_likelihood
    print("\nPASS: synthetic AWD series scored higher than continuous-flood series.")


if __name__ == "__main__":
    main()
