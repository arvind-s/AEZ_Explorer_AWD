"""GDD-based phenophase classification (120-day paddy, D-6072 scaled)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import GDDStageThreshold


@dataclass
class StageResult:
    stage_id: str
    stage_name: str
    cumulative_gdd: float
    days_after_transplant: int
    confidence: float
    gdd_stage_margin: float
    satellite_stage_hint: str
    anomaly_flags: list[str]


def classify_gdd_stage(
    cumulative_gdd: float,
    stages: tuple[GDDStageThreshold, ...],
) -> tuple[str, str, float]:
    """Return (stage_id, stage_name, margin to nearest boundary inside stage)."""
    for st in stages:
        if st.gdd_min <= cumulative_gdd < st.gdd_max:
            margin = min(cumulative_gdd - st.gdd_min, st.gdd_max - cumulative_gdd)
            return st.name, st.name, float(margin)
    last = stages[-1]
    if cumulative_gdd >= last.gdd_max:
        return last.name, last.name, 0.0
    return stages[0].name, stages[0].name, 0.0


def validate_stage_with_satellite(
    gdd_stage: str,
    satellite_hint: str,
    gdd_margin: float,
    source_meta: dict,
) -> tuple[float, list[str]]:
    """
    Adjust confidence when GDD stage disagrees with coarse satellite hint.
    """
    flags: list[str] = []
    agreement = gdd_stage == satellite_hint or satellite_hint == "unknown"

    # Stage mapping for partial agreement (vegetative vs establishment)
    partial_ok = {
        ("establishment", "vegetative"),
        ("vegetative", "establishment"),
        ("reproductive", "vegetative"),
        ("ripening", "reproductive"),
        ("maturity", "ripening"),
    }
    if not agreement and (gdd_stage, satellite_hint) in partial_ok:
        agreement = True
        flags.append("satellite_stage_adjacent")

    if not agreement and satellite_hint != "unknown":
        flags.append(f"gdd_satellite_mismatch_{gdd_stage}_vs_{satellite_hint}")

    s2_count = source_meta.get("S2", 0)
    s1_count = source_meta.get("S1", 0)
    total = s2_count + s1_count + source_meta.get("missing", 0)
    data_score = (s2_count + 0.7 * s1_count) / max(total, 1)

    margin_score = min(1.0, gdd_margin / 80.0)
    agree_score = 1.0 if agreement else 0.45

    confidence = 0.55 * margin_score + 0.25 * agree_score + 0.20 * data_score
    return float(min(0.98, max(0.2, confidence))), flags


def build_stage_result(
    cumulative_gdd: float,
    transplant_date: pd.Timestamp,
    assessment_date: pd.Timestamp,
    stages: tuple[GDDStageThreshold, ...],
    satellite_hint: str,
    source_meta: dict,
) -> StageResult:
    stage_id, stage_name, margin = classify_gdd_stage(cumulative_gdd, stages)
    dat = int((pd.Timestamp(assessment_date) - pd.Timestamp(transplant_date)).days)
    conf, flags = validate_stage_with_satellite(stage_id, satellite_hint, margin, source_meta)

    return StageResult(
        stage_id=stage_id,
        stage_name=stage_name,
        cumulative_gdd=cumulative_gdd,
        days_after_transplant=dat,
        confidence=conf,
        gdd_stage_margin=margin,
        satellite_stage_hint=satellite_hint,
        anomaly_flags=flags,
    )
