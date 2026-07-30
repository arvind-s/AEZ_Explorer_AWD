#!/usr/bin/env python3
"""
Two validation utilities for the v0 model, in increasing order of rigor:

1. `check_against_rice_calendar` -- a cheap, open-source sanity check with NO
   AWD-specific ground truth needed: does the model's detected
   flooding/transplant date (the wetness-index peak) fall inside the
   planting window RiceAtlas publishes for this district/season? If not,
   something upstream (AOI, dates, calibration, polarization) is probably
   wrong, before you even get to AWD-specific accuracy questions.

   RiceAtlas (Laborte et al., 2017, Scientific Data,
   DOI 10.7910/DVN/JE6R2R) is not bundled here -- download the calendar CSV
   from the Harvard Dataverse and point --calendar-csv at it.

2. `score_against_labels` -- the real accuracy check, once you have actual
   AWD/non-AWD labels for the AOI's fields. Point this at:
     - Varaha's own field survey data (per the agreed plan: use this ONLY as
       a final held-out check, not for tuning thresholds), or
     - one of the open regional datasets flagged in the literature review
       (Mekong / Punjab / Philippines-Japan / Bangladesh) IF its data-sharing
       terms allow reuse -- confirm with each paper's data-availability
       statement/authors first, per the open item in the roadmap doc.

   Expects a CSV with columns: aoi_id, awd_label (0/1). Compares against a
   CSV of model outputs with columns: aoi_id, awd_likelihood.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime


def check_against_rice_calendar(detected_transplant_date: datetime,
                                 calendar_csv: str, state: str, district: str) -> None:
    with open(calendar_csv) as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("STATE", "").strip().lower() == state.lower()
                and r.get("DISTRICT", "").strip().lower() == district.lower()]
    if not rows:
        print(f"No RiceAtlas entry found for {district}, {state} in {calendar_csv} "
              f"-- check column names/spelling match the RiceAtlas schema you downloaded.")
        return

    for r in rows:
        # RiceAtlas fields are typically named like HSTART_1 / HEND_1 for
        # season 1 harvest, and there's a corresponding planting start/end.
        # Adjust these keys to match whichever RiceAtlas export you have.
        print(f"RiceAtlas entry: {r}")

    print(f"\nModel detected transplant/flood date: {detected_transplant_date.date()}")
    print("Compare this manually (or extend this function) against the "
          "planting window(s) printed above -- a match is a basic sanity "
          "check, not a substitute for real AWD label validation below.")


def score_against_labels(labels_csv: str, predictions_csv: str, threshold: float = 0.5) -> dict:
    labels = {}
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            labels[r["aoi_id"]] = int(r["awd_label"])

    preds = {}
    with open(predictions_csv) as f:
        for r in csv.DictReader(f):
            preds[r["aoi_id"]] = float(r["awd_likelihood"])

    tp = fp = tn = fn = 0
    for aoi_id, true_label in labels.items():
        if aoi_id not in preds:
            continue
        pred_label = int(preds[aoi_id] >= threshold)
        if pred_label == 1 and true_label == 1:
            tp += 1
        elif pred_label == 1 and true_label == 0:
            fp += 1
        elif pred_label == 0 and true_label == 0:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    accuracy = (tp + tn) / n if n else float("nan")

    metrics = {
        "n_matched": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }
    print(metrics)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("calendar-check")
    p1.add_argument("--date", required=True, help="Detected transplant date, YYYY-MM-DD")
    p1.add_argument("--calendar-csv", required=True)
    p1.add_argument("--state", required=True)
    p1.add_argument("--district", required=True)

    p2 = sub.add_parser("score")
    p2.add_argument("--labels-csv", required=True)
    p2.add_argument("--predictions-csv", required=True)
    p2.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()
    if args.mode == "calendar-check":
        check_against_rice_calendar(datetime.fromisoformat(args.date), args.calendar_csv, args.state, args.district)
    else:
        score_against_labels(args.labels_csv, args.predictions_csv, args.threshold)
