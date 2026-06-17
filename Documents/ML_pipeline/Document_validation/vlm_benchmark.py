"""Benchmark mini VLMs against the labeled document validation test set.

Usage:
    python vlm_benchmark.py                           # all models, 72-file subset
    python vlm_benchmark.py --models qwen2-vl-2b smolvlm   # specific models
    python vlm_benchmark.py --full qwen2-vl-2b        # one model, full dataset
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_validation.ground_truth import load_ground_truth, make_subset
from document_validation.validator import ValidationConfig, _load_image, _render_pdf_pages
from document_validation.vlm_detector import MODEL_REGISTRY, VlmDetector

PROJECT_ROOT = Path(__file__).parent
TEST_DATA = PROJECT_ROOT / "test_data"
OUT_DIR = PROJECT_ROOT / "eval_results"
OUT_DIR.mkdir(exist_ok=True)

SUBSET_CSV = OUT_DIR / "vlm_subset.csv"
BENCHMARK_CSV = OUT_DIR / "vlm_benchmark.csv"
SUMMARY_CSV = OUT_DIR / "vlm_benchmark_summary.csv"

LABELS = ["accepted", "blur", "cut", "not_document"]
CONFIG = ValidationConfig(pdf_dpi=200)


def get_or_create_subset(gt_df: pd.DataFrame) -> pd.DataFrame:
    if SUBSET_CSV.exists():
        print(f"Loading existing subset from {SUBSET_CSV}")
        sub = pd.read_csv(SUBSET_CSV)
        sub["file_path"] = sub["file_path"].apply(Path)
        return sub
    sub = make_subset(gt_df, n_per_class=18, seed=42)
    sub.to_csv(SUBSET_CSV, index=False)
    print(f"Subset created: {len(sub)} files → {SUBSET_CSV}")
    return sub


def load_first_page(file_path: Path):
    import numpy as np
    if file_path.suffix.lower() == ".pdf":
        pages = list(_render_pdf_pages(file_path, CONFIG))
        return pages[0] if pages else np.zeros((100, 100, 3), dtype=np.uint8)
    return _load_image(file_path)


def run_model(model_key: str, eval_df: pd.DataFrame) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Model: {model_key}  ({len(eval_df)} files)")
    print(f"{'='*60}")

    detector = VlmDetector(model_key)
    rows = []

    for i, (_, row) in enumerate(eval_df.iterrows()):
        t0 = time.time()
        try:
            page = load_first_page(row["file_path"])
            vote = detector.detect(page, row["file_path"], 1, CONFIG)
            predicted = vote.label or "accepted"
        except Exception as exc:
            predicted = "error"
            print(f"  ERROR on {Path(row['file_path']).name}: {exc}")
        latency = time.time() - t0

        rows.append({
            "model": model_key,
            "file_path": str(row["file_path"]),
            "expected_label": row["expected_label"],
            "predicted_label": predicted,
            "latency_s": round(latency, 2),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eval_df)}", end="\r", flush=True)

    del detector
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    valid = [r for r in rows if r["predicted_label"] != "error"]
    y_true = [r["expected_label"] for r in valid]
    y_pred = [r["predicted_label"] for r in valid]
    print(f"\n{classification_report(y_true, y_pred, labels=LABELS, zero_division=0)}")
    return rows


def build_summary(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for model, grp in benchmark_df.groupby("model"):
        valid = grp[grp["predicted_label"] != "error"]
        yt = valid["expected_label"]
        yp = valid["predicted_label"]
        row: dict = {
            "model": model,
            "n_files": len(valid),
            "accuracy": round((yt == yp).mean(), 3),
            "macro_f1": round(
                f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0), 3
            ),
            "median_latency_s": round(grp["latency_s"].median(), 2),
        }
        for cls in LABELS:
            row[f"{cls}_f1"] = round(f1_score(yt == cls, yp == cls, zero_division=0), 3)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY),
        help="Model keys to benchmark (default: all)"
    )
    parser.add_argument(
        "--full", metavar="MODEL_KEY",
        help="Run one model on the full dataset instead of the subset"
    )
    args = parser.parse_args()

    gt_df = load_ground_truth(TEST_DATA)
    print(f"Ground truth: {len(gt_df)} files, {gt_df['expected_label'].value_counts().to_dict()}")

    if args.full:
        eval_df = gt_df
        model_keys = [args.full]
    else:
        eval_df = get_or_create_subset(gt_df)
        model_keys = args.models

    all_rows: list[dict] = []
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            print(f"Unknown model key: {key!r} — skipping")
            continue
        rows = run_model(key, eval_df)
        all_rows.extend(rows)

    if not all_rows:
        print("No results produced.")
        return

    benchmark_df = pd.DataFrame(all_rows)
    benchmark_df.to_csv(BENCHMARK_CSV, index=False)
    print(f"\nResults → {BENCHMARK_CSV}")

    summary = build_summary(benchmark_df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print("\n=== Model Ranking ===")
    print(summary.to_string(index=False))
    print(f"\nSummary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
