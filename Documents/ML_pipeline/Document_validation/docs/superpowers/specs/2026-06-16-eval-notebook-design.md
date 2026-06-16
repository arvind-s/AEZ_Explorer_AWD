# Evaluation Notebook Design
**Date:** 2026-06-16
**Scope:** Separate notebook for measuring ensemble document validation accuracy against labeled test data

---

## Context

The existing `ensemble_document_validation.ipynb` runs validation and shows results but never compares against ground truth. Labeled test data already exists in `test_data/<DocType>/<State>/<category>/` but is unused for evaluation. The primary gap: no metrics to know whether errors skew toward false positives (good docs rejected) or false negatives (bad docs accepted).

---

## Notebook: `eval_document_validation.ipynb`

Six sections, designed to be run top-to-bottom.

---

### Section 1 — Setup

Imports, `ValidationConfig` (current production config), detector probe, and path to `test_data/`. Same pattern as the existing notebook.

---

### Section 2 — Ground Truth Loader

Walks `test_data/<DocType>/<State>/<category>/` and maps folder names to expected labels:

| Folder name | Expected label |
|---|---|
| `is_blur` | `blur` |
| `is_not_full_document(cut)` | `cut` |
| `is_not_document` | `not_document` |
| `is_clear` | `accepted` |
| anything else | skip |

Produces a flat DataFrame: `[file_path, doc_type, state, expected_label]`.

---

### Section 3 — Batch Eval Run

Runs `validate_document_file_ensemble` on every file in the ground truth DataFrame. Shows a progress counter. Saves raw `EnsembleFileResult` objects and caches `DetectorVote` objects per file/page for reuse in Sections 5–6.

**File-level prediction:** Aggregate page-level labels using `TIE_BREAK_ORDER` (`not_document > blur > not_clear > cut > accepted`). A file with one `blur` page and one `accepted` page predicts `blur`.

Produces `results_df`: ground truth columns + `[predicted_label, page_count, per_page_labels]`.

---

### Section 4 — Metrics & Confusion Matrix

**Per-class metrics (one-vs-rest):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| accepted | | | | |
| blur | | | | |
| cut | | | | |
| not_document | | | | |
| **macro avg** | | | | |

**Confusion matrix:** 4×4 heatmap (rows = expected, cols = predicted) rendered with `matplotlib`.

**Per-state breakdown table:** rows = states, columns = `[n_files, accuracy, blur_F1, cut_F1, not_document_F1, accepted_F1]`. Surfaces state-specific systematic errors.

---

### Section 5 — Detector Ablation

Re-runs `validate_document_file_ensemble` with each detector in isolation (`detectors=["opencv"]`, `detectors=["doctr"]`), then compares all three to the full ensemble.

**Output — per-detector metrics table:**

| Class | opencv P/R/F1 | doctr P/R/F1 | ensemble P/R/F1 |
|---|---|---|---|
| blur | | | |
| cut | | | |
| not_document | | | |
| accepted | | | |

**Purpose:** Shows whether doctr adds value over opencv alone and for which classes. Directly informs whether detector importance weights need adjustment.

---

### Section 6 — Threshold Calibration

Sweeps three key thresholds and plots how per-class precision/recall/F1 change:

| Threshold | Range | Step | Re-run strategy |
|---|---|---|---|
| `min_reject_confidence` | 0.30–0.95 | 0.05 | Recompute from cached `DetectorVote` objects — no detector re-run |
| `blur_laplacian_threshold` | 20–150 | 10 | Re-run OpenCV only (fast, deterministic); reuse cached doctr votes |
| `min_document_confidence` | 0.40–0.95 | 0.05 | Re-run OpenCV only; reuse cached doctr votes |

**Plots:** One figure per threshold. Each figure has 4 subplots (one per class), showing precision, recall, and F1 as curves with a vertical marker at the current config value.

**Summary table:** For each threshold, the value that maximises macro-F1 — gives a concrete recommended config as a byproduct.

---

## What This Does Not Cover

- `is_full_concent_form` and `is_proper_concent_form` folders are excluded — they test document completeness, not image quality, which this validator does not check.
- Multi-label ground truth (a file that is both blurry and cut) is not in the current test data structure. If added later, the file-level aggregation logic will need revisiting.
- `deepdoctection` is excluded from ablation (adapter is a placeholder).
- `deqa_doc` is excluded from ablation (requires external command; not available in standard eval runs).
