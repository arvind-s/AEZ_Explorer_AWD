# Evaluation Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `eval_document_validation.ipynb` — a standalone notebook that measures ensemble validation accuracy against labeled test data, with per-state breakdown, detector ablation, and threshold calibration.

**Architecture:** The notebook reads labeled ground truth from `test_data/`, runs the ensemble once, caches `EnsembleFileResult` objects and page BGR images, then derives all further metrics/sweeps from those caches. A new public `recompute_page_from_votes` helper in `ensemble.py` lets the threshold calibration section replay decisions with different configs without re-running full inference.

**Tech Stack:** Python 3.10+, OpenCV, PyMuPDF (fitz), NumPy, pandas, matplotlib, scikit-learn, document_validation (local package at `src/`)

---

### Task 1: Add `recompute_page_from_votes` to `ensemble.py`

This public function takes pre-weighted `DetectorVote` objects and a `ValidationConfig`, and produces an `EnsemblePageResult` — essentially the last half of `_vote_page` after the detector calls. The threshold calibration section needs it to replay cached votes under different `min_reject_confidence` values without re-running detectors.

**Files:**
- Modify: `src/document_validation/ensemble.py`
- Test: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validator.py`:

```python
from document_validation.ensemble import (
    DetectorVote,
    EnsemblePageResult,
    recompute_page_from_votes,
    _decision_columns_from_votes,
)
import dataclasses


def test_recompute_accepts_with_low_min_reject_confidence():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="blur",
            importance=1.0,
            issues=["blur"],
            confidence=0.55,
            scores={"blur_confidence": 0.55},
        )
    ]
    config = ValidationConfig(min_reject_confidence=0.70)
    result = recompute_page_from_votes(1, votes, config)

    # confidence 0.55 < min_reject_confidence 0.70 → should not trigger rejection
    assert result.accepted
    assert result.label == "accepted"
    assert result.Decision == "accept"


def test_recompute_rejects_with_high_min_reject_confidence():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="blur",
            importance=1.0,
            issues=["blur"],
            confidence=0.55,
            scores={"blur_confidence": 0.55},
        )
    ]
    config = ValidationConfig(min_reject_confidence=0.50)
    result = recompute_page_from_votes(1, votes, config)

    # confidence 0.55 >= min_reject_confidence 0.50 → should reject
    assert not result.accepted
    assert result.label == "blur"
    assert result.is_blur
    assert result.Decision == "reject"


def test_recompute_preserves_vote_list():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="accepted",
            importance=1.0,
            issues=[],
            confidence=0.9,
        ),
        DetectorVote(
            detector="doctr",
            available=True,
            label="accepted",
            importance=1.25,
            issues=[],
            confidence=0.88,
        ),
    ]
    config = ValidationConfig()
    result = recompute_page_from_votes(2, votes, config)

    assert result.page_number == 2
    assert result.votes == votes
    assert result.label_votes["accepted"] == 2
    assert abs(result.weighted_label_votes["accepted"] - 2.25) < 0.001


def test_recompute_aggregates_weighted_votes_correctly():
    votes = [
        DetectorVote(
            detector="opencv_gate",
            available=True,
            label="not_document",
            importance=1.0,
            issues=["not_document"],
            confidence=0.80,
        ),
        DetectorVote(
            detector="doctr",
            available=True,
            label="accepted",
            importance=1.25,
            issues=[],
            confidence=0.85,
        ),
    ]
    config = ValidationConfig(min_reject_confidence=0.70)
    result = recompute_page_from_votes(1, votes, config)

    assert result.label_votes["not_document"] == 1
    assert result.label_votes["accepted"] == 1
    assert abs(result.weighted_label_votes["not_document"] - 1.0) < 0.001
    assert abs(result.weighted_label_votes["accepted"] - 1.25) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/mipl/Documents/ML_pipeline/Document_validation
python -m pytest tests/test_validator.py::test_recompute_accepts_with_low_min_reject_confidence tests/test_validator.py::test_recompute_rejects_with_high_min_reject_confidence tests/test_validator.py::test_recompute_preserves_vote_list tests/test_validator.py::test_recompute_aggregates_weighted_votes_correctly -v
```

Expected: `FAILED` with `ImportError: cannot import name 'recompute_page_from_votes'`

- [ ] **Step 3: Add `recompute_page_from_votes` to `ensemble.py`**

Add this function after the `ensemble_result_to_rows` function in `src/document_validation/ensemble.py` (around line 409):

```python
def recompute_page_from_votes(
    page_number: int,
    votes: list[DetectorVote],
    config: ValidationConfig,
) -> EnsemblePageResult:
    """Replay ensemble decision on pre-weighted votes under a new config.

    Useful for threshold sweeps: cache votes from a full run, then call this
    with different ValidationConfig values to recompute decisions without
    re-running any detector.

    The votes must already have importance weights set (i.e., come from a
    previous EnsemblePageResult.votes list).
    """
    available_votes = [v for v in votes if v.available and v.label]
    label_votes = {label: 0 for label in ALL_LABELS}
    weighted_label_votes = {label: 0.0 for label in ALL_LABELS}

    for vote in available_votes:
        label_votes[vote.label or "accepted"] += 1
        weighted_label_votes[vote.label or "accepted"] += vote.importance

    weighted_label_votes = {
        label: round(score, 4) for label, score in weighted_label_votes.items()
    }
    decision_columns = _decision_columns_from_votes(available_votes, config)
    strong_issue_labels = _strong_reject_labels(available_votes, config)
    label = (
        _label_from_issues(sorted(strong_issue_labels))
        if decision_columns["Decision"] == "reject"
        else "accepted"
    )
    issues = [] if label == "accepted" else [label]

    return EnsemblePageResult(
        page_number=page_number,
        accepted=decision_columns["Decision"] == "accept",
        label=label,
        issues=issues,
        is_blur=bool(decision_columns["is_blur"]),
        is_cut=bool(decision_columns["is_cut"]),
        is_clear=bool(decision_columns["is_clear"]),
        is_image=bool(decision_columns["is_image"]),
        Decision=str(decision_columns["Decision"]),
        label_votes=label_votes,
        weighted_label_votes=weighted_label_votes,
        votes=votes,
    )
```

Also add `recompute_page_from_votes` to the public exports in `src/document_validation/__init__.py`. First read `__init__.py` to see its current contents, then add the import.

- [ ] **Step 4: Export `recompute_page_from_votes` from `__init__.py`**

In `src/document_validation/__init__.py`, change the ensemble import block from:

```python
from document_validation.ensemble import (
    DetectorVote,
    EnsembleFileResult,
    EnsemblePageResult,
    ensemble_result_to_rows,
    probe_available_detectors,
    validate_document_file_ensemble,
)
```

to:

```python
from document_validation.ensemble import (
    DetectorVote,
    EnsembleFileResult,
    EnsemblePageResult,
    ensemble_result_to_rows,
    probe_available_detectors,
    recompute_page_from_votes,
    validate_document_file_ensemble,
)
```

Also add `"recompute_page_from_votes"` to the `__all__` list.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/mipl/Documents/ML_pipeline/Document_validation
python -m pytest tests/test_validator.py::test_recompute_accepts_with_low_min_reject_confidence tests/test_validator.py::test_recompute_rejects_with_high_min_reject_confidence tests/test_validator.py::test_recompute_preserves_vote_list tests/test_validator.py::test_recompute_aggregates_weighted_votes_correctly -v
```

Expected: `4 passed`

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
cd /Users/mipl/Documents/ML_pipeline/Document_validation
python -m pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/document_validation/ensemble.py src/document_validation/__init__.py tests/test_validator.py
git commit -m "feat: add recompute_page_from_votes for threshold calibration"
```

---

### Task 2: Notebook — Sections 1–3 (Setup, Ground Truth, Batch Eval)

**Files:**
- Create: `eval_document_validation.ipynb`

- [ ] **Step 1: Create the notebook with Section 1 (Setup)**

Create `eval_document_validation.ipynb` with the following cells.

**Cell 1 — markdown:**
```
# Ensemble Document Validation — Evaluation

Measures validation accuracy against labeled test data in `test_data/`.

Sections:
1. Setup
2. Ground truth loader
3. Batch eval run (with caching)
4. Metrics & confusion matrix
5. Detector ablation
6. Threshold calibration
```

**Cell 2 — markdown:**
```
## 1. Setup
```

**Cell 3 — code:**
```python
import dataclasses
import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

PROJECT_ROOT = Path.cwd()
SRC_PATH = str(PROJECT_ROOT / "src")

missing = [name for name in ("cv2", "fitz") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

for mod in list(sys.modules):
    if mod == "document_validation" or mod.startswith("document_validation."):
        del sys.modules[mod]

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from document_validation import (
    ValidationConfig,
    probe_available_detectors,
    recompute_page_from_votes,
    validate_document_file_ensemble,
)
from document_validation.ensemble import OpenCvGateDetector
from document_validation.validator import _load_image, _render_pdf_pages

config = ValidationConfig(
    blur_laplacian_threshold=40.0,
    blur_tenengrad_threshold=30.0,
    min_readability_contrast=80.0,
    max_low_readability_gray_std=65.0,
    min_document_confidence=0.95,
    min_reject_confidence=0.70,
    pdf_dpi=200,
)

requested_detectors = ["opencv", "doctr"]
detectors, detector_warnings = probe_available_detectors(requested_detectors)
for w in detector_warnings:
    print("Note:", w)

detector_importance = {
    "opencv": 1.0,
    "doctr": 1.25,
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pdf"}
LABELS = ["accepted", "blur", "cut", "not_document"]
TIE_BREAK = ("not_document", "blur", "not_clear", "cut", "accepted")

print("Using detectors:", detectors)
print("Ready")
```

- [ ] **Step 2: Add Section 2 (Ground Truth Loader)**

**Cell 4 — markdown:**
```
## 2. Ground Truth Loader

Walks `test_data/<DocType>/<State>/<category>/` and maps folder names to expected labels.
Only `is_blur`, `is_not_full_document(cut)`, `is_not_document`, and `is_clear` are used.
```

**Cell 5 — code:**
```python
FOLDER_TO_LABEL = {
    "is_blur": "blur",
    "is_not_full_document(cut)": "cut",
    "is_not_document": "not_document",
    "is_clear": "accepted",
}

def load_ground_truth(test_data_dir: Path) -> pd.DataFrame:
    rows = []
    for doc_type_dir in sorted(test_data_dir.iterdir()):
        if not doc_type_dir.is_dir():
            continue
        for state_dir in sorted(doc_type_dir.iterdir()):
            if not state_dir.is_dir():
                continue
            for category_dir in sorted(state_dir.iterdir()):
                if not category_dir.is_dir():
                    continue
                label = FOLDER_TO_LABEL.get(category_dir.name)
                if label is None:
                    continue
                # rglob handles both flat folders and subdirs like is_blur/max/, is_blur/min/
                for f in sorted(category_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in EXTENSIONS:
                        rows.append({
                            "file_path": f,
                            "doc_type": doc_type_dir.name,
                            "state": state_dir.name,
                            "expected_label": label,
                        })
    return pd.DataFrame(rows)

TEST_DATA_DIR = PROJECT_ROOT / "test_data"
gt_df = load_ground_truth(TEST_DATA_DIR)
print(f"Ground truth: {len(gt_df)} files")
gt_df.groupby(["doc_type", "state", "expected_label"]).size().unstack(fill_value=0)
```

- [ ] **Step 3: Add Section 3 (Batch Eval Run)**

**Cell 6 — markdown:**
```
## 3. Batch Eval Run

Runs the ensemble on every labeled file. Caches `EnsembleFileResult` objects and page
BGR images for reuse in threshold calibration (Section 6).
```

**Cell 7 — code:**
```python
def _file_label_from_issues(issues: list) -> str:
    if not issues:
        return "accepted"
    for label in TIE_BREAK:
        if label in issues:
            return label
    return issues[0]


# RESULT_CACHE: str(file_path) -> EnsembleFileResult (contains all DetectorVote objects)
# PAGE_CACHE:   str(file_path) -> list of BGR page images (for OpenCV threshold sweeps)
RESULT_CACHE: dict = {}
PAGE_CACHE: dict = {}

eval_rows = []
n = len(gt_df)

for i, (_, row) in enumerate(gt_df.iterrows()):
    if (i + 1) % 50 == 0 or i == n - 1:
        print(f"  {i + 1}/{n}", end="\r")
    file_path = row["file_path"]
    key = str(file_path)
    try:
        result = validate_document_file_ensemble(
            file_path,
            config=config,
            detectors=detectors,
            detector_importance=detector_importance,
        )
        RESULT_CACHE[key] = result

        # Cache page images for OpenCV threshold sweeps
        if file_path.suffix.lower() == ".pdf":
            PAGE_CACHE[key] = list(_render_pdf_pages(file_path, config))
        else:
            PAGE_CACHE[key] = [_load_image(file_path)]

        predicted = _file_label_from_issues(result.issues)
        eval_rows.append({**row.to_dict(), "predicted_label": predicted, "page_count": result.page_count})
    except Exception as exc:
        eval_rows.append({**row.to_dict(), "predicted_label": "error", "page_count": 0, "error": str(exc)})

results_df = pd.DataFrame(eval_rows)
errors = (results_df["predicted_label"] == "error").sum()
print(f"\nDone. {len(results_df)} files evaluated. Errors: {errors}")
if errors:
    print(results_df[results_df["predicted_label"] == "error"][["file_path", "error"]])
```

- [ ] **Step 4: Run sections 1–3 top-to-bottom and verify no errors**

Open `eval_document_validation.ipynb` in Jupyter and run all cells. Confirm:
- `gt_df` has rows with columns `[file_path, doc_type, state, expected_label]`
- `results_df` has no error rows (or if it does, they're reported clearly)
- `RESULT_CACHE` and `PAGE_CACHE` are populated

- [ ] **Step 5: Commit**

```bash
git add eval_document_validation.ipynb
git commit -m "feat: add eval notebook sections 1-3 (setup, ground truth, batch eval)"
```

---

### Task 3: Notebook — Section 4 (Metrics & Confusion Matrix)

**Files:**
- Modify: `eval_document_validation.ipynb`

- [ ] **Step 1: Add Section 4 cells**

**Cell 8 — markdown:**
```
## 4. Metrics & Confusion Matrix

Per-class precision/recall/F1, overall confusion matrix, and per-state breakdown.
```

**Cell 9 — code (per-class metrics):**
```python
valid_df = results_df[results_df["predicted_label"] != "error"].copy()
y_true = valid_df["expected_label"]
y_pred = valid_df["predicted_label"]

print(f"Files evaluated: {len(valid_df)}")
print()
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
```

**Cell 10 — code (confusion matrix):**
```python
cm = confusion_matrix(y_true, y_pred, labels=LABELS)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(LABELS)))
ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted")
ax.set_ylabel("Expected")
ax.set_title("Confusion Matrix — Ensemble")
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
```

**Cell 11 — code (per-state breakdown):**
```python
state_rows = []
for state, grp in valid_df.groupby("state"):
    yt = grp["expected_label"]
    yp = grp["predicted_label"]
    row = {
        "state": state,
        "n_files": len(grp),
        "accuracy": round((yt == yp).mean(), 3),
    }
    for cls in LABELS:
        if (yt == cls).any():
            row[f"{cls}_F1"] = round(f1_score(yt == cls, yp == cls, zero_division=0), 3)
        else:
            row[f"{cls}_F1"] = float("nan")
    state_rows.append(row)

state_df = pd.DataFrame(state_rows).set_index("state").sort_values("accuracy")
f1_cols = [c for c in state_df.columns if "F1" in c]
state_df.style.format("{:.3f}", subset=f1_cols + ["accuracy"]).background_gradient(
    cmap="RdYlGn", subset=f1_cols + ["accuracy"]
)
```

- [ ] **Step 2: Run section 4 and confirm**

Run cells 8–11. Confirm:
- `classification_report` prints 4 class rows + macro avg
- Confusion matrix renders correctly (rows = expected, columns = predicted)
- Per-state table shows one row per state with no Python errors

- [ ] **Step 3: Commit**

```bash
git add eval_document_validation.ipynb
git commit -m "feat: add eval notebook section 4 (metrics and confusion matrix)"
```

---

### Task 4: Notebook — Section 5 (Detector Ablation)

**Files:**
- Modify: `eval_document_validation.ipynb`

- [ ] **Step 1: Add Section 5 cells**

**Cell 12 — markdown:**
```
## 5. Detector Ablation

Runs each detector in isolation and compares precision/recall/F1 with the full ensemble.
Shows whether doctr adds value over opencv alone and for which issue classes.
Note: this re-runs validation, so it takes ~same time as Section 3.
```

**Cell 13 — code (per-detector eval):**
```python
ablation_configs = {
    "opencv": ["opencv"],
    "doctr": [d for d in ["doctr"] if d in detectors],  # skip if doctr not installed
    "ensemble": detectors,
}

ablation_results: dict[str, pd.DataFrame] = {}

for det_name, det_list in ablation_configs.items():
    if not det_list:
        print(f"Skipping {det_name}: not available")
        continue
    rows = []
    for i, (_, row) in enumerate(gt_df.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  {det_name}: {i + 1}/{len(gt_df)}", end="\r")
        try:
            result = validate_document_file_ensemble(
                row["file_path"],
                config=config,
                detectors=det_list,
                detector_importance=detector_importance,
            )
            predicted = _file_label_from_issues(result.issues)
            rows.append({**row.to_dict(), "predicted_label": predicted})
        except Exception:
            rows.append({**row.to_dict(), "predicted_label": "error"})
    ablation_results[det_name] = pd.DataFrame(rows)
    print(f"\n  {det_name} done.")
```

**Cell 14 — code (ablation comparison table):**
```python
ablation_rows = []
for det_name, df in ablation_results.items():
    valid = df[df["predicted_label"] != "error"]
    yt = valid["expected_label"]
    yp = valid["predicted_label"]
    for cls in LABELS:
        ablation_rows.append({
            "detector": det_name,
            "class": cls,
            "precision": round(precision_score(yt == cls, yp == cls, zero_division=0), 3),
            "recall":    round(recall_score(yt == cls, yp == cls, zero_division=0), 3),
            "f1":        round(f1_score(yt == cls, yp == cls, zero_division=0), 3),
        })

ablation_df = pd.DataFrame(ablation_rows)
ablation_pivot = ablation_df.pivot_table(
    index="class", columns="detector", values=["precision", "recall", "f1"]
).round(3)
ablation_pivot
```

- [ ] **Step 2: Run section 5 and confirm**

Run cells 12–14. Confirm:
- `ablation_pivot` shows a multi-level column table with one row per label class
- No Python errors; if `doctr` is not installed it's skipped gracefully

- [ ] **Step 3: Commit**

```bash
git add eval_document_validation.ipynb
git commit -m "feat: add eval notebook section 5 (detector ablation)"
```

---

### Task 5: Notebook — Section 6 (Threshold Calibration)

**Files:**
- Modify: `eval_document_validation.ipynb`

- [ ] **Step 1: Add Section 6 — helper and `min_reject_confidence` sweep**

**Cell 15 — markdown:**
```
## 6. Threshold Calibration

Sweeps three key ValidationConfig thresholds and plots per-class precision/recall/F1.
A vertical marker shows where the current config sits on each curve.

- `min_reject_confidence`: replayed from cached votes — no detector re-run
- `blur_laplacian_threshold`: re-runs OpenCV only; reuses cached doctr votes
- `min_document_confidence`: re-runs OpenCV only; reuses cached doctr votes
```

**Cell 16 — code (sweep helpers):**
```python
import numpy as np


def _sweep_metrics(gt_df, pred_fn) -> pd.DataFrame:
    """
    pred_fn(row) -> predicted label string.
    Returns a DataFrame with columns [threshold, class, precision, recall, f1].
    Called once per threshold value by each sweep cell.
    """
    rows = []
    for _, row in gt_df.iterrows():
        try:
            rows.append({"expected_label": row["expected_label"], "predicted_label": pred_fn(row)})
        except Exception:
            rows.append({"expected_label": row["expected_label"], "predicted_label": "error"})
    df = pd.DataFrame(rows)
    valid = df[df["predicted_label"] != "error"]
    return valid


def _compute_class_metrics(valid_df, threshold_value) -> list[dict]:
    yt = valid_df["expected_label"]
    yp = valid_df["predicted_label"]
    return [
        {
            "threshold": threshold_value,
            "class": cls,
            "precision": round(precision_score(yt == cls, yp == cls, zero_division=0), 4),
            "recall":    round(recall_score(yt == cls, yp == cls, zero_division=0), 4),
            "f1":        round(f1_score(yt == cls, yp == cls, zero_division=0), 4),
        }
        for cls in LABELS
    ]


def _plot_threshold_sweep(sweep_df, title, current_value, xlabel):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
    for idx, cls in enumerate(LABELS):
        ax = axes[idx]
        sub = sweep_df[sweep_df["class"] == cls]
        ax.plot(sub["threshold"], sub["precision"], marker="o", label="precision")
        ax.plot(sub["threshold"], sub["recall"],    marker="s", label="recall")
        ax.plot(sub["threshold"], sub["f1"],        marker="^", linewidth=2, label="F1")
        ax.axvline(x=current_value, color="gray", linestyle="--", alpha=0.7, label="current")
        ax.set_title(f"class: {cls}")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
```

**Cell 17 — code (`min_reject_confidence` sweep):**
```python
MRC_VALUES = np.round(np.arange(0.30, 0.96, 0.05), 2)
mrc_rows = []

for thresh in MRC_VALUES:
    new_config = dataclasses.replace(config, min_reject_confidence=thresh)

    def _pred(row, nc=new_config):
        cached = RESULT_CACHE.get(str(row["file_path"]))
        if cached is None:
            raise ValueError("not in cache")
        new_pages = [
            recompute_page_from_votes(page.page_number, page.votes, nc)
            for page in cached.pages
        ]
        return _file_label_from_issues([iss for p in new_pages for iss in p.issues])

    valid = _sweep_metrics(gt_df, _pred)
    mrc_rows.extend(_compute_class_metrics(valid, thresh))

mrc_df = pd.DataFrame(mrc_rows)
_plot_threshold_sweep(
    mrc_df,
    title="Threshold sweep: min_reject_confidence",
    current_value=config.min_reject_confidence,
    xlabel="min_reject_confidence",
)
```

**Cell 18 — code (`min_reject_confidence` best-value summary):**
```python
macro_mrc = mrc_df.groupby("threshold")["f1"].mean().reset_index()
best_mrc = macro_mrc.loc[macro_mrc["f1"].idxmax()]
print(f"Best macro-F1 for min_reject_confidence: {best_mrc['f1']:.4f} at threshold={best_mrc['threshold']:.2f}")
print(f"Current config: min_reject_confidence={config.min_reject_confidence}")
```

- [ ] **Step 2: Add `blur_laplacian_threshold` sweep cells**

**Cell 19 — code (`blur_laplacian_threshold` sweep):**
```python
BLT_VALUES = np.arange(20, 151, 10, dtype=float)
opencv_detector = OpenCvGateDetector()
blt_rows = []

for thresh in BLT_VALUES:
    new_config = dataclasses.replace(config, blur_laplacian_threshold=thresh)

    def _pred(row, nc=new_config):
        key = str(row["file_path"])
        cached = RESULT_CACHE.get(key)
        page_images = PAGE_CACHE.get(key)
        if cached is None or page_images is None:
            raise ValueError("not in cache")
        new_pages = []
        for page, page_bgr in zip(cached.pages, page_images):
            # Re-run OpenCV with the new blur threshold
            new_opencv = opencv_detector.detect(page_bgr, row["file_path"], page.page_number, nc)
            # Apply same importance weight as original
            orig_opencv = next((v for v in page.votes if v.detector == "opencv_gate"), None)
            importance = orig_opencv.importance if orig_opencv is not None else 1.0
            new_opencv = dataclasses.replace(new_opencv, importance=importance)
            # Rebuild vote list: new opencv + all other cached votes unchanged
            other_votes = [v for v in page.votes if v.detector != "opencv_gate"]
            new_votes = [new_opencv] + other_votes
            new_pages.append(recompute_page_from_votes(page.page_number, new_votes, nc))
        return _file_label_from_issues([iss for p in new_pages for iss in p.issues])

    valid = _sweep_metrics(gt_df, _pred)
    blt_rows.extend(_compute_class_metrics(valid, thresh))

blt_df = pd.DataFrame(blt_rows)
_plot_threshold_sweep(
    blt_df,
    title="Threshold sweep: blur_laplacian_threshold",
    current_value=config.blur_laplacian_threshold,
    xlabel="blur_laplacian_threshold",
)

macro_blt = blt_df.groupby("threshold")["f1"].mean().reset_index()
best_blt = macro_blt.loc[macro_blt["f1"].idxmax()]
print(f"Best macro-F1 for blur_laplacian_threshold: {best_blt['f1']:.4f} at threshold={best_blt['threshold']:.0f}")
print(f"Current config: blur_laplacian_threshold={config.blur_laplacian_threshold}")
```

- [ ] **Step 3: Add `min_document_confidence` sweep cells**

**Cell 20 — code (`min_document_confidence` sweep):**
```python
MDC_VALUES = np.round(np.arange(0.40, 0.96, 0.05), 2)
mdc_rows = []

for thresh in MDC_VALUES:
    new_config = dataclasses.replace(config, min_document_confidence=thresh)

    def _pred(row, nc=new_config):
        key = str(row["file_path"])
        cached = RESULT_CACHE.get(key)
        page_images = PAGE_CACHE.get(key)
        if cached is None or page_images is None:
            raise ValueError("not in cache")
        new_pages = []
        for page, page_bgr in zip(cached.pages, page_images):
            new_opencv = opencv_detector.detect(page_bgr, row["file_path"], page.page_number, nc)
            orig_opencv = next((v for v in page.votes if v.detector == "opencv_gate"), None)
            importance = orig_opencv.importance if orig_opencv is not None else 1.0
            new_opencv = dataclasses.replace(new_opencv, importance=importance)
            other_votes = [v for v in page.votes if v.detector != "opencv_gate"]
            new_votes = [new_opencv] + other_votes
            new_pages.append(recompute_page_from_votes(page.page_number, new_votes, nc))
        return _file_label_from_issues([iss for p in new_pages for iss in p.issues])

    valid = _sweep_metrics(gt_df, _pred)
    mdc_rows.extend(_compute_class_metrics(valid, thresh))

mdc_df = pd.DataFrame(mdc_rows)
_plot_threshold_sweep(
    mdc_df,
    title="Threshold sweep: min_document_confidence",
    current_value=config.min_document_confidence,
    xlabel="min_document_confidence",
)

macro_mdc = mdc_df.groupby("threshold")["f1"].mean().reset_index()
best_mdc = macro_mdc.loc[macro_mdc["f1"].idxmax()]
print(f"Best macro-F1 for min_document_confidence: {best_mdc['f1']:.4f} at threshold={best_mdc['threshold']:.2f}")
print(f"Current config: min_document_confidence={config.min_document_confidence}")
```

- [ ] **Step 4: Add recommended config summary cell**

**Cell 21 — markdown:**
```
### Recommended Config

The cells below summarise the best-performing threshold value for each swept parameter.
Combine these only if improvements are independent — validate the combined config with
a fresh run of Section 3.
```

**Cell 22 — code:**
```python
print("Threshold calibration summary")
print("=" * 45)
print(f"  min_reject_confidence:  {config.min_reject_confidence:.2f}  →  best: {best_mrc['threshold']:.2f}  (macro-F1: {best_mrc['f1']:.4f})")
print(f"  blur_laplacian_threshold: {config.blur_laplacian_threshold:.0f}  →  best: {best_blt['threshold']:.0f}  (macro-F1: {best_blt['f1']:.4f})")
print(f"  min_document_confidence: {config.min_document_confidence:.2f}  →  best: {best_mdc['threshold']:.2f}  (macro-F1: {best_mdc['f1']:.4f})")
print()
print("Suggested config (validate with a fresh Section 3 run):")
print(f"""
config = ValidationConfig(
    blur_laplacian_threshold={best_blt['threshold']:.0f},
    blur_tenengrad_threshold={config.blur_tenengrad_threshold},
    min_readability_contrast={config.min_readability_contrast},
    max_low_readability_gray_std={config.max_low_readability_gray_std},
    min_document_confidence={best_mdc['threshold']:.2f},
    min_reject_confidence={best_mrc['threshold']:.2f},
    pdf_dpi={config.pdf_dpi},
)
""")
```

- [ ] **Step 5: Run section 6 end-to-end and confirm**

Run cells 15–22 in order. Confirm:
- All three sweep plots render (2×2 subplots, vertical current-config marker visible)
- Summary cell prints without errors
- No `KeyError` or `ValueError` from the caches

- [ ] **Step 6: Commit**

```bash
git add eval_document_validation.ipynb
git commit -m "feat: add eval notebook sections 4-6 (metrics, ablation, threshold calibration)"
```
