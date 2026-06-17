# VLM Document Validation — Design Spec
**Date:** 2026-06-17
**Status:** Approved

## Background

The current OpenCV-based document validator achieves 37% accuracy on the 319-file test set (macro-F1 0.31). Blur detection was completely broken (F1=0.00) until threshold recalibration, and even after fixing reaches only F1=0.49. The core limitation is that pixel statistics (Laplacian variance, Tenengrad, readability contrast) cannot reliably separate blur from cut or accepted in real-world phone photos of Indian agricultural documents.

This spec covers replacing the OpenCV detector entirely with a mini Vision-Language Model (VLM) that can reason semantically about document image quality.

## Goal

Benchmark 6 mini VLMs against the labeled test set, identify the best performer, and integrate it as a drop-in replacement for `OpenCvGateDetector` in the existing ensemble infrastructure.

## Scope

**In scope:**
- Ground truth loader fix (expanded folder name mapping)
- 72-file stratified benchmark subset
- `vlm_benchmark.py` — unattended multi-model benchmark script
- `vlm_eval.ipynb` — interactive exploration notebook
- `src/document_validation/vlm_detector.py` — `VlmDetector` class implementing `PageDetector` protocol
- Minor extension to `probe_available_detectors` in `ensemble.py`

**Out of scope (Phase 2):**
- Confidence scoring (sampled or logprob-based)
- Fine-tuning any VLM on this dataset
- Multi-VLM ensemble voting

## Ground Truth Loader Fix

The current loader misses ~470 files because Landrecord uses different folder names for cut and accepted. Expanded `FOLDER_TO_LABEL`:

```python
FOLDER_TO_LABEL = {
    "is_blur": "blur",
    "is_clear": "accepted",
    "is_proper_doc": "accepted",
    "is_proper_concent_form": "accepted",
    "is_land_record": "accepted",
    "is_full_concent_form": "accepted",
    "is_not_document": "not_document",
    "is_not_full_document(cut)": "cut",   # ConcentForm naming
    "is_not_full_doc(cut)": "cut",        # Landrecord naming
    "is_not_proper_doc(cut)": "cut",      # Karnataka/WB/MP naming
}
```

`phone_photo` is excluded — quality is ambiguous with no reliable label.

Full labeled set: ~789 files across 10 states, 2 doc types (ConcentForm, Landrecord).

## Benchmark Subset

**Size:** 72 files — 18 per class (accepted, blur, cut, not_document).

**Sampling:** Stratified by doc_type × state × label, random seed fixed. Written once to `eval_results/vlm_subset.csv` and reused across all model runs for a fair comparison.

**Purpose:** Screen all 6 models quickly (~25 min total). Top 2–3 models by macro-F1 proceed to full ~789-file eval.

## Models

| Key | Model | Parameters | Backend |
|---|---|---|---|
| `qwen2-vl-2b` | Qwen2-VL-2B-Instruct | 2B | HuggingFace + MPS |
| `internvl2-2b` | InternVL2-2B | 2B | HuggingFace + MPS |
| `moondream2` | moondream2 | 1.8B | HuggingFace + MPS |
| `smolvlm` | SmolVLM-Instruct | 2B | HuggingFace + MPS |
| `phi35-vision` | Phi-3.5-vision-instruct | 4.2B | HuggingFace + MPS |
| `llava-phi3` | llava-phi3:latest | 4B | Ollama REST API |

All HuggingFace models use `device="mps"` (Apple M4). Models are loaded one at a time and deleted between runs to free memory.

## Prompt

Identical prompt for all models:

```
You are a document quality inspector for Indian agricultural documents.
Classify this image into exactly one of these categories:

accepted      - document is clear, complete and readable
blur          - image is blurry, washed out, low contrast, or text is not legible
cut           - document is physically cut off or edges not fully visible in the frame
not_document  - not a document (random photo, blank page, non-document content)

Reply with ONLY the category name, nothing else.
```

Zero-shot, no examples. Prompt is fixed across all models for a fair comparison.

## Response Parsing

VLMs rarely output exactly one word. Normalisation:
1. Lowercase + strip whitespace
2. Scan for any of the 4 label strings in the response
3. If multiple match, take the first found
4. If none match, return `"accepted"` and log a parse-failure warning

## `vlm_benchmark.py`

Standalone script (not inside `src/`). Usage:

```bash
python vlm_benchmark.py                      # all 6 models on 72-file subset
python vlm_benchmark.py --full qwen2-vl-2b  # one model on full ~789-file set
```

**Loop:**
```python
for model_name, loader in MODELS.items():
    predict_fn = loader()           # loads weights
    for row in subset:
        t0 = time.time()
        label = predict_fn(page_image)
        latency = time.time() - t0
        record(model_name, row, label, latency)
    del predict_fn                  # free memory
```

**Outputs:**
- `eval_results/vlm_benchmark.csv` — columns: `model, file_path, expected_label, predicted_label, latency_s`
- `eval_results/vlm_benchmark_summary.csv` — ranked table: `model, accuracy, macro_f1, accepted_f1, blur_f1, cut_f1, not_document_f1, median_latency_s`
- Prints ranked summary table to stdout on completion

## `vlm_eval.ipynb`

Interactive notebook for exploration and understanding. Sections:

1. **Setup** — choose model key, subset vs full dataset toggle
2. **Ground truth** — expanded loader, file counts per class shown as table
3. **Single image test** — load one file, run model, display raw VLM output + parsed label (for prompt debugging)
4. **Bulk run** — same loop as benchmark script, progress bar, caches to CSV
5. **Metrics** — `classification_report`, confusion matrix heatmap, per-state breakdown
6. **Model comparison** — loads all `vlm_benchmark.csv` results, renders ranked summary

## `VlmDetector` Class

**File:** `src/document_validation/vlm_detector.py`

Implements the existing `PageDetector` protocol so it drops straight into the ensemble:

```python
class VlmDetector:
    name: str   # e.g. "qwen2-vl-2b"

    def __init__(self, model_key: str, confidence_mode: str = "fixed"): ...
    def detect(self, page_bgr, source_path, page_number, config) -> DetectorVote: ...
    def _predict(self, page_bgr: np.ndarray) -> str: ...
```

**Phase 1 (this plan):** `confidence_mode="fixed"` — confidence always 1.0. `min_reject_confidence` in `ValidationConfig` has no effect.

**Phase 2 (future):** `confidence_mode="sampled"` runs N inference passes with temperature > 0, uses label agreement rate as confidence. `confidence_mode="logprob"` reads token log-probabilities where supported.

**Integration — one-line swap:**
```python
# ensemble call, no other changes needed
validate_document_file_ensemble(
    file_path,
    detectors=["qwen2-vl-2b"],
    ...
)
```

`probe_available_detectors` in `ensemble.py` extended to recognise VLM keys and check model availability (HuggingFace cache or Ollama).

## File Layout

```
src/document_validation/
    vlm_detector.py          ← new
    ensemble.py              ← minor: probe_available_detectors extended

vlm_benchmark.py             ← new
vlm_eval.ipynb               ← new
eval_results/
    vlm_subset.csv           ← written on first run, reused
    vlm_benchmark.csv        ← per-model per-file results
    vlm_benchmark_summary.csv
```

**Unchanged:** `validator.py`, `eval_document_validation.ipynb`, `run_eval.py`, `ValidationConfig`. The OpenCV pipeline remains runnable for comparison.

## Success Criteria

- At least one VLM achieves macro-F1 > 0.50 on the 72-file subset (vs 0.31 for OpenCV)
- Blur F1 > 0.60 (current best: 0.49)
- Median latency < 10s/image on Apple M4 (acceptable for batch processing)
- `VlmDetector` passes existing tests in `tests/`
