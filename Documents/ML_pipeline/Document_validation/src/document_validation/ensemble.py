from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from document_validation.validator import (
    ValidationConfig,
    _load_image,
    _render_pdf_pages,
    validate_document_image,
)

ISSUE_LABELS = ("blur", "cut", "not_document", "not_clear")
ALL_LABELS = ("accepted", *ISSUE_LABELS)
TIE_BREAK_ORDER = ("not_document", "blur", "not_clear", "cut", "accepted")
DEFAULT_DETECTOR_IMPORTANCE = {
    "opencv": 1.0,
    "opencv_gate": 1.0,
    "doctr": 1.25,
    "deepdoctection": 1.5,
    "deqa_doc": 1.5,
}
DECISION_COLUMNS = ("is_blur", "is_cut", "is_clear", "is_image", "Decision")


@dataclass(frozen=True)
class DetectorVote:
    detector: str
    available: bool
    label: str | None
    importance: float = 0.0
    issues: list[str] = field(default_factory=list)
    confidence: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class EnsemblePageResult:
    page_number: int
    accepted: bool
    label: str
    issues: list[str]
    is_blur: bool
    is_cut: bool
    is_clear: bool
    is_image: bool
    Decision: str
    label_votes: dict[str, int]
    weighted_label_votes: dict[str, float]
    votes: list[DetectorVote]


@dataclass(frozen=True)
class EnsembleFileResult:
    accepted: bool
    issues: list[str]
    file_type: str
    page_count: int
    strategy: str
    pages: list[EnsemblePageResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_decision_rows(self) -> list[dict[str, bool | int | str]]:
        return [
            {
                "page_number": page.page_number,
                "is_blur": page.is_blur,
                "is_cut": page.is_cut,
                "is_clear": page.is_clear,
                "is_image": page.is_image,
                "Decision": page.Decision,
            }
            for page in self.pages
        ]


class PageDetector(Protocol):
    name: str

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        ...


class OpenCvGateDetector:
    name = "opencv_gate"

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        result = validate_document_image(page_bgr, config=config)
        label = _label_from_issues(result.issues)
        return DetectorVote(
            detector=self.name,
            available=True,
            label=label,
            issues=result.issues,
            confidence=_confidence_from_opencv_result(label, result),
            scores={
                "blur_confidence": result.blur.confidence,
                "document_confidence": result.document.confidence,
                "cut_confidence": result.cut.confidence,
                "clear_confidence": result.clear.confidence,
            },
            details=result.to_dict(),
        )


class DoctrDetector:
    """Optional OCR signal from docTR.

    This detector votes `not_document` when very little text is detected and
    `blur` when OCR confidence is weak despite some text-like regions.
    """

    name = "doctr"

    def __init__(self, min_words: int = 3, min_mean_confidence: float = 0.45) -> None:
        self.min_words = min_words
        self.min_mean_confidence = min_mean_confidence
        self._predictor: Any | None = None

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        try:
            predictor = self._get_predictor()
            page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
            exported = predictor([page_rgb]).export()
            word_confidences = _extract_doctr_word_confidences(exported)
            word_count = len(word_confidences)
            mean_confidence = (
                float(np.mean(word_confidences)) if word_confidences else 0.0
            )

            if word_count < self.min_words:
                label = "not_document"
                issues = ["not_document"]
            elif mean_confidence < self.min_mean_confidence:
                label = "blur"
                issues = ["blur"]
            else:
                label = "accepted"
                issues = []

            return DetectorVote(
                detector=self.name,
                available=True,
                label=label,
                issues=issues,
                confidence=round(mean_confidence, 3),
                scores={
                    "word_count": float(word_count),
                    "mean_word_confidence": round(mean_confidence, 3),
                },
            )
        except ImportError as exc:
            return _unavailable_vote(
                self.name,
                "Install the optional docTR dependency: pip install 'python-doctr[torch]'",
                exc,
            )
        except Exception as exc:
            return _error_vote(self.name, exc)

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            from doctr.models import ocr_predictor

            self._predictor = ocr_predictor(pretrained=True)
        return self._predictor


class DeepdoctectionDetector:
    """Optional placeholder for deepdoctection layout/OCR confidence.

    deepdoctection is best integrated as a file-level pipeline. The ensemble
    keeps this adapter explicit so projects can wire their chosen analyzer
    config without making the lightweight validator depend on Detectron2/OCR
    stacks by default.
    """

    name = "deepdoctection"

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        try:
            import deepdoctection  # noqa: F401
        except ImportError as exc:
            return _unavailable_vote(
                self.name,
                "Install and configure deepdoctection before enabling this detector.",
                exc,
            )

        return DetectorVote(
            detector=self.name,
            available=False,
            label=None,
            error=(
                "deepdoctection is installed, but this project needs a configured "
                "analyzer adapter for your model/OCR backend before it can vote."
            ),
        )


class DeqaDocDetector:
    """Optional command adapter for DeQA-Doc or another DIQA model.

    Set DEQA_DOC_COMMAND to a command that accepts an image path as its final
    argument and prints JSON like:
    {"label": "accepted", "issues": [], "confidence": 0.91}
    """

    name = "deqa_doc"

    def __init__(self, command: str | None = None) -> None:
        self.command = command or os.getenv("DEQA_DOC_COMMAND")

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        if not self.command:
            return DetectorVote(
                detector=self.name,
                available=False,
                label=None,
                error=(
                    "Set DEQA_DOC_COMMAND to enable DeQA-Doc voting. The command "
                    "must output JSON with label/issues/confidence."
                ),
            )

        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
                cv2.imwrite(image_file.name, page_bgr)
                completed = subprocess.run(
                    [*self.command.split(), image_file.name],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            import json

            payload = json.loads(completed.stdout)
            label = payload.get("label")
            issues = list(payload.get("issues", []))
            if label not in ALL_LABELS:
                label = _label_from_issues(issues)

            return DetectorVote(
                detector=self.name,
                available=True,
                label=label,
                issues=issues,
                confidence=payload.get("confidence"),
                scores=payload.get("scores", {}),
                details=payload,
            )
        except Exception as exc:
            return _error_vote(self.name, exc)


def validate_document_file_ensemble(
    file_path: str | Path,
    config: ValidationConfig | None = None,
    detectors: list[str] | None = None,
    detector_importance: dict[str, float] | None = None,
) -> EnsembleFileResult:
    """Validate an image/PDF with weighted max-voting across detectors."""

    config = config or ValidationConfig()
    path = Path(file_path)
    page_images, file_type = _load_pages(path, config)
    detector_instances = _build_detectors(detectors)
    importance = _merge_detector_importance(detector_importance)

    pages = [
        _vote_page(page_bgr, path, page_number, config, detector_instances, importance)
        for page_number, page_bgr in enumerate(page_images, start=1)
    ]
    issues = sorted({issue for page in pages for issue in page.issues})

    return EnsembleFileResult(
        accepted=all(page.accepted for page in pages),
        issues=issues,
        file_type=file_type,
        page_count=len(pages),
        strategy="weighted_max_vote",
        pages=pages,
    )


def probe_available_detectors(
    requested: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return detectors that can vote, plus warnings for unavailable ones."""

    requested = requested or ["opencv"]
    available: list[str] = []
    warnings: list[str] = []

    for name in requested:
        if name in {"opencv", "opencv_gate"}:
            available.append("opencv")
            continue
        if name == "doctr":
            try:
                import doctr  # noqa: F401
            except ImportError:
                warnings.append(
                    "doctr skipped: install with pip install -e '.[ensemble]'"
                )
                continue
            available.append("doctr")
            continue
        if name == "deepdoctection":
            try:
                import deepdoctection  # noqa: F401
            except ImportError:
                warnings.append("deepdoctection skipped: not installed")
                continue
            warnings.append(
                "deepdoctection installed but needs a configured analyzer adapter"
            )
            continue
        if name == "deqa_doc":
            if os.getenv("DEQA_DOC_COMMAND"):
                available.append("deqa_doc")
            else:
                warnings.append(
                    "deqa_doc skipped: set DEQA_DOC_COMMAND to enable it"
                )
            continue

        # VLM keys
        from document_validation.vlm_detector import MODEL_REGISTRY as _VLM_REGISTRY
        if name in _VLM_REGISTRY:
            if name == "llava-phi3":
                try:
                    import requests as _req
                    _req.get("http://localhost:11434", timeout=2)
                    available.append(name)
                except Exception:
                    warnings.append(
                        f"{name} skipped: Ollama not running at localhost:11434"
                    )
            else:
                available.append(name)
            continue

    if not available:
        available = ["opencv"]
    return list(dict.fromkeys(available)), warnings


def ensemble_result_to_rows(
    result: EnsembleFileResult,
    file_path: str | Path | None = None,
    include_detector_details: bool = False,
) -> list[dict[str, Any]]:
    """Build flat rows for notebooks/CSV with the requested decision columns."""

    rows: list[dict[str, Any]] = []
    for page in result.pages:
        row: dict[str, Any] = {
            "page_number": page.page_number,
            "is_blur": page.is_blur,
            "is_cut": page.is_cut,
            "is_clear": page.is_clear,
            "is_image": page.is_image,
            "Decision": page.Decision,
        }
        if file_path is not None:
            row = {"file": str(file_path), **row}

        if include_detector_details:
            row["final_label"] = page.label
            row["issues"] = ",".join(page.issues)
            row["raw_votes"] = page.label_votes
            row["weighted_votes"] = page.weighted_label_votes
            for vote in page.votes:
                if not vote.available:
                    continue
                row[f"{vote.detector}_label"] = vote.label
                row[f"{vote.detector}_importance"] = vote.importance
                row[f"{vote.detector}_confidence"] = vote.confidence
                for score_name, score_value in vote.scores.items():
                    row[f"{vote.detector}_{score_name}"] = score_value

        rows.append(row)
    return rows


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


def _load_pages(path: Path, config: ValidationConfig) -> tuple[list[np.ndarray], str]:
    if path.suffix.lower() == ".pdf":
        return list(_render_pdf_pages(path, config)), "pdf"
    return [_load_image(path)], "image"


def _build_detectors(detectors: list[str] | None) -> list[PageDetector]:
    detector_names = detectors or ["opencv", "doctr", "deepdoctection", "deqa_doc"]
    detector_map: dict[str, PageDetector] = {
        "opencv": OpenCvGateDetector(),
        "opencv_gate": OpenCvGateDetector(),
        "doctr": DoctrDetector(),
        "deepdoctection": DeepdoctectionDetector(),
        "deqa_doc": DeqaDocDetector(),
    }
    return [detector_map[name] for name in detector_names]


def _vote_page(
    page_bgr: np.ndarray,
    source_path: Path,
    page_number: int,
    config: ValidationConfig,
    detectors: list[PageDetector],
    detector_importance: dict[str, float],
) -> EnsemblePageResult:
    votes = [
        _with_importance(
            detector.detect(page_bgr, source_path, page_number, config),
            detector_importance,
        )
        for detector in detectors
    ]
    available_votes = [vote for vote in votes if vote.available and vote.label]
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


def _label_from_issues(issues: list[str]) -> str:
    if not issues:
        return "accepted"
    for label in TIE_BREAK_ORDER:
        if label in issues:
            return label
    return issues[0]


def _max_vote_label(label_votes: dict[str, int] | dict[str, float]) -> str:
    max_count = max(label_votes.values())
    tied = {label for label, count in label_votes.items() if count == max_count}
    for label in TIE_BREAK_ORDER:
        if label in tied:
            return label
    return "accepted"


def _with_importance(
    vote: DetectorVote,
    detector_importance: dict[str, float],
) -> DetectorVote:
    importance = _detector_importance(vote.detector, detector_importance)
    if not vote.available or not vote.label:
        importance = 0.0

    return DetectorVote(
        detector=vote.detector,
        available=vote.available,
        label=vote.label,
        importance=importance,
        issues=vote.issues,
        confidence=vote.confidence,
        scores=vote.scores,
        details=vote.details,
        error=vote.error,
    )


def _detector_importance(
    detector: str,
    detector_importance: dict[str, float],
) -> float:
    if detector in detector_importance:
        return float(detector_importance[detector])
    return 1.0


def _merge_detector_importance(
    detector_importance: dict[str, float] | None,
) -> dict[str, float]:
    importance = dict(DEFAULT_DETECTOR_IMPORTANCE)
    if not detector_importance:
        return importance

    for detector, weight in detector_importance.items():
        importance[detector] = float(weight)
        if detector in {"opencv", "opencv_gate"}:
            importance["opencv"] = float(weight)
            importance["opencv_gate"] = float(weight)
    return importance


def _confidence_from_opencv_result(label: str, result: Any) -> float:
    if label == "blur":
        return result.blur.confidence
    if label == "cut":
        return result.cut.confidence
    if label == "not_document":
        return 1.0 - result.document.confidence
    if label == "not_clear":
        return 1.0 - result.clear.confidence
    return min(
        1.0,
        (1.0 - result.blur.confidence)
        * result.document.confidence
        * (1.0 - result.cut.confidence),
    )


def _decision_columns_from_votes(
    votes: list[DetectorVote],
    config: ValidationConfig,
) -> dict[str, bool | str]:
    labels = _strong_reject_labels(votes, config)

    is_blur = "blur" in labels
    is_cut = "cut" in labels
    is_image = "not_document" in labels
    is_clear = not is_blur and "not_clear" not in labels
    accepted = not is_blur and not is_cut and is_clear and not is_image

    return {
        "is_blur": is_blur,
        "is_cut": is_cut,
        "is_clear": is_clear,
        "is_image": is_image,
        "Decision": "accept" if accepted else "reject",
    }


def _strong_reject_labels(
    votes: list[DetectorVote],
    config: ValidationConfig,
) -> set[str]:
    labels: set[str] = set()
    for vote in votes:
        for issue in vote.issues:
            if issue in ISSUE_LABELS and _has_high_issue_confidence(
                vote,
                issue,
                config,
            ):
                labels.add(issue)
        if vote.label in ISSUE_LABELS and _has_high_issue_confidence(
            vote,
            vote.label,
            config,
        ):
            labels.add(vote.label)

    return labels


def _has_high_issue_confidence(
    vote: DetectorVote,
    issue: str,
    config: ValidationConfig,
) -> bool:
    if not vote.available or vote.label == "accepted":
        return False
    confidence = _issue_confidence(vote, issue)
    return confidence is not None and confidence >= config.min_reject_confidence


def _issue_confidence(vote: DetectorVote, issue: str) -> float | None:
    if issue == "blur" and "blur_confidence" in vote.scores:
        return float(vote.scores["blur_confidence"])
    if issue == "cut" and "cut_confidence" in vote.scores:
        return float(vote.scores["cut_confidence"])
    if issue == "not_document" and "document_confidence" in vote.scores:
        return 1.0 - float(vote.scores["document_confidence"])
    if issue == "not_clear" and "clear_confidence" in vote.scores:
        return 1.0 - float(vote.scores["clear_confidence"])
    if vote.label == issue and vote.confidence is not None:
        return float(vote.confidence)
    return None


def _extract_doctr_word_confidences(exported: dict[str, Any]) -> list[float]:
    confidences: list[float] = []
    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    confidence = word.get("confidence")
                    if confidence is not None:
                        confidences.append(float(confidence))
    return confidences


def _unavailable_vote(detector: str, message: str, exc: Exception) -> DetectorVote:
    return DetectorVote(
        detector=detector,
        available=False,
        label=None,
        error=f"{message} ({exc})",
    )


def _error_vote(detector: str, exc: Exception) -> DetectorVote:
    return DetectorVote(
        detector=detector,
        available=False,
        label=None,
        error=str(exc),
    )
