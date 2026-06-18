from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ValidationConfig:
    """Thresholds are intentionally visible so they can be tuned on real data."""

    blur_laplacian_threshold: float = 900.0
    blur_tenengrad_threshold: float = 60.0
    blur_patch_ratio: float = 0.30
    min_readability_contrast: float = 35.0
    max_low_readability_gray_std: float = 65.0
    min_document_confidence: float = 0.75
    min_reject_confidence: float = 0.75
    min_cut_confidence: float = 0.85
    min_page_area_ratio: float = 0.25
    border_touch_ratio: float = 0.015
    max_safe_border_ink_ratio: float = 0.012
    max_resize_side: int = 1400
    pdf_dpi: int = 200


@dataclass(frozen=True)
class BlurResult:
    is_blurry: bool
    watermark_blur: bool
    laplacian_variance: float
    tenengrad: float
    raw_laplacian_variance: float
    readability_contrast: float
    grayscale_std: float
    confidence: float


@dataclass(frozen=True)
class DocumentResult:
    is_document: bool
    confidence: float
    page_area_ratio: float
    bright_low_saturation_ratio: float
    ink_ratio: float
    edge_density: float
    saturation_p90: float
    page_bbox: tuple[int, int, int, int] | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CutResult:
    is_cut: bool
    confidence: float
    touches_border: bool
    content_touches_border: bool
    border_ink_ratio: float
    page_bbox: tuple[int, int, int, int] | None
    content_bbox: tuple[int, int, int, int] | None
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClearResult:
    is_clear: bool
    is_low_readability: bool
    is_pixelated_text: bool
    readability_contrast: float
    grayscale_std: float
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    issues: list[str]
    blur: BlurResult
    document: DocumentResult
    cut: CutResult
    clear: ClearResult

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_decision_row(self) -> dict[str, bool | str]:
        is_blur = self.blur.is_blurry
        is_cut = self.cut.is_cut
        is_image = not self.document.is_document
        is_clear = self.clear.is_clear
        accepted = not is_blur and not is_cut and is_clear and not is_image
        return {
            "is_blur": is_blur,
            "is_cut": is_cut,
            "is_clear": is_clear,
            "is_image": is_image,
            "Decision": "accept" if accepted else "reject",
        }


@dataclass(frozen=True)
class PageValidationResult:
    page_number: int
    result: ValidationResult


@dataclass(frozen=True)
class FileValidationResult:
    accepted: bool
    issues: list[str]
    file_type: str
    page_count: int
    pages: list[PageValidationResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_decision_rows(self) -> list[dict[str, bool | int | str]]:
        return [
            {
                "page_number": page.page_number,
                **page.result.to_decision_row(),
            }
            for page in self.pages
        ]


def validate_document_image(
    image: str | Path | np.ndarray,
    config: ValidationConfig | None = None,
) -> ValidationResult:
    """Validate whether an image is usable for document LLM verification."""

    config = config or ValidationConfig()
    bgr = _load_image(image)
    bgr = _resize_for_processing(bgr, config.max_resize_side)

    blur = _detect_blur(bgr, config)
    document = _detect_document(bgr, config)
    cut = _detect_cut(bgr, document, config)
    clear = _detect_clear(blur, document, config)

    issues: list[str] = []
    if blur.is_blurry:
        issues.append("blur")
    if not document.is_document:
        issues.append("not_document")
    if document.is_document and not clear.is_clear:
        issues.append("not_clear")
    if cut.is_cut:
        issues.append("cut")

    return ValidationResult(
        accepted=len(issues) == 0,
        issues=issues,
        blur=blur,
        document=document,
        cut=cut,
        clear=clear,
    )


def validate_document_file(
    file_path: str | Path,
    config: ValidationConfig | None = None,
) -> FileValidationResult:
    """Validate an image file or every rendered page in a PDF."""

    config = config or ValidationConfig()
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        page_images = list(_render_pdf_pages(path, config))
        file_type = "pdf"
    else:
        page_images = [_load_image(path)]
        file_type = "image"

    if not page_images:
        raise ValueError(f"no pages found in file: {path}")

    pages = [
        PageValidationResult(
            page_number=idx,
            result=validate_document_image(page_image, config=config),
        )
        for idx, page_image in enumerate(page_images, start=1)
    ]
    issues = sorted({issue for page in pages for issue in page.result.issues})

    return FileValidationResult(
        accepted=all(page.result.accepted for page in pages),
        issues=issues,
        file_type=file_type,
        page_count=len(pages),
        pages=pages,
    )


def _render_pdf_pages(path: Path, config: ValidationConfig) -> list[np.ndarray]:
    import fitz

    zoom = config.pdf_dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[np.ndarray] = []

    with fitz.open(path) as pdf:
        for page in pdf:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )
            if pixmap.n == 1:
                bgr = cv2.cvtColor(rgb[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
            images.append(bgr)

    return images


def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy()
        raise ValueError("image array must be grayscale or BGR/RGB with 3 channels")

    path = Path(image)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image: {path}")
    return bgr


def _resize_for_processing(bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_side:
        return bgr
    scale = max_side / float(largest_side)
    return cv2.resize(
        bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _full_image_sharpness(gray: np.ndarray) -> tuple[float, float]:
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return lap, float(np.mean(np.sqrt(gx * gx + gy * gy)))


def _patch_sharpness_p25(
    gray: np.ndarray, rows: int = 3, cols: int = 3
) -> tuple[float, float]:
    """
    Laplacian variance and Tenengrad at the 25th-percentile across non-blank
    patches.  A sharp watermark inflates full-image metrics but cannot rescue
    the blurry content patches that drag down the percentile.
    """
    h, w = gray.shape
    ph, pw = h // rows, w // cols
    if ph < 40 or pw < 40:
        return _full_image_sharpness(gray)

    lap_vals, ten_vals = [], []
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            if np.mean(patch > 220) > 0.80:  # skip mostly-blank/white patches
                continue
            lap_vals.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            ten_vals.append(float(np.mean(np.sqrt(gx * gx + gy * gy))))

    if len(lap_vals) < 3:
        return _full_image_sharpness(gray)

    return float(np.percentile(lap_vals, 25)), float(np.percentile(ten_vals, 25))


def _detect_blur(bgr: np.ndarray, config: ValidationConfig) -> BlurResult:
    gray_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray_raw)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ink = _ink_mask(gray_raw, hsv[:, :, 2])

    raw_laplacian_variance = float(cv2.Laplacian(gray_raw, cv2.CV_64F).var())
    laplacian_variance, tenengrad = _full_image_sharpness(gray)
    readability_contrast = _readability_contrast(gray_raw, ink)
    grayscale_std = float(gray_raw.std())

    # Watermark-robust check: a clear watermark over blurry content inflates the
    # full-image Laplacian but cannot lift the 25th-percentile patch score.
    patch_lap_p25, _patch_ten_p25 = _patch_sharpness_p25(gray)
    wm_threshold = config.blur_laplacian_threshold * config.blur_patch_ratio
    watermark_masked_blur = (
        laplacian_variance >= config.blur_laplacian_threshold
        and patch_lap_p25 < wm_threshold
    )

    # Require BOTH Laplacian AND Tenengrad to fail — avoids single-metric false positives
    # and keeps rejection confidence high.
    normal_blur = (
        laplacian_variance < config.blur_laplacian_threshold
        and tenengrad < config.blur_tenengrad_threshold
    )
    optical_blur = normal_blur or watermark_masked_blur

    lap_score = _below_threshold_score(laplacian_variance, config.blur_laplacian_threshold)
    ten_score = _below_threshold_score(tenengrad, config.blur_tenengrad_threshold)
    patch_score = _below_threshold_score(patch_lap_p25, wm_threshold)

    if normal_blur:
        # Both metrics agree — confidence is the weaker of the two
        confidence = min(lap_score, ten_score)
    elif watermark_masked_blur:
        confidence = patch_score
    else:
        # Not blurry — report the strongest remaining blur signal (for ensemble use)
        confidence = max(lap_score, ten_score)

    return BlurResult(
        is_blurry=optical_blur,
        watermark_blur=watermark_masked_blur,
        laplacian_variance=round(laplacian_variance, 3),
        tenengrad=round(tenengrad, 3),
        raw_laplacian_variance=round(raw_laplacian_variance, 3),
        readability_contrast=round(readability_contrast, 3),
        grayscale_std=round(grayscale_std, 3),
        confidence=round(confidence, 3),
    )


def _detect_clear(
    blur: BlurResult,
    document: DocumentResult,
    config: ValidationConfig,
) -> ClearResult:
    low_readability = (
        blur.readability_contrast < config.min_readability_contrast
        and blur.grayscale_std < config.max_low_readability_gray_std
        and document.is_document
    )
    # Pixelated text often escapes global blur checks because block edges remain sharp.
    is_pixelated_text = low_readability and not blur.is_blurry
    confidence = 1.0
    reasons: list[str] = []

    if blur.is_blurry:
        confidence = min(confidence, 0.25)
        reasons.append("document is optically blurry")
    if low_readability:
        readability_score = max(
            _below_threshold_score(
                blur.readability_contrast,
                config.min_readability_contrast,
            ),
            _below_threshold_score(
                blur.grayscale_std,
                config.max_low_readability_gray_std,
            ),
        )
        if blur.readability_contrast < config.min_readability_contrast * 0.65:
            readability_score = max(readability_score, 0.75)
        confidence = min(confidence, 1.0 - readability_score)
        reasons.append("printed text is pixelated or not readable")

    return ClearResult(
        is_clear=document.is_document and not blur.is_blurry and not low_readability,
        is_low_readability=low_readability,
        is_pixelated_text=is_pixelated_text,
        readability_contrast=blur.readability_contrast,
        grayscale_std=blur.grayscale_std,
        confidence=round(max(0.0, confidence), 3),
        reasons=reasons,
    )


def _detect_document(bgr: np.ndarray, config: ValidationConfig) -> DocumentResult:
    height, width = bgr.shape[:2]
    image_area = float(height * width)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright_low_sat_mask = cv2.inRange(hsv, (0, 0, 115), (179, 85, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_low_sat_mask = cv2.morphologyEx(bright_low_sat_mask, cv2.MORPH_CLOSE, kernel)
    bright_low_sat_mask = cv2.morphologyEx(bright_low_sat_mask, cv2.MORPH_OPEN, kernel)

    contour = _largest_external_contour(bright_low_sat_mask)
    page_bbox: tuple[int, int, int, int] | None = None
    page_area_ratio = 0.0
    shape_score = 0.0

    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        page_bbox = (int(x), int(y), int(w), int(h))
        page_area_ratio = float(cv2.contourArea(contour) / image_area)
        aspect = w / float(max(h, 1))
        aspect_score = 1.0 if 0.45 <= aspect <= 2.3 else 0.45

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        rectangle_score = 1.0 if len(approx) == 4 else 0.65
        shape_score = min(1.0, page_area_ratio / 0.65) * aspect_score * rectangle_score

    bright_low_saturation_ratio = float(np.mean(bright_low_sat_mask > 0))
    ink_mask = _ink_mask(gray, value)
    ink_ratio = float(np.mean(ink_mask > 0))
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    saturation_p90 = float(np.percentile(saturation, 90))

    page_score = min(1.0, bright_low_saturation_ratio / 0.55)
    ink_score = _range_score(ink_ratio, lower=0.003, upper=0.28, ideal=0.045)
    edge_score = _range_score(edge_density, lower=0.004, upper=0.22, ideal=0.055)
    low_saturation_score = 1.0 - min(1.0, max(0.0, saturation_p90 - 75.0) / 95.0)
    structured_document_score = (
        0.55 * ink_score + 0.30 * edge_score + 0.15 * low_saturation_score
    )

    confidence = (
        0.34 * page_score
        + 0.24 * shape_score
        + 0.18 * ink_score
        + 0.14 * edge_score
        + 0.10 * low_saturation_score
    )
    confidence = max(confidence, 0.78 * structured_document_score)

    reasons: list[str] = []
    if (
        page_area_ratio < config.min_page_area_ratio
        and structured_document_score < 0.68
    ):
        reasons.append("no large white or low-saturation page region found")
    if saturation_p90 > 130:
        reasons.append("image has high color saturation for a document")
    if ink_ratio < 0.003:
        reasons.append("too little text or dark document content found")
    if edge_density < 0.004:
        reasons.append("too few edges for a readable document")

    return DocumentResult(
        is_document=confidence >= config.min_document_confidence,
        confidence=round(float(confidence), 3),
        page_area_ratio=round(page_area_ratio, 3),
        bright_low_saturation_ratio=round(bright_low_saturation_ratio, 3),
        ink_ratio=round(ink_ratio, 4),
        edge_density=round(edge_density, 4),
        saturation_p90=round(saturation_p90, 2),
        page_bbox=page_bbox,
        reasons=reasons,
    )


def _detect_cut(
    bgr: np.ndarray,
    document: DocumentResult,
    config: ValidationConfig,
) -> CutResult:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    margin = max(3, int(min(height, width) * config.border_touch_ratio))

    ink = _ink_mask(gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2])
    border = np.zeros_like(ink, dtype=np.uint8)
    border[:margin, :] = 255
    border[-margin:, :] = 255
    border[:, :margin] = 255
    border[:, -margin:] = 255
    border_pixels = max(int(np.count_nonzero(border)), 1)
    border_ink_ratio = float(np.count_nonzero((ink > 0) & (border > 0)) / border_pixels)
    content_bbox = _main_content_bbox(ink, margin)
    content_margin = max(margin * 2, int(min(height, width) * 0.025))
    content_touches_border = False
    if content_bbox is not None:
        cx, cy, cw, ch = content_bbox
        content_touches_border = (
            cx <= content_margin
            or cy <= content_margin
            or cx + cw >= width - content_margin
            or cy + ch >= height - content_margin
        )

    touches_border = False
    if document.page_bbox is not None:
        x, y, w, h = document.page_bbox
        touches_border = (
            x <= margin
            or y <= margin
            or x + w >= width - margin
            or y + h >= height - margin
        )

    full_frame_document = touches_border and document.page_area_ratio >= 0.95
    suspicious_border_ink = border_ink_ratio > config.max_safe_border_ink_ratio
    border_ink_indicates_cut = suspicious_border_ink and (
        not full_frame_document or content_touches_border
    )

    reasons: list[str] = []
    if touches_border and document.page_area_ratio < 0.93:
        reasons.append("detected page region touches the image border")
    if border_ink_indicates_cut:
        reasons.append("text or strong content appears on the image border")

    confidence = 0.0
    if touches_border and document.page_area_ratio < 0.93:
        confidence += 0.45
    if border_ink_indicates_cut:
        confidence += min(0.55, border_ink_ratio / config.max_safe_border_ink_ratio * 0.25)
    confidence = min(1.0, confidence)

    return CutResult(
        is_cut=confidence >= config.min_cut_confidence,
        confidence=round(confidence, 3),
        touches_border=touches_border,
        content_touches_border=content_touches_border,
        border_ink_ratio=round(border_ink_ratio, 4),
        page_bbox=document.page_bbox,
        content_bbox=content_bbox,
        reasons=reasons,
    )


def _largest_external_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _main_content_bbox(
    ink: np.ndarray,
    border_margin: int,
) -> tuple[int, int, int, int] | None:
    """Find the main content box while ignoring artifacts stuck to page edges."""

    height, width = ink.shape[:2]
    image_area = float(height * width)
    clean = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    content = np.zeros_like(clean)
    min_component_area = max(10, int(image_area * 0.000005))

    for idx in range(1, component_count):
        x, y, w, h, area = stats[idx]
        if area < min_component_area or w < 2 or h < 2:
            continue

        touches_outer_edge = (
            x <= border_margin
            or y <= border_margin
            or x + w >= width - border_margin
            or y + h >= height - border_margin
        )
        if touches_outer_edge:
            continue

        content[labels == idx] = 255

    ys, xs = np.where(content > 0)
    if len(xs) == 0:
        return None

    # Use a robust box so isolated specks near a page edge do not make a
    # full-frame PDF look cropped when the real content has safe margins.
    x0 = int(np.floor(np.percentile(xs, 1)))
    y0 = int(np.floor(np.percentile(ys, 1)))
    x1 = int(np.ceil(np.percentile(xs, 99))) + 1
    y1 = int(np.ceil(np.percentile(ys, 99))) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _ink_mask(gray: np.ndarray, value: np.ndarray) -> np.ndarray:
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )
    dark = cv2.inRange(value, 0, 180)
    return cv2.bitwise_and(adaptive, dark)


def _readability_contrast(gray: np.ndarray, ink: np.ndarray) -> float:
    ink_pixels = ink > 0
    if not np.any(ink_pixels):
        return 0.0

    local_region = cv2.dilate(
        ink,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    ) > 0
    background_pixels = local_region & ~ink_pixels
    if not np.any(background_pixels):
        return 0.0

    return float(gray[background_pixels].mean() - gray[ink_pixels].mean())


def _below_threshold_score(value: float, threshold: float) -> float:
    if value >= threshold:
        return 0.0
    return float(min(1.0, (threshold - value) / max(threshold, 1e-6)))


def _range_score(value: float, lower: float, upper: float, ideal: float) -> float:
    if value < lower or value > upper:
        return 0.0
    if value <= ideal:
        return float((value - lower) / max(ideal - lower, 1e-6))
    return float((upper - value) / max(upper - ideal, 1e-6))
