"""
extract_only.py

Keep ONLY the document-type header, name, and father's name on Indian KYC
images. Every other word, number, date, address, and face is redacted.

Supports: Aadhaar, PAN, Passport, Voter ID, Driving Licence, National ID.
Accepts images (jpg/png/bmp/tiff) and PDFs (each page processed separately).

Usage
-----
    # Single image or PDF
    python extract_only.py aadhar.jpg --out aadhar_clean.jpg
    python extract_only.py eaadhaar.pdf --out aadhar_clean.jpg

    # Batch: all images/PDFs in a directory
    python extract_only.py --dir /path/to/docs --out-dir /path/to/redacted

    # Noisy phone photos
    python extract_only.py card.jpg --backend easyocr --langs en,hi

Output
------
    <stem>.extracted.<ext>   -- redacted image (doc header + name + father only)
    <stem>.extracted.audit.json -- doc_type, name, father_name, word count
"""

import argparse
import json
import os
import re
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import statistics

from redact import (
    OCR_BACKENDS, _LANG_DEFAULTS,
    build_text_and_index, union_box, group_lines,
    detect_faces, redact_region, _tesseract_bin,
)
from detectors import find_entities


def _group_lines_robust(words):
    """Like group_lines but pre-filters pathologically tall boxes.
    Tesseract PSM 11 sometimes emits block-level bounding boxes (e.g., a single
    token with h=267 spanning three card rows) that bridge unrelated text lines
    into one group. Any word taller than 3× the median height is excluded from
    grouping so it cannot stretch the running cur_bottom."""
    heights = [words[i]["box"][3] for i in range(len(words)) if words[i]["box"][3] > 0]
    if not heights:
        return group_lines(words)
    med_h = statistics.median(heights)
    threshold = max(med_h * 3.0, 60)
    normal = [i for i in range(len(words)) if words[i]["box"][3] <= threshold]
    if len(normal) == len(words):
        return group_lines(words)
    fake_lines = group_lines([words[i] for i in normal])
    return {lid: [normal[fi] for fi in idxs] for lid, idxs in fake_lines.items()}


# ─── Document-type signal phrases ─────────────────────────────────────────────

_DOC_PATTERNS = [
    ("AADHAAR",         [r"aadhaar", r"आधार", r"unique\s+identification", r"uidai"]),
    ("PAN",             [r"permanent\s+account", r"income\s+tax"]),
    ("PASSPORT",        [r"passport", r"passeport", r"republic\s+of\s+india",
                         r"भारत\s+गणराज्य", r"भारत\s+गणराज"]),
    ("VOTER_ID",        [r"election\s+commission", r"electors?\s+photo", r"voter\s+id"]),
    ("DRIVING_LICENCE", [r"driving\s+licen[cs]e"]),
    ("NATIONAL_ID",     [r"national\s+id", r"national\s+identity"]),
]


def detect_doc_type(full_text):
    """Return the document-type label, or 'UNKNOWN'.
    First tries header keywords; falls back to entity patterns in the text."""
    nt = full_text.lower()
    for label, patterns in _DOC_PATTERNS:
        if any(re.search(p, nt) for p in patterns):
            return label
    # Fallback: infer from structured IDs found in text.
    entity_label_map = {
        "AADHAAR": "AADHAAR", "VID": "AADHAAR",
        "PAN": "PAN",
        "VOTER_ID": "VOTER_ID",
        "DRIVING_LICENCE": "DRIVING_LICENCE",
    }
    for ent in find_entities(full_text):
        if ent.label in entity_label_map and ent.score >= 0.7:
            return entity_label_map[ent.label]
    return "UNKNOWN"


def header_word_indices(words, lines):
    """Return indices of all words on lines that contain a doc-type signal."""
    keep = set()
    for idxs in lines.values():
        line_text = " ".join(words[i]["text"] for i in idxs).lower()
        for _, patterns in _DOC_PATTERNS:
            if any(re.search(p, line_text) for p in patterns):
                keep.update(idxs)
                break
    return keep


# ─── Unlabeled name detection (modern Aadhaar / cards without "Name:" label) ──

_NON_NAME_RE = re.compile(
    r"(government|india|aadhaar|republic|election|voter|commission|"
    r"authority|identification|unique|address|colony|office|road|"
    r"भारत|सरकार|आधार|पहचान|आयोग|आयकर|विभाग|प्राधिकरण|"
    r"male|female|पुरुष|महिला|dob|pan|"
    r"permanent|account|number|signature|department|ministry|"
    r"income|tax|nationality|expiry|issue|birth)",
    re.IGNORECASE,
)

# Latin name words must contain at least one vowel (filters OCR consonant garbage
# like "ATT", "TRTT" from Kannada/Tamil text misread as Latin clusters).
_HAS_VOWEL_RE = re.compile(r"[AEIOUaeiou]")


def _name_tokens(text):
    """Return the clean name-word tokens from text (skip noise, abbrevs, clusters)."""
    good = []
    for w in text.strip().split():
        w_clean = w.strip().rstrip(",.;:!?")
        if not _NAME_WORD_RE.match(w_clean):
            continue
        if re.match(r"^[A-Za-z]{2}$", w_clean):       # 2-char abbrev (HM, ST)
            continue
        if re.match(r"^[A-Za-z]{3,}$", w_clean) and not _HAS_VOWEL_RE.search(w_clean):
            continue                                    # consonant cluster (ATT, TRTT)
        good.append(w_clean)
    return good


def _is_name_candidate(text):
    """True if text looks like a standalone person name (no labels, no numbers).
    Noise/abbreviation tokens are skipped rather than causing a full-line rejection,
    so 'HM RAHUL SHARMA' still yields a 2-word match.  Long lines (>8 raw tokens)
    are address/header rows, not names.
    Mixed-script lines (Devanagari + Latin): accepted only when the Latin portion
    alone is 2+ tokens, so a single Devanagari noise token ('काम') next to a Latin
    name is transparent rather than a disqualifier."""
    if _NON_NAME_RE.search(text):
        return False
    # Reject lines that contain a digit-leading token (date, ID number, Devanagari
    # digit-noise like "४ञ", etc.).  "Starts with digit" is stricter than
    # "pure numeric" because it also catches mixed tokens like "४ञ" (Devanagari
    # digit + letter) while still accepting "Tadala9i" (Latin letter leading, OCR
    # digit glitch in the middle).  Lines with embedded "LABEL: value" (like
    # "5o: Sidramappa Tadalaai") are handled upstream by keep_by_labels, so
    # rejecting them here is safe.
    for _tok in text.strip().split():
        _t = _tok.strip(",.;:!?")
        if _t and re.match(r'^\d', _t):
            return False
    # Long lines are addresses, not names
    if len(text.strip().split()) > 8:
        return False
    good = _name_tokens(text)
    if not (2 <= len(good) <= 5):
        return False
    deva = [w for w in good if re.match(r'^[ऀ-ॿ]+$', w)]
    latin = [w for w in good if re.match(r'^[A-Za-z]', w)]
    # Mixed-script line: accept only when the Latin portion alone forms a 2+ word name
    if deva and latin:
        return len(latin) >= 2
    return True


def keep_name_candidates(words, lines):
    """Return (indices, texts) of line-groups that look like person names.
    Used when label-anchored detection finds nothing (no 'Name:' on the card).
    Returns the cleaned text (noise tokens removed) so that 'HM RAHUL SHARMA'
    is reported as 'RAHUL SHARMA'.  For mixed-script lines where the Latin portion
    is the name, only the Latin tokens are used in the display text."""
    idxs, texts = set(), []
    for line_idxs in lines.values():
        line_text = " ".join(words[i]["text"] for i in line_idxs)
        if _is_name_candidate(line_text):
            idxs.update(line_idxs)
            good = _name_tokens(line_text)
            deva = [w for w in good if re.match(r'^[ऀ-ॿ]+$', w)]
            latin = [w for w in good if re.match(r'^[A-Za-z]', w)]
            # For mixed-script lines keep only the Latin portion as display text
            clean = " ".join(latin if (deva and latin) else good)
            texts.append(clean)
    return idxs, texts


# ─── Label-anchored keep (name + father) ──────────────────────────────────────

_KEEP_LABELS = {
    "NAME":   ["name", "नाम", "surname", "उपनाम", "given", "दिया"],
    "PARENT": ["father", "fathers", "s/o", "d/o", "w/o", "c/o", "पिता",
               "husband", "mother", "guardian", "care of"],
}
_KEEP_LOOKUP = {kw: lab for lab, kws in _KEEP_LABELS.items() for kw in kws}

# Labels that are only matched when they appear as the SOLE word on a line.
# "To" introduces the addressee block in Aadhaar letter-format printouts:
#   To
#   Lingaraj Ramappa Neshvi
#   S/O Ramappa
# Adding it to _KEEP_LOOKUP would cause false positives everywhere ("to" is
# common English), so it lives here and is checked separately in keep_by_labels.
_STANDALONE_LABELS = {"to": "NAME"}

# Extended lookup used ONLY when matching an embedded "PREFIX: value" segment.
# Shorter slash-free aliases (so, do, wo, co) are safe here because they are
# anchored by the colon separator — standalone OCR noise never reaches this path.
_KEEP_LOOKUP_EMBEDDED = {
    **_KEEP_LOOKUP,
    "so": "PARENT",   # s/o with "/" dropped by OCR
    "do": "PARENT",   # d/o
    "wo": "PARENT",   # w/o
    "co": "PARENT",   # c/o
}
# Python \w does not match Devanagari combining marks (vowel signs Mc category),
# so 'नाम' → 'नम' and 'पिता' → 'पत' if we use [^\w/].  Explicitly allow the
# full Devanagari block (U+0900–U+097F) so matras are preserved.
_norm_token = lambda t: re.sub(r"[^a-zA-Z0-9ऀ-ॿ/]", "", t.lower())

# OCR digit-for-letter confusion fixes for label matching only (not for name tokens).
_LABEL_OCR_FIX = str.maketrans("5018", "soib")


def _resolve_label(word_text):
    """Match a word (or EasyOCR segment) against the label lookup.

    Returns (label_key, embedded_value) where:
      - label_key is None if no label matched
      - embedded_value is a string when the label AND value were in the same
        EasyOCR segment (e.g. '5o: Sidramappa Tadalaai'), None otherwise
        (value comes from subsequent words, handled by the caller).

    Handles three cases in order:
      1. Direct match (e.g. 'Name:', 'S/O:')
      2. Digit-fixed direct match (e.g. '5/O:' → 's/o')
      3. Embedded 'LABEL: value' segment: split at ':', try prefix as label
         with and without OCR digit fix and without the slash (so 's/o' → 'so').
    """
    n = _norm_token(word_text)
    # Case 1: direct
    lab = _KEEP_LOOKUP.get(n)
    if lab:
        return lab, None
    # Case 2: digit-fixed
    lab = _KEEP_LOOKUP.get(n.translate(_LABEL_OCR_FIX))
    if lab:
        return lab, None
    # Case 3: embedded "PREFIX: value" in a single EasyOCR segment.
    # Uses _KEEP_LOOKUP_EMBEDDED (includes slash-free aliases like "so" for "s/o")
    # because the colon anchor makes false positives rare here.
    # Restricted to Latin-script prefixes only: Devanagari labels (नाम:, पिता:)
    # are always standalone OCR tokens and are matched by Cases 1/2 above.
    # This prevents EasyOCR grouping "नाम: [Hindi-name]" from overriding the
    # fallback name-candidate path that finds the clean Latin-script name.
    if ":" in word_text:
        colon_i = word_text.index(":")
        prefix = word_text[:colon_i].strip()
        suffix = word_text[colon_i + 1:].strip()
        if prefix and suffix and not re.search(r'[ऀ-ॿ]', prefix):
            p_norm = _norm_token(prefix)
            for candidate in [
                p_norm,
                p_norm.replace("/", ""),
                p_norm.translate(_LABEL_OCR_FIX),
                p_norm.replace("/", "").translate(_LABEL_OCR_FIX),
            ]:
                lab = _KEEP_LOOKUP_EMBEDDED.get(candidate)
                if lab:
                    val_toks = [w.rstrip(",.;:!?") for w in suffix.split()
                                if _NAME_WORD_RE.match(w.rstrip(",.;:!?"))]
                    return lab, " ".join(val_toks) if val_toks else None
    return None, None


_LABEL_ONLY_RE = re.compile(
    r"^(name[\(s\)]*|names|given|surname|उपनाम|दिया|नाम|"
    r"का|की|के|है|गया|"    # Hindi particles that appear in label phrases
    r"type|code|nationality|sex|date|place|expiry|issue|birth|of|/|namets\)?)$",
    re.IGNORECASE,
)

# A word is a valid name component if it's:
#   all-caps Latin (KUMAR, SPECIMEN, G), or
#   Devanagari (फिरदोस), or
#   contains only alphabetic + space chars (no trailing commas, dots, ~)
_NAME_WORD_RE = re.compile(
    r"^[A-Z][A-Z\-\’’]+$"          # all-caps Latin (2+ chars) + curly apostrophe
    r"|^[A-Z][a-z][a-z\-\’’]*$"    # Title-case: Sanket, Sarvabhoum, Bagali
    r"|^[A-Z]$"                      # single uppercase initial
    r"|^[ऀ-ॿ]+$"                   # Devanagari
)


def _value_words(words, idxs, label_filter=True):
    """Return only word indices that look like valid name tokens.
    label_filter=False skips the LABEL_ONLY_RE check — use for next-line values
    where the entire line is a value (e.g. 'APPLICANT NAME' on a PAN card, where
    'NAME' is part of the name, not a label)."""
    return [
        k for k in idxs
        if words[k]["box"][3] >= 5          # discard sub-pixel noise boxes
        and (not label_filter or not _LABEL_ONLY_RE.match(words[k]["text"].strip()))
        and _NAME_WORD_RE.match(words[k]["text"].strip().rstrip(",.;:!?"))
    ]


def keep_by_labels(words, lines):
    """
    Return (keep_idx: set[int], found: dict[str, str]).
    Handles both same-line values (Aadhaar: "Name: Ravi Kumar") and
    next-line values (passport: "Surname\\nSPECIMEN").
    """
    keep_idx = set()
    found = {}
    line_index = sorted(lines.keys())
    # Lines already consumed as value lines — skip them to prevent a word like
    # 'NAME' inside a value (e.g. "APPLICANT NAME") from re-triggering as a label.
    value_lines = set()

    for li in line_index:
        if li in value_lines:
            continue
        idxs = lines[li]
        for pos, wi in enumerate(idxs):
            lab, embedded_val = _resolve_label(words[wi]["text"])
            if not lab:
                # Standalone-only labels (e.g. "To" in Aadhaar letter format):
                # only match when this word is the sole token on the line.
                if len(idxs) == 1:
                    lab = _STANDALONE_LABELS.get(
                        _norm_token(words[wi]["text"]))
                    embedded_val = None
            if not lab:
                continue

            if embedded_val is not None:
                # Embedded case: label and value were in the same EasyOCR segment
                # (e.g. '5o: Sidramappa Tadalaai' → PARENT = 'Sidramappa Tadalaai').
                keep_idx.add(wi)
                if embedded_val:
                    if lab in found:
                        found[lab] = found[lab] + " " + embedded_val
                    else:
                        found[lab] = embedded_val
                break

            # Try same-line value first (words after the label on this line).
            inline = _value_words(words, idxs[pos + 1:])

            if inline:
                keep_idx.update(inline)
                val_text = " ".join(
                    words[k]["text"].rstrip(",.;:!?") for k in inline)
            else:
                # Passport style: value is on the NEXT line.
                next_idxs = lines.get(li + 1, [])
                # Only use next line if it doesn't start with another label.
                next_lab, _ = _resolve_label(
                    words[next_idxs[0]]["text"]) if next_idxs else (None, None)
                if next_idxs and not next_lab:
                    useful = _value_words(words, next_idxs, label_filter=False)
                    keep_idx.update(useful)
                    val_text = " ".join(
                        words[k]["text"].rstrip(",.;:!?") for k in useful)
                    value_lines.add(li + 1)  # consumed; don't re-scan for labels
                else:
                    val_text = None

            if val_text:
                # Accumulate multiple NAME fields (Surname + Given Name).
                if lab in found:
                    found[lab] = found[lab] + " " + val_text
                else:
                    found[lab] = val_text
            break
    return keep_idx, found


# ─── VLM fallback (Ollama) ────────────────────────────────────────────────────

_VLM_PROMPT = (
    "This is an Indian government ID card (Aadhaar, PAN, Passport, Voter ID, "
    "or Driving Licence). The image may be rotated or low quality. "
    "Identify:\n"
    "1. The card holder's full name (Latin/English script only)\n"
    "2. The father's, mother's, or spouse's name "
    "(look for S/O: D/O: W/O: Father: पिता: Husband: or similar labels)\n\n"
    "Reply in EXACTLY this format, nothing else:\n"
    "NAME: <full name>\n"
    "PARENT: <parent or spouse name, or None>\n"
)


def _vlm_extract_names(img_bgr, model="moondream", host="http://localhost:11434"):
    """Call an Ollama vision model to extract name + parent from an ID card.

    Returns a dict with "NAME" and/or "PARENT" keys (only fields that were
    found).  Returns {} on any error so callers can treat it as a no-op."""
    import base64, json, urllib.request, urllib.error

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return {}
    b64 = base64.b64encode(buf.tobytes()).decode()

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": _VLM_PROMPT, "images": [b64]}],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {}

    text = (data.get("message") or {}).get("content") or data.get("response", "")
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("NAME:"):
            val = line[5:].strip()
            if val and val.lower() not in ("none", "n/a", "-", "unknown", ""):
                result["NAME"] = val
        elif line.upper().startswith("PARENT:"):
            val = line[7:].strip()
            if val and val.lower() not in ("none", "n/a", "-", "unknown", ""):
                result["PARENT"] = val
    return result


def _vlm_keep_idx(words, vlm_found):
    """Return word indices whose text matches a token in VLM-extracted names.
    Keeps those words visible when VLM fills in what OCR missed."""
    name_toks = set()
    for v in vlm_found.values():
        if isinstance(v, str):
            for tok in v.split():
                t = tok.strip(",.;:!?").lower()
                if len(t) >= 3:
                    name_toks.add(t)
    keep = set()
    for i, wd in enumerate(words):
        wt = wd["text"].strip().rstrip(",.;:!?").lower()
        if len(wt) >= 3 and wt in name_toks:
            keep.add(i)
    return keep


# ─── QR / barcode locator ─────────────────────────────────────────────────────

def _find_qr_boxes(img, upscale=4, pad_frac=0.30):
    """
    Return a list of (x,y,w,h) boxes covering any QR codes found.
    Strategy (layered, stops at first success):
      1. zxing-cpp decoder (fast, accurate when readable)
      2. OpenCV QRCodeDetector on upscaled image
      3. Finder-pattern heuristic: look for triple of nested-square contours
    """
    import zxingcpp

    h_orig, w_orig = img.shape[:2]
    results = []

    # --- Layer 1: zxing-cpp ---
    for scale in (1, 2, 4):
        up = cv2.resize(img, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC) if scale > 1 else img
        for r in zxingcpp.read_barcodes(up):
            pts = r.position
            xs = [pts.top_left.x, pts.top_right.x,
                  pts.bottom_right.x, pts.bottom_left.x]
            ys = [pts.top_left.y, pts.top_right.y,
                  pts.bottom_right.y, pts.bottom_left.y]
            x0 = int(min(xs) / scale); y0 = int(min(ys) / scale)
            x1 = int(max(xs) / scale); y1 = int(max(ys) / scale)
            results.append((x0, y0, x1 - x0, y1 - y0))
        if results:
            return results

    # --- Layer 2: OpenCV QRCodeDetector ---
    qr_det = cv2.QRCodeDetector()
    for scale in (2, 4):
        up = cv2.resize(img, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, pts, _ = qr_det.detectAndDecode(bw)
        if pts is not None:
            xs = pts[0, :, 0]; ys = pts[0, :, 1]
            x0 = int(xs.min() / scale); y0 = int(ys.min() / scale)
            x1 = int(xs.max() / scale); y1 = int(ys.max() / scale)
            return [(x0, y0, x1 - x0, y1 - y0)]

    # --- Layer 3: finder-pattern heuristic ---
    up = cv2.resize(img, None, fx=upscale, fy=upscale,
                    interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hier = cv2.findContours(bw, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]

    # Collect candidate nested-square centroids.
    centres = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if not (0.65 < cw / (ch + 1e-5) < 1.55):
            continue
        # Must have both a child and a parent (middle ring of finder pattern).
        if hier[i][2] >= 0 and hier[i][3] >= 0:
            cx = (x + cw / 2) / upscale
            cy = (y + ch / 2) / upscale
            centres.append((cx, cy, cw / upscale, ch / upscale, area))

    if len(centres) < 3:
        return []

    # Cluster: pick the densest group of ≥3 centres.
    centres.sort(key=lambda v: v[4], reverse=True)
    best = None
    best_spread = float("inf")
    for i in range(len(centres) - 2):
        cluster = centres[i : i + 4]
        xs = [c[0] for c in cluster]; ys = [c[1] for c in cluster]
        spread = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if spread < best_spread:
            best_spread = spread
            best = cluster

    if best is None or best_spread > max(w_orig, h_orig) * 0.5:
        return []

    xs = [c[0] for c in best]; ys = [c[1] for c in best]
    sw = [c[2] for c in best]; sh = [c[3] for c in best]
    x0 = int(min(xs) - max(sw))
    y0 = int(min(ys) - max(sh))
    x1 = int(max(xs) + max(sw) * 2)
    y1 = int(max(ys) + max(sh) * 2)
    pad_x = int((x1 - x0) * pad_frac)
    pad_y = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(w_orig, x1 + pad_x); y1 = min(h_orig, y1 + pad_y)
    return [(x0, y0, x1 - x0, y1 - y0)]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = os.path.join(_HERE, "face_detection_yunet_2023mar.onnx")

# If the long edge is below this, OCR text is likely unreliable.
_MIN_RELIABLE_PX = 600


def _crop_to_card(img, threshold=200, pad=40):
    """Crop a document-page PDF scan to the union bounding box of all
    significant non-white regions.  Using the UNION (not just the largest
    contour) handles two-sided Aadhaar scans where the front card has a white
    background and the back card has a colourful one — both are included.
    For real-world photos taken on dark backgrounds (gravel, table, etc.) the
    logic is inverted: we find the bright card area instead of non-white content.
    Returns (cropped_img, (ocr_x0, ocr_y0), (card_x0, card_y0, card_x1, card_y1)).
    ocr_x0/y0 is the crop offset used for word-box mapping (0,0 when not cropping).
    card_x0/y0/x1/y1 is the detected card extent in original-image coordinates
    (always set, even when no crop is applied — useful for full_redact blanket)."""
    import numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect background type: clean scan (bright bg) vs real-world photo (dark bg)
    white_frac = float((gray > 220).mean())
    oh, ow = img.shape[:2]

    if white_frac < 0.35:
        # Real-world photo on dark background (gravel, table, etc.).
        # Locate the card using row/column mean brightness.  PDF page margins
        # can produce thin pure-white strips at the edges; isolated strips are
        # excluded by keeping only the LARGEST contiguous bright band on each axis.
        card_thresh = 120

        def _largest_bright_band(means):
            """Return (start, end) indices of the longest run of means > card_thresh."""
            bright = means > card_thresh
            if not bright.any():
                return 0, len(means)
            changes = np.diff(bright.astype(np.int8))
            starts = list(np.where(changes == 1)[0] + 1)
            ends   = list(np.where(changes == -1)[0] + 1)
            if bright[0]:  starts = [0] + starts
            if bright[-1]: ends   = ends + [len(means)]
            lengths = [e - s for s, e in zip(starts, ends)]
            best = int(np.argmax(lengths))
            return starts[best], ends[best]

        row_means = gray.mean(axis=1)
        ry0, ry1 = _largest_bright_band(row_means)
        if ry1 <= ry0:
            return img, (0, 0), (0, 0, ow, oh)
        # Use full image width — the card spans the full width after rotation;
        # column-mean detection would cut off darker areas (e.g. the photo region).
        y0 = max(0, ry0 - pad); y1 = min(oh, ry1 + pad)
        x0 = 0; x1 = ow
    else:
        # Clean document scan: find union of all non-white content regions.
        # UNION (not just largest) handles two-sided Aadhaar scans where the front
        # card (white bg) and back card (colourful) are separate contours.
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, (0, 0), (0, 0, ow, oh)
        min_area = oh * ow * 0.005
        xs, ys, x1s, y1s = [], [], [], []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            cx, cy, cw, ch = cv2.boundingRect(c)
            xs.append(cx); ys.append(cy); x1s.append(cx + cw); y1s.append(cy + ch)
        if not xs:
            cx, cy, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            xs, ys, x1s, y1s = [cx], [cy], [cx + cw], [cy + ch]
        x0 = max(0, min(xs) - pad); y0 = max(0, min(ys) - pad)
        x1 = min(ow, max(x1s) + pad); y1 = min(oh, max(y1s) + pad)

    # Only crop when there is a meaningful margin (≥10% on at least one side).
    card_bounds = (x0, y0, x1, y1)
    if x0/ow < 0.10 and y0/oh < 0.10 and (ow-x1)/ow < 0.10 and (oh-y1)/oh < 0.10:
        return img, (0, 0), card_bounds
    return img[y0:y1, x0:x1], (x0, y0), card_bounds


def _preprocess(img):
    """CLAHE + unsharp mask to improve contrast before OCR."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    return cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)


def _tesseract_osd_angle(img):
    """Use Tesseract OSD (fast, no text extraction) to get suggested rotation.
    Returns degrees to rotate CCW, or 0 if detection fails."""
    import subprocess, tempfile
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=os.getcwd())
    os.close(fd)
    try:
        cv2.imwrite(tmp, small)
        proc = subprocess.run(
            [_tesseract_bin(), tmp, "stdout", "--psm", "0", "-l", "eng"],
            capture_output=True, text=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("Rotate:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return 0


_CV_ROTATE = {
    90:  cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_CLOCKWISE,
}


def _quick_word_count(img, scale=0.5):
    """Count high-confidence words from a Tesseract pass (fast quality probe)."""
    import subprocess, tempfile
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=os.getcwd())
    os.close(fd)
    count = 0
    try:
        cv2.imwrite(tmp, small)
        proc = subprocess.run(
            [_tesseract_bin(), tmp, "stdout", "-l", "eng",
             "--oem", "1", "--psm", "11", "tsv"],
            capture_output=True, text=True,
        )
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) > 11:
                txt = parts[11].strip()
                try:
                    conf = float(parts[10])
                except ValueError:
                    continue
                if txt and conf >= 50:
                    count += 1
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return count


def _auto_rotate(img):
    """Correct image orientation for portrait-captured ID cards.
    Only attempts rotation when the image is portrait (h > w) — landscape
    images are already correctly oriented and returned unchanged.
    Returns (rotated_img, rotation_applied_degrees)."""
    h, w = img.shape[:2]

    # Already landscape (or square) — no rotation needed.
    if w >= h:
        return img, 0

    # Portrait image: try Tesseract OSD first.
    angle = _tesseract_osd_angle(img)
    if angle in _CV_ROTATE:
        return cv2.rotate(img, _CV_ROTATE[angle]), angle

    # OSD inconclusive — probe original orientation plus both 90° rotations
    # and pick the one with the most high-confidence Tesseract words.
    # Include 0° so portrait documents (Aadhaar letters, scanned pages) that
    # happen to be narrower than they are tall are never wrongly rotated.
    count_0   = _quick_word_count(img)
    ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cw  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    count_ccw = _quick_word_count(ccw)
    count_cw  = _quick_word_count(cw)
    best = max(count_0, count_ccw, count_cw)
    if best == count_0:
        return img, 0
    if count_ccw >= count_cw:
        return ccw, 270
    return cw, 90


def run(in_path, out_path, backend="tesseract", langs=None, mode="blackout",
        model_path=_DEFAULT_MODEL, redact_faces=True, _img=None,
        _skip_rotate=False, vlm_model=None, vlm_host="http://localhost:11434",
        full_redact=False):
    """
    Returns (result_dict, audit_log_path).
    result_dict keys: doc_type, name, father_name, words_ocr, redactions,
                      warnings (list of str).
    Pass _img (BGR ndarray) to skip disk read (used for PDF page processing).
    Pass _skip_rotate=True for PDF-derived images (already in correct orientation;
    page aspect ratio would otherwise trigger the ID-card rotation heuristic).
    """
    if _img is not None:
        img = _img
    else:
        img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"Could not read image: {in_path}")

    # Auto-rotate portrait images to landscape (ID cards are landscape).
    # Skip for clean-scan PDF pages (already correctly oriented).
    orig_img = img  # keep pre-crop image; word-box offsets are relative to this
    if _skip_rotate:
        rotation = 0
    else:
        img, rotation = _auto_rotate(img)
        orig_img = img      # rotated image IS the output canvas
    # Crop to card region — handles both document-page PDFs (card in white A4 margin)
    # and real-world photos (card against dark gravel/table background).
    img, _crop_offset, _card_bounds = _crop_to_card(img)

    # Indian ID documents always contain Hindi — use eng+hin for Tesseract by default.
    if langs is None:
        langs = "eng+hin" if backend == "tesseract" else "en,hi"
    warnings = []

    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge < _MIN_RELIABLE_PX:
        warnings.append(
            f"Image is only {w}×{h}px — OCR accuracy may be poor. "
            f"Provide a scan or photo with long edge ≥{_MIN_RELIABLE_PX}px."
        )

    # Upscale to 3200px for genuinely low-res images; 1600px for medium-res.
    # CLAHE preprocessing only helps noisy/low-contrast photos — applying it to
    # clean digital images (screenshots, e-docs) amplifies watermark artifacts
    # and degrades OCR.  Only preprocess when the original was below threshold.
    target_px = 3200 if long_edge < _MIN_RELIABLE_PX else 1600
    scale = 1.0
    if long_edge < target_px:
        scale = target_px / long_edge
        ocr_img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        if long_edge < _MIN_RELIABLE_PX:   # only preprocess truly low-res images
            ocr_img = _preprocess(ocr_img)
    else:
        ocr_img = img

    words = OCR_BACKENDS[backend](ocr_img, langs)
    cx0, cy0 = _crop_offset
    for wd in words:
        x, y, bw, bh = wd["box"]
        # Scale back from OCR resolution, then add crop offset so boxes are
        # in the coordinate space of the original (pre-crop) image.
        wd["box"] = (int(x / scale) + cx0, int(y / scale) + cy0,
                     int(bw / scale), int(bh / scale))

    # Measure average OCR confidence using only words Tesseract is somewhat
    # confident about (≥30%).  Very-low-confidence reads are scanner noise /
    # white-margin artifacts on document-page scans and inflate the error rate
    # even when the actual card text is perfectly legible.
    reliable = [wd["conf"] for wd in words if wd["conf"] >= 0.30]
    avg_conf = (sum(reliable) / len(reliable)) if reliable else 0.0

    # Auto-fallback to EasyOCR when Tesseract produces low-quality results.
    # Triggers when Tesseract finds almost nothing (len<5) OR has low avg confidence
    # (<0.60) — the latter catches photos of physical cards where Tesseract misreads
    # Kannada/Tamil script as garbled Hindi, giving many fake "reliable" words.
    # Only switch if EasyOCR achieves higher avg confidence than Tesseract.
    _used_backend = backend
    if (len(reliable) < 5 or avg_conf < 0.60) and backend == "tesseract":
        try:
            easy_langs = "en,hi" if langs == "eng+hin" else langs.replace("+", ",")
            easy_words = OCR_BACKENDS["easyocr"](ocr_img, easy_langs)
            easy_rel = [w for w in easy_words if w["conf"] >= 0.30]
            easy_avg = (sum(w["conf"] for w in easy_rel) / len(easy_rel)) if easy_rel else 0.0
            if easy_avg > avg_conf:
                cx0e, cy0e = _crop_offset
                for wd in easy_words:
                    x, y, bw, bh = wd["box"]
                    wd["box"] = (int(x / scale) + cx0e, int(y / scale) + cy0e,
                                 int(bw / scale), int(bh / scale))
                words = easy_words
                reliable = [wd["conf"] for wd in words if wd["conf"] >= 0.30]
                avg_conf = easy_avg
                _used_backend = "easyocr(auto)"
        except Exception:
            pass  # EasyOCR not installed or failed — keep Tesseract result

    low_conf = avg_conf < 0.45

    full_text, _ = build_text_and_index(words)
    doc_type = detect_doc_type(full_text)
    lines = _group_lines_robust(words)

    # Decide what to keep.
    if full_redact:
        # Full-redact mode: keep nothing — redact all text, faces, QR codes.
        # Low-confidence pages also get a card-region blanket (applied below in the
        # redaction block) so upside-down / unreadable text can't leak through.
        keep_idx = set()
        found_labels = {}
    else:
        keep_idx = header_word_indices(words, lines)
        label_keep, found_labels = keep_by_labels(words, lines)
        keep_idx |= label_keep

        # Fallback: if NAME not found (modern Aadhaar / unlabeled format),
        # keep any text block that looks like a person name.
        if "NAME" not in found_labels:
            name_idx, name_texts = keep_name_candidates(words, lines)
            keep_idx |= name_idx
            if name_texts:
                # Detect bilingual format: one candidate in Devanagari, one in Latin
                # (Aadhaar prints the same name in both scripts on adjacent lines).
                # In that case both belong to NAME.  If both are Latin (old PAN/DL
                # layout), the first is the person's name and the second is the
                # father's name.
                _deva = [t for t in name_texts if re.search(r'[ऀ-ॿ]', t)]
                _latin = [t for t in name_texts if re.match(r'^[A-Za-z\' \-]+$', t)]
                if _deva and _latin:
                    # Bilingual — all candidates are the same person
                    found_labels["NAME"] = " / ".join(name_texts[:2])
                    if len(name_texts) > 2:
                        found_labels["NAME"] += " / " + " / ".join(name_texts[2:])
                else:
                    # Unlabeled sequential layout (old PAN/DL): line 1 = name, line 2 = father
                    found_labels["NAME"] = name_texts[0]
                    if len(name_texts) > 1 and "PARENT" not in found_labels:
                        found_labels["PARENT"] = name_texts[1]
                    if len(name_texts) > 2:
                        found_labels["NAME"] += " / " + " / ".join(name_texts[2:])

        # VLM fallback: call Ollama when OCR-based extraction missed the name.
        # Fires when vlm_model is set AND primary pipeline found no name
        # (which happens on Kannada/Tamil cards, dark photos, etc.).
        # Also fires on low-confidence reads even if a name was tentatively found,
        # since VLM can correct OCR hallucinations.
        _vlm_used = False
        if vlm_model and ("NAME" not in found_labels or avg_conf < 0.60):
            vlm_result = _vlm_extract_names(ocr_img, vlm_model, vlm_host)
            if vlm_result:
                for k, v in vlm_result.items():
                    if k not in found_labels or not found_labels[k]:
                        found_labels[k] = v
                _vlm_used = True
                _used_backend = _used_backend + f"+vlm({vlm_model})"
                warnings.append(f"VLM fallback used: {vlm_model}")
                # Try to find OCR word boxes that match the VLM-extracted names so
                # they remain visible in the output image rather than being redacted.
                keep_idx |= _vlm_keep_idx(words, vlm_result)

    redactions = []
    audit_items = []

    # full_redact → always apply card-region blanket (keeps page background).
    # Extend upward by 10% of page height so white card margins missed by the
    # contour-based crop (e.g. upside-down cards with blank top strips) are covered.
    # Always-on in full_redact mode also handles barcodes that OCR/QR detection miss.
    if full_redact:
        cih, ciw = img.shape[:2]
        oh = orig_img.shape[0]
        up_pad = min(int(cy0), max(300, int(oh * 0.10)))
        bx0, by0 = int(cx0), max(0, int(cy0) - up_pad)
        card_box = (bx0, by0, int(ciw), int(cy0) - by0 + int(cih))
        redactions.append(card_box)
        audit_items.append({"type": "CARD_BLANKET", "box": list(card_box)})

    nothing_kept = not found_labels and not (keep_idx - header_word_indices(words, lines))
    if not full_redact and (low_conf or (nothing_kept and doc_type == "UNKNOWN")):
        # Apply safety blanket when:
        #   (a) OCR confidence too low to trust word boxes, OR
        #   (b) Nothing useful found (no name/father, no doc type) — nothing to preserve.
        reason = (f"OCR confidence too low (avg={avg_conf:.2f})"
                  if low_conf else
                  "no name/father/doc-type found — nothing to preserve")
        warnings.append(
            f"{reason}. Applying full-image safety redaction."
        )
        pad = 4
        oh, ow = orig_img.shape[:2]
        full_box = (pad, pad, ow - 2 * pad, oh - 2 * pad)
        redactions.append(full_box)
        audit_items.append({"type": "SAFETY_BLANKET", "box": list(full_box)})
    else:
        # Redact every word NOT in keep_idx.
        for i, wd in enumerate(words):
            if i not in keep_idx:
                redactions.append(wd["box"])
                audit_items.append({"type": "WORD", "text": wd["text"],
                                    "box": [int(v) for v in wd["box"]]})

    # Redact QR / barcodes (encode full PII — always redact).
    for qr_box in _find_qr_boxes(orig_img):
        redactions.append(qr_box)
        audit_items.append({"type": "QR_CODE", "box": [int(v) for v in qr_box]})

    # Redact MRZ lines (passport bottom zone: P<<NAME..., Z99999...).
    # Detect as OCR word boxes whose text matches MRZ character patterns.
    _MRZ_RE = re.compile(r'^[A-Z0-9<]{10,}$')
    for i, wd in enumerate(words):
        if _MRZ_RE.match(wd["text"].replace(" ", "")) and i not in keep_idx:
            redactions.append(wd["box"])
            audit_items.append({"type": "MRZ", "text": wd["text"],
                                "box": [int(v) for v in wd["box"]]})

    # Redact face / portrait.  Run detection on the cropped card image so the
    # face is proportionally larger (full-page images from real-world photos
    # and document-page PDFs make the face too small for reliable detection).
    # Shift detected boxes by _crop_offset to land in orig_img space.
    if redact_faces:
        for fbox in detect_faces(img, model_path):
            fx, fy, fw, fh = fbox
            fx += cx0; fy += cy0   # map cropped-card coords → orig_img space
            ex, ey = int(fw * 0.45), int(fh * 0.6)
            fbox = (max(0, fx - ex), max(0, fy - ey), fw + 2 * ex, fh + 2 * ey)
            redactions.append(fbox)
            audit_items.append({"type": "FACE", "box": [int(v) for v in fbox]})

    for box in redactions:
        redact_region(orig_img, box, mode=mode)
    cv2.imwrite(out_path, orig_img)

    result = {
        "input": in_path,
        "output": out_path,
        "doc_type": doc_type,
        "name": found_labels.get("NAME"),
        "father_name": found_labels.get("PARENT"),
        "backend": _used_backend,
        "mode": mode,
        "words_ocr": len(words),
        "avg_ocr_conf": round(avg_conf, 3),
        "rotation_applied": rotation,
        "redactions": len(redactions),
        "warnings": warnings,
        "items": audit_items,
    }
    log_path = os.path.splitext(out_path)[0] + ".audit.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result, log_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_PDF_EXTS = {".pdf"}


def pdf_to_images(pdf_path, dpi=300):
    """Convert each page of a PDF to a BGR numpy array (cv2-compatible).
    Requires PyMuPDF (pip install pymupdf).
    Returns list of (page_num, img_bgr)."""
    try:
        import fitz
    except ImportError:
        raise SystemExit("PDF support requires PyMuPDF: pip install pymupdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append((i + 1, img_bgr))
    doc.close()
    return pages


def main():
    ap = argparse.ArgumentParser(
        description="Keep only doc-type header + name + father's name; redact everything else."
    )
    ap.add_argument("input", nargs="?", help="Single image file")
    ap.add_argument("--dir", metavar="DIR",
                    help="Directory of images (batch mode)")
    ap.add_argument("--out", default=None,
                    help="Output path (single-file mode); default <stem>.extracted.<ext>")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (batch mode); default <dir>/extracted/")
    ap.add_argument("--backend", choices=list(OCR_BACKENDS), default="tesseract")
    ap.add_argument("--langs", default=None,
                    help="OCR language codes (tesseract: eng+hin  /  easyocr: en,hi)")
    ap.add_argument("--mode", choices=["blackout", "blur", "pixelate"],
                    default="blackout")
    ap.add_argument("--no-faces", action="store_true",
                    help="Skip face/portrait redaction")
    ap.add_argument("--model", default=_DEFAULT_MODEL,
                    help="Path to YuNet face detection ONNX model")
    ap.add_argument("--vlm-fallback", action="store_true",
                    help="Use a local Ollama VLM when OCR fails to find the name")
    ap.add_argument("--vlm-model", default="moondream",
                    help="Ollama model name for VLM fallback (default: moondream). "
                         "Recommended options: moondream, minicpm-v, qwen2-vl, llava")
    ap.add_argument("--vlm-host", default="http://localhost:11434",
                    help="Ollama host URL (default: http://localhost:11434)")
    ap.add_argument("--full-redact", action="store_true",
                    help="Redact all text, faces and QR codes — keep nothing visible")
    args = ap.parse_args()

    if not args.input and not args.dir:
        ap.error("Provide an input file or --dir for batch mode.")
    if args.input and args.dir:
        ap.error("Provide either an input file or --dir, not both.")

    vlm_model = args.vlm_model if args.vlm_fallback else None
    kwargs = dict(backend=args.backend, langs=args.langs, mode=args.mode,
                  model_path=args.model, redact_faces=not args.no_faces,
                  vlm_model=vlm_model, vlm_host=args.vlm_host,
                  full_redact=args.full_redact)

    if args.dir:
        out_dir = args.out_dir or os.path.join(args.dir, "extracted")
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(args.dir)
                       if os.path.splitext(f)[1].lower() in _IMG_EXTS | _PDF_EXTS)
        if not files:
            print(f"No images or PDFs found in {args.dir}", file=sys.stderr)
            sys.exit(1)
        for fname in files:
            stem, ext = os.path.splitext(fname)
            in_path = os.path.join(args.dir, fname)
            try:
                if ext.lower() in _PDF_EXTS:
                    pages = pdf_to_images(in_path)
                    for pnum, pimg in pages:
                        suffix = f"_p{pnum}" if len(pages) > 1 else ""
                        out_path = os.path.join(out_dir, stem + suffix + ".extracted.jpg")
                        _gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
                        _skip_rot = (not kwargs.get('full_redact', False)
                                     and float((_gray > 220).mean()) >= 0.35)
                        res, _ = run(in_path, out_path, _img=pimg, _skip_rotate=_skip_rot, **kwargs)
                        print(f"[{res['doc_type']:>16}]  {fname} page {pnum}")
                        print(f"   Name  : {res['name']}")
                        print(f"   Father: {res['father_name']}")
                        print(f"   -> {out_path}")
                else:
                    out_path = os.path.join(out_dir, stem + ".extracted" + ext)
                    res, _ = run(in_path, out_path, **kwargs)
                    print(f"[{res['doc_type']:>16}]  {fname}")
                    print(f"   Name  : {res['name']}")
                    print(f"   Father: {res['father_name']}")
                    print(f"   -> {out_path}")
            except Exception as exc:
                print(f"ERROR {fname}: {exc}", file=sys.stderr)
    else:
        in_path = args.input
        stem, ext = os.path.splitext(args.out or args.input)
        out_ext = os.path.splitext(args.out)[1].lower() if args.out else (ext or ".jpg")

        if os.path.splitext(in_path)[1].lower() in _PDF_EXTS:
            pages = pdf_to_images(in_path)
            # When the requested output is a PDF, write per-page JPGs to a temp
            # dir then stitch them back into a single PDF.
            want_pdf_out = (out_ext == ".pdf")
            import tempfile
            tmp_dir = tempfile.mkdtemp() if want_pdf_out else None
            page_imgs = []

            for pnum, pimg in pages:
                suffix = f"_p{pnum}" if (len(pages) > 1 and not want_pdf_out) else ""
                if want_pdf_out:
                    out = os.path.join(tmp_dir, f"page_{pnum}.jpg")
                elif args.out:
                    s, e = os.path.splitext(args.out)
                    out = s + suffix + (e or ".jpg")
                else:
                    out = stem + suffix + ".extracted.jpg"
                try:
                    _gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
                    _skip_rot = float((_gray > 220).mean()) >= 0.35
                    res, log = run(in_path, out, _img=pimg, _skip_rotate=_skip_rot, **kwargs)
                except ValueError as exc:
                    sys.exit(str(exc))
                if want_pdf_out:
                    page_imgs.append(out)
                print(f"Page     : {pnum}/{len(pages)}")
                print(f"Doc type : {res['doc_type']}")
                print(f"Name     : {res['name']}")
                print(f"Father   : {res['father_name']}")
                print(f"OCR conf : {res['avg_ocr_conf']:.2f}")
                for w in res.get("warnings", []):
                    print(f"WARNING  : {w}")
                if not want_pdf_out:
                    print(f"Output   -> {out}")
                    print(f"Audit    -> {log}")
                print()

            if want_pdf_out and page_imgs:
                import fitz as _fitz
                pdf_out_path = args.out
                out_doc = _fitz.open()
                for img_path in page_imgs:
                    img_doc = _fitz.open(img_path)
                    pdf_bytes = img_doc.convert_to_pdf()
                    img_pdf = _fitz.open("pdf", pdf_bytes)
                    out_doc.insert_pdf(img_pdf)
                    os.remove(img_path)
                out_doc.save(pdf_out_path)
                out_doc.close()
                import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
                print(f"Output   -> {pdf_out_path}")
        else:
            out = args.out or (stem + ".extracted" + (ext or ".jpg"))
            try:
                res, log = run(in_path, out, **kwargs)
            except ValueError as exc:
                sys.exit(str(exc))
            print(f"Doc type : {res['doc_type']}")
            print(f"Name     : {res['name']}")
            print(f"Father   : {res['father_name']}")
            print(f"OCR conf : {res['avg_ocr_conf']:.2f}")
            for w in res.get("warnings", []):
                print(f"WARNING  : {w}")
            print(f"Output   -> {out}")
            print(f"Audit    -> {log}")


if __name__ == "__main__":
    main()
