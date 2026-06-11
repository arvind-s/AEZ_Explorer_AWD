"""
extract_only.py

Keep ONLY the document-type header, name, and father's name on Indian KYC
images. Every other word, number, date, address, and face is redacted.

Supports: Aadhaar, PAN, Passport, Voter ID, Driving Licence, National ID.

Usage
-----
    # Single image
    python extract_only.py aadhar.jpg --out aadhar_clean.jpg

    # Batch: all images in a directory
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

from redact import (
    OCR_BACKENDS, _LANG_DEFAULTS,
    build_text_and_index, union_box, group_lines,
    detect_faces, redact_region,
)
from detectors import find_entities


# ─── Document-type signal phrases ─────────────────────────────────────────────

_DOC_PATTERNS = [
    ("AADHAAR",         [r"aadhaar", r"आधार", r"unique\s+identification", r"uidai"]),
    ("PAN",             [r"permanent\s+account", r"income\s+tax"]),
    ("PASSPORT",        [r"passport", r"passeport", r"republic\s+of\s+india"]),
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


# ─── Label-anchored keep (name + father) ──────────────────────────────────────

_KEEP_LABELS = {
    "NAME":   ["name", "नाम"],
    "PARENT": ["father", "s/o", "d/o", "w/o", "c/o", "पिता",
               "husband", "mother", "guardian", "care of"],
}
_KEEP_LOOKUP = {kw: lab for lab, kws in _KEEP_LABELS.items() for kw in kws}
_norm_token = lambda t: re.sub(r"[^\w/]", "", t.lower())


def keep_by_labels(words, lines):
    """
    Return (keep_idx: set[int], found: dict[str, str]).
    keep_idx  -- word indices for name + father values only (NOT the label words).
    found     -- {"NAME": "Ravi Kumar", "PARENT": "Suresh Kumar"} for the audit.
    """
    keep_idx = set()
    found = {}
    for li in sorted(lines.keys()):
        idxs = lines[li]
        for pos, wi in enumerate(idxs):
            lab = _KEEP_LOOKUP.get(_norm_token(words[wi]["text"]))
            if not lab:
                continue
            value_idxs = idxs[pos + 1:]
            if value_idxs:
                keep_idx.update(value_idxs)
                found[lab] = " ".join(words[k]["text"] for k in value_idxs)
            break
    return keep_idx, found


# ─── Pipeline ─────────────────────────────────────────────────────────────────

_DEFAULT_MODEL = os.path.join(_HERE, "face_detection_yunet_2023mar.onnx")


def run(in_path, out_path, backend="tesseract", langs=None, mode="blackout",
        model_path=_DEFAULT_MODEL, redact_faces=True):
    """
    Returns (result_dict, audit_log_path).
    result_dict keys: doc_type, name, father_name, words_ocr, redactions.
    """
    img = cv2.imread(in_path)
    if img is None:
        raise ValueError(f"Could not read image: {in_path}")

    langs = langs or _LANG_DEFAULTS.get(backend, "eng")

    # Upscale small images so OCR has enough resolution.
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) < 1600:
        scale = 1600.0 / max(h, w)
        ocr_img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
    else:
        ocr_img = img

    words = OCR_BACKENDS[backend](ocr_img, langs)
    for wd in words:
        x, y, bw, bh = wd["box"]
        wd["box"] = (int(x / scale), int(y / scale),
                     int(bw / scale), int(bh / scale))

    full_text, _ = build_text_and_index(words)
    doc_type = detect_doc_type(full_text)
    lines = group_lines(words)

    # Decide what to keep.
    keep_idx = header_word_indices(words, lines)
    label_keep, found_labels = keep_by_labels(words, lines)
    keep_idx |= label_keep

    # Redact every word NOT in keep_idx.
    redactions = []
    audit_items = []
    for i, wd in enumerate(words):
        if i not in keep_idx:
            redactions.append(wd["box"])
            audit_items.append({"type": "WORD", "text": wd["text"],
                                "box": [int(v) for v in wd["box"]]})

    # Redact face / portrait.
    if redact_faces:
        for fbox in detect_faces(img, model_path):
            fx, fy, fw, fh = fbox
            ex, ey = int(fw * 0.45), int(fh * 0.6)
            fbox = (max(0, fx - ex), max(0, fy - ey), fw + 2 * ex, fh + 2 * ey)
            redactions.append(fbox)
            audit_items.append({"type": "FACE", "box": [int(v) for v in fbox]})

    for box in redactions:
        redact_region(img, box, mode=mode)
    cv2.imwrite(out_path, img)

    result = {
        "input": in_path,
        "output": out_path,
        "doc_type": doc_type,
        "name": found_labels.get("NAME"),
        "father_name": found_labels.get("PARENT"),
        "backend": backend,
        "mode": mode,
        "words_ocr": len(words),
        "redactions": len(redactions),
        "items": audit_items,
    }
    log_path = os.path.splitext(out_path)[0] + ".audit.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result, log_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


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
    args = ap.parse_args()

    if not args.input and not args.dir:
        ap.error("Provide an input file or --dir for batch mode.")
    if args.input and args.dir:
        ap.error("Provide either an input file or --dir, not both.")

    kwargs = dict(backend=args.backend, langs=args.langs, mode=args.mode,
                  model_path=args.model, redact_faces=not args.no_faces)

    if args.dir:
        out_dir = args.out_dir or os.path.join(args.dir, "extracted")
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(args.dir)
                       if os.path.splitext(f)[1].lower() in _IMG_EXTS)
        if not files:
            print(f"No images found in {args.dir}", file=sys.stderr)
            sys.exit(1)
        for fname in files:
            stem, ext = os.path.splitext(fname)
            out_path = os.path.join(out_dir, stem + ".extracted" + ext)
            try:
                res, _ = run(os.path.join(args.dir, fname), out_path, **kwargs)
                print(f"[{res['doc_type']:>16}]  {fname}")
                print(f"   Name  : {res['name']}")
                print(f"   Father: {res['father_name']}")
                print(f"   -> {out_path}")
            except Exception as exc:
                print(f"ERROR {fname}: {exc}", file=sys.stderr)
    else:
        stem, ext = os.path.splitext(args.input)
        out = args.out or (stem + ".extracted" + (ext or ".jpg"))
        try:
            res, log = run(args.input, out, **kwargs)
        except ValueError as exc:
            sys.exit(str(exc))
        print(f"Doc type : {res['doc_type']}")
        print(f"Name     : {res['name']}")
        print(f"Father   : {res['father_name']}")
        print(f"Output   -> {out}")
        print(f"Audit    -> {log}")


if __name__ == "__main__":
    main()
