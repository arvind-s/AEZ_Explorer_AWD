"""
Indian KYC image redaction pipeline (classic open stack).

Stages
------
1. Pre-process       : grayscale + upscale to help OCR on phone photos.
2. OCR with boxes    : Tesseract by default (word-level boxes). Swappable backend.
3. Entity detection  : regex + checksums (detectors.py) over the OCR text.
4. Face detection    : OpenCV YuNet -> blur the ID portrait.
5. Redaction         : irreversible -- pixels are painted over / blurred and the
                       image is re-encoded as raster (no recoverable text layer).
6. Audit log         : JSON of what was redacted, where, by which recognizer.

Usage
-----
    python redact.py input.jpg --out redacted.jpg --mode blackout
    python redact.py input.jpg --backend easyocr --langs en,hi

The OCR backend is abstracted so you can upgrade Tesseract -> EasyOCR/PaddleOCR
for accuracy on noisy images without touching the rest of the pipeline.
"""

import argparse
import json
import os
import re

import cv2
import numpy as np

from detectors import find_entities


# ---------------------------------------------------------------------------
# OCR backends -- each returns a list of word dicts:
#   {"text": str, "box": (x, y, w, h), "conf": float}
# ---------------------------------------------------------------------------

def _tesseract_bin():
    import shutil
    for cand in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract",
                 shutil.which("tesseract")):
        if cand and os.path.exists(cand):
            return cand
    return "tesseract"


def ocr_tesseract(img_bgr, langs="eng"):
    """Call the tesseract binary directly (TSV output) to avoid pytesseract's
    temp-file location, which some sandboxes block the binary from reading."""
    import subprocess
    import tempfile
    # write the temp image next to the cwd so the binary can read it
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=os.getcwd())
    os.close(fd)
    try:
        cv2.imwrite(tmp, img_bgr)
        proc = subprocess.run(
            [_tesseract_bin(), tmp, "stdout", "-l", langs,
             "--oem", "1", "--psm", "11", "tsv"],
            capture_output=True, text=True,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.strip()}")

    words = []
    lines = proc.stdout.splitlines()
    if not lines:
        return words
    header = lines[0].split("\t")
    col = {name: i for i, name in enumerate(header)}
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) <= col.get("text", 11):
            continue
        txt = f[col["text"]].strip()
        try:
            conf = float(f[col["conf"]])
        except ValueError:
            conf = -1.0
        if txt and conf >= 0:
            words.append({
                "text": txt,
                "box": (int(f[col["left"]]), int(f[col["top"]]),
                        int(f[col["width"]]), int(f[col["height"]])),
                "conf": conf / 100.0,
            })
    return words


def ocr_easyocr(img_bgr, langs="en"):
    import easyocr
    reader = ocr_easyocr._reader  # set lazily below
    if reader is None or ocr_easyocr._langs != langs:
        reader = easyocr.Reader([l.strip() for l in langs.split(",")], gpu=False)
        ocr_easyocr._reader = reader
        ocr_easyocr._langs = langs
    words = []
    for box, txt, conf in reader.readtext(img_bgr):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        words.append({"text": txt, "box": (x, y, w, h), "conf": float(conf)})
    return words


ocr_easyocr._reader = None
ocr_easyocr._langs = None

# Tesseract uses 3-letter codes (eng, hin); easyocr uses 2-letter (en, hi).
_LANG_DEFAULTS = {"tesseract": "eng", "easyocr": "en"}
OCR_BACKENDS = {"tesseract": ocr_tesseract, "easyocr": ocr_easyocr}


# ---------------------------------------------------------------------------
# GLiNER zero-shot NER for free-form PII (names, addresses, organisations).
# ---------------------------------------------------------------------------

_GLINER_LABELS = ["person name", "father name", "address", "city",
                  "organization", "date of birth"]
_GLINER_MAP = {"person name": "NAME", "father name": "PARENT", "address": "ADDRESS",
               "city": "ADDRESS", "organization": "ORG", "date of birth": "DOB"}


def gliner_entities(text, threshold=0.4):
    from gliner import GLiNER
    model = gliner_entities._model
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        gliner_entities._model = model
    out = []
    for e in model.predict_entities(text, _GLINER_LABELS, threshold=threshold):
        out.append({"label": _GLINER_MAP.get(e["label"], e["label"].upper()),
                    "text": e["text"], "start": e["start"], "end": e["end"],
                    "score": e["score"]})
    return out


gliner_entities._model = None


# ---------------------------------------------------------------------------
# Mapping detected character spans back onto OCR word boxes.
# ---------------------------------------------------------------------------

def build_text_and_index(words):
    """Concatenate word texts with single spaces, recording each word's char span."""
    parts, spans, cursor = [], [], 0
    for i, w in enumerate(words):
        t = w["text"]
        spans.append((cursor, cursor + len(t), i))
        parts.append(t)
        cursor += len(t) + 1  # +1 for the joining space
    return " ".join(parts), spans


def words_for_span(spans, start, end):
    """Indices of OCR words whose char range overlaps [start, end)."""
    hit = []
    for s, e, idx in spans:
        if s < end and start < e:
            hit.append(idx)
    return hit


# ---------------------------------------------------------------------------
# Label-anchored redaction: redact whatever follows a known field label on the
# same OCR line (and continuation lines for addresses). Dependency-free; catches
# free-form values (names, parentage, address) that regex/checksums can't.
# ---------------------------------------------------------------------------

_LABELS = {
    "NAME":    ["name", "नाम"],
    "PARENT":  ["father", "s/o", "d/o", "w/o", "c/o", "पिता", "husband"],
    "ADDRESS": ["address", "addr", "पता", "add"],
}
_LABEL_LOOKUP = {kw: lab for lab, kws in _LABELS.items() for kw in kws}
_STOP_WORDS = set(_LABEL_LOOKUP) | {"dob", "gender", "sex", "mobile", "phone",
                                    "year", "birth", "male", "female"}


def group_lines(words):
    """Assign each word a line id by clustering on vertical overlap.
    Returns {line_id: [word indices sorted left-to-right]}."""
    order = sorted(range(len(words)), key=lambda i: words[i]["box"][1])
    lines, line_id = {}, -1
    cur_bottom = None
    assign = {}
    for i in order:
        x, y, w, h = words[i]["box"]
        if cur_bottom is None or y > cur_bottom - h * 0.4:
            if cur_bottom is None or y > cur_bottom:
                line_id += 1
        assign[i] = line_id
        cur_bottom = y + h if cur_bottom is None else max(cur_bottom, y + h)
    # regroup using a simpler band test (robust for cards)
    lines = {}
    used = [False] * len(words)
    bands = sorted(range(len(words)), key=lambda i: words[i]["box"][1])
    lid = 0
    for i in bands:
        if used[i]:
            continue
        yi, hi = words[i]["box"][1], words[i]["box"][3]
        cy = yi + hi / 2
        members = []
        for j in bands:
            if used[j]:
                continue
            yj, hj = words[j]["box"][1], words[j]["box"][3]
            if yj <= cy <= yj + hj or yi <= (yj + hj / 2) <= yi + hi:
                members.append(j)
                used[j] = True
        members.sort(key=lambda k: words[k]["box"][0])
        lines[lid] = members
        lid += 1
    return lines


def redact_by_labels(words):
    """Return [(box, label)] for values following field labels."""
    lines = group_lines(words)
    line_index = sorted(lines.keys())
    norm = lambda t: re.sub(r"[^\w/]", "", t.lower())
    out = []
    for li in line_index:
        idxs = lines[li]
        for pos, wi in enumerate(idxs):
            lab = _LABEL_LOOKUP.get(norm(words[wi]["text"]))
            if not lab:
                continue
            # redact rest of this line after the label
            tail = [words[k]["box"] for k in idxs[pos + 1:]]
            if tail:
                out.append((union_box(tail), lab))
            # address: also redact the next line if it isn't another labelled field
            if lab == "ADDRESS":
                nxt = lines.get(li + 1, [])
                if nxt and norm(words[nxt[0]]["text"]) not in _STOP_WORDS:
                    out.append((union_box([words[k]["box"] for k in nxt]), "ADDRESS"))
            break  # one label per line
    return out


# ---------------------------------------------------------------------------
# Face detection (OpenCV YuNet).
# ---------------------------------------------------------------------------

def detect_faces(img_bgr, model_path):
    if not os.path.exists(model_path):
        return []
    h, w = img_bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h),
                                    score_threshold=0.6, nms_threshold=0.3)
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = [int(v) for v in f[:4]]
            out.append((max(0, x), max(0, y), fw, fh))
    return out


# ---------------------------------------------------------------------------
# Redaction primitives -- irreversible.
# ---------------------------------------------------------------------------

def redact_region(img, box, mode="blackout", pad=2):
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return
    if mode == "blackout":
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    elif mode == "blur":
        roi = img[y0:y1, x0:x1]
        k = max(15, ((x1 - x0) // 4) | 1)  # odd kernel scaled to region width
        img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    elif mode == "pixelate":
        roi = img[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                           interpolation=cv2.INTER_LINEAR)
        img[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0),
                                       interpolation=cv2.INTER_NEAREST)


def union_box(boxes):
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[0] + b[2] for b in boxes]; ys1 = [b[1] + b[3] for b in boxes]
    x0, y0 = min(xs0), min(ys0)
    return (x0, y0, max(xs1) - x0, max(ys1) - y0)


# ---------------------------------------------------------------------------
# Pipeline.
# ---------------------------------------------------------------------------

def run(in_path, out_path, backend="tesseract", langs=None, mode="blackout",
        model_path="face_detection_yunet_2023mar.onnx", redact_faces=True,
        min_score=0.55, use_labels=True, use_gliner=False):
    img = cv2.imread(in_path)
    if img is None:
        raise SystemExit(f"Could not read image: {in_path}")

    langs = langs or _LANG_DEFAULTS.get(backend, "eng")

    # 1+2. Upscale small images, then OCR.
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) < 1600:
        scale = 1600.0 / max(h, w)
        ocr_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        ocr_img = img
    words = OCR_BACKENDS[backend](ocr_img, langs)
    # rescale boxes back to original coords
    for wd in words:
        x, y, bw, bh = wd["box"]
        wd["box"] = (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))

    # 3. Entity detection over reconstructed text.
    full_text, spans = build_text_and_index(words)
    matches = find_entities(full_text)

    audit = []
    redactions = []
    for m in matches:
        if m.score < min_score:
            continue
        idxs = words_for_span(spans, m.start, m.end)
        if not idxs:
            continue
        box = union_box([words[i]["box"] for i in idxs])
        redactions.append(box)
        audit.append({
            "type": m.label, "matched_text": m.text, "score": round(m.score, 2),
            "checksum_valid": m.validated, "box": [int(v) for v in box],
            "source": "regex+checksum",
        })

    # 3b. Label-anchored values (names, parentage, address).
    if use_labels:
        for box, lab in redact_by_labels(words):
            redactions.append(box)
            audit.append({"type": lab, "matched_text": None, "score": 0.75,
                          "checksum_valid": None, "box": [int(v) for v in box],
                          "source": "label-anchor"})

    # 3c. GLiNER NER for free-form PII (optional).
    if use_gliner:
        for ent in gliner_entities(full_text):
            idxs = words_for_span(spans, ent["start"], ent["end"])
            if not idxs:
                continue
            box = union_box([words[i]["box"] for i in idxs])
            redactions.append(box)
            audit.append({"type": ent["label"], "matched_text": ent["text"],
                          "score": round(ent["score"], 2), "checksum_valid": None,
                          "box": [int(v) for v in box], "source": "gliner"})

    # 4. Faces. Expand the face box outward so the whole portrait (hair, ears,
    #    chin) is covered, not just the detected face rectangle.
    if redact_faces:
        for fbox in detect_faces(img, model_path):
            fx, fy, fw, fh = fbox
            ex, ey = int(fw * 0.45), int(fh * 0.6)
            fbox = (max(0, fx - ex), max(0, fy - ey), fw + 2 * ex, fh + 2 * ey)
            redactions.append(fbox)
            audit.append({"type": "FACE", "matched_text": None, "score": 0.9,
                          "checksum_valid": None, "box": [int(v) for v in fbox],
                          "source": "yunet"})

    # 5. Apply redactions (irreversible) and re-encode as raster.
    for box in redactions:
        redact_region(img, box, mode=mode)
    cv2.imwrite(out_path, img)

    # 6. Audit log.
    log_path = os.path.splitext(out_path)[0] + ".audit.json"
    with open(log_path, "w") as f:
        json.dump({
            "input": in_path, "output": out_path, "backend": backend,
            "mode": mode, "words_ocr": len(words),
            "redactions": len(redactions), "items": audit,
        }, f, indent=2)

    return audit, log_path


def main():
    ap = argparse.ArgumentParser(description="Indian KYC image redaction")
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=list(OCR_BACKENDS), default="tesseract")
    ap.add_argument("--langs", default=None, help="OCR langs (tesseract: eng,hin / easyocr: en,hi)")
    ap.add_argument("--mode", choices=["blackout", "blur", "pixelate"], default="blackout")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-labels", action="store_true", help="disable label-anchored redaction")
    ap.add_argument("--gliner", action="store_true", help="enable GLiNER NER (needs `pip install gliner`)")
    ap.add_argument("--model", default="face_detection_yunet_2023mar.onnx")
    args = ap.parse_args()

    out = args.out or (os.path.splitext(args.input)[0] + ".redacted.jpg")
    audit, log_path = run(args.input, out, backend=args.backend, langs=args.langs,
                          mode=args.mode, model_path=args.model,
                          redact_faces=not args.no_faces,
                          use_labels=not args.no_labels, use_gliner=args.gliner)
    print(f"Redacted -> {out}")
    print(f"Audit    -> {log_path}")
    for a in audit:
        print(f"  [{a['type']:>16}] '{a['matched_text']}' score={a['score']} "
              f"checksum={a['checksum_valid']} box={a['box']}")


if __name__ == "__main__":
    main()
