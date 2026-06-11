# Indian KYC Image Redaction (classic open stack)

Detects and irreversibly redacts PII on images of Indian identity documents
(Aadhaar, PAN, voter ID, driving licence) plus generic PII (phone, email, DOB).

## Refined model stack (image input)

| Stage | Model / method | Why |
|---|---|---|
| OCR + boxes | **Tesseract 5 (LSTM)** via subprocess, TSV output | Lightweight, word-level boxes, installs cleanly on py3.13/arm64. Swap to **EasyOCR** for noisy photos. |
| Layout/region | (implicit) OCR word boxes + label anchors | Sufficient for cards; add **DocLayout-YOLO/Surya** for complex forms. |
| Structured IDs | **regex + Verhoeff (Aadhaar) / format (PAN, EPIC, DL)** | Highest precision; checksums kill false positives. |
| Free-form PII | **GLiNER** (optional) | Zero-shot names/addresses without retraining. |
| Face / portrait | **OpenCV YuNet** ONNX | Fast, accurate frontal-face detection; blur the photo. |
| Redaction | OpenCV blackout / blur / pixelate + raster re-encode | Irreversible — no recoverable text layer. |

## Install

```bash
brew install tesseract tesseract-lang
pip install opencv-contrib-python pillow numpy
# face model:
curl -L -o face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Run

```bash
python redact.py path/to/id.jpg --out redacted.jpg --mode blackout
python redact.py id.jpg --gliner                       # + zero-shot NER for names/parentage/address
python redact.py id.jpg --backend easyocr --langs en,hi --mode blur   # noisy-photo accuracy upgrade
python redact.py id.jpg --no-faces --no-labels         # toggles
```

Outputs the redacted image + a `*.audit.json` log of every redaction
(type, matched text, confidence, checksum validity, bounding box, source).

## Three detection layers (recall-biased, union of all)
1. **regex + checksum** — Aadhaar (Verhoeff), VID, PAN, voter EPIC, driving licence,
   Indian mobile, email, DOB. High precision.
2. **label anchors** (default, dependency-free) — redacts values after
   `Name:` / `Father:` / `S/O` / `Address:`, incl. one address continuation line.
3. **GLiNER NER** (`--gliner`) — zero-shot person/parent/address/org from free text.

Faces are detected with YuNet and the box is expanded to cover the whole portrait.
PINCODE is detected but below the default threshold (tune `min_score`).

## Notes / further upgrades
- **Noisy phone photos** — switch `--backend easyocr` (or PaddleOCR) for higher OCR recall.
- **Reversibility** — output is flattened raster; never ship a PDF with the original text layer intact.
- **Over-redaction** — label-anchor address-continuation can occasionally grab an
  adjacent line; safe for redaction (errs toward covering more).
