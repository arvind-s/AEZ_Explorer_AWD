"""
Indian KYC entity detectors: high-precision regex + checksum validators.

These run over OCR text tokens. Each detector returns a label and a confidence.
Structured IDs (Aadhaar, PAN) are validated with checksums / strict formats so we
get very low false-positive rates -- the right bias for redaction is high recall,
but for structured numbers we can also be highly precise.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Aadhaar: 12 digits, last digit is a Verhoeff checksum.
# ---------------------------------------------------------------------------

# Verhoeff multiplication (d), permutation (p) and inverse (inv) tables.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """True if the digit string passes the Verhoeff checksum (Aadhaar uses this)."""
    c = 0
    for i, item in enumerate(reversed(number)):
        if not item.isdigit():
            return False
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
    return c == 0


@dataclass
class Match:
    label: str          # e.g. "AADHAAR", "PAN"
    text: str           # the matched substring
    start: int          # char offset in the source string
    end: int
    score: float        # 0..1 confidence
    validated: bool     # passed a checksum / strict format check


# Aadhaar appears as 12 digits, usually grouped 4-4-4 (space or hyphen separated).
_AADHAAR_RE = re.compile(r"(?<!\d)(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)")
# VID: 16 digits (virtual Aadhaar id).
_VID_RE = re.compile(r"(?<!\d)(\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})(?!\d)")
# PAN: 5 letters, 4 digits, 1 letter. 4th char is entity type, 5th is name initial.
_PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
# Voter EPIC: 3 letters + 7 digits.
_EPIC_RE = re.compile(r"\b([A-Z]{3}[0-9]{7})\b")
# Driving licence: state(2) + region(2) + space/0 + 11 digits (loose).
_DL_RE = re.compile(r"\b([A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{7})\b")
# Indian mobile: optional +91/0, then 6-9 leading 10-digit.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+91[\s\-]?|0)?([6-9]\d{9})(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
# Dates of birth: dd/mm/yyyy, dd-mm-yyyy, yyyy.
_DOB_RE = re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b")
_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# Length-preserving map of common OCR letter->digit confusions, so a misread
# Aadhaar like "6Z5O 2SSO 71O5" can still be recovered. Length is preserved so
# character offsets stay valid for box mapping.
_OCR_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "|": "1", "i": "1",
    "Z": "2", "z": "2", "S": "5", "s": "5",
    "B": "8", "G": "6", "T": "7",
})


def _normalize_digits(s: str) -> str:
    return s.translate(_OCR_DIGIT_FIX)


def find_entities(text: str):
    """Return a list of Match objects found in `text`."""
    out = []

    # PAN (check before generic, strict format -> validated).
    for m in _PAN_RE.finditer(text):
        out.append(Match("PAN", m.group(1), m.start(1), m.end(1), 0.97, True))

    # VID first (16 digits) so it isn't swallowed by Aadhaar's 12-digit pattern.
    vid_spans = []
    for m in _VID_RE.finditer(text):
        d = _digits(m.group(1))
        if len(d) == 16:
            out.append(Match("VID", m.group(1), m.start(1), m.end(1), 0.9, True))
            vid_spans.append((m.start(1), m.end(1)))

    for m in _AADHAAR_RE.finditer(text):
        # skip if inside a VID match
        if any(s <= m.start(1) and m.end(1) <= e for s, e in vid_spans):
            continue
        d = _digits(m.group(1))
        if len(d) != 12 or d[0] in "01":  # Aadhaar never starts 0/1
            continue
        valid = verhoeff_valid(d)
        # Even if checksum fails (OCR error), still redact -- but lower score.
        out.append(Match("AADHAAR", m.group(1), m.start(1), m.end(1),
                         0.95 if valid else 0.6, valid))

    for m in _EPIC_RE.finditer(text):
        out.append(Match("VOTER_ID", m.group(1), m.start(1), m.end(1), 0.85, True))

    for m in _DL_RE.finditer(text):
        out.append(Match("DRIVING_LICENCE", m.group(1), m.start(1), m.end(1), 0.8, True))

    for m in _PHONE_RE.finditer(text):
        out.append(Match("PHONE", m.group(1), m.start(1), m.end(1), 0.8, True))

    for m in _EMAIL_RE.finditer(text):
        out.append(Match("EMAIL", m.group(0), m.start(), m.end(), 0.9, True))

    for m in _DOB_RE.finditer(text):
        out.append(Match("DOB", m.group(1), m.start(1), m.end(1), 0.7, False))

    for m in _PIN_RE.finditer(text):
        out.append(Match("PINCODE", m.group(1), m.start(1), m.end(1), 0.4, False))

    # Loose recovery pass: re-scan with OCR letter->digit fixes so a garbled
    # Aadhaar (e.g. "6Z5O 2SSO 71O5") is still redacted. Over-redaction here is
    # the safe direction; we only emit if not already covered above.
    covered = [(m.start, m.end) for m in out if m.label in ("AADHAAR", "VID")]
    fixed = _normalize_digits(text)
    for m in _AADHAAR_RE.finditer(fixed):
        d = _digits(m.group(1))
        if len(d) != 12 or d[0] in "01":
            continue
        if any(s <= m.start(1) and m.end(1) <= e for s, e in covered):
            continue
        valid = verhoeff_valid(d)
        out.append(Match("AADHAAR", text[m.start(1):m.end(1)], m.start(1), m.end(1),
                         0.85 if valid else 0.6, valid))

    return out
