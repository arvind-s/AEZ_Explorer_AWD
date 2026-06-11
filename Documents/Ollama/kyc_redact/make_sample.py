"""Generate a synthetic mock Aadhaar-style card with FAKE data for testing.

No real PII. The Aadhaar number is generated with a valid Verhoeff checksum so the
pipeline's checksum path is exercised. The portrait is a simple synthetic face.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from detectors import _VERHOEFF_D, _VERHOEFF_P

# inverse table for Verhoeff checksum-digit generation
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_checksum(number: str) -> str:
    c = 0
    for i, item in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(item)]]
    return str(_INV[c])


def make_aadhaar():
    base = "23456789012"          # 11 digits, leading 2 (valid: not 0/1)
    return base + verhoeff_checksum(base)


def font(size):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf",
              "/Library/Fonts/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_face(draw, x, y, w, h):
    # very rough frontal face so YuNet has a chance to fire
    cx, cy = x + w // 2, y + h // 2
    draw.ellipse([x, y, x + w, y + h], fill=(225, 200, 175))           # head
    draw.ellipse([cx - w//4 - 8, cy - 8, cx - w//4 + 8, cy + 8], fill=(40, 40, 40))   # eye L
    draw.ellipse([cx + w//4 - 8, cy - 8, cx + w//4 + 8, cy + 8], fill=(40, 40, 40))   # eye R
    draw.line([cx, cy, cx, cy + 18], fill=(120, 90, 70), width=3)       # nose
    draw.arc([cx - 22, cy + 14, cx + 22, cy + 40], 20, 160, fill=(120, 60, 60), width=3)  # mouth


def main():
    W, H = 1000, 620
    img = Image.new("RGB", (W, H), (252, 250, 245))
    d = ImageDraw.Draw(img)

    aadhaar = make_aadhaar()
    aadhaar_fmt = f"{aadhaar[0:4]} {aadhaar[4:8]} {aadhaar[8:12]}"

    # header band
    d.rectangle([0, 0, W, 70], fill=(247, 130, 30))
    d.rectangle([0, H - 30, W, H], fill=(20, 110, 60))
    d.text((90, 22), "Government of India", font=font(30), fill=(255, 255, 255))

    # portrait
    draw_face(d, 60, 130, 200, 230)
    d.rectangle([60, 130, 260, 360], outline=(120, 120, 120), width=2)

    x = 300
    d.text((x, 120), "Name:  Ravi Kumar Sharma", font=font(28), fill=(0, 0, 0))
    d.text((x, 170), "DOB:   14/08/1990", font=font(26), fill=(0, 0, 0))
    d.text((x, 210), "Gender: Male", font=font(26), fill=(0, 0, 0))
    d.text((x, 260), "Mobile: 9876543210", font=font(26), fill=(0, 0, 0))
    d.text((x, 300), "Address: 12 MG Road, Bengaluru 560001",
           font=font(22), fill=(0, 0, 0))

    # big Aadhaar number
    d.text((300, 430), aadhaar_fmt, font=font(54), fill=(0, 0, 0))
    d.text((300, 500), "PAN: ABCPK1234L", font=font(28), fill=(0, 0, 0))

    img.save("sample_card.png")
    print("Wrote sample_card.png")
    print("Fake Aadhaar:", aadhaar_fmt, "(valid checksum)")


if __name__ == "__main__":
    main()
