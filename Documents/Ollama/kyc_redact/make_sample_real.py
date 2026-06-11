"""Mock Aadhaar-style card with a REAL (GAN-generated, not a real person) face
photo composited into the portrait box, so YuNet face detection can be exercised."""
import os
from PIL import Image, ImageDraw, ImageFont
from make_sample import make_aadhaar, font


def main():
    W, H = 1000, 620
    img = Image.new("RGB", (W, H), (252, 250, 245))
    d = ImageDraw.Draw(img)

    aadhaar = make_aadhaar()
    aadhaar_fmt = f"{aadhaar[0:4]} {aadhaar[4:8]} {aadhaar[8:12]}"

    d.rectangle([0, 0, W, 70], fill=(247, 130, 30))
    d.rectangle([0, H - 30, W, H], fill=(20, 110, 60))
    d.text((90, 22), "Government of India", font=font(30), fill=(255, 255, 255))

    # composite real face into portrait box
    box = (60, 130, 260, 380)  # x0,y0,x1,y1
    if os.path.exists("face_real.jpg"):
        face = Image.open("face_real.jpg").convert("RGB")
        face = face.resize((box[2] - box[0], box[3] - box[1]))
        img.paste(face, (box[0], box[1]))
    d.rectangle(box, outline=(120, 120, 120), width=2)

    x = 300
    d.text((x, 120), "Name:  Ravi Kumar Sharma", font=font(28), fill=(0, 0, 0))
    d.text((x, 165), "Father: Mohan Lal Sharma", font=font(26), fill=(0, 0, 0))
    d.text((x, 205), "DOB:   14/08/1990", font=font(26), fill=(0, 0, 0))
    d.text((x, 245), "Mobile: 9876543210", font=font(26), fill=(0, 0, 0))
    d.text((x, 290), "Address: 12 MG Road,", font=font(22), fill=(0, 0, 0))
    d.text((x, 320), "Bengaluru, Karnataka 560001", font=font(22), fill=(0, 0, 0))

    d.text((300, 430), aadhaar_fmt, font=font(54), fill=(0, 0, 0))
    d.text((300, 500), "PAN: ABCPK1234L", font=font(28), fill=(0, 0, 0))

    img.save("sample_real.png")
    print("Wrote sample_real.png  | Aadhaar:", aadhaar_fmt)


if __name__ == "__main__":
    main()
