from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from document_validation.ensemble import DetectorVote
from document_validation.validator import ValidationConfig

LABELS = ("accepted", "blur", "cut", "not_document")
ISSUE_LABELS = ("blur", "cut", "not_document")

PROMPT = """You are a document quality inspector for Indian agricultural documents.
Classify this image into exactly one of these categories:

accepted      - document is clear, complete and readable
blur          - image is blurry, washed out, low contrast, or text is not legible
cut           - document is physically cut off or edges not fully visible in the frame
not_document  - not a document (random photo, blank page, non-document content)

Reply with ONLY the category name, nothing else."""


def parse_label(response: str) -> str:
    text = response.lower().strip()
    found = {label: text.index(label) for label in LABELS if label in text}
    if not found:
        return "accepted"
    return min(found, key=found.__getitem__)


class VlmDetector:
    name: str

    def __init__(self, model_key: str) -> None:
        self.name = model_key
        self._model_key = model_key
        self._predict_fn: Callable[[np.ndarray], str] | None = None

    def _load(self) -> None:
        if self._model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model key: {self._model_key!r}. "
                f"Available: {list(MODEL_REGISTRY)}"
            )
        self._predict_fn = MODEL_REGISTRY[self._model_key]()

    def detect(
        self,
        page_bgr: np.ndarray,
        source_path: Path,
        page_number: int,
        config: ValidationConfig,
    ) -> DetectorVote:
        try:
            if self._predict_fn is None:
                self._load()
            raw = self._predict_fn(page_bgr)  # type: ignore[misc]
            label = parse_label(raw)
            issues = [label] if label in ISSUE_LABELS else []
            return DetectorVote(
                detector=self.name,
                available=True,
                label=label,
                issues=issues,
                confidence=1.0,
            )
        except Exception as exc:
            return DetectorVote(
                detector=self.name,
                available=False,
                label=None,
                error=str(exc),
            )


# Populated in loaders below
MODEL_REGISTRY: dict[str, Callable[[], Callable[[np.ndarray], str]]] = {}


# ── HuggingFace loaders ────────────────────────────────────────────────────────

def _load_qwen2_vl() -> Callable[[np.ndarray], str]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="mps"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("mps")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16)
        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    return predict


MODEL_REGISTRY["qwen2-vl-2b"] = _load_qwen2_vl


def _load_internvl2() -> Callable[[np.ndarray], str]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    model_id = "OpenGVLab/InternVL2-2B"
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="mps", trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pixel_values = transform(pil).unsqueeze(0).to(torch.bfloat16).to("mps")
        generation_config = dict(max_new_tokens=16, do_sample=False)
        response = model.chat(tokenizer, pixel_values, f"<image>\n{PROMPT}", generation_config)
        return response.strip()

    return predict


MODEL_REGISTRY["internvl2-2b"] = _load_internvl2


def _load_moondream2() -> Callable[[np.ndarray], str]:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "vikhyatk/moondream2"
    revision = "2025-01-09"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="mps",
    ).eval()

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        enc = model.encode_image(image)
        answer = model.answer_question(enc, PROMPT, tokenizer)
        return answer.strip()

    return predict


MODEL_REGISTRY["moondream2"] = _load_moondream2


def _load_smolvlm() -> Callable[[np.ndarray], str]:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    model_id = "HuggingFaceTB/SmolVLM-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="mps"
    ).eval()

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}
        ]}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to("mps")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16)
        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    return predict


MODEL_REGISTRY["smolvlm"] = _load_smolvlm


def _load_phi35_vision() -> Callable[[np.ndarray], str]:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/Phi-3.5-vision-instruct"
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, num_crops=4
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="mps", trust_remote_code=True,
        torch_dtype=torch.bfloat16, _attn_implementation="eager",
    ).eval()

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        messages = [{"role": "user", "content": f"<|image_1|>\n{PROMPT}"}]
        prompt_text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to("mps")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=16, do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        decoded = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    return predict


MODEL_REGISTRY["phi35-vision"] = _load_phi35_vision


def _load_llava_phi3() -> Callable[[np.ndarray], str]:
    import base64
    import io
    import requests
    from PIL import Image

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def predict(page_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = requests.post(OLLAMA_URL, json={
            "model": "llava-phi3",
            "prompt": PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 16},
        }, timeout=60)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    return predict


MODEL_REGISTRY["llava-phi3"] = _load_llava_phi3
