"""
Modal serverless GPU deployment for BlueprintBid LLM.

Deploy:
    modal deploy modal_deploy.py

Local test:
    modal run modal_deploy.py

The deployed function auto-scales 0 → N based on demand
and uses A10G GPUs ($1.00/hr, 24GB VRAM).
"""
from __future__ import annotations

import modal

# ── Docker image definition ───────────────────────────────────────────

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # opencv deps
    .pip_install(
        # Core inference
        "torch>=2.2",
        "transformers>=4.40",
        "accelerate>=0.28",
        "bitsandbytes>=0.43",
        "Pillow>=10.0",
        "numpy>=1.26",
        # OCR
        "easyocr>=1.7",
        # Server utils
        "pydantic>=2.6",
    )
)

app = modal.App("blueprintbid-inference")

# ── Model volume (cache HuggingFace weights across cold starts) ───────

model_volume = modal.Volume.from_name(
    "blueprintbid-model-cache",
    create_if_missing=True,
)

MODEL_CACHE_PATH = "/model_cache"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


# ── Warm-up: download model into the volume ──────────────────────────

@app.function(
    image=gpu_image,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=1800,
)
def download_model():
    """Download model weights into the persistent volume (run once)."""
    import os
    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"Downloading {MODEL_ID} …")
    AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model_volume.commit()
    print("Download complete.")


# ── Inference class (keeps model warm between requests) ───────────────

@app.cls(
    image=gpu_image,
    gpu="A10G",
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=120,
    container_idle_timeout=300,  # keep warm 5 min
    allow_concurrent_inputs=3,
)
class DrawingAnalyzer:
    """Serverless GPU container that holds the VLM in memory."""

    @modal.enter()
    def load_model(self):
        import os
        os.environ["HF_HOME"] = MODEL_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        print("Loading model …")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            trust_remote_code=True,
        )
        self.model.eval()

        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Model loaded. VRAM: {vram:.1f} GB")

    @modal.method()
    def analyze(
        self,
        image_base64: str,
        text: str = "",
        trade: str = "millwork",
    ) -> dict:
        """
        Main inference entry point.

        Parameters
        ----------
        image_base64 : str – base64-encoded PNG
        text : str – optional OCR text
        trade : str – millwork | plumbing | electrical | flooring

        Returns dict with scopeItems, specSheet, specifications.
        """
        import base64
        import io
        import time

        import torch
        from PIL import Image

        t0 = time.perf_counter()

        # Decode image
        img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

        # Resize if needed
        max_dim = 2048
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # OCR fallback
        if not text.strip():
            try:
                import easyocr
                reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                import numpy as np
                results = reader.readtext(np.array(img), detail=0)
                text = "\n".join(results)
            except Exception as e:
                print(f"OCR failed: {e}")

        # Build prompts (inline to avoid import issues in Modal container)
        system_prompt, user_prompt = _build_prompt(trade, text)

        # Build conversation
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input], images=[img], padding=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.05,
                do_sample=True,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        raw_output = self.processor.decode(generated, skip_special_tokens=True)

        elapsed = time.perf_counter() - t0
        print(f"Inference done in {elapsed:.1f}s")

        # Parse JSON
        result = _parse_result(raw_output)
        return result


# ── Web endpoint wrapper ──────────────────────────────────────────────

@app.function(image=gpu_image, timeout=10)
@modal.web_endpoint(method="POST")
def analyze_drawing_endpoint(request: dict) -> dict:
    """
    HTTP POST endpoint:
      {
        "image_base64": "...",
        "text": "...",
        "trade": "millwork"
      }
    """
    image_b64 = request.get("image_base64", "")
    text = request.get("text", "")
    trade = request.get("trade", "millwork")

    if not image_b64:
        return {"error": "image_base64 is required"}

    analyzer = DrawingAnalyzer()
    return analyzer.analyze.remote(image_b64, text, trade)


# ── Prompt builder (self-contained for Modal) ─────────────────────────

def _build_prompt(trade: str, ocr_text: str) -> tuple:
    """Build system + user prompt for the given trade."""
    trade = trade.lower().strip()

    system_prompts = {
        "millwork": "You are an expert construction estimator specializing in millwork and casework. You analyze architectural construction drawings to extract quantities for bidding. Your expertise includes cabinetry (wall, base, tall), countertops, shelving, paneling, hardware (pulls, knobs, hinges), drawers, and trim. You understand dimension lines, finish codes (PL-1, SS-3), room labels, and standard units (LF, EA, SF).",
        "plumbing": "You are an expert construction estimator specializing in plumbing. You analyze MEP drawings to extract plumbing quantities. Your expertise includes fixtures (sinks, toilets, urinals), faucets, drains, pipes, valves, water heaters, and accessories. You understand plumbing symbols, pipe sizing, and fixture schedules.",
        "electrical": "You are an expert construction estimator specializing in electrical. You analyze electrical drawings to extract quantities. Your expertise includes receptacles, switches, lighting, panels, conduit, wire, fire alarm, and data/comm. You understand electrical symbols, circuit numbers, and wire gauge notation.",
        "flooring": "You are an expert construction estimator specializing in flooring. You analyze architectural drawings to extract flooring quantities. Your expertise includes carpet, VCT, LVT, tile, base, transitions, underlayment, and floor prep. You understand room finish schedules, floor patterns, and area calculations.",
    }

    system = system_prompts.get(trade, system_prompts["millwork"])

    categories = {
        "millwork": "Cabinetry, Countertops, Hardware, Shelving, Paneling, Drawers, Accessories, Trim",
        "plumbing": "Fixtures, Faucets, Drains, Pipes, Valves, Water Heaters, Accessories",
        "electrical": "Receptacles, Switches, Lighting, Panels, Conduit, Wire, Fire Alarm, Data/Comm",
        "flooring": "Floor Covering, Base, Transitions, Underlayment, Floor Prep, Accessories",
    }

    ocr_section = ""
    if ocr_text.strip():
        ocr_section = f"OCR-EXTRACTED TEXT:\n---\n{ocr_text[:3000]}\n---\n"

    user = f"""Analyze this construction drawing image for {trade} scope items.

{ocr_section}
CATEGORIES: {categories.get(trade, categories["millwork"])}

Respond with ONLY valid JSON:
{{
  "scopeItems": [
    {{"category": "...", "description": "...", "quantity": <number>, "unit": "<ea|lf|sf|lot>", "confidence": <0-1>, "pageRefs": [1], "textSnippets": ["..."]}}
  ],
  "specSheet": [
    {{"key": "...", "value": "...", "confidence": <0-1>, "pageRefs": [1], "textSnippets": ["..."]}}
  ],
  "specifications": [
    {{"finish_code": "...", "finish_name": "...", "category": "...", "manufacturer": "...", "description": "...", "finish": "..."}}
  ]
}}

RULES: unit must be lowercase (ea/lf/sf/lot). confidence 0-1. Convert dimensions (13'-6" = 13.5). If no items found, return empty arrays."""

    return system, user


# ── JSON parser (self-contained for Modal) ────────────────────────────

def _parse_result(raw: str) -> dict:
    """Extract and parse JSON from model output."""
    import json
    import re
    import uuid

    # Extract JSON
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    json_str = match.group(1) if match else None

    if not json_str:
        start = raw.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = raw[start:i + 1]
                        break

    if not json_str:
        return {"scopeItems": [], "specSheet": [], "specifications": []}

    # Fix trailing commas
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {"scopeItems": [], "specSheet": [], "specifications": []}

    # Normalize
    for item in data.get("scopeItems", []):
        if "id" not in item:
            item["id"] = f"item-{uuid.uuid4().hex[:8]}"
        if "unit" in item:
            item["unit"] = item["unit"].lower()
        if "confidence" in item and item["confidence"] > 1:
            item["confidence"] = item["confidence"] / 100

    for entry in data.get("specSheet", []):
        if "confidence" in entry and entry["confidence"] > 1:
            entry["confidence"] = entry["confidence"] / 100

    return {
        "scopeItems": data.get("scopeItems", []),
        "specSheet": data.get("specSheet", []),
        "specifications": data.get("specifications", []),
    }
