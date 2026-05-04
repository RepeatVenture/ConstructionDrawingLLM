"""
Qwen2.5-VL vision-language model provider.

This is the recommended model for construction drawing analysis:
- Excellent document and diagram understanding
- Strong structured output (JSON) capability
- 7B version runs on 24GB VRAM with INT4 quantization
- Good OCR built into the vision encoder

Supports quantization: none, int8, int4 (bitsandbytes), gptq, awq.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Optional

import torch
from PIL import Image

import config
from models.base import BaseVLMProvider

logger = logging.getLogger(__name__)


class Qwen2VLProvider(BaseVLMProvider):
    """Qwen2.5-VL inference backend with optional LoRA adapter."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        quantization: Optional[str] = None,
        lora_path: Optional[str] = None,
    ):
        self._model_id = model_id or config.MODEL_IDS[config.ModelBackend.QWEN2_VL]
        self._quantization = quantization or config.QUANTIZATION.value
        self._lora_path = lora_path or os.environ.get("LORA_PATH")
        self._model = None
        self._processor = None

    # ── lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return

        logger.info(
            "Loading Qwen2.5-VL model=%s  quant=%s",
            self._model_id,
            self._quantization,
        )

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        load_kwargs: dict = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }

        if self._quantization == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self._quantization == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif self._quantization in ("gptq", "awq"):
            # For GPTQ/AWQ the model repo itself should be a quantized checkpoint.
            # Just load normally; the config inside the repo handles it.
            pass
        # else "none" → full precision

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_id,
            **load_kwargs,
        )

        # Load LoRA adapter if specified (fine-tuned weights)
        if self._lora_path:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from %s", self._lora_path)
            self._model = PeftModel.from_pretrained(self._model, self._lora_path)

        self._model.eval()

        vram = _gpu_mem_used()
        logger.info("Model loaded. GPU memory used: %.1f GB", vram)

    def is_loaded(self) -> bool:
        return self._model is not None

    def model_name(self) -> str:
        return self._model_id

    def unload(self) -> None:
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── inference ──────────────────────────────────────────────────────

    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not self.is_loaded():
            self.load()

        # Build conversation in Qwen2-VL chat format
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Processor handles image conversion + chat template
        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repetition_penalty=config.REPETITION_PENALTY,
                do_sample=config.TEMPERATURE > 0,
            )

        # Decode only the generated portion (strip the input tokens)
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        text = self._processor.decode(generated, skip_special_tokens=True)

        logger.debug("Generated %d tokens", len(generated))
        return text


# ── helpers ────────────────────────────────────────────────────────────

def _gpu_mem_used() -> float:
    """Return GPU memory used in GB (0 if no GPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3
