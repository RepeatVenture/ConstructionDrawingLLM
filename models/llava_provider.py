"""
LLaVA vision-language model provider.

Alternative to Qwen2.5-VL. LLaVA-NeXT (v1.6) with Mistral-7B backbone
offers good general vision understanding and can be fine-tuned on
domain-specific data.

Supports quantization via bitsandbytes (int4, int8).
"""
from __future__ import annotations

import gc
import logging
from typing import Optional

import torch
from PIL import Image

import config
from models.base import BaseVLMProvider

logger = logging.getLogger(__name__)


class LLaVAProvider(BaseVLMProvider):
    """LLaVA-NeXT inference backend."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        self._model_id = model_id or config.MODEL_IDS[config.ModelBackend.LLAVA]
        self._quantization = quantization or config.QUANTIZATION.value
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading LLaVA model=%s  quant=%s", self._model_id, self._quantization)

        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

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
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._processor = LlavaNextProcessor.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            self._model_id,
            **load_kwargs,
        )
        self._model.eval()

        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        logger.info("LLaVA loaded. GPU memory: %.1f GB", vram)

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

    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not self.is_loaded():
            self.load()

        # LLaVA uses "[INST] <image>\n{prompt} [/INST]" style
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt},
                ],
            },
        ]

        text_input = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=text_input,
            images=[image],
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
                do_sample=config.TEMPERATURE > 0,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        text = self._processor.decode(generated, skip_special_tokens=True)
        return text
