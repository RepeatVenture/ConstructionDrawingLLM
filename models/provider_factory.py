"""
Factory for instantiating the configured VLM provider.
"""
from __future__ import annotations

import logging

import config
from models.base import BaseVLMProvider

logger = logging.getLogger(__name__)

# Singleton provider instance
_provider: BaseVLMProvider | None = None


def get_provider() -> BaseVLMProvider:
    """Return the singleton VLM provider, creating it on first call."""
    global _provider
    if _provider is not None:
        return _provider

    backend = config.MODEL_BACKEND

    if backend == config.ModelBackend.QWEN2_VL:
        from models.qwen2_vl import Qwen2VLProvider
        _provider = Qwen2VLProvider()
    elif backend == config.ModelBackend.LLAVA:
        from models.llava_provider import LLaVAProvider
        _provider = LLaVAProvider()
    elif backend == config.ModelBackend.INTERNVL2:
        # InternVL2 uses the same transformers API pattern as Qwen;
        # provide basic support via Qwen provider with overridden model_id.
        from models.qwen2_vl import Qwen2VLProvider
        _provider = Qwen2VLProvider(
            model_id=config.MODEL_IDS[config.ModelBackend.INTERNVL2],
        )
    else:
        raise ValueError(f"Unknown model backend: {backend}")

    logger.info("Created VLM provider: %s", backend.value)
    return _provider


def preload_model() -> None:
    """Eagerly load model weights (call at server startup)."""
    provider = get_provider()
    if not provider.is_loaded():
        provider.load()
