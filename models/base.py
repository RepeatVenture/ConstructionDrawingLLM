"""
Abstract base class for vision-language model providers.
"""
from __future__ import annotations

import abc
from typing import Optional

from PIL import Image

from schemas.extraction import ExtractionResult


class BaseVLMProvider(abc.ABC):
    """Interface every VLM backend must implement."""

    @abc.abstractmethod
    def load(self) -> None:
        """Load model weights into memory / GPU."""

    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""

    @abc.abstractmethod
    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Run inference with the image + text prompt.

        Returns the raw text output from the model
        (should be JSON, but caller is responsible for parsing).
        """

    @abc.abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    def unload(self) -> None:
        """Free GPU memory (optional override)."""
        pass
