"""
Image preprocessing for construction drawings.

Handles:
- Decoding from base64 / raw bytes / file path
- Resizing to fit model input constraints
- Contrast enhancement for line drawings
- Rotation correction (deskewing)
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import config

logger = logging.getLogger(__name__)


def load_image(source: Union[str, bytes, Path]) -> Image.Image:
    """
    Load an image from base64 string, raw bytes, or file path.
    Returns an RGB PIL Image.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.exists():
            img = Image.open(path)
        else:
            # Assume base64 string
            raw = source if isinstance(source, str) else source.decode()
            # Strip optional data-URI prefix
            if "," in raw[:80]:
                raw = raw.split(",", 1)[1]
            img = Image.open(io.BytesIO(base64.b64decode(raw)))
    elif isinstance(source, bytes):
        try:
            img = Image.open(io.BytesIO(source))
        except Exception:
            img = Image.open(io.BytesIO(base64.b64decode(source)))
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")

    return img.convert("RGB")


def resize_image(
    img: Image.Image,
    max_dim: int = config.MAX_IMAGE_DIM,
) -> Image.Image:
    """Resize so the longest side is at most *max_dim* while preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.debug("Resizing image from %dx%d to %dx%d", w, h, new_w, new_h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def enhance_drawing(img: Image.Image) -> Image.Image:
    """
    Enhance a construction drawing for better model understanding.
    - Boost contrast (line drawings benefit from sharper lines)
    - Sharpen slightly
    """
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # Mild sharpening
    img = img.filter(ImageFilter.SHARPEN)

    return img


def preprocess(
    source: Union[str, bytes, Path],
    *,
    enhance: bool = True,
    max_dim: Optional[int] = None,
) -> Image.Image:
    """
    Full preprocessing pipeline.

    Returns a clean RGB PIL Image ready for model input.
    """
    img = load_image(source)
    img = resize_image(img, max_dim or config.MAX_IMAGE_DIM)
    if enhance:
        img = enhance_drawing(img)
    return img


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()
