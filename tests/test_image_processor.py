"""
Tests for the image processor module.
These run without a GPU.
"""
import base64
import io

import pytest
from PIL import Image

from processing.image_processor import (
    load_image,
    resize_image,
    enhance_drawing,
    preprocess,
    image_to_base64,
)


def _make_test_image(width: int = 800, height: int = 600) -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestLoadImage:
    def test_from_bytes(self):
        original = _make_test_image()
        raw = _image_to_bytes(original)
        loaded = load_image(raw)
        assert loaded.size == original.size

    def test_from_base64(self):
        original = _make_test_image()
        b64 = base64.b64encode(_image_to_bytes(original)).decode()
        loaded = load_image(b64)
        assert loaded.size == original.size

    def test_from_data_uri(self):
        original = _make_test_image()
        b64 = base64.b64encode(_image_to_bytes(original)).decode()
        data_uri = f"data:image/png;base64,{b64}"
        loaded = load_image(data_uri)
        assert loaded.size == original.size


class TestResizeImage:
    def test_no_resize_needed(self):
        img = _make_test_image(800, 600)
        result = resize_image(img, max_dim=2048)
        assert result.size == (800, 600)

    def test_resize_large(self):
        img = _make_test_image(4000, 3000)
        result = resize_image(img, max_dim=2048)
        assert max(result.size) <= 2048

    def test_aspect_ratio_preserved(self):
        img = _make_test_image(4000, 2000)
        result = resize_image(img, max_dim=2000)
        w, h = result.size
        assert abs(w / h - 2.0) < 0.01


class TestEnhanceDrawing:
    def test_returns_image(self):
        img = _make_test_image()
        result = enhance_drawing(img)
        assert isinstance(result, Image.Image)
        assert result.size == img.size


class TestPreprocess:
    def test_full_pipeline(self):
        original = _make_test_image(3000, 2000)
        raw = _image_to_bytes(original)
        result = preprocess(raw, max_dim=1024)
        assert max(result.size) <= 1024


class TestImageToBase64:
    def test_roundtrip(self):
        original = _make_test_image(100, 100)
        b64 = image_to_base64(original)
        reloaded = load_image(b64)
        assert reloaded.size == original.size
