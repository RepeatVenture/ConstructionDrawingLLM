"""
Integration tests for the FastAPI server.
Run only when the server is NOT running (uses TestClient).
Does NOT require a GPU – uses a mock VLM provider.
"""
import base64
import io
import json
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image
from fastapi.testclient import TestClient


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_LLM_OUTPUT = json.dumps({
    "scopeItems": [
        {
            "id": "item-1",
            "category": "Cabinetry",
            "description": "Wall cabinet 17 l/f",
            "quantity": 17,
            "unit": "lf",
            "confidence": 0.95,
            "pageRefs": [1],
            "textSnippets": ["Wall cabinet 17 LF"],
        }
    ],
    "specSheet": [
        {
            "key": "Finish",
            "value": "PL-1",
            "confidence": 0.9,
            "pageRefs": [1],
            "textSnippets": ["Finish: PL-1"],
        }
    ],
    "specifications": [],
})


@pytest.fixture
def client():
    """Create a test client with a mocked VLM provider."""
    mock_provider = MagicMock()
    mock_provider.is_loaded.return_value = True
    mock_provider.model_name.return_value = "test-model"
    mock_provider.infer.return_value = MOCK_LLM_OUTPUT

    with patch("server.preload_model"), \
         patch("server.get_provider", return_value=mock_provider), \
         patch("server.extract_text", return_value=""):
        # Import after patching so lifespan uses mock
        from server import app
        yield TestClient(app)


class TestAnalyzeDrawingMultipart:
    def test_success(self, client):
        img_bytes = _make_test_image_bytes()
        resp = client.post(
            "/analyze-drawing",
            files={"image": ("page.png", img_bytes, "image/png")},
            data={"trade": "millwork", "text": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "scopeItems" in data
        assert len(data["scopeItems"]) == 1
        assert data["scopeItems"][0]["unit"] == "lf"

    def test_invalid_trade(self, client):
        img_bytes = _make_test_image_bytes()
        resp = client.post(
            "/analyze-drawing",
            files={"image": ("page.png", img_bytes, "image/png")},
            data={"trade": "hvac"},
        )
        assert resp.status_code == 400

    def test_empty_image(self, client):
        resp = client.post(
            "/analyze-drawing",
            files={"image": ("page.png", b"", "image/png")},
            data={"trade": "millwork"},
        )
        assert resp.status_code == 400


class TestAnalyzeDrawingJson:
    def test_success(self, client):
        img_b64 = base64.b64encode(_make_test_image_bytes()).decode()
        resp = client.post(
            "/analyze-drawing-json",
            json={
                "image_base64": img_b64,
                "text": "",
                "trade": "millwork",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["scopeItems"]) == 1


class TestHealth:
    def test_health(self, client):
        with patch("server.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] in ("ok", "loading")
