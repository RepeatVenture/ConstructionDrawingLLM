"""
Lightweight demo server that runs without a GPU or large model.
Uses a mock VLM provider that returns realistic sample output.
This lets you verify the API, integration, and output format work correctly.

Start:  python server_demo.py
Test:   curl -X POST http://localhost:8000/analyze-drawing \
            -F "image=@any_image.png" -F "trade=millwork"
"""
from __future__ import annotations

import json
import random
import uuid
from PIL import Image

from models.base import BaseVLMProvider

# ── Example outputs per trade (realistic construction data) ───────────

DEMO_OUTPUTS = {
    "millwork": {
        "scopeItems": [
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Cabinetry", "description": "Wall cabinet 17 l/f", "quantity": 17, "unit": "lf", "confidence": 0.95, "pageRefs": [1], "textSnippets": ["Wall cabinet 17 LF", "Room 129"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Cabinetry", "description": "Base cabinet 11 l/f", "quantity": 11, "unit": "lf", "confidence": 0.93, "pageRefs": [1], "textSnippets": ["Base cabinet 11 LF"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Countertops", "description": "Solid surface top with loose 6\" backsplash", "quantity": 13.5, "unit": "lf", "confidence": 0.88, "pageRefs": [1], "textSnippets": ["Countertop 13'-6\""]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Hardware", "description": "Hafele Bar pull 155.00.961", "quantity": 22, "unit": "ea", "confidence": 0.90, "pageRefs": [1], "textSnippets": ["22 Hafele pulls"]},
        ],
        "specSheet": [
            {"key": "Cabinet Finish", "value": "PL-1", "confidence": 0.95, "pageRefs": [1], "textSnippets": ["Finish: PL-1"]},
            {"key": "Countertop Material", "value": "Solid Surface", "confidence": 0.88, "pageRefs": [1], "textSnippets": ["SS countertop"]},
        ],
        "specifications": [
            {"finish_code": "PL-1", "finish_name": "Plastic Laminate", "category": "Casework", "manufacturer": "Wilsonart", "description": "High pressure laminate", "finish": "Matte"},
        ],
    },
    "plumbing": {
        "scopeItems": [
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Fixtures", "description": "Lavatory sink - undermount", "quantity": 3, "unit": "ea", "confidence": 0.92, "pageRefs": [1], "textSnippets": ["LAV (3)"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Fixtures", "description": "Water closet - floor mount", "quantity": 4, "unit": "ea", "confidence": 0.94, "pageRefs": [1], "textSnippets": ["WC (4)"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Faucets", "description": "Single handle lavatory faucet, sensor", "quantity": 3, "unit": "ea", "confidence": 0.85, "pageRefs": [1], "textSnippets": ["Sensor faucet"]},
        ],
        "specSheet": [
            {"key": "Fixture Manufacturer", "value": "Kohler", "confidence": 0.80, "pageRefs": [1], "textSnippets": ["Kohler"]},
        ],
        "specifications": [],
    },
    "electrical": {
        "scopeItems": [
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Receptacles", "description": "Duplex receptacle 20A", "quantity": 12, "unit": "ea", "confidence": 0.91, "pageRefs": [1], "textSnippets": ["duplex outlet"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Switches", "description": "Single pole switch", "quantity": 6, "unit": "ea", "confidence": 0.89, "pageRefs": [1], "textSnippets": ["switch"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Lighting", "description": "2x4 recessed LED troffer", "quantity": 18, "unit": "ea", "confidence": 0.93, "pageRefs": [1], "textSnippets": ["2x4 LED"]},
        ],
        "specSheet": [],
        "specifications": [],
    },
    "flooring": {
        "scopeItems": [
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Floor Covering", "description": "LVT plank flooring", "quantity": 450, "unit": "sf", "confidence": 0.87, "pageRefs": [1], "textSnippets": ["LVT 450 SF"]},
            {"id": f"item-{uuid.uuid4().hex[:8]}", "category": "Base", "description": "4\" rubber base", "quantity": 120, "unit": "lf", "confidence": 0.85, "pageRefs": [1], "textSnippets": ["rubber base"]},
        ],
        "specSheet": [],
        "specifications": [],
    },
}


class DemoVLMProvider(BaseVLMProvider):
    """Mock provider that returns realistic sample data without a model."""

    def load(self) -> None:
        pass

    def is_loaded(self) -> bool:
        return True

    def model_name(self) -> str:
        return "demo-mock-provider (no GPU required)"

    def infer(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        # Detect trade from prompt
        trade = "millwork"
        for t in ("plumbing", "electrical", "flooring", "millwork"):
            if t in user_prompt.lower():
                trade = t
                break

        output = DEMO_OUTPUTS.get(trade, DEMO_OUTPUTS["millwork"])

        # Regenerate unique IDs each call
        for item in output.get("scopeItems", []):
            item["id"] = f"item-{uuid.uuid4().hex[:8]}"

        return json.dumps(output)


# ── Monkey-patch the provider factory to use demo provider ────────────

import models.provider_factory as pf

_demo = DemoVLMProvider()

def _get_demo_provider():
    return _demo

def _preload_noop():
    pass

pf.get_provider = _get_demo_provider
pf.preload_model = _preload_noop

# ── Now import and run the real server ────────────────────────────────

from server import app  # noqa: F401
import uvicorn
import config

if __name__ == "__main__":
    print("=" * 60)
    print("  BlueprintBid LLM – DEMO MODE (no GPU required)")
    print("  Returns realistic mock data for all trades")
    print(f"  http://localhost:{config.PORT}/health")
    print(f"  http://localhost:{config.PORT}/docs")
    print("=" * 60)
    uvicorn.run(
        "server_demo:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
    )
