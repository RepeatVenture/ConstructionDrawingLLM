"""
Configuration for the BlueprintBid Local LLM inference system.
All settings can be overridden via environment variables.
"""
import os
from pathlib import Path
from enum import Enum


class ModelBackend(str, Enum):
    QWEN2_VL = "qwen2-vl"
    LLAVA = "llava"
    INTERNVL2 = "internvl2"


class QuantizationType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"


# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "model_cache")))

# ── Model settings ─────────────────────────────────────────────────────
MODEL_BACKEND = ModelBackend(os.getenv("MODEL_BACKEND", ModelBackend.QWEN2_VL))

MODEL_IDS = {
    ModelBackend.QWEN2_VL: os.getenv("QWEN2_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ModelBackend.LLAVA: os.getenv("LLAVA_MODEL_ID", "llava-hf/llava-v1.6-mistral-7b-hf"),
    ModelBackend.INTERNVL2: os.getenv("INTERNVL2_MODEL_ID", "OpenGVLab/InternVL2-8B"),
}

QUANTIZATION = QuantizationType(os.getenv("QUANTIZATION", QuantizationType.INT4))

# Maximum image dimension (longest side) before resize
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "2048"))

# ── Server settings ────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))

# API key for authentication (required in production)
API_KEY = os.getenv("BLUEPRINTBID_API_KEY", "")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

# ── Inference settings ─────────────────────────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.05"))

# ── OCR settings ───────────────────────────────────────────────────────
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "en")

# ── Concurrency ────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# ── Logging ────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_IMAGES = os.getenv("LOG_IMAGES", "false").lower() == "true"  # never log customer data in prod
