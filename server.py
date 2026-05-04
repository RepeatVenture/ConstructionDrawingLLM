"""
FastAPI server for BlueprintBid local LLM construction drawing analysis.

Endpoints:
  POST /analyze-drawing          (multipart/form-data)
  POST /analyze-drawing-json     (application/json with base64 image)
  GET  /health                   (readiness check)

Authentication is enforced when REQUIRE_AUTH=true and
BLUEPRINTBID_API_KEY is set.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from models.provider_factory import get_provider, preload_model
from processing.image_processor import preprocess, image_to_base64
from processing.ocr_provider import extract_text
from processing.output_parser import parse_extraction_result
from prompts.trade_prompts import build_prompt
from schemas.extraction import (
    AnalyzeRequest,
    ExtractionResult,
    HealthResponse,
    Trade,
)

logger = logging.getLogger(__name__)

# ── concurrency limiter ───────────────────────────────────────────────
_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
    return _semaphore


# ── authentication ────────────────────────────────────────────────────

async def verify_api_key(
    authorization: Optional[str] = Header(None),
) -> None:
    """Check Bearer token when auth is required."""
    if not config.REQUIRE_AUTH:
        return
    if not config.API_KEY:
        raise HTTPException(500, "Server misconfiguration: API key not set")
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, config.API_KEY):
        raise HTTPException(403, "Invalid API key")


# ── lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights at startup."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    logger.info("Starting BlueprintBid LLM server …")
    logger.info(
        "Config: model=%s  quant=%s  max_concurrent=%d  auth=%s",
        config.MODEL_BACKEND.value,
        config.QUANTIZATION.value,
        config.MAX_CONCURRENT_REQUESTS,
        config.REQUIRE_AUTH,
    )
    preload_model()
    logger.info("Model ready – accepting requests")
    yield
    logger.info("Shutting down …")


# ── app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BlueprintBid Local LLM",
    description="Construction drawing analysis for scope extraction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── core inference logic ──────────────────────────────────────────────

async def _run_inference(
    image_bytes: bytes,
    text: str,
    trade: str,
) -> ExtractionResult:
    """Shared inference pipeline used by both endpoints."""
    sem = _get_semaphore()
    async with sem:
        t0 = time.perf_counter()

        # 1. Preprocess image
        img = preprocess(image_bytes)

        # 2. OCR (if enabled and no text provided)
        if not text.strip() and config.ENABLE_OCR:
            text = await asyncio.to_thread(extract_text, img)

        # 3. Build prompts
        system_prompt, user_prompt = build_prompt(trade, text)

        # 4. Run VLM inference (CPU-bound → run in thread)
        provider = get_provider()
        raw_output = await asyncio.to_thread(
            provider.infer, img, system_prompt, user_prompt
        )

        # 5. Parse and validate
        result = parse_extraction_result(raw_output)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Inference complete: trade=%s  items=%d  specs=%d  time=%.1fs",
            trade,
            len(result.scopeItems),
            len(result.specSheet),
            elapsed,
        )
        return result


# ── endpoints ─────────────────────────────────────────────────────────

@app.post(
    "/analyze-drawing",
    response_model=ExtractionResult,
    dependencies=[Depends(verify_api_key)],
)
async def analyze_drawing_multipart(
    image: UploadFile = File(..., description="PNG image of a construction drawing page"),
    text: str = Form("", description="Optional OCR-extracted text"),
    trade: str = Form("millwork", description="Trade to analyze"),
):
    """Analyze a construction drawing (multipart/form-data)."""
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Empty image file")

    # Validate trade
    trade = trade.lower().strip()
    valid_trades = {t.value for t in Trade}
    if trade not in valid_trades:
        raise HTTPException(400, f"Invalid trade '{trade}'. Must be one of: {valid_trades}")

    try:
        return await _run_inference(image_bytes, text, trade)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {e}")


@app.post(
    "/analyze-drawing-json",
    response_model=ExtractionResult,
    dependencies=[Depends(verify_api_key)],
)
async def analyze_drawing_json(req: AnalyzeRequest):
    """Analyze a construction drawing (JSON body with base64 image)."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    try:
        return await _run_inference(image_bytes, req.text, req.trade.value)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness / health check."""
    provider = get_provider()
    gpu_available = torch.cuda.is_available()

    return HealthResponse(
        status="ok" if provider.is_loaded() else "loading",
        model=provider.model_name(),
        quantization=config.QUANTIZATION.value,
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        max_concurrent=config.MAX_CONCURRENT_REQUESTS,
    )


# ── error handlers ────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower(),
    )
