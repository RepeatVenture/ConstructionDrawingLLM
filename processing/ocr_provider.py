"""
OCR provider for extracting text from construction drawings.

Uses EasyOCR as default (good accuracy, easy install).
Also supports PaddleOCR as alternative for better accuracy on dense text.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PIL import Image
import numpy as np

import config

logger = logging.getLogger(__name__)

# Lazy-loaded OCR engine singleton
_ocr_engine = None


def _get_ocr_engine():
    """Lazy-init the OCR engine."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        import easyocr
        _ocr_engine = easyocr.Reader(
            [config.OCR_LANGUAGE],
            gpu=True,
            verbose=False,
        )
        logger.info("Initialized EasyOCR engine (GPU=%s)", True)
    except Exception as e:
        logger.warning("EasyOCR init failed (%s), trying PaddleOCR...", e)
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=config.OCR_LANGUAGE,
                use_gpu=True,
                show_log=False,
            )
            logger.info("Initialized PaddleOCR engine")
        except Exception as e2:
            logger.error("No OCR engine available: %s", e2)
            _ocr_engine = None

    return _ocr_engine


def extract_text(image: Image.Image) -> str:
    """
    Run OCR on a PIL Image and return all detected text joined by newlines.
    """
    if not config.ENABLE_OCR:
        return ""

    engine = _get_ocr_engine()
    if engine is None:
        logger.warning("OCR engine not available, returning empty text")
        return ""

    img_array = np.array(image)

    try:
        # EasyOCR
        import easyocr
        if isinstance(engine, easyocr.Reader):
            results = engine.readtext(img_array, detail=1, paragraph=False)
            # results = [(bbox, text, confidence), ...]
            lines = _merge_easyocr_results(results)
            text = "\n".join(lines)
            logger.debug("EasyOCR extracted %d lines", len(lines))
            return text
    except (ImportError, TypeError):
        pass

    try:
        # PaddleOCR
        results = engine.ocr(img_array, cls=True)
        if results and results[0]:
            lines = [line[1][0] for line in results[0]]
            text = "\n".join(lines)
            logger.debug("PaddleOCR extracted %d lines", len(lines))
            return text
    except Exception as e:
        logger.error("OCR extraction failed: %s", e)

    return ""


def _merge_easyocr_results(
    results: List[Tuple],
    y_threshold: int = 15,
) -> List[str]:
    """
    Merge EasyOCR detections that are on the same line (similar y-coordinate)
    into single lines, sorted top-to-bottom then left-to-right.
    """
    if not results:
        return []

    # Each result: (bbox, text, confidence)
    # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    entries = []
    for bbox, text, conf in results:
        if conf < 0.25:  # skip low confidence noise
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        entries.append((y_center, x_center, text.strip()))

    # Sort by y then x
    entries.sort(key=lambda e: (e[0], e[1]))

    # Merge entries with similar y into lines
    lines: List[str] = []
    current_line_parts: List[Tuple[float, str]] = []
    current_y: Optional[float] = None

    for y, x, text in entries:
        if current_y is None or abs(y - current_y) > y_threshold:
            # New line
            if current_line_parts:
                current_line_parts.sort(key=lambda p: p[0])
                lines.append(" ".join(p[1] for p in current_line_parts))
            current_line_parts = [(x, text)]
            current_y = y
        else:
            current_line_parts.append((x, text))

    if current_line_parts:
        current_line_parts.sort(key=lambda p: p[0])
        lines.append(" ".join(p[1] for p in current_line_parts))

    return lines
