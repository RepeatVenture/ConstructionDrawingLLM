"""
Parse and validate raw LLM text output into the BlueprintBid ExtractionResult schema.

Handles common LLM output quirks:
- Markdown code fences around JSON
- Trailing commas
- Missing fields (fills defaults)
- Confidence values in 0-100 range (normalizes to 0-1)
- Uppercase units (lowercases)
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from schemas.extraction import (
    ExtractionResult,
    ScopeItem,
    SpecSheetEntry,
    Specification,
    UnitType,
)

logger = logging.getLogger(__name__)


def parse_extraction_result(raw_text: str) -> ExtractionResult:
    """
    Parse the model's raw text output into a validated ExtractionResult.

    Tries multiple strategies to extract JSON from the model output.
    """
    # 1. Try to extract JSON from the text
    json_str = _extract_json_string(raw_text)
    if json_str is None:
        logger.warning("No JSON found in model output, returning empty result")
        return ExtractionResult()

    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s – attempting repair", e)
        data = _repair_and_parse(json_str)
        if data is None:
            logger.error("Could not parse JSON from model output")
            return ExtractionResult()

    # 3. Validate and normalize
    return _normalize(data)


def _extract_json_string(text: str) -> Optional[str]:
    """Extract JSON object from model output text."""
    # Strategy 1: Look for ```json ... ``` code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Strategy 2: Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Strategy 3: Just take from { to end
    return text[start:]


def _repair_and_parse(json_str: str) -> Optional[dict]:
    """Attempt to fix common JSON issues and re-parse."""
    s = json_str

    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Fix single quotes → double quotes (careful with apostrophes)
    # Only do this if there are no double quotes at all
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')

    # Remove control characters
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)

    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return {"scopeItems": parsed, "specSheet": [], "specifications": []}
        return parsed
    except json.JSONDecodeError:
        pass

    return None


def _normalize(data: dict) -> ExtractionResult:
    """Normalize a parsed dict into a validated ExtractionResult."""
    scope_items = _normalize_scope_items(data.get("scopeItems", []))
    spec_sheet = _normalize_spec_sheet(data.get("specSheet", []))
    specifications = _normalize_specifications(data.get("specifications", []))

    return ExtractionResult(
        scopeItems=scope_items,
        specSheet=spec_sheet,
        specifications=specifications,
    )


def _normalize_scope_items(items: list) -> List[ScopeItem]:
    """Validate and normalize scope items."""
    result = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        try:
            item = _normalize_single_scope_item(raw, i)
            if item is not None:
                result.append(item)
        except Exception as e:
            logger.warning("Skipping invalid scope item %d: %s", i, e)
    return result


def _normalize_single_scope_item(raw: dict, index: int) -> Optional[ScopeItem]:
    """Normalize a single scope item dict."""
    # Ensure required fields
    description = raw.get("description", "").strip()
    if not description:
        return None

    # Normalize quantity
    quantity = raw.get("quantity", 0)
    if isinstance(quantity, str):
        quantity = _parse_quantity_string(quantity)
    quantity = float(quantity) if quantity else 0

    # Normalize unit
    unit = str(raw.get("unit", "ea")).lower().strip()
    if unit not in ("ea", "lf", "sf", "lot"):
        # Try to guess
        if unit in ("each", "pc", "pcs", "piece", "pieces"):
            unit = "ea"
        elif unit in ("lf", "lft", "linear feet", "linear ft", "lin ft"):
            unit = "lf"
        elif unit in ("sf", "sqft", "square feet", "sq ft"):
            unit = "sf"
        else:
            unit = "ea"

    # Normalize confidence
    confidence = float(raw.get("confidence", 0.5))
    if confidence > 1.0:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))

    # Category
    category = raw.get("category", "General").strip()

    # IDs
    item_id = raw.get("id", f"item-{uuid.uuid4().hex[:8]}")

    # Page refs
    page_refs = raw.get("pageRefs", [1])
    if not isinstance(page_refs, list):
        page_refs = [page_refs]

    # Text snippets
    snippets = raw.get("textSnippets", [])
    if isinstance(snippets, str):
        snippets = [snippets]

    return ScopeItem(
        id=item_id,
        category=category,
        description=description,
        quantity=quantity,
        unit=UnitType(unit),
        confidence=confidence,
        pageRefs=page_refs,
        textSnippets=snippets,
    )


def _parse_quantity_string(s: str) -> float:
    """Parse quantity from strings like '13.5', '13'-6\"', '17 LF'."""
    s = s.strip()

    # Handle feet-inches: 13'-6" → 13.5
    ft_in = re.match(r"(\d+)['\u2032]\s*-?\s*(\d+)[\"″\u2033]?", s)
    if ft_in:
        feet = int(ft_in.group(1))
        inches = int(ft_in.group(2))
        return feet + inches / 12.0

    # Just extract the first number
    num = re.search(r"[\d.]+", s)
    if num:
        return float(num.group())

    return 0


def _normalize_spec_sheet(items: list) -> List[SpecSheetEntry]:
    """Validate spec sheet entries."""
    result = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        key = raw.get("key", "").strip()
        value = raw.get("value", "").strip()
        if not key or not value:
            continue

        confidence = float(raw.get("confidence", 0.5))
        if confidence > 1.0:
            confidence /= 100.0
        confidence = max(0.0, min(1.0, confidence))

        page_refs = raw.get("pageRefs", [1])
        if not isinstance(page_refs, list):
            page_refs = [page_refs]

        snippets = raw.get("textSnippets", [])
        if isinstance(snippets, str):
            snippets = [snippets]

        result.append(SpecSheetEntry(
            key=key,
            value=value,
            confidence=confidence,
            pageRefs=page_refs,
            textSnippets=snippets,
        ))
    return result


def _normalize_specifications(items: list) -> List[Specification]:
    """Validate specification entries."""
    result = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        # At least one field should be non-empty
        if not any(raw.get(k, "") for k in ("finish_code", "finish_name", "category", "manufacturer")):
            continue
        result.append(Specification(
            finish_code=str(raw.get("finish_code", "")),
            finish_name=str(raw.get("finish_name", "")),
            category=str(raw.get("category", "")),
            manufacturer=str(raw.get("manufacturer", "")),
            description=str(raw.get("description", "")),
            finish=str(raw.get("finish", "")),
        ))
    return result
