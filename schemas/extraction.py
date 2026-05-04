"""
Pydantic models matching the exact BlueprintBid extraction output format.

These schemas are the contract between the local LLM server and the
BlueprintBid Next.js application.  **Do not change field names or types**
without updating the corresponding TypeScript types in
  lib/document-ai/types.ts
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────

class UnitType(str, Enum):
    EA = "ea"
    LF = "lf"
    SF = "sf"
    LOT = "lot"


class Trade(str, Enum):
    MILLWORK = "millwork"
    PLUMBING = "plumbing"
    ELECTRICAL = "electrical"
    FLOORING = "flooring"


# ── Scope Item ─────────────────────────────────────────────────────────

class ScopeItem(BaseModel):
    id: str = Field(default_factory=lambda: f"item-{uuid.uuid4().hex[:8]}")
    category: str = Field(..., description="e.g. Cabinetry, Hardware, Countertops, Drawers")
    description: str = Field(..., description="Specific description with details")
    quantity: float = Field(..., ge=0, description="Numeric quantity")
    unit: UnitType = Field(..., description="One of: ea, lf, sf, lot")
    confidence: float = Field(..., ge=0, le=1, description="0-1 confidence score")
    pageRefs: List[int] = Field(default_factory=lambda: [1])
    textSnippets: List[str] = Field(default_factory=list)

    @field_validator("unit", mode="before")
    @classmethod
    def lowercase_unit(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower().strip()
        return v


# ── Spec Sheet Entry ──────────────────────────────────────────────────

class SpecSheetEntry(BaseModel):
    key: str = Field(..., description="Specification attribute name")
    value: str = Field(..., description="Specification value")
    confidence: float = Field(..., ge=0, le=1)
    pageRefs: List[int] = Field(default_factory=lambda: [1])
    textSnippets: List[str] = Field(default_factory=list)


# ── Specification ──────────────────────────────────────────────────────

class Specification(BaseModel):
    finish_code: str = Field(default="", description="e.g. PL-1, SS-3")
    finish_name: str = Field(default="", description="e.g. Plastic Laminate")
    category: str = Field(default="", description="e.g. Casework, Flooring")
    manufacturer: str = Field(default="")
    description: str = Field(default="")
    finish: str = Field(default="")


# ── Full Extraction Result ─────────────────────────────────────────────

class ExtractionResult(BaseModel):
    """Top-level response matching the BlueprintBid expected format."""
    scopeItems: List[ScopeItem] = Field(default_factory=list)
    specSheet: List[SpecSheetEntry] = Field(default_factory=list)
    specifications: List[Specification] = Field(default_factory=list)


# ── Request model ──────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """JSON request body (alternative to multipart)."""
    image_base64: str = Field(..., description="Base64-encoded PNG image")
    text: str = Field(default="", description="OCR-extracted text (optional)")
    trade: Trade = Field(default=Trade.MILLWORK)


# ── Health / meta ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    quantization: str = ""
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    max_concurrent: int = 1
