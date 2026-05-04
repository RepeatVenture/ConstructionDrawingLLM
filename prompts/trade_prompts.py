"""
Trade-specific prompt templates for construction drawing analysis.

Each trade has:
- A system prompt explaining the role and construction knowledge
- A user prompt template that receives image description + OCR text
- Category lists and examples to guide structured output

The prompts are designed for vision-language models that receive both
the image and text context together.
"""
from __future__ import annotations

from typing import Dict

# ── JSON format instructions (shared across all trades) ────────────────

JSON_FORMAT_INSTRUCTIONS = """
You MUST respond with ONLY valid JSON. No markdown, no explanation, no extra text.

The JSON must have this exact structure:
{
  "scopeItems": [
    {
      "category": "<category from the list below>",
      "description": "<specific description with measurements/details>",
      "quantity": <number>,
      "unit": "<ea|lf|sf|lot>",
      "confidence": <0.0 to 1.0>,
      "pageRefs": [1],
      "textSnippets": ["<relevant text from the drawing>"]
    }
  ],
  "specSheet": [
    {
      "key": "<specification attribute>",
      "value": "<specification value>",
      "confidence": <0.0 to 1.0>,
      "pageRefs": [1],
      "textSnippets": ["<relevant text>"]
    }
  ],
  "specifications": [
    {
      "finish_code": "<code like PL-1>",
      "finish_name": "<full name>",
      "category": "<category>",
      "manufacturer": "<if specified>",
      "description": "<description>",
      "finish": "<finish type>"
    }
  ]
}

CRITICAL RULES:
- "unit" must be LOWERCASE: "ea", "lf", "sf", or "lot" (never "EA", "LF", etc.)
- "confidence" must be between 0 and 1 (not 0-100)
- "quantity" must be a number (convert dimensions: 13'-6" = 13.5)
- If no items found for this trade, return {"scopeItems": [], "specSheet": [], "specifications": []}
- Include ALL items you can identify, even with lower confidence
- Each scope item needs a unique, specific description
"""

# ── System prompts per trade ───────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {
    "millwork": """You are an expert construction estimator specializing in millwork and casework.
You analyze architectural construction drawings to extract quantities for bidding.

Your expertise includes:
- Cabinetry: wall cabinets, base cabinets, tall cabinets, custom built-ins
- Countertops: laminate, solid surface, quartz, granite, with backsplash details
- Shelving: adjustable, fixed, open, enclosed
- Paneling: wainscoting, wall panels, column wraps
- Hardware: pulls, knobs, hinges, drawer slides, locks
- Drawers: file drawers, box drawers, pencil drawers
- Accessories: paper towel holders, soap dispensers, towel bars

Construction drawing conventions you understand:
- LF = linear feet, EA = each, SF = square feet
- Dimension lines show measurements (e.g., 13'-6" means 13.5 linear feet)
- Finish codes (PL-1 = plastic laminate type 1, SS-3 = solid surface type 3)
- Room numbers and names identify locations
- Elevation drawings show cabinet heights and configurations
- Plan views show layout and dimensions""",

    "plumbing": """You are an expert construction estimator specializing in plumbing.
You analyze architectural and MEP construction drawings to extract plumbing quantities.

Your expertise includes:
- Fixtures: sinks, toilets, urinals, lavatories, drinking fountains
- Faucets: single-handle, double-handle, sensor-operated, commercial
- Drains: floor drains, trench drains, hub drains
- Pipes: supply lines, waste lines, vent lines
- Valves: shut-off valves, check valves, mixing valves
- Water heaters: tank, tankless, commercial
- Accessories: soap dispensers, paper towel dispensers, grab bars

Construction drawing conventions you understand:
- Plumbing symbols (circle with letter for fixture type)
- Pipe sizing notation (¾", 1", 1½", 2", 3", 4")
- CW = cold water, HW = hot water, W = waste
- Fixture unit counts
- Riser diagrams showing vertical pipe runs""",

    "electrical": """You are an expert construction estimator specializing in electrical.
You analyze electrical construction drawings to extract quantities.

Your expertise includes:
- Receptacles: duplex, GFCI, dedicated, floor outlets
- Switches: single-pole, 3-way, dimmer, occupancy sensor
- Lighting: recessed, pendant, surface mount, exit signs, emergency
- Panels: electrical panels, sub-panels, disconnects
- Conduit: EMT, rigid, flexible, underground
- Wire: by gauge and type
- Fire alarm: pull stations, detectors, horns/strobes
- Data/comm: data outlets, phone jacks, WAP locations

Construction drawing conventions you understand:
- Electrical symbols (circles, half-circles for outlets/switches)
- Circuit numbers and panel designations
- Home run lines (arrows pointing to panel)
- Voltage ratings (120V, 208V, 277V, 480V)
- Wire gauge notation (#12, #10, #8)""",

    "flooring": """You are an expert construction estimator specializing in flooring.
You analyze architectural construction drawings to extract flooring quantities.

Your expertise includes:
- Floor types: carpet, VCT, LVT/LVP, tile (ceramic, porcelain), concrete, epoxy
- Base: rubber base, wood base, tile base
- Transitions: reducer strips, thresholds, edge strips
- Underlayment: plywood, cement board, moisture barrier
- Floor preparation: leveling, patching
- Area calculations from room dimensions

Construction drawing conventions you understand:
- Room finish schedules showing floor types per room
- Floor pattern layouts and tile sizes
- Dimension lines and room areas
- Finish codes matching schedule legends
- Transition locations at doorways and material changes""",
}

# ── Categories per trade ───────────────────────────────────────────────

TRADE_CATEGORIES: Dict[str, list] = {
    "millwork": [
        "Cabinetry", "Countertops", "Hardware", "Shelving",
        "Paneling", "Drawers", "Accessories", "Trim",
    ],
    "plumbing": [
        "Fixtures", "Faucets", "Drains", "Pipes",
        "Valves", "Water Heaters", "Accessories",
    ],
    "electrical": [
        "Receptacles", "Switches", "Lighting", "Panels",
        "Conduit", "Wire", "Fire Alarm", "Data/Comm",
    ],
    "flooring": [
        "Floor Covering", "Base", "Transitions",
        "Underlayment", "Floor Prep", "Accessories",
    ],
}

# ── User prompt template ───────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """Analyze this construction drawing image for {trade} scope items.

{ocr_section}

CATEGORIES to use for this trade: {categories}

INSTRUCTIONS:
1. Examine the drawing carefully for all {trade}-related components
2. For each item found, extract: description, quantity, unit, and category
3. Convert all dimensions to numbers (e.g., 13'-6" = 13.5 LF, 4'-0" = 4.0 LF)
4. Count discrete items (fixtures, outlets, pulls) as "ea" (each)
5. Measure linear items (cabinets, countertops, base) as "lf" (linear feet)
6. Calculate area items (flooring, paneling) as "sf" (square feet)
7. Extract any finish codes, manufacturer details, or specifications
8. Set confidence based on how clearly visible/readable the information is

{JSON_FORMAT_INSTRUCTIONS}"""


def build_prompt(trade: str, ocr_text: str = "") -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for a given trade.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt)
    """
    trade_key = trade.lower().strip()
    if trade_key not in SYSTEM_PROMPTS:
        trade_key = "millwork"  # safe default

    system_prompt = SYSTEM_PROMPTS[trade_key]

    ocr_section = ""
    if ocr_text and ocr_text.strip():
        ocr_section = f"""OCR-EXTRACTED TEXT from this page (use as supplementary context):
---
{ocr_text.strip()[:3000]}
---
"""

    categories = ", ".join(TRADE_CATEGORIES[trade_key])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        trade=trade_key,
        ocr_section=ocr_section,
        categories=categories,
        JSON_FORMAT_INSTRUCTIONS=JSON_FORMAT_INSTRUCTIONS,
    )

    return system_prompt, user_prompt
