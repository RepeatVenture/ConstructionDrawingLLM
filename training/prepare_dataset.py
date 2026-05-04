"""
Dataset preparation pipeline for fine-tuning Qwen2.5-VL on construction drawings.

Combines multiple public datasets into a unified VQA-style training format:
  { "image": <path>, "conversations": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}] }

Supported datasets:
  - CubiCasa5K       (5K floor plans with room/wall/symbol annotations)
  - SESYD            (synthetic eng. drawings with symbol bounding boxes)
  - DIDI / ds4sd     (P&ID piping & instrumentation diagrams)
  - FloorPlanCAD     (10K CAD-derived floor plan images)
  - CVC-FP           (122 well-annotated floor plans)
  - Swiss Dwellings  (22K apartment plans with room areas)
  - Custom           (your own annotated construction drawings)
"""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.getenv("TRAINING_DATA_DIR", "training/data"))
PREPARED_DIR = DATA_ROOT / "prepared"
RAW_DIR = DATA_ROOT / "raw"


# ═══════════════════════════════════════════════════════════════════════
#  Conversation builder — creates VQA pairs from annotations
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a construction-estimating AI that analyzes architectural and "
    "engineering drawings. Extract every scope item with category, description, "
    "quantity, unit (ea, lf, sf, lot), and a confidence score (0-1). "
    "Also extract spec sheet entries and specifications when visible. "
    "Return ONLY valid JSON matching the required schema."
)


def build_extraction_conversation(
    image_path: str,
    scope_items: List[Dict[str, Any]],
    spec_sheet: List[Dict[str, Any]] | None = None,
    specifications: List[Dict[str, Any]] | None = None,
    trade: str = "general",
) -> Dict[str, Any]:
    """Build a single training sample in the Qwen2.5-VL chat format."""
    answer = {
        "scopeItems": scope_items,
        "specSheet": spec_sheet or [],
        "specifications": specifications or [],
    }

    return {
        "image": image_path,
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analyze this {trade} construction drawing. "
                    "Extract ALL scope items, quantities, sizes, types, "
                    "specifications, and finishes. Return JSON."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, indent=2),
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Base adapter — each dataset implements this
# ═══════════════════════════════════════════════════════════════════════

class DatasetAdapter(ABC):
    """Converts a specific public dataset into our training format."""

    name: str = "base"

    @abstractmethod
    def download(self, dest: Path) -> None:
        """Download raw data to dest/."""

    @abstractmethod
    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        """Convert raw annotations → list of training samples."""

    def _save_samples(
        self, samples: List[Dict[str, Any]], out_dir: Path, prefix: str
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prefix}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        logger.info("Wrote %d samples → %s", len(samples), out_path)
        return out_path


# ═══════════════════════════════════════════════════════════════════════
#  CubiCasa5K adapter
# ═══════════════════════════════════════════════════════════════════════

class CubiCasa5KAdapter(DatasetAdapter):
    """
    CubiCasa5K: 5,000 real scanned floor plans with SVG annotations.
    Rooms, walls, doors, windows, fixtures, icons.
    https://github.com/CubiCasa/CubiCasa5k
    """

    name = "cubicasa5k"

    # Room type → estimating category mapping
    ROOM_CATEGORIES = {
        "Kitchen": "Cabinetry",
        "Bathroom": "Plumbing Fixtures",
        "Toilet": "Plumbing Fixtures",
        "LivingRoom": "Flooring",
        "Bedroom": "Flooring",
        "Hallway": "Flooring",
        "Storage": "Millwork",
        "Garage": "Concrete",
        "Balcony": "Exterior",
        "Outdoor": "Exterior",
    }

    ICON_ITEMS = {
        "Sink": ("Plumbing Fixtures", "Sink fixture", "ea"),
        "Toilet": ("Plumbing Fixtures", "Toilet fixture", "ea"),
        "Bathtub": ("Plumbing Fixtures", "Bathtub", "ea"),
        "Shower": ("Plumbing Fixtures", "Shower unit", "ea"),
        "Oven": ("Appliances", "Oven/range", "ea"),
        "Refrigerator": ("Appliances", "Refrigerator", "ea"),
        "Dishwasher": ("Appliances", "Dishwasher", "ea"),
        "WashingMachine": ("Appliances", "Washing machine", "ea"),
        "Stairs": ("Carpentry", "Staircase assembly", "ea"),
        "Fireplace": ("Masonry", "Fireplace", "ea"),
    }

    def download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading CubiCasa5K...")
        os.system(
            f"git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git {dest / 'CubiCasa5k'}"
        )

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        repo = raw_dir / "CubiCasa5k"
        samples = []

        # CubiCasa5K structure: {high_quality,high_quality_architectural,colorful}/*/
        for split_dir in ["high_quality", "high_quality_architectural", "colorful"]:
            split_path = repo / split_dir
            if not split_path.exists():
                continue

            for plan_dir in sorted(split_path.iterdir()):
                if not plan_dir.is_dir():
                    continue

                img_path = plan_dir / "F1_original.png"
                svg_path = plan_dir / "model.svg"

                if not img_path.exists():
                    # Try alternative names
                    for alt in ["F1_scaled.png", "image.png"]:
                        alt_path = plan_dir / alt
                        if alt_path.exists():
                            img_path = alt_path
                            break
                    else:
                        continue

                scope_items = self._parse_cubicasa_annotations(plan_dir, svg_path)
                if not scope_items:
                    continue

                # Copy image to output
                out_img = out_dir / "images" / f"cubicasa_{plan_dir.name}.png"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, out_img)

                sample = build_extraction_conversation(
                    image_path=str(out_img),
                    scope_items=scope_items,
                    trade="general",
                )
                samples.append(sample)

        self._save_samples(samples, out_dir, "cubicasa5k")
        return samples

    def _parse_cubicasa_annotations(
        self, plan_dir: Path, svg_path: Path
    ) -> List[Dict[str, Any]]:
        """Parse room types, icons, and structural elements from CubiCasa."""
        items = []
        idx = 1

        # Parse room annotations if available
        rooms_file = plan_dir / "rooms.txt"
        if rooms_file.exists():
            for line in rooms_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    room_type = parts[0]
                    try:
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        area_px = abs(x2 - x1) * abs(y2 - y1)
                        # Rough px→sf conversion (varies per plan scale)
                        area_sf = max(round(area_px / 500, 1), 10)
                    except (ValueError, IndexError):
                        area_sf = 100.0

                    cat = self.ROOM_CATEGORIES.get(room_type, "General")
                    items.append({
                        "id": f"cc-{idx}",
                        "category": cat,
                        "description": f"{room_type} — floor area",
                        "quantity": area_sf,
                        "unit": "sf",
                        "confidence": 0.80,
                        "pageRefs": [1],
                        "textSnippets": [room_type],
                    })
                    idx += 1

        # Parse icon/fixture annotations
        icons_file = plan_dir / "icons.txt"
        if icons_file.exists():
            icon_counts: Dict[str, int] = {}
            for line in icons_file.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    icon_type = parts[0]
                    icon_counts[icon_type] = icon_counts.get(icon_type, 0) + 1

            for icon_type, count in icon_counts.items():
                if icon_type in self.ICON_ITEMS:
                    cat, desc, unit = self.ICON_ITEMS[icon_type]
                    items.append({
                        "id": f"cc-{idx}",
                        "category": cat,
                        "description": desc,
                        "quantity": float(count),
                        "unit": unit,
                        "confidence": 0.85,
                        "pageRefs": [1],
                        "textSnippets": [icon_type],
                    })
                    idx += 1

        # Count doors/windows from SVG or annotation files
        for element, cat, desc in [
            ("Door", "Hardware", "Door"),
            ("Window", "Glazing", "Window unit"),
        ]:
            elem_file = plan_dir / f"{element.lower()}s.txt"
            if elem_file.exists():
                count = sum(1 for line in elem_file.read_text().splitlines() if line.strip())
                if count > 0:
                    items.append({
                        "id": f"cc-{idx}",
                        "category": cat,
                        "description": desc,
                        "quantity": float(count),
                        "unit": "ea",
                        "confidence": 0.85,
                        "pageRefs": [1],
                        "textSnippets": [f"{count} {element.lower()}s"],
                    })
                    idx += 1

        return items


# ═══════════════════════════════════════════════════════════════════════
#  SESYD adapter (synthetic architectural/electrical symbols)
# ═══════════════════════════════════════════════════════════════════════

class SESYDAdapter(DatasetAdapter):
    """
    SESYD: Synthetic engineering drawings with symbol ground truth.
    Includes floor plans and electrical diagrams.
    https://sesyd.labri.fr/
    """

    name = "sesyd"

    ARCH_SYMBOLS = {
        "door": ("Hardware", "Door", "ea"),
        "window": ("Glazing", "Window", "ea"),
        "sink": ("Plumbing Fixtures", "Sink", "ea"),
        "toilet": ("Plumbing Fixtures", "Toilet", "ea"),
        "bathtub": ("Plumbing Fixtures", "Bathtub", "ea"),
        "shower": ("Plumbing Fixtures", "Shower", "ea"),
        "sofa": ("Furniture", "Sofa", "ea"),
        "bed": ("Furniture", "Bed", "ea"),
        "table": ("Furniture", "Table", "ea"),
        "chair": ("Furniture", "Chair", "ea"),
        "stairs": ("Carpentry", "Staircase", "ea"),
        "washbasin": ("Plumbing Fixtures", "Wash basin", "ea"),
    }

    ELEC_SYMBOLS = {
        "outlet": ("Electrical", "Power outlet", "ea"),
        "switch": ("Electrical", "Light switch", "ea"),
        "light": ("Electrical", "Light fixture", "ea"),
        "receptacle": ("Electrical", "Receptacle", "ea"),
        "junction": ("Electrical", "Junction box", "ea"),
        "panel": ("Electrical", "Electrical panel", "ea"),
        "transformer": ("Electrical", "Transformer", "ea"),
        "motor": ("Electrical", "Motor", "ea"),
        "resistor": ("Electrical", "Resistor", "ea"),
        "capacitor": ("Electrical", "Capacitor", "ea"),
    }

    def download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(
            "SESYD must be downloaded manually from https://sesyd.labri.fr/\n"
            "  1. Download 'floorplans' and 'electrical' archives\n"
            "  2. Extract to %s/sesyd/",
            dest,
        )

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        sesyd_dir = raw_dir / "sesyd"
        samples = []

        # Process floor plan sets
        for subset in ["floorplans1", "floorplans2", "floorplans3"]:
            subset_dir = sesyd_dir / subset
            if not subset_dir.exists():
                continue
            samples.extend(
                self._process_sesyd_subset(subset_dir, out_dir, self.ARCH_SYMBOLS, "millwork")
            )

        # Process electrical diagram sets
        for subset in ["electrical1", "electrical2"]:
            subset_dir = sesyd_dir / subset
            if not subset_dir.exists():
                continue
            samples.extend(
                self._process_sesyd_subset(subset_dir, out_dir, self.ELEC_SYMBOLS, "electrical")
            )

        self._save_samples(samples, out_dir, "sesyd")
        return samples

    def _process_sesyd_subset(
        self,
        subset_dir: Path,
        out_dir: Path,
        symbol_map: Dict[str, Tuple[str, str, str]],
        trade: str,
    ) -> List[Dict[str, Any]]:
        samples = []
        gt_dir = subset_dir / "GT"
        img_dir = subset_dir / "images"

        if not gt_dir.exists() or not img_dir.exists():
            return []

        for gt_file in sorted(gt_dir.glob("*.xml")):
            img_name = gt_file.stem + ".png"
            img_path = img_dir / img_name
            if not img_path.exists():
                img_name = gt_file.stem + ".jpg"
                img_path = img_dir / img_name
                if not img_path.exists():
                    continue

            items = self._parse_sesyd_gt(gt_file, symbol_map)
            if not items:
                continue

            out_img = out_dir / "images" / f"sesyd_{subset_dir.name}_{gt_file.stem}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_img)

            sample = build_extraction_conversation(
                image_path=str(out_img),
                scope_items=items,
                trade=trade,
            )
            samples.append(sample)

        return samples

    def _parse_sesyd_gt(
        self, gt_file: Path, symbol_map: Dict[str, Tuple[str, str, str]]
    ) -> List[Dict[str, Any]]:
        """Parse SESYD XML ground truth file."""
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(gt_file)
        except ET.ParseError:
            return []

        root = tree.getroot()
        symbol_counts: Dict[str, int] = {}

        for elem in root.iter():
            symbol_type = elem.get("type", "").lower()
            if not symbol_type:
                symbol_type = elem.tag.lower()
            for key in symbol_map:
                if key in symbol_type:
                    symbol_counts[key] = symbol_counts.get(key, 0) + 1

        items = []
        idx = 1
        for sym, count in symbol_counts.items():
            cat, desc, unit = symbol_map[sym]
            items.append({
                "id": f"sesyd-{idx}",
                "category": cat,
                "description": desc,
                "quantity": float(count),
                "unit": unit,
                "confidence": 0.90,
                "pageRefs": [1],
                "textSnippets": [f"{count}x {desc}"],
            })
            idx += 1

        return items


# ═══════════════════════════════════════════════════════════════════════
#  DIDI adapter (P&ID diagrams from HuggingFace)
# ═══════════════════════════════════════════════════════════════════════

class DIDIAdapter(DatasetAdapter):
    """
    DIDI (ds4sd/DIDI): P&ID piping & instrumentation diagrams.
    Apache 2.0 license. Available on HuggingFace.
    """

    name = "didi"

    PIPE_SYMBOLS = {
        "valve": ("Plumbing", "Valve", "ea"),
        "gate_valve": ("Plumbing", "Gate valve", "ea"),
        "ball_valve": ("Plumbing", "Ball valve", "ea"),
        "check_valve": ("Plumbing", "Check valve", "ea"),
        "globe_valve": ("Plumbing", "Globe valve", "ea"),
        "butterfly_valve": ("Plumbing", "Butterfly valve", "ea"),
        "reducer": ("Plumbing", "Pipe reducer", "ea"),
        "tee": ("Plumbing", "Pipe tee", "ea"),
        "elbow": ("Plumbing", "Pipe elbow", "ea"),
        "flange": ("Plumbing", "Pipe flange", "ea"),
        "pump": ("Mechanical", "Pump", "ea"),
        "compressor": ("Mechanical", "Compressor", "ea"),
        "heat_exchanger": ("Mechanical", "Heat exchanger", "ea"),
        "tank": ("Mechanical", "Storage tank", "ea"),
        "vessel": ("Mechanical", "Pressure vessel", "ea"),
        "instrument": ("Instrumentation", "Instrument", "ea"),
        "controller": ("Instrumentation", "Controller", "ea"),
        "transmitter": ("Instrumentation", "Transmitter", "ea"),
        "indicator": ("Instrumentation", "Indicator", "ea"),
    }

    def download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading DIDI dataset from HuggingFace...")
        try:
            from datasets import load_dataset

            ds = load_dataset("ds4sd/DIDI", cache_dir=str(dest / "didi_cache"))
            # Save images to disk
            img_dir = dest / "didi" / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            meta = []
            for split_name in ds:
                for i, sample in enumerate(ds[split_name]):
                    img = sample.get("image")
                    if img is not None:
                        img_path = img_dir / f"{split_name}_{i:04d}.png"
                        if isinstance(img, Image.Image):
                            img.save(img_path)
                        meta.append({
                            "image": str(img_path),
                            "split": split_name,
                            "annotations": {
                                k: v for k, v in sample.items() if k != "image"
                            },
                        })

            meta_path = dest / "didi" / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)

            logger.info("Downloaded %d DIDI samples", len(meta))
        except ImportError:
            logger.error("Install 'datasets' package: pip install datasets")
            raise

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        meta_path = raw_dir / "didi" / "metadata.json"
        if not meta_path.exists():
            logger.warning("DIDI metadata not found. Run download first.")
            return []

        with open(meta_path) as f:
            meta = json.load(f)

        samples = []
        for entry in meta:
            img_path = entry["image"]
            if not Path(img_path).exists():
                continue

            annotations = entry.get("annotations", {})
            items = self._annotations_to_scope_items(annotations)

            out_img = out_dir / "images" / f"didi_{Path(img_path).stem}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_img)

            sample = build_extraction_conversation(
                image_path=str(out_img),
                scope_items=items,
                trade="plumbing",
            )
            samples.append(sample)

        self._save_samples(samples, out_dir, "didi")
        return samples

    def _annotations_to_scope_items(
        self, annotations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert DIDI annotation format to scope items."""
        items = []
        idx = 1

        # DIDI uses bounding boxes with class labels
        labels = annotations.get("labels", annotations.get("objects", []))
        if isinstance(labels, list):
            symbol_counts: Dict[str, int] = {}
            for label in labels:
                if isinstance(label, dict):
                    cls = label.get("category", label.get("label", "")).lower()
                else:
                    cls = str(label).lower()
                cls = cls.replace(" ", "_").replace("-", "_")
                for key in self.PIPE_SYMBOLS:
                    if key in cls:
                        symbol_counts[key] = symbol_counts.get(key, 0) + 1
                        break

            for sym, count in symbol_counts.items():
                cat, desc, unit = self.PIPE_SYMBOLS[sym]
                items.append({
                    "id": f"didi-{idx}",
                    "category": cat,
                    "description": desc,
                    "quantity": float(count),
                    "unit": unit,
                    "confidence": 0.88,
                    "pageRefs": [1],
                    "textSnippets": [f"{count}x {desc}"],
                })
                idx += 1

        # If no structured labels, create a generic P&ID item
        if not items:
            items.append({
                "id": "didi-1",
                "category": "Piping",
                "description": "P&ID diagram — manual review required",
                "quantity": 1.0,
                "unit": "lot",
                "confidence": 0.50,
                "pageRefs": [1],
                "textSnippets": ["P&ID sheet"],
            })

        return items


# ═══════════════════════════════════════════════════════════════════════
#  FloorPlanCAD adapter
# ═══════════════════════════════════════════════════════════════════════

class FloorPlanCADAdapter(DatasetAdapter):
    """
    FloorPlanCAD: 10,000+ CAD floor plans with panoptic annotations.
    https://floorplancad.github.io/
    """

    name = "floorplancad"

    ELEMENT_MAP = {
        "Wall": ("Structure", "Wall", "lf"),
        "Door": ("Hardware", "Door", "ea"),
        "Window": ("Glazing", "Window", "ea"),
        "Room": ("Flooring", "Room area", "sf"),
        "Railing": ("Carpentry", "Railing", "lf"),
        "Stairs": ("Carpentry", "Staircase", "ea"),
        "Column": ("Structure", "Column", "ea"),
        "Parking": ("Site Work", "Parking space", "ea"),
    }

    def download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FloorPlanCAD must be downloaded from https://floorplancad.github.io/\n"
            "  1. Request access and download the dataset\n"
            "  2. Extract to %s/floorplancad/",
            dest,
        )

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        fpcad_dir = raw_dir / "floorplancad"
        samples = []

        if not fpcad_dir.exists():
            logger.warning("FloorPlanCAD not found at %s", fpcad_dir)
            return []

        # FloorPlanCAD provides JSON annotations per image
        for ann_file in sorted(fpcad_dir.rglob("*.json")):
            try:
                with open(ann_file) as f:
                    ann = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            img_name = ann.get("image_name", ann_file.stem + ".png")
            img_path = ann_file.parent / img_name
            if not img_path.exists():
                # Try common paths
                for ext in [".png", ".jpg", ".svg"]:
                    alt = ann_file.parent / (ann_file.stem + ext)
                    if alt.exists():
                        img_path = alt
                        break
                else:
                    continue

            items = self._parse_floorplancad_annotation(ann)
            if not items:
                continue

            out_img = out_dir / "images" / f"fpcad_{ann_file.stem}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_img)

            sample = build_extraction_conversation(
                image_path=str(out_img),
                scope_items=items,
                trade="general",
            )
            samples.append(sample)

        self._save_samples(samples, out_dir, "floorplancad")
        return samples

    def _parse_floorplancad_annotation(
        self, ann: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        items = []
        idx = 1

        # Count elements by type
        elements = ann.get("elements", ann.get("annotations", []))
        if isinstance(elements, list):
            counts: Dict[str, int] = {}
            for elem in elements:
                etype = ""
                if isinstance(elem, dict):
                    etype = elem.get("type", elem.get("category", "")).strip()
                if etype:
                    counts[etype] = counts.get(etype, 0) + 1

            for etype, count in counts.items():
                matched = False
                for key, (cat, desc, unit) in self.ELEMENT_MAP.items():
                    if key.lower() in etype.lower():
                        items.append({
                            "id": f"fpcad-{idx}",
                            "category": cat,
                            "description": desc,
                            "quantity": float(count),
                            "unit": unit,
                            "confidence": 0.85,
                            "pageRefs": [1],
                            "textSnippets": [f"{count}x {etype}"],
                        })
                        idx += 1
                        matched = True
                        break
                if not matched and count > 0:
                    items.append({
                        "id": f"fpcad-{idx}",
                        "category": "General",
                        "description": etype,
                        "quantity": float(count),
                        "unit": "ea",
                        "confidence": 0.70,
                        "pageRefs": [1],
                        "textSnippets": [etype],
                    })
                    idx += 1

        return items


# ═══════════════════════════════════════════════════════════════════════
#  Swiss Dwellings adapter
# ═══════════════════════════════════════════════════════════════════════

class SwissDwellingsAdapter(DatasetAdapter):
    """
    Swiss Dwellings: 22K+ apartment floor plans with room areas.
    CC BY-SA 4.0.
    https://zenodo.org/record/7070952
    """

    name = "swiss_dwellings"

    def download(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Download Swiss Dwellings from https://zenodo.org/record/7070952\n"
            "  Extract to %s/swiss_dwellings/",
            dest,
        )

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        sw_dir = raw_dir / "swiss_dwellings"
        samples = []

        if not sw_dir.exists():
            return []

        # Swiss Dwellings provides floor plan images + CSV with room data
        csv_files = list(sw_dir.rglob("*.csv"))
        for csv_file in csv_files:
            import csv as csv_mod

            with open(csv_file, newline="") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    plan_id = row.get("plan_id", row.get("id", ""))
                    if not plan_id:
                        continue

                    # Find matching image
                    img_path = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        candidate = sw_dir / "images" / f"{plan_id}{ext}"
                        if candidate.exists():
                            img_path = candidate
                            break
                    if not img_path:
                        continue

                    items = self._row_to_scope_items(row)
                    if not items:
                        continue

                    out_img = out_dir / "images" / f"swiss_{plan_id}.png"
                    out_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img_path, out_img)

                    sample = build_extraction_conversation(
                        image_path=str(out_img),
                        scope_items=items,
                        trade="general",
                    )
                    samples.append(sample)

        self._save_samples(samples, out_dir, "swiss_dwellings")
        return samples

    def _row_to_scope_items(self, row: Dict[str, str]) -> List[Dict[str, Any]]:
        items = []
        idx = 1

        room_columns = {
            k: v for k, v in row.items()
            if "area" in k.lower() or "room" in k.lower()
        }

        for col_name, value in room_columns.items():
            try:
                area = float(value)
                if area <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            room_name = col_name.replace("_area", "").replace("area_", "").replace("_", " ").title()
            items.append({
                "id": f"swiss-{idx}",
                "category": "Flooring",
                "description": f"{room_name} — floor area",
                "quantity": round(area, 1),
                "unit": "sf",
                "confidence": 0.92,
                "pageRefs": [1],
                "textSnippets": [f"{room_name}: {area} sqm"],
            })
            idx += 1

        return items


# ═══════════════════════════════════════════════════════════════════════
#  Custom dataset adapter (your own drawings)
# ═══════════════════════════════════════════════════════════════════════

class CustomAdapter(DatasetAdapter):
    """
    Load your own annotated construction drawings.

    Expected format:
      training/data/raw/custom/
        images/
          drawing_001.png
          drawing_002.png
        annotations/
          drawing_001.json
          drawing_002.json

    Each annotation JSON should match:
    {
      "trade": "millwork",
      "scopeItems": [...],
      "specSheet": [...],
      "specifications": [...]
    }
    """

    name = "custom"

    def download(self, dest: Path) -> None:
        template_dir = dest / "custom" / "annotations"
        template_dir.mkdir(parents=True, exist_ok=True)
        (dest / "custom" / "images").mkdir(parents=True, exist_ok=True)

        # Write a template annotation file
        template = {
            "trade": "millwork",
            "scopeItems": [
                {
                    "category": "Cabinetry",
                    "description": "Wall cabinet, plastic laminate, 36\"W x 12\"D x 30\"H",
                    "quantity": 4.0,
                    "unit": "ea",
                    "confidence": 0.95,
                    "textSnippets": ["Type A wall cabinet"],
                },
                {
                    "category": "Countertops",
                    "description": "Solid surface countertop, 25.5\" deep",
                    "quantity": 17.0,
                    "unit": "lf",
                    "confidence": 0.90,
                    "textSnippets": ["CT-1 solid surface"],
                },
            ],
            "specSheet": [
                {"key": "Finish", "value": "PL-1 Plastic Laminate", "confidence": 0.9},
            ],
            "specifications": [
                {
                    "finish_code": "PL-1",
                    "finish_name": "Plastic Laminate",
                    "category": "Casework",
                    "manufacturer": "Wilsonart",
                    "description": "High-pressure laminate, Grade 10",
                    "finish": "Matte",
                },
            ],
        }
        template_path = template_dir / "_TEMPLATE.json"
        with open(template_path, "w") as f:
            json.dump(template, f, indent=2)

        logger.info(
            "Custom dataset template created at %s\n"
            "  1. Place drawing images in %s\n"
            "  2. Create matching JSON annotations in %s\n"
            "  3. See %s for the annotation format",
            dest / "custom",
            dest / "custom" / "images",
            template_dir,
            template_path,
        )

    def prepare(self, raw_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
        custom_dir = raw_dir / "custom"
        img_dir = custom_dir / "images"
        ann_dir = custom_dir / "annotations"
        samples = []

        if not ann_dir.exists():
            return []

        for ann_file in sorted(ann_dir.glob("*.json")):
            if ann_file.name.startswith("_"):
                continue  # Skip template

            try:
                with open(ann_file) as f:
                    ann = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to parse %s", ann_file)
                continue

            # Find matching image
            stem = ann_file.stem
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".tiff", ".pdf"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if not img_path:
                logger.warning("No image found for %s", ann_file)
                continue

            trade = ann.get("trade", "general")
            scope_items = ann.get("scopeItems", [])
            spec_sheet = ann.get("specSheet", [])
            specifications = ann.get("specifications", [])

            # Validate and add IDs if missing
            for i, item in enumerate(scope_items):
                if "id" not in item:
                    item["id"] = f"custom-{i + 1}"
                if "pageRefs" not in item:
                    item["pageRefs"] = [1]
                if "textSnippets" not in item:
                    item["textSnippets"] = []

            out_img = out_dir / "images" / f"custom_{stem}.png"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_img)

            sample = build_extraction_conversation(
                image_path=str(out_img),
                scope_items=scope_items,
                spec_sheet=spec_sheet,
                specifications=specifications,
                trade=trade,
            )
            samples.append(sample)

        self._save_samples(samples, out_dir, "custom")
        return samples


# ═══════════════════════════════════════════════════════════════════════
#  Registry & CLI
# ═══════════════════════════════════════════════════════════════════════

ADAPTERS: Dict[str, type] = {
    "cubicasa5k": CubiCasa5KAdapter,
    "sesyd": SESYDAdapter,
    "didi": DIDIAdapter,
    "floorplancad": FloorPlanCADAdapter,
    "swiss_dwellings": SwissDwellingsAdapter,
    "custom": CustomAdapter,
}


def download_datasets(names: List[str] | None = None) -> None:
    """Download specified datasets (or all if None)."""
    targets = names or list(ADAPTERS.keys())
    for name in targets:
        if name not in ADAPTERS:
            logger.error("Unknown dataset: %s (available: %s)", name, list(ADAPTERS.keys()))
            continue
        adapter = ADAPTERS[name]()
        adapter.download(RAW_DIR)


def prepare_all(
    names: List[str] | None = None,
    val_split: float = 0.1,
) -> Tuple[Path, Path]:
    """
    Prepare all datasets and merge into train/val JSONL files.

    Returns (train_path, val_path).
    """
    targets = names or list(ADAPTERS.keys())
    all_samples: List[Dict[str, Any]] = []

    for name in targets:
        if name not in ADAPTERS:
            continue
        adapter = ADAPTERS[name]()
        try:
            samples = adapter.prepare(RAW_DIR, PREPARED_DIR)
            logger.info("Dataset '%s': %d samples", name, len(samples))
            all_samples.extend(samples)
        except Exception as e:
            logger.warning("Failed to prepare dataset '%s': %s", name, e)

    if not all_samples:
        logger.error("No training samples produced! Download datasets first.")
        return Path(""), Path("")

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = max(1, int(len(all_samples) * (1 - val_split)))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PREPARED_DIR / "train.jsonl"
    val_path = PREPARED_DIR / "val.jsonl"

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    logger.info(
        "Dataset ready: %d train, %d val → %s",
        len(train_samples),
        len(val_samples),
        PREPARED_DIR,
    )
    return train_path, val_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare construction drawing datasets")
    parser.add_argument(
        "command",
        choices=["download", "prepare", "all"],
        help="download: fetch raw data | prepare: convert to training format | all: both",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=f"Specific datasets to process (default: all). Options: {list(ADAPTERS.keys())}",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )

    args = parser.parse_args()

    if args.command in ("download", "all"):
        download_datasets(args.datasets)

    if args.command in ("prepare", "all"):
        train_path, val_path = prepare_all(args.datasets, args.val_split)
        if train_path.exists():
            print(f"\nTrain: {train_path}")
            print(f"Val:   {val_path}")
            print("\nNext step: python training/finetune.py --train-data", train_path)
