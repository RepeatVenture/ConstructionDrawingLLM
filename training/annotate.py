"""
Simple annotation helper for labeling your own construction drawings.

Generates the JSON annotation files needed by the CustomAdapter
in prepare_dataset.py.

Usage:
  # Interactive mode — walks through each image in a directory
  python training/annotate.py --images path/to/drawings/

  # Validate existing annotations
  python training/annotate.py --validate training/data/raw/custom/

  # Convert from a CSV schedule export
  python training/annotate.py --from-csv schedule.csv --images drawings/ --trade millwork
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


VALID_UNITS = {"ea", "lf", "sf", "lot"}
VALID_TRADES = {"millwork", "plumbing", "electrical", "flooring", "general", "mechanical", "hvac"}

CATEGORY_SUGGESTIONS = {
    "millwork": [
        "Cabinetry", "Countertops", "Hardware", "Drawers", "Shelving",
        "Millwork Panels", "Door Frames", "Trim", "Crown Molding",
    ],
    "plumbing": [
        "Plumbing Fixtures", "Piping", "Valves", "Water Heater",
        "Drain", "Vent", "Backflow Preventer", "Cleanout",
    ],
    "electrical": [
        "Electrical", "Lighting", "Switches", "Outlets", "Panel",
        "Conduit", "Fire Alarm", "Data/Telecom", "Generator",
    ],
    "flooring": [
        "Floor Covering", "Base", "Carpet", "Tile", "VCT",
        "LVT", "Hardwood", "Epoxy", "Transition Strips",
    ],
    "general": [
        "Structure", "Glazing", "Hardware", "Painting",
        "Drywall", "Ceiling", "Doors", "Windows",
    ],
}


def create_annotation_interactive(image_path: str, trade: str = "general") -> Dict[str, Any]:
    """Walk user through annotating a single drawing image."""
    print(f"\n{'=' * 60}")
    print(f"  Annotating: {image_path}")
    print(f"  Trade: {trade}")
    print(f"{'=' * 60}")

    if trade in CATEGORY_SUGGESTIONS:
        print(f"\n  Suggested categories: {', '.join(CATEGORY_SUGGESTIONS[trade])}")

    scope_items = []
    print("\n  Enter scope items (blank description to finish):\n")

    idx = 1
    while True:
        print(f"  --- Item {idx} ---")
        description = input("    Description: ").strip()
        if not description:
            break

        category = input(f"    Category: ").strip()
        if not category:
            category = "General"

        qty_str = input("    Quantity: ").strip()
        try:
            quantity = float(qty_str)
        except ValueError:
            print("    Invalid quantity, skipping item")
            continue

        unit = input(f"    Unit ({'/'.join(VALID_UNITS)}): ").strip().lower()
        if unit not in VALID_UNITS:
            print(f"    Invalid unit '{unit}', defaulting to 'ea'")
            unit = "ea"

        conf_str = input("    Confidence (0-1, default 0.9): ").strip()
        try:
            confidence = float(conf_str) if conf_str else 0.9
            confidence = max(0, min(1, confidence))
        except ValueError:
            confidence = 0.9

        snippet = input("    Text snippet from drawing (optional): ").strip()

        scope_items.append({
            "id": f"custom-{idx}",
            "category": category,
            "description": description,
            "quantity": quantity,
            "unit": unit,
            "confidence": confidence,
            "pageRefs": [1],
            "textSnippets": [snippet] if snippet else [],
        })
        idx += 1

    # Spec sheet entries
    spec_sheet = []
    print("\n  Enter spec sheet entries (blank key to skip):\n")
    while True:
        key = input("    Spec key: ").strip()
        if not key:
            break
        value = input("    Spec value: ").strip()
        conf_str = input("    Confidence (0-1, default 0.9): ").strip()
        try:
            confidence = float(conf_str) if conf_str else 0.9
        except ValueError:
            confidence = 0.9

        spec_sheet.append({
            "key": key,
            "value": value,
            "confidence": confidence,
            "pageRefs": [1],
            "textSnippets": [],
        })

    # Specifications
    specifications = []
    print("\n  Enter finish specifications (blank code to skip):\n")
    while True:
        code = input("    Finish code (e.g., PL-1): ").strip()
        if not code:
            break
        name = input("    Finish name: ").strip()
        cat = input("    Category: ").strip()
        mfr = input("    Manufacturer: ").strip()
        desc = input("    Description: ").strip()
        finish = input("    Finish: ").strip()

        specifications.append({
            "finish_code": code,
            "finish_name": name,
            "category": cat,
            "manufacturer": mfr,
            "description": desc,
            "finish": finish,
        })

    annotation = {
        "trade": trade,
        "scopeItems": scope_items,
        "specSheet": spec_sheet,
        "specifications": specifications,
    }

    print(f"\n  Created {len(scope_items)} scope items, "
          f"{len(spec_sheet)} specs, {len(specifications)} finishes")

    return annotation


def validate_annotations(custom_dir: str) -> bool:
    """Validate all annotation files in a custom dataset directory."""
    ann_dir = Path(custom_dir) / "annotations"
    img_dir = Path(custom_dir) / "images"

    if not ann_dir.exists():
        print(f"ERROR: No annotations/ directory in {custom_dir}")
        return False

    errors = 0
    warnings = 0
    valid = 0

    for ann_file in sorted(ann_dir.glob("*.json")):
        if ann_file.name.startswith("_"):
            continue

        prefix = f"  {ann_file.name}: "

        try:
            with open(ann_file) as f:
                ann = json.load(f)
        except json.JSONDecodeError as e:
            print(f"{prefix}INVALID JSON — {e}")
            errors += 1
            continue

        # Check image exists
        stem = ann_file.stem
        img_found = False
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            if (img_dir / f"{stem}{ext}").exists():
                img_found = True
                break
        if not img_found:
            print(f"{prefix}WARNING — no matching image found")
            warnings += 1

        # Check trade
        trade = ann.get("trade", "")
        if trade not in VALID_TRADES:
            print(f"{prefix}WARNING — unknown trade '{trade}'")
            warnings += 1

        # Check scope items
        items = ann.get("scopeItems", [])
        if not items:
            print(f"{prefix}WARNING — no scope items")
            warnings += 1

        for i, item in enumerate(items):
            for key in ["category", "description", "quantity", "unit"]:
                if key not in item:
                    print(f"{prefix}ERROR — scopeItems[{i}] missing '{key}'")
                    errors += 1

            if "unit" in item and item["unit"] not in VALID_UNITS:
                print(f"{prefix}ERROR — scopeItems[{i}] invalid unit: {item['unit']}")
                errors += 1

            if "quantity" in item:
                try:
                    q = float(item["quantity"])
                    if q < 0:
                        print(f"{prefix}ERROR — scopeItems[{i}] negative quantity")
                        errors += 1
                except (ValueError, TypeError):
                    print(f"{prefix}ERROR — scopeItems[{i}] non-numeric quantity")
                    errors += 1

        valid += 1
        if not any(
            ann_file.name in line
            for line in [f"{prefix}WARNING", f"{prefix}ERROR"]
        ):
            print(f"{prefix}OK — {len(items)} items")

    print(f"\n  Validated {valid} files: {errors} errors, {warnings} warnings")
    return errors == 0


def convert_csv_to_annotations(
    csv_path: str,
    images_dir: str,
    trade: str = "general",
    output_dir: Optional[str] = None,
) -> int:
    """
    Convert a CSV schedule/takeoff export to annotation files.

    Expected CSV columns (flexible — uses best match):
      image, file, drawing     → matches to image file
      category, type           → scope item category
      description, item, name  → scope item description
      quantity, qty, count     → numeric quantity
      unit, uom                → ea/lf/sf/lot
    """
    import csv

    output = Path(output_dir or "training/data/raw/custom")
    (output / "annotations").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        # Auto-detect column mapping
        def find_col(*names):
            for n in names:
                for h in headers:
                    if n in h:
                        return reader.fieldnames[headers.index(h)]
            return None

        img_col = find_col("image", "file", "drawing", "filename")
        cat_col = find_col("category", "type", "trade")
        desc_col = find_col("description", "item", "name", "detail")
        qty_col = find_col("quantity", "qty", "count", "amount")
        unit_col = find_col("unit", "uom")

        if not desc_col:
            print("ERROR: Could not find a description/item column in CSV")
            return 0

        # Group rows by image
        image_items: Dict[str, List[Dict]] = {}
        for row in reader:
            img_name = row.get(img_col, "default") if img_col else "default"
            if img_name not in image_items:
                image_items[img_name] = []

            qty = 1.0
            if qty_col and row.get(qty_col):
                try:
                    qty = float(row[qty_col])
                except ValueError:
                    qty = 1.0

            unit = "ea"
            if unit_col and row.get(unit_col):
                u = row[unit_col].lower().strip()
                if u in VALID_UNITS:
                    unit = u

            image_items[img_name].append({
                "category": row.get(cat_col, "General") if cat_col else "General",
                "description": row.get(desc_col, ""),
                "quantity": qty,
                "unit": unit,
            })

    # Write annotation files
    count = 0
    for img_name, items in image_items.items():
        stem = Path(img_name).stem
        scope_items = []
        for i, item in enumerate(items):
            if not item["description"]:
                continue
            scope_items.append({
                "id": f"csv-{i + 1}",
                "category": item["category"],
                "description": item["description"],
                "quantity": item["quantity"],
                "unit": item["unit"],
                "confidence": 0.95,
                "pageRefs": [1],
                "textSnippets": [],
            })

        if scope_items:
            ann = {
                "trade": trade,
                "scopeItems": scope_items,
                "specSheet": [],
                "specifications": [],
            }
            ann_path = output / "annotations" / f"{stem}.json"
            with open(ann_path, "w") as f:
                json.dump(ann, f, indent=2)
            count += 1

    print(f"Created {count} annotation files in {output / 'annotations'}")
    print(f"Place matching images in {output / 'images'}")
    return count


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate construction drawings for training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to annotate interactively")
    group.add_argument("--validate", help="Validate annotations in a custom dataset dir")
    group.add_argument("--from-csv", help="Convert CSV schedule to annotation files")

    parser.add_argument("--trade", default="general", help="Trade for annotation")
    parser.add_argument("--output", default=None, help="Output directory for annotations")

    args = parser.parse_args()

    if args.validate:
        ok = validate_annotations(args.validate)
        sys.exit(0 if ok else 1)

    elif args.from_csv:
        convert_csv_to_annotations(
            csv_path=args.from_csv,
            images_dir=args.images or ".",
            trade=args.trade,
            output_dir=args.output,
        )

    elif args.images:
        img_dir = Path(args.images)
        output_dir = Path(args.output or "training/data/raw/custom")
        ann_dir = output_dir / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        image_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in image_exts
        )

        if not images:
            print(f"No images found in {img_dir}")
            sys.exit(1)

        print(f"Found {len(images)} images to annotate")
        print("Press Ctrl+C to stop at any time\n")

        for img_path in images:
            # Skip if already annotated
            ann_path = ann_dir / f"{img_path.stem}.json"
            if ann_path.exists():
                print(f"  Skipping {img_path.name} (already annotated)")
                continue

            try:
                annotation = create_annotation_interactive(str(img_path), args.trade)
            except (KeyboardInterrupt, EOFError):
                print("\n\nStopped. Progress saved.")
                break

            with open(ann_path, "w") as f:
                json.dump(annotation, f, indent=2)
            print(f"  Saved → {ann_path}")

            # Copy image to dataset
            import shutil
            dest_img = output_dir / "images" / img_path.name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_img)
