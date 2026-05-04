"""
Convert the DCI Mt Kisco millwork scope document into training annotations.
Matches scope items to drawing page references and creates JSON annotations.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Parsed scope data from RM-DCI 25-006-P4 SCOPE.pdf
SCOPE_DATA = [
    {
        "item": 1,
        "rm_number": "122",
        "room": "MACHINE REPAIR",
        "dwg_refs": ["8A-B/A8.02"],
        "items": [
            {"category": "Cabinetry", "description": "Wall cabinets", "finish": "PL-1", "count": 5, "quantity": 12.5, "unit": "lf", "textSnippets": ["Wall cabinets", "PL-1", "12.5 L/F"]},
            {"category": "Cabinetry", "description": "Base cabinets", "finish": "PL-1", "count": 2, "quantity": 5, "unit": "lf", "textSnippets": ["Base cabinets", "PL-1", "5 L/F"]},
            {"category": "Cabinetry", "description": "Base cabinets (Drawer boxes)", "finish": "PL-1", "count": 1, "quantity": 1.5, "unit": "lf", "textSnippets": ["Base cabinets (Drawer boxes)", "1.5 L/F"]},
            {"category": "Cabinetry", "description": "Sink cabinet", "finish": "PL-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Sink cabinet", "2.5 L/F"]},
            {"category": "Drawers", "description": "Particleboard box with melamine", "finish": "PL-1", "count": 5, "quantity": 5, "unit": "ea", "textSnippets": ["Particleboard box", "melamine"]},
            {"category": "Paneling", "description": "Slopped soffit panel", "finish": "PL-1", "count": 1, "quantity": 15, "unit": "lf", "textSnippets": ["Slopped soffit panel", "15 L/F"]},
            {"category": "Hardware", "description": "Timberline cam locks", "finish": "Satin Nickel", "count": 13, "quantity": 13, "unit": "ea", "textSnippets": ["Timberline cam locks", "Satin Nickel"]},
            {"category": "Countertops", "description": "Solid surface top with 4\"h loose backsplash", "finish": "SS-1", "count": 2, "quantity": 17, "unit": "lf", "textSnippets": ["Solid surface top", "4\"h loose backsplash", "SS-1", "17 L/F"]},
            {"category": "Misc", "description": "Shop install solid surface integral sink (Sink supplied by plumber)", "finish": "SB-3", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["integral sink", "SB-3"]},
            {"category": "Hardware", "description": "2 1/2\" Plastic grommets", "finish": "Black", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["2 1/2\" Plastic grommets", "Black"]},
        ]
    },
    {
        "item": 2,
        "rm_number": "119",
        "room": "STAFF BREAK",
        "dwg_refs": ["9/A8.02"],
        "items": [
            {"category": "Cabinetry", "description": "Wall cabinets", "finish": "PL-1", "count": 3, "quantity": 7.5, "unit": "lf", "textSnippets": ["Wall cabinets", "PL-1", "7.5 L/F"]},
            {"category": "Cabinetry", "description": "Base cabinets", "finish": "PL-1", "count": 1, "quantity": 3, "unit": "lf", "textSnippets": ["Base cabinets", "3 L/F"]},
            {"category": "Cabinetry", "description": "Base cabinets (Drawer boxes)", "finish": "PL-1", "count": 1, "quantity": 2, "unit": "lf", "textSnippets": ["Base cabinets (Drawer boxes)", "2 L/F"]},
            {"category": "Cabinetry", "description": "Sink cabinet", "finish": "PL-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Sink cabinet", "2.5 L/F"]},
            {"category": "Drawers", "description": "Particleboard box with melamine", "finish": "PL-1", "count": 5, "quantity": 5, "unit": "ea", "textSnippets": ["Particleboard box", "melamine"]},
            {"category": "Paneling", "description": "Slopped soffit panel", "finish": "PL-1", "count": 1, "quantity": 9, "unit": "lf", "textSnippets": ["Slopped soffit panel", "9 L/F"]},
            {"category": "Hardware", "description": "Timberline cam locks", "finish": "Satin Nickel", "count": 9, "quantity": 9, "unit": "ea", "textSnippets": ["Timberline cam locks"]},
            {"category": "Countertops", "description": "Solid surface top with 4\"h loose backsplash", "finish": "SS-1", "count": 1, "quantity": 7.5, "unit": "lf", "textSnippets": ["Solid surface top", "SS-1", "7.5 L/F"]},
            {"category": "Misc", "description": "Shop install solid surface integral sink", "finish": "SB-3", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["integral sink", "SB-3"]},
        ]
    },
    {
        "item": 3,
        "rm_number": "117",
        "room": "SOILED WORK",
        "dwg_refs": ["7A/A8.01", "7B/A8.02"],
        "items": [
            {"category": "Cabinetry", "description": "Wall cabinets", "finish": "PL-1", "count": 5, "quantity": 12, "unit": "lf", "textSnippets": ["Wall cabinets", "12 L/F"]},
            {"category": "Cabinetry", "description": "37 1/2\"h Base cabinets", "finish": "PL-1", "count": 2, "quantity": 5.5, "unit": "lf", "textSnippets": ["37 1/2\"h Base cabinets", "5.5 L/F"]},
            {"category": "Cabinetry", "description": "Sink cabinet with slopped removable panel", "finish": "PL-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Sink cabinet", "slopped removable panel"]},
            {"category": "Drawers", "description": "Particleboard box with melamine", "finish": "PL-1", "count": 2, "quantity": 2, "unit": "ea", "textSnippets": ["Particleboard box"]},
            {"category": "Paneling", "description": "Slopped soffit panel", "finish": "PL-1", "count": 1, "quantity": 15, "unit": "lf", "textSnippets": ["Slopped soffit panel", "15 L/F"]},
            {"category": "Supports", "description": "37 1/2\" Plam end panel/support", "finish": "PL-1", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["end panel/support"]},
            {"category": "Hardware", "description": "Timberline cam locks", "finish": "Satin Nickel", "count": 9, "quantity": 9, "unit": "ea", "textSnippets": ["Timberline cam locks"]},
            {"category": "Countertops", "description": "Solid surface top with 4\"h loose backsplash", "finish": "SS-1", "count": 2, "quantity": 12, "unit": "lf", "textSnippets": ["Solid surface top", "12 L/F"]},
            {"category": "Misc", "description": "Shop install solid surface integral sink", "finish": "SB-2", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["integral sink", "SB-2"]},
            {"category": "Hardware", "description": "2 1/2\" Plastic grommets", "finish": "Black", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["grommets"]},
        ]
    },
    {
        "item": 4,
        "rm_number": "116",
        "room": "CLEAN WORK",
        "dwg_refs": ["6A,B/A8.01"],
        "items": [
            {"category": "Cabinetry", "description": "Wall cabinets", "finish": "PL-1", "count": 5, "quantity": 12, "unit": "lf", "textSnippets": ["Wall cabinets", "12 L/F"]},
            {"category": "Cabinetry", "description": "Base cabinets", "finish": "PL-1", "count": 3, "quantity": 5.5, "unit": "lf", "textSnippets": ["Base cabinets", "5.5 L/F"]},
            {"category": "Cabinetry", "description": "Sink cabinet with slopped removable panel", "finish": "PL-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Sink cabinet"]},
            {"category": "Drawers", "description": "Particleboard box with melamine", "finish": "PL-1", "count": 3, "quantity": 3, "unit": "ea", "textSnippets": ["Particleboard box"]},
            {"category": "Paneling", "description": "Slopped soffit panel", "finish": "PL-1", "count": 1, "quantity": 15, "unit": "lf", "textSnippets": ["Slopped soffit panel"]},
            {"category": "Hardware", "description": "Timberline cam locks", "finish": "Satin Nickel", "count": 11, "quantity": 11, "unit": "ea", "textSnippets": ["cam locks"]},
            {"category": "Countertops", "description": "Solid surface top with 4\"h loose backsplash", "finish": "SS-1", "count": 2, "quantity": 12, "unit": "lf", "textSnippets": ["Solid surface", "12 L/F"]},
            {"category": "Misc", "description": "Shop install solid surface integral sink", "finish": "SB-2", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["integral sink"]},
            {"category": "Hardware", "description": "2 1/2\" Plastic grommets", "finish": "Black", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["grommets"]},
        ]
    },
    {
        "item": 5,
        "rm_number": "109",
        "room": "TREATMENT",
        "dwg_refs": ["3A-C/A8.01", "4A-D/A8.01", "5A-D/A8.01"],
        "nurse_stations": ["A", "B", "C"],
        "items": [
            # Nurse Station A
            {"category": "Cabinetry", "description": "Base cabinets (Drawer boxes)", "finish": "PL-1", "count": 2, "quantity": 3, "unit": "lf", "textSnippets": ["NURSE STATION A", "Base cabinets", "3 L/F"], "station": "A"},
            {"category": "Cabinetry", "description": "Sink cabinet", "finish": "PL-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Sink cabinet", "2.5 L/F"], "station": "A"},
            {"category": "Drawers", "description": "Particleboard box with melamine", "finish": "PL-1", "count": 6, "quantity": 6, "unit": "ea", "textSnippets": ["Particleboard box"], "station": "A"},
            {"category": "Countertops", "description": "Plastic laminate top with 4\"h loose backsplash per ID1.20", "finish": "PL-2", "count": 1, "quantity": 10, "unit": "lf", "textSnippets": ["Plastic laminate top", "ID1.20"], "station": "A"},
            {"category": "Countertops", "description": "Solid surface top with 4\"h loose backsplash at sink cabinet", "finish": "SS-1", "count": 1, "quantity": 2.5, "unit": "lf", "textSnippets": ["Solid surface", "sink cabinet"], "station": "A"},
            {"category": "Countertops", "description": "11\" wall cap with 1\"edge on both sides with field seams", "finish": "SS-1", "count": 3, "quantity": 21, "unit": "lf", "textSnippets": ["wall cap", "21 L/F"], "station": "A"},
            {"category": "Hardware", "description": "Timberline cam locks", "finish": "Satin Nickel", "count": 6, "quantity": 6, "unit": "ea", "textSnippets": ["cam locks"], "station": "A"},
            {"category": "Supports", "description": "Surface mount counter brackets", "finish": "Black", "count": 2, "quantity": 2, "unit": "ea", "textSnippets": ["counter brackets"], "station": "A"},
            {"category": "Hardware", "description": "2 1/2\" Plastic grommets", "finish": "Black", "count": 2, "quantity": 2, "unit": "ea", "textSnippets": ["grommets"], "station": "A"},
            {"category": "Misc", "description": "Shop install solid surface integral sink", "finish": "SB-2", "count": 1, "quantity": 1, "unit": "ea", "textSnippets": ["integral sink"], "station": "A"},
            # Additional items for Nurse Stations B and C would go here...
        ]
    },
]

# Finish specifications
FINISH_SPECS = {
    "PL-1": {
        "finish_code": "PL-1",
        "finish_name": "Studio Teak",
        "category": "Plastic Laminate",
        "manufacturer": "Wilsonart",
        "description": "High-pressure decorative laminate",
        "finish": "Studio Teak 7960K-18"
    },
    "PL-2": {
        "finish_code": "PL-2",
        "finish_name": "Crisp Linen",
        "category": "Plastic Laminate",
        "manufacturer": "Wilsonart",
        "description": "High-pressure decorative laminate",
        "finish": "Crisp Linen 4942-38"
    },
    "SS-1": {
        "finish_code": "SS-1",
        "finish_name": "Pearl Mirage",
        "category": "Solid Surface",
        "manufacturer": "Wilsonart",
        "description": "Solid surface countertop material",
        "finish": "Pearl Mirage 9199MG"
    },
    "SB-1": {
        "finish_code": "SB-1",
        "finish_name": "Calm White",
        "category": "Sink",
        "manufacturer": "Wilsonart",
        "description": "Integral solid surface sink",
        "finish": "AK2615 Calm White"
    },
    "SB-2": {
        "finish_code": "SB-2",
        "finish_name": "Calm White",
        "category": "Sink",
        "manufacturer": "Wilsonart",
        "description": "Integral solid surface sink",
        "finish": "AV1513 Calm White"
    },
    "SB-3": {
        "finish_code": "SB-3",
        "finish_name": "Calm White",
        "category": "Sink",
        "manufacturer": "Wilsonart",
        "description": "Integral solid surface sink",
        "finish": "AK1413 Calm White"
    },
}


def create_annotation(room_item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a room's scope items into training annotation format."""
    
    scope_items = []
    for item in room_item["items"]:
        scope_item = {
            "category": item["category"],
            "description": item["description"],
            "quantity": item["quantity"],
            "unit": item["unit"],
            "confidence": 0.95,  # High confidence since this is ground truth
            "pageRefs": [1],  # Will be updated when we match to actual pages
            "textSnippets": item["textSnippets"]
        }
        scope_items.append(scope_item)
    
    # Build spec sheet entries
    spec_sheet = []
    finish_codes = set()
    for item in room_item["items"]:
        if item["finish"] and item["finish"] in FINISH_SPECS:
            finish_codes.add(item["finish"])
    
    for finish_code in finish_codes:
        if finish_code in FINISH_SPECS:
            spec_sheet.append({
                "key": "Finish Code",
                "value": finish_code,
                "confidence": 0.95,
                "pageRefs": [1],
                "textSnippets": [finish_code]
            })
    
    # Build specifications list
    specifications = []
    for finish_code in finish_codes:
        if finish_code in FINISH_SPECS:
            specifications.append(FINISH_SPECS[finish_code])
    
    return {
        "trade": "millwork",
        "room_number": room_item["rm_number"],
        "room_name": room_item["room"],
        "drawing_refs": room_item["dwg_refs"],
        "scopeItems": scope_items,
        "specSheet": spec_sheet,
        "specifications": specifications
    }


def main():
    """Generate training annotations from scope data."""
    
    output_dir = Path("training/data/raw/custom/annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Converting DCI Mt Kisco Scope to Training Annotations")
    print("=" * 70)
    
    # Create annotations for each room/area
    for room_data in SCOPE_DATA[:5]:  # Process first 5 items for now
        annotation = create_annotation(room_data)
        
        # Create filename from room number
        filename = f"room_{room_data['rm_number']}_{room_data['room'].lower().replace(' ', '_')}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"✓ Created: {filename}")
        print(f"  Room: {room_data['room']} (#{room_data['rm_number']})")
        print(f"  Drawing refs: {', '.join(room_data['dwg_refs'])}")
        print(f"  Scope items: {len(annotation['scopeItems'])}")
        print()
    
    print(f"\n✓ Annotations saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Extract corresponding drawing pages from the PDF")
    print(f"  2. Match annotations to drawing images")
    print(f"  3. Run: python training/prepare_dataset.py custom")


if __name__ == "__main__":
    main()
