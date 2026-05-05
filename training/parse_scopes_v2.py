"""
Parse all scope sheets into JSON training annotations.
Handles both horizontal (185 Marcy/67 Irving) and vertical (DCI/Brightview) formats.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def detect_format(text: str) -> str:
    """Detect which scope format this document uses."""
    # All projects use vertical format (each field on separate line)
    return "vertical"


def parse_horizontal_format(text: str) -> List[Dict[str, Any]]:
    """Parse horizontal format (185 Marcy, 67 Irving)."""
    lines = text.split('\n')
    items = []
    current_room = None
    current_dwg = None
    
    # Pre-identify room headers
    room_headers = set()
    for i, line in enumerate(lines):
        line = line.strip()
        if line and line.isupper() and len(line) < 50 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^[\d\-]+/[A-Z]-\d+$', next_line):
                room_headers.add(line)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Room header
        if line in room_headers:
            current_room = line
            current_dwg = None
            continue
        
        # Drawing reference
        if re.match(r'^[\d\-]+/[A-Z]-\d+$', line) and len(line) < 20:
            current_dwg = line
            continue
        
        # Skip headers
        if 'ITEM' in line or 'SCOPE OF WORK' in line or 'Project Number' in line:
            continue
        
        # Item line with category
        if any(cat in line.upper() for cat in ['CABINETRY', 'COUNTERTOP', 'HARDWARE', 'DRAWER', 'PANELING', 'TRIM']):
            if not current_room:
                continue
            
            # Extract components
            desc_match = re.search(r'(CABINETRY|COUNTERTOPS?|HARDWARE|DRAWERS?|PANELING|TRIM|CABINET)(.*)', line, re.IGNORECASE)
            if not desc_match:
                continue
            
            category = desc_match.group(1).title()
            remaining = desc_match.group(2).strip()
            
            # Parse remaining: description finish? qty? unit?
            parts = remaining.split()
            desc = []
            finish = ""
            qty = 1.0
            unit = "ea"
            
            for i, part in enumerate(parts):
                if re.match(r'^[A-Z]{2}\d+[A-Z]?$', part):  # Finish code
                    finish = part
                elif re.match(r'^\d+(\.\d+)?$', part) and i + 1 < len(parts) and parts[i + 1].upper() in ['LF', 'EA', 'SF', 'SQFT']:
                    qty = float(part)
                    unit = parts[i + 1].lower().replace('l/f', 'lf')
                elif part.upper() not in ['LF', 'EA', 'SF', 'SQFT', 'L/F'] and not re.match(r'^\d+(\.\d+)?$', part):
                    desc.append(part)
            
            items.append({
                'room_name': current_room,
                'room_number': "",
                'drawing_ref': current_dwg or "",
                'category': category,
                'description': ' '.join(desc),
                'finish': finish,
                'quantity': qty,
                'unit': unit
            })
    
    return items


def parse_vertical_format(text: str) -> List[Dict[str, Any]]:
    """Parse vertical format (all projects use this)."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    items = []
    
    i = 0
    current_room = None
    current_rm_num = ""
    current_dwg = None
    
    # Category keywords to identify item lines (use exact matches only)
    category_exact = {'Cabinetry', 'Cabinets', 'Counters', 'Countertops', 'Hardware', 'Drawers', 'Drawer', 
                     'Paneling', 'Trim', 'Shelving', 'Glass', 'Portal', 'Column', 'Beams', 'Stairs', 'Misc',
                     'CABINETRY', 'CABINETS', 'PANELING', 'TRIM', 'HARDWARE', 'DRAWER', 'DRAWERS', 'COUNTERTOPS', 
                     'COUNTERS', 'SHELVING', 'GLASS', 'PORTAL', 'COLUMN', 'BEAMS', 'STAIRS', 'MISC'}
    
    while i < len(lines):
        line = lines[i]
        
        # Skip headers and totals
        if any(x in line for x in ['ITEM', 'SCOPE OF WORK', 'Project Number', 'Job:', 'FIRST FLOOR', 'SECOND FLOOR', 'TOTALS', 'RM #', 'ROOM', 'DWG#', 'CATEGORY', 'DESCRIPTION', 'FINISH', 'COUNT', 'QTY']):
            i += 1
            continue
        
        # Pattern 1: Number(s) at start, followed by room name, then DWG
        # Examples: "1 122" then "MACHINE REPAIR" then "8A-B/A8.02"
        #           "1.1 1142" then "LOBBY" then "1,2/ID3.1"
        if re.match(r'^[\d\.]+(\s+\d+)?$', line):
            # This is an item number / room number line
            parts = line.split()
            if len(parts) > 1:
                current_rm_num = parts[-1]
            else:
                current_rm_num = parts[0]
            
            i += 1
            
            # Next line should be room name
            if i < len(lines):
                potential_room = lines[i]
                # Check if it's a room name (not a category, not a DWG)
                if potential_room.isupper() and potential_room not in category_exact and \
                   not re.match(r'^[\w\-,/]+\.\d+$', potential_room) and \
                   not re.match(r'^[\w\-,]+/[\w\d\.]+$', potential_room):
                    current_room = potential_room
                    i += 1
                    
                    # Next line might be drawing ref
                    if i < len(lines) and (re.match(r'^[\w\-,/]+\.\d+$', lines[i]) or re.match(r'^[\w\-,]+/[\w\d\.]+$', lines[i])):
                        current_dwg = lines[i]
                        i += 1
            continue
        
        # Pattern 2: Room name directly (uppercase, short), then DWG on next line
        # Examples: "BATH" then "1/A-600"
        #           "KITCHEN A" then "2/A-610"
        if line.isupper() and len(line) < 50 and line not in category_exact:
            # Check if next line is a DWG reference
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^[\d\-]+/[A-Z]-\d+$', next_line):
                    # This is a room header
                    current_room = line
                    current_rm_num = ""
                    i += 1
                    current_dwg = lines[i]
                    i += 1
                    continue
        
        # Check if this is a category line (start of an item)
        is_category = line in category_exact
        
        if is_category and current_room:
            category = line.title()
            description = ""
            finish = ""
            count = 1
            qty = 1.0
            unit = "ea"
            
            # Next lines: description, finish?, count?, qty/unit
            i += 1
            
            # Get description (may be multi-line)
            desc_parts = []
            while i < len(lines):
                desc_line = lines[i]
                # Stop if we hit another category, finish code, or count/qty
                if desc_line in category_exact:
                    break
                # If it looks like a finish code pattern, stop
                if re.match(r'^[A-Z]{1,4}[-\d]+[A-Z]?$', desc_line) or \
                   re.match(r'^[MS]\d+/[SC]\d+$', desc_line) or \
                   desc_line.upper() in ['SATIN NICKEL', 'BLACK', 'WHITE', 'WHTE', 'PAINTED', 'PRIMED'] or \
                   re.match(r'^[A-Z]\d+$', desc_line):
                    break
                # If it looks like qty/unit, stop
                if re.match(r'^[\d\.]+\s*[A-Z/]+$', desc_line):
                    break
                # If it's a single digit (count), stop
                if re.match(r'^\d$', desc_line):
                    break
                # Otherwise, it's part of the description
                desc_parts.append(desc_line)
                i += 1
            
            description = ' '.join(desc_parts)
            
            if not description:
                continue
            
            # Check for finish code (optional)
            if i < len(lines):
                finish_line = lines[i]
                # Finish codes: WD01A, PL-1, M1/S1, C8, etc. or words like "Satin Nickel", "PAINTED"
                if re.match(r'^[A-Z]{1,4}[-\d]+[A-Z]?$', finish_line) or \
                   re.match(r'^[MS]\d+/[SC]\d+$', finish_line) or \
                   finish_line.upper() in ['SATIN NICKEL', 'BLACK', 'WHITE', 'WHTE', 'PAINTED', 'PRIMED'] or \
                   re.match(r'^[A-Z]\d+$', finish_line):
                    finish = finish_line
                    i += 1
            
            # Check for count (single digit, optional)
            if i < len(lines) and re.match(r'^\d$', lines[i]):
                count = int(lines[i])
                i += 1
            
            # Get qty and unit (usually together: "12.5 L/F", "5 EA")
            if i < len(lines):
                qty_line = lines[i]
                qty_match = re.match(r'^([\d\.]+)\s*([A-Z/]+)$', qty_line)
                if qty_match:
                    qty = float(qty_match.group(1))
                    unit = qty_match.group(2).lower().replace('l/f', 'lf').replace('sqft', 'sf')
                    i += 1
            
            # Add item
            items.append({
                'room_name': current_room,
                'room_number': current_rm_num,
                'drawing_ref': current_dwg or "",
                'category': category,
                'description': description,
                'finish': finish,
                'count': count,
                'quantity': qty,
                'unit': unit
            })
            continue
        
        # If we didn't match anything, advance
        i += 1
    
    return items


def create_annotations_from_scope(project_name: str) -> int:
    """Create JSON annotations from parsed scope data."""
    
    scope_file = Path(f"data/raw/{project_name}/annotations/_scope_text.txt")
    output_dir = Path(f"data/raw/{project_name}/annotations")
    
    if not scope_file.exists():
        print(f"  ⚠ No scope file found for {project_name}")
        return 0
    
    # Read scope
    with open(scope_file, 'r', encoding='utf-8', errors='ignore') as f:
        scope_text = f.read()
    
    # Detect format and parse
    fmt = detect_format(scope_text)
    if fmt == "horizontal":
        items = parse_horizontal_format(scope_text)
    else:
        items = parse_vertical_format(scope_text)
    
    if not items:
        print(f"  ⚠ No items parsed from {project_name} scope")
        return 0
    
    # Group by room
    rooms = {}
    for item in items:
        room_key = f"{item['room_number']}_{item['room_name']}"
        if room_key not in rooms:
            rooms[room_key] = {
                'room_number': item['room_number'],
                'room_name': item['room_name'],
                'drawing_ref': item['drawing_ref'],
                'items': []
            }
        rooms[room_key]['items'].append(item)
    
    # Create annotation file for each room
    count = 0
    for room_key, room_data in rooms.items():
        scope_items = []
        finish_codes = set()
        
        for item in room_data['items']:
            scope_item = {
                'category': item['category'],
                'description': item['description'],
                'quantity': item['quantity'],
                'unit': item['unit'],
                'confidence': 0.95,
                'pageRefs': [1],
                'textSnippets': [item['description'][:50]]
            }
            scope_items.append(scope_item)
            
            if item['finish']:
                finish_codes.add(item['finish'])
        
        # Build spec sheet
        spec_sheet = []
        for finish in finish_codes:
            spec_sheet.append({
                'key': 'Finish Code',
                'value': finish,
                'confidence': 0.95,
                'pageRefs': [1],
                'textSnippets': [finish]
            })
        
        # Create annotation
        annotation = {
            'trade': 'millwork',
            'room_number': room_data['room_number'],
            'room_name': room_data['room_name'],
            'drawing_refs': [room_data['drawing_ref']] if room_data['drawing_ref'] else [],
            'scopeItems': scope_items,
            'specSheet': spec_sheet,
            'specifications': []
        }
        
        # Save annotation file
        safe_name = room_key.lower().replace(' ', '_').replace('/', '_').replace(',', '')
        output_file = output_dir / f"room_{safe_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        count += 1
    
    return count


def main():
    """Parse all project scope sheets."""
    
    print("\n" + "="*70)
    print("Parsing All Scope Sheets (Multi-Format)")
    print("="*70)
    
    projects = ['DCI', '185 Marcy', '67 Irving', 'Brightview']
    
    total_annotations = 0
    
    for project in projects:
        print(f"\n{project}:")
        count = create_annotations_from_scope(project)
        print(f"  ✓ Created {count} room annotations")
        total_annotations += count
    
    print(f"\n{'='*70}")
    print(f"Total: {total_annotations} annotations created")
    print(f"\nNext step:")
    print(f"  python training/prepare_combined_dataset.py")


if __name__ == "__main__":
    main()
