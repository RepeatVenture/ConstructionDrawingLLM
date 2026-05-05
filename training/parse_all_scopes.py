"""
Parse all scope sheets and create JSON training annotations.
Handles the standard Cavan/IMC scope format with room-based organization.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_scope_table(text: str) -> List[Dict[str, Any]]:
    """Parse the scope table into structured data.
    
    Handles three different formats:
    1. Format A (185 Marcy, 67 Irving): Room name line, then DWG# line, then items
    2. Format B (DCI): ITEM# RM# ROOM DWG# header line, then items
    3. Format C (Brightview): ITEM# RM# ROOM DWG# on every item line
    """
    
    lines = text.split('\n')
    items = []
    current_room = None
    current_rm_num = ""
    current_dwg = None
    
    # Pre-process to identify room headers (Format A)
    room_headers = set()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) > 60:
            continue
        
        # If this line is short, uppercase, and the NEXT line is a drawing ref, it's likely a room
        if line.isupper() and len(line) < 50 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if next line is a drawing reference (Format A pattern)
            if re.match(r'^[\d\-]+/[A-Z]-\d+$', next_line):
                room_headers.add(line)
    
    for i, line in enumerate(lines):
        orig_line = line
        line = line.strip()
        
        # Skip empty lines, headers, totals
        if not line or line.startswith('ITEM') or line.startswith('SCOPE OF WORK') or \
           line.startswith('Project Number') or line.startswith('Job:') or 'TOTALS' in line.upper() or \
           line.startswith('FIRST FLOOR') or line.startswith('SECOND FLOOR'):
            continue
        
        # Check if this is a drawing reference ONLY (Format A)
        if re.match(r'^[\d\-]+/[A-Z]-\d+$', line) and len(line) < 20:
            current_dwg = line
            continue
        
        # Check if this is a known room header (Format A)
        if line in room_headers:
            current_room = line
            current_rm_num = ""
            current_dwg = None  # Will be set on next DWG line
            continue
        
        # Check if this is a Format B/C header line: starts with number, room number, room name, dwg
        # Pattern: "1 122 MACHINE REPAIR 8A-B/A8.02" or "1 1142 LOBBY 1/ID3.1"
        header_match = re.match(r'^[\d\.]+\s+(\d+)\s+([A-Z\s]+?)\s+([\w\-,/]+\.\d+|[\w\-]+/[\w\d\.]+)(?:\s|$)', line)
        if header_match:
            current_rm_num = header_match.group(1)
            current_room = header_match.group(2).strip()
            current_dwg = header_match.group(3)
            continue
        
        # Parse item lines (has category + description)
        is_category = any(cat in line.upper() for cat in ['CABINETRY', 'COUNTERTOP', 'COUNTER', 'HARDWARE', 'DRAWER', 
                                                             'PANELING', 'TRIM', 'SHELVING', 'GLASS',
                                                             'PORTAL', 'COLUMN', 'BEAM', 'STAIR', 'CABINET', 'MISC'])
        
        if is_category:
            # First check if this line starts with item#/rm#/room/dwg (Format C - Brightview)
            item_line_match = re.match(r'^[\d\.]+\s+(\d+)\s+([A-Z\s]+?)\s+([\w\-,/]+\.\d+|[\w\-]+/[\w\d\.]+)\s+(.+)$', line)
            if item_line_match:
                # Brightview format - extract room info from this line
                line_rm_num = item_line_match.group(1)
                line_room = item_line_match.group(2).strip()
                line_dwg = item_line_match.group(3)
                line_content = item_line_match.group(4)
                
                # Update current room info
                current_rm_num = line_rm_num
                current_room = line_room
                current_dwg = line_dwg
                
                # Continue parsing from line_content
                line = line_content
            
            # Try to extract: category, description, finish, count, qty, unit
            item = {
                'room_number': current_rm_num,
                'room_name': current_room or "",
                'drawing_ref': current_dwg or "",
                'category': "",
                'description': "",
                'finish': "",
                'count': 1,
                'quantity': 0,
                'unit': "ea"
            }
            
            # Find and extract category
            remaining = line
            for cat in ['CABINETRY', 'COUNTERTOPS', 'COUNTERTOP', 'COUNTERS', 'COUNTER', 'HARDWARE', 'DRAWERS', 'DRAWER',
                       'PANELING', 'TRIM', 'SHELVING', 'GLASS', 'PORTAL', 'COLUMN', 'BEAMS', 'BEAM',
                       'STAIRS', 'STAIR', 'CABINET', 'MISC']:
                if cat in line.upper():
                    item['category'] = cat.title()
                    idx = line.upper().index(cat)
                    remaining = line[idx + len(cat):].strip()
                    break
            
            # Parse remaining text into components
            if remaining:
                # Split into tokens
                parts = remaining.split()
                
                # Collect description, finish, qty, unit
                desc_parts = []
                finish_parts = []
                qty = None
                unit = None
                count = 1
                
                idx = 0
                while idx < len(parts):
                    part = parts[idx]
                    
                    # Check if it's a finish code pattern
                    if re.match(r'^[A-Z]{2,4}[-\d]+[A-Z]?$', part) or re.match(r'^[MS]\d+/[SC]\d+$', part) or \
                       re.match(r'^[A-Z]{2}\d+$', part) or re.match(r'^[A-Z]\d+$', part):
                        finish_parts.append(part)
                        idx += 1
                        continue
                    
                    # Check if it's a quantity (digit possibly with decimal)
                    if re.match(r'^\d+(\.\d+)?$', part):
                        # Next token might be a unit
                        if idx + 1 < len(parts):
                            next_part = parts[idx + 1]
                            if next_part.upper() in ['EA', 'LF', 'SQFT', 'SF', 'LOT', 'L/F', 'SQ', 'FT', 'SQFT']:
                                qty = float(part)
                                unit = next_part.lower().replace('l/f', 'lf').replace('sqft', 'sf').replace('sq', 'sf')
                                idx += 2
                                continue
                        # Could still be a quantity or count
                        if qty is None:
                            qty = float(part)
                        idx += 1
                        continue
                    
                    # Check if it's a unit by itself
                    if part.upper() in ['EA', 'LF', 'SQFT', 'SF', 'LOT', 'L/F', 'SQ', 'FT']:
                        unit = part.lower().replace('l/f', 'lf').replace('sqft', 'sf').replace('sq', 'sf')
                        idx += 1
                        continue
                    
                    # Otherwise it's part of description
                    desc_parts.append(part)
                    idx += 1
                
                item['description'] = ' '.join(desc_parts)
                item['finish'] = ' '.join(finish_parts) if finish_parts else ""
                item['quantity'] = qty if qty else 1.0
                item['unit'] = unit if unit else 'ea'
                item['count'] = count
            
            if item['description'] and current_room:  # Only add if we have both description and room
                items.append(item)
    
    return items


def create_annotations_from_scope(project_name: str) -> int:
    """Create JSON annotations from parsed scope data."""
    
    scope_file = Path(f"data/raw/{project_name}/annotations/_scope_text.txt")
    output_dir = Path(f"data/raw/{project_name}/annotations")
    
    if not scope_file.exists():
        print(f"  ⚠ No scope file found for {project_name}")
        return 0
    
    # Read and parse scope
    with open(scope_file, 'r', encoding='utf-8', errors='ignore') as f:
        scope_text = f.read()
    
    items = parse_scope_table(scope_text)
    
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
        # Build scope items
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
        safe_name = room_key.lower().replace(' ', '_').replace('/', '_')
        output_file = output_dir / f"room_{safe_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        count += 1
    
    return count


def main():
    """Parse all project scope sheets."""
    
    print("\n" + "="*70)
    print("Parsing All Scope Sheets")
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
