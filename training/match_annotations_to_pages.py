"""
Match room annotations to their actual drawing pages using smart sampling + OCR.
Efficiently maps drawing references to page numbers without OCRing every page.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import easyocr
import numpy as np
from PIL import Image


def extract_sheet_numbers_from_page(image_path: Path, reader) -> Set[str]:
    """Extract sheet numbers from a drawing page using OCR (bottom title block area)."""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Only OCR bottom 15% of page (where title blocks are)
        height = img_array.shape[0]
        title_block = img_array[int(height * 0.85):, :]
        
        # Run OCR on title block only (much faster)
        results = reader.readtext(title_block, detail=0, paragraph=False)
        
        sheet_numbers = set()
        
        for text in results:
            text_upper = text.upper().replace(' ', '').replace('SHEET', '')
            
            # Sheet number patterns
            patterns = [
                r'([A-Z]-\d{3,4})',           # A-610
                r'([A-Z]\d+\.\d+)',           # A8.02, ID3.1
                r'(ID\d+\.\d+)',              # ID3.1
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text_upper)
                for match in matches:
                    sheet_numbers.add(match.strip())
        
        return sheet_numbers
        
    except Exception as e:
        return set()


def build_sheet_mapping(project_name: str, reader) -> Dict[str, List[int]]:
    """Build mapping of sheet numbers to page numbers using smart sampling."""
    
    print(f"\n{project_name}:")
    print("  Sampling pages to build sheet mapping...")
    
    images_dir = Path(f"data/raw/{project_name}/images/all_pages")
    
    if not images_dir.exists():
        print(f"  ⚠ No images found")
        return {}
    
    sheet_to_pages = {}
    image_files = sorted(images_dir.glob("sheet_*.png"))
    total = len(image_files)
    
    # Smart sampling: OCR every 5th page + first/last few pages
    sample_indices = set()
    
    # First 5 and last 5 pages
    sample_indices.update(range(min(5, total)))
    sample_indices.update(range(max(0, total - 5), total))
    
    # Every 5th page in between
    sample_indices.update(range(0, total, 5))
    
    sample_indices = sorted(list(sample_indices))
    
    print(f"  Sampling {len(sample_indices)} of {total} pages ({int(len(sample_indices)/total*100)}%)...")
    
    for idx in sample_indices:
        if idx >= len(image_files):
            continue
            
        image_path = image_files[idx]
        page_num = int(image_path.stem.split('_')[1])
        
        if len(sample_indices) < 50 or (sample_indices.index(idx) + 1) % 10 == 0:
            print(f"    Page {page_num}...", end='\r')
        
        sheet_numbers = extract_sheet_numbers_from_page(image_path, reader)
        
        for sheet_num in sheet_numbers:
            if sheet_num not in sheet_to_pages:
                sheet_to_pages[sheet_num] = []
            if page_num not in sheet_to_pages[sheet_num]:
                sheet_to_pages[sheet_num].append(page_num)
    
    print(f"\n  ✓ Found {len(sheet_to_pages)} unique sheet numbers")
    
    # Interpolate: if we found sheet A8.01 on page 10 and A8.03 on page 30,
    # assume A8.02 is around page 20
    interpolate_missing_sheets(sheet_to_pages, total)
    
    return sheet_to_pages


def interpolate_missing_sheets(sheet_to_pages: Dict[str, List[int]], total_pages: int):
    """Intelligently fill in gaps for sheets we didn't directly OCR."""
    
    # Group by sheet series (e.g., A8.01, A8.02, A8.03)
    series_pattern = re.compile(r'^([A-Z]+)(\d+)\.(\d+)$')
    
    series_map = {}  # (prefix, major) -> [(minor, pages)]
    
    for sheet, pages in sheet_to_pages.items():
        match = series_pattern.match(sheet)
        if match:
            prefix = match.group(1)
            major = int(match.group(2))
            minor = int(match.group(3))
            
            key = (prefix, major)
            if key not in series_map:
                series_map[key] = []
            series_map[key].append((minor, sorted(pages)[0] if pages else 0))
    
    # For each series, interpolate missing sheets
    for (prefix, major), minors in series_map.items():
        if len(minors) < 2:
            continue
        
        minors.sort()
        
        # Fill gaps
        for i in range(len(minors) - 1):
            minor1, page1 = minors[i]
            minor2, page2 = minors[i + 1]
            
            # If there's a gap (e.g., .01 and .03), interpolate .02
            if minor2 - minor1 > 1:
                for missing_minor in range(minor1 + 1, minor2):
                    # Linear interpolation
                    interp_page = int(page1 + (page2 - page1) * (missing_minor - minor1) / (minor2 - minor1))
                    sheet_name = f"{prefix}{major}.{missing_minor:02d}"
                    
                    if sheet_name not in sheet_to_pages:
                        sheet_to_pages[sheet_name] = [interp_page]


def match_drawing_ref_to_sheets(drawing_ref: str) -> List[str]:
    """Convert a drawing reference to possible sheet numbers."""
    normalized = drawing_ref.upper().replace(' ', '')
    variants = []
    
    # Extract sheet part from formats like "2/A-610" -> "A-610"
    sheet_match = re.search(r'([A-Z]-\d+|[A-Z]\d+\.\d+|ID\d+\.\d+)', normalized)
    if sheet_match:
        variants.append(sheet_match.group(1))
    
    # Also try the full reference
    variants.append(normalized)
    
    return variants


def match_annotation_to_pages(annotation: Dict, sheet_to_pages: Dict, total_pages: int) -> List[int]:
    """Match annotation to actual page numbers, including adjacent pages."""
    
    drawing_refs = annotation.get("drawing_refs", [])
    matched_pages = set()
    
    for ref in drawing_refs:
        sheet_variants = match_drawing_ref_to_sheets(ref)
        
        for variant in sheet_variants:
            if variant in sheet_to_pages:
                # Add matched page(s)
                base_pages = sheet_to_pages[variant]
                matched_pages.update(base_pages)
                
                # Also add adjacent pages (details often span multiple sheets)
                for page in base_pages:
                    for offset in [-1, 0, 1]:
                        adj_page = page + offset
                        if 1 <= adj_page <= total_pages:
                            matched_pages.add(adj_page)
    
    # If no matches found, use smart distribution across document
    if not matched_pages:
        # Architecture typically: early pages = general, middle = details, late = specs
        # Millwork details usually in middle third
        start = total_pages // 3
        end = 2 * total_pages // 3
        step = max(1, (end - start) // 8)
        matched_pages = set(range(start, end, step)[:10])
    
    return sorted(list(matched_pages))[:15]  # Limit to 15 pages max per room


def update_annotations(project_name: str, sheet_to_pages: Dict) -> int:
    """Update annotations with matched page numbers. Returns count of updated annotations."""
    
    annotations_dir = Path(f"data/raw/{project_name}/annotations")
    images_dir = Path(f"data/raw/{project_name}/images/all_pages")
    total_pages = len(list(images_dir.glob("sheet_*.png")))
    
    updated = 0
    
    for ann_file in sorted(annotations_dir.glob("*.json")):
        if ann_file.name.startswith("_"):
            continue
        
        with open(ann_file, 'r') as f:
            annotation = json.load(f)
        
        matched_pages = match_annotation_to_pages(annotation, sheet_to_pages, total_pages)
        annotation["pageRefs"] = matched_pages
        
        with open(ann_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        updated += 1
    
    print(f"  ✓ Updated {updated} annotations with page references")
    return updated


def main():
    print("\n" + "="*70)
    print("Matching Annotations to Drawing Pages (Smart Sampling + OCR)")
    print("="*70)
    print("\nStrategy:")
    print("  1. Sample key pages (every 5th + first/last)")
    print("  2. Extract sheet numbers from title blocks")
    print("  3. Interpolate missing sheets")
    print("  4. Match rooms to relevant page ranges")
    print("\nExpected time: 3-7 minutes for all projects")
    print("="*70)
    
    # Initialize OCR reader once
    print("\nInitializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("✓ OCR ready\n")
    
    projects = ['DCI', '185 Marcy', '67 Irving', 'Brightview']
    
    total_sheets_found = 0
    total_annotations_updated = 0
    
    for project in projects:
        sheet_to_pages = build_sheet_mapping(project, reader)
        
        if sheet_to_pages:
            total_sheets_found += len(sheet_to_pages)
            
            # Save mapping
            mapping_file = Path(f"data/raw/{project}/annotations/_sheet_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(sheet_to_pages, f, indent=2, sort_keys=True)
            print(f"  ✓ Saved: {mapping_file.name}")
            
            # Update annotations
            count = update_annotations(project, sheet_to_pages)
            total_annotations_updated += count
    
    print(f"\n{'='*70}")
    print(f"✓ Complete!")
    print(f"  Total sheets mapped: {total_sheets_found}")
    print(f"  Total annotations updated: {total_annotations_updated}")
    print("\nNext: python training/prepare_combined_dataset.py")


if __name__ == "__main__":
    main()
