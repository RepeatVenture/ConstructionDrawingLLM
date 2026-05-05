"""
Extract specific drawing sheets from the DCI Mt Kisco PDF based on scope references.
Uses OCR to identify sheet numbers and extracts matching pages.
"""
import fitz
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import easyocr

# Initialize OCR reader
reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        print("Initializing EasyOCR (this may take a moment)...")
        reader = easyocr.Reader(['en'], gpu=False)
    return reader


def extract_sheet_number_from_page(page: fitz.Page) -> str:
    """Try to find the sheet number on a drawing page using OCR."""
    
    # Get a smaller region (typically title block is at bottom right)
    # Convert page to image at lower resolution for faster OCR
    mat = fitz.Matrix(0.5, 0.5)  # 50% scale
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    
    # Use OCR to find text
    ocr = get_ocr_reader()
    results = ocr.readtext(img_data)
    
    # Look for patterns like "A8.01", "A8.02", etc.
    sheet_patterns = [
        r'\bA\d+\.\d+\b',  # A8.01, A8.02
        r'\b[A-Z]\d+\.\d+\b',  # Generic sheet numbers
    ]
    
    for (bbox, text, confidence) in results:
        for pattern in sheet_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].upper()
    
    return None


def find_drawings_by_reference(pdf_path: str) -> Dict[str, List[int]]:
    """
    Scan the PDF and build a map of sheet numbers to page numbers.
    Returns: {"A8.01": [5, 12], "A8.02": [6], ...}
    """
    doc = fitz.open(pdf_path)
    sheet_map = {}
    
    print(f"\nScanning PDF for sheet numbers (this will take a few minutes)...")
    print("=" * 70)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        sheet_num = extract_sheet_number_from_page(page)
        
        if sheet_num:
            if sheet_num not in sheet_map:
                sheet_map[sheet_num] = []
            sheet_map[sheet_num].append(page_num + 1)  # 1-indexed
            print(f"  Page {page_num + 1:3d}: Found sheet {sheet_num}")
        
        # Only scan first 30 pages for now (most relevant details are early)
        if page_num >= 29:
            print(f"\n  (Scanned first 30 pages)")
            break
    
    doc.close()
    return sheet_map


def parse_drawing_reference(ref: str) -> List[str]:
    """
    Parse drawing references like "8A-B/A8.02" into list of sheet numbers.
    Returns: ["A8.02"]
    """
    # Extract the main sheet number (after the /)
    match = re.search(r'/([A-Z]\d+\.\d+)', ref)
    if match:
        return [match.group(1)]
    
    # Try without slash
    match = re.search(r'\b([A-Z]\d+\.\d+)\b', ref)
    if match:
        return [match.group(1)]
    
    return []


def extract_training_pages(pdf_path: str, annotations_dir: Path, images_dir: Path):
    """Extract drawing pages that match the scope annotations."""
    
    # Build sheet number map
    sheet_map = find_drawings_by_reference(pdf_path)
    
    print(f"\n\nFound sheets: {list(sheet_map.keys())}")
    print("=" * 70)
    
    # Load annotations and extract corresponding pages
    doc = fitz.open(pdf_path)
    
    for ann_file in sorted(annotations_dir.glob("*.json")):
        with open(ann_file) as f:
            annotation = json.load(f)
        
        room_num = annotation["room_number"]
        room_name = annotation["room_name"]
        dwg_refs = annotation["drawing_refs"]
        
        print(f"\nRoom {room_num} ({room_name}):")
        print(f"  Drawing refs: {dwg_refs}")
        
        # Find matching pages
        pages_to_extract = set()
        for ref in dwg_refs:
            sheets = parse_drawing_reference(ref)
            for sheet in sheets:
                if sheet in sheet_map:
                    pages_to_extract.update(sheet_map[sheet])
                    print(f"    {ref} → Sheet {sheet} → Page(s) {sheet_map[sheet]}")
                else:
                    print(f"    {ref} → Sheet {sheet} → NOT FOUND")
        
        # Extract pages
        if pages_to_extract:
            for page_num in sorted(pages_to_extract):
                page = doc[page_num - 1]  # Convert to 0-indexed
                
                # Render at 150 DPI
                mat = fitz.Matrix(150/72, 150/72)
                pix = page.get_pixmap(matrix=mat)
                
                # Save with room-specific name
                img_name = f"room_{room_num}_{room_name.lower().replace(' ', '_')}_page_{page_num:03d}.png"
                img_path = images_dir / img_name
                pix.save(str(img_path))
                
                print(f"    ✓ Extracted page {page_num} → {img_name}")
                
                # Update annotation with correct page reference
                annotation["pageRefs"] = sorted(pages_to_extract)
        else:
            # No pages found, use a placeholder
            print(f"    ⚠ No matching pages found - will need manual page mapping")
            annotation["pageRefs"] = [1]
        
        # Save updated annotation
        with open(ann_file, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    doc.close()
    print(f"\n✓ Extraction complete!")


def main():
    pdf_path = "DCI/DCI Mt Kisco_100 CDs_R1 and R2_Stamped_04.30.25.pdf"
    annotations_dir = Path("data/raw/custom/annotations")
    images_dir = Path("data/raw/custom/images")
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("DCI Mt Kisco - Extract Training Pages")
    print("=" * 70)
    
    extract_training_pages(pdf_path, annotations_dir, images_dir)
    
    print(f"\n✓ Training data ready at:")
    print(f"    Images:      {images_dir}")
    print(f"    Annotations: {annotations_dir}")
    print(f"\nNext: python training/prepare_dataset.py custom")


if __name__ == "__main__":
    main()
