"""
Process all projects at once - extract pages and analyze scope sheets.
"""
import fitz
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re


def extract_pages_from_pdf(pdf_path: Path, output_dir: Path, prefix: str = "sheet") -> int:
    """Extract all pages from PDF as images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"  Extracting {total_pages} pages from {pdf_path.name}...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)
        
        output_path = output_dir / f"{prefix}_{page_num+1:03d}.png"
        pix.save(str(output_path))
        
        if (page_num + 1) % 25 == 0:
            print(f"    ✓ {page_num + 1}/{total_pages}...")
    
    doc.close()
    print(f"  ✓ Extracted {total_pages} pages\n")
    return total_pages


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc[page_num].get_text()
    doc.close()
    return text


def identify_project_files(project_dir: Path) -> Dict[str, Path]:
    """Identify which PDF is drawings, scope, and specs."""
    pdfs = list(project_dir.glob("*.pdf"))
    
    files = {
        "drawings": None,
        "scope": None,
        "specs": None
    }
    
    for pdf in pdfs:
        name_lower = pdf.name.lower()
        
        # Identify scope sheet
        if any(word in name_lower for word in ["scope", "takeoff"]):
            files["scope"] = pdf
        # Identify spec document
        elif any(word in name_lower for word in ["spec", "manual"]):
            files["specs"] = pdf
        # Identify drawings (usually largest or has "drawing", "architectural", "cd")
        elif any(word in name_lower for word in ["drawing", "architectural", "cd", "set"]):
            if files["drawings"] is None or pdf.stat().st_size > files["drawings"].stat().st_size:
                files["drawings"] = pdf
    
    # If no clear drawings file, use the largest PDF
    if files["drawings"] is None and pdfs:
        files["drawings"] = max(pdfs, key=lambda p: p.stat().st_size)
    
    return files


def process_project(project_name: str) -> Dict:
    """Process a single project - extract pages and prepare structure."""
    
    project_dir = Path("training") / project_name
    data_dir = Path("training/data/raw") / project_name
    images_dir = data_dir / "images" / "all_pages"
    annotations_dir = data_dir / "annotations"
    
    print(f"\n{'='*70}")
    print(f"Processing: {project_name}")
    print('='*70)
    
    # Identify files
    files = identify_project_files(project_dir)
    
    print(f"Files identified:")
    print(f"  Drawings: {files['drawings'].name if files['drawings'] else 'None'}")
    print(f"  Scope:    {files['scope'].name if files['scope'] else 'None'}")
    print(f"  Specs:    {files['specs'].name if files['specs'] else 'None'}")
    print()
    
    # Extract drawing pages
    if files["drawings"]:
        page_count = extract_pages_from_pdf(files["drawings"], images_dir, "sheet")
    else:
        print("  ⚠ No drawings found!")
        page_count = 0
    
    # Extract scope text (for later parsing)
    scope_text = ""
    if files["scope"]:
        print(f"  Reading scope document...")
        scope_text = extract_text_from_pdf(files["scope"])
        print(f"  ✓ Extracted {len(scope_text)} characters of scope text\n")
    
    # Extract specs text (for reference)
    specs_text = ""
    if files["specs"]:
        print(f"  Reading specifications...")
        specs_text = extract_text_from_pdf(files["specs"])
        print(f"  ✓ Extracted {len(specs_text)} characters of spec text\n")
    
    # Create annotations directory
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Save extracted text for manual review
    if scope_text:
        scope_txt_path = annotations_dir / "_scope_text.txt"
        with open(scope_txt_path, 'w', encoding='utf-8') as f:
            f.write(scope_text)
        print(f"  ✓ Saved scope text: {scope_txt_path}")
    
    if specs_text:
        specs_txt_path = annotations_dir / "_specs_text.txt"
        with open(specs_txt_path, 'w', encoding='utf-8') as f:
            f.write(specs_text)
        print(f"  ✓ Saved specs text: {specs_txt_path}")
    
    return {
        "project_name": project_name,
        "page_count": page_count,
        "has_scope": bool(scope_text),
        "has_specs": bool(specs_text),
        "scope_text": scope_text[:5000],  # First 5k chars for preview
        "specs_text": specs_text[:5000],
    }


def main():
    """Process all projects."""
    
    print("\n" + "="*70)
    print("Processing All Projects")
    print("="*70)
    
    # Find all project directories
    training_dir = Path("training")
    projects = []
    
    for item in training_dir.iterdir():
        if item.is_dir() and not item.name.startswith(('.', '_', 'data')):
            projects.append(item.name)
    
    print(f"\nFound {len(projects)} projects: {', '.join(projects)}\n")
    
    results = []
    for project in sorted(projects):
        result = process_project(project)
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    total_pages = 0
    for result in results:
        total_pages += result["page_count"]
        print(f"{result['project_name']:15s}: {result['page_count']:3d} pages | "
              f"Scope: {'✓' if result['has_scope'] else '✗'} | "
              f"Specs: {'✓' if result['has_specs'] else '✗'}")
    
    print(f"\nTotal: {total_pages} pages extracted")
    
    print(f"\nNext steps:")
    print(f"  1. Review scope text files in each project's annotations folder")
    print(f"  2. Create JSON annotations (or I can parse them)")
    print(f"  3. Run: python training/prepare_combined_dataset.py")


if __name__ == "__main__":
    main()
