"""
Process a new project's drawings and scope into training data.
This script handles any project - just provide the folder structure.

Usage:
  python add_new_project.py PROJECT_NAME

Example:
  python add_new_project.py NewYorkOffice

Expected folder structure:
  training/
    PROJECT_NAME/
      drawings.pdf         (construction drawings)
      scope.pdf           (takeoff/scope document)
      scope_data.json     (or create manually)
"""

import sys
import json
import fitz
from pathlib import Path
from typing import Dict, List, Any
import shutil


def extract_pages_from_pdf(pdf_path: Path, output_dir: Path):
    """Extract all pages from PDF as images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    print(f"Extracting {len(doc)} pages from {pdf_path.name}...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(150/72, 150/72)
        pix = page.get_pixmap(matrix=mat)
        
        output_path = output_dir / f"sheet_{page_num+1:03d}.png"
        pix.save(str(output_path))
        
        if (page_num + 1) % 10 == 0:
            print(f"  ✓ {page_num + 1}/{len(doc)}...")
    
    doc.close()
    print(f"✓ Extracted {len(doc)} pages\n")


def create_project_structure(project_name: str):
    """Set up the training data structure for a new project."""
    
    project_dir = Path("training") / project_name
    
    if not project_dir.exists():
        print(f"Error: Project folder not found: {project_dir}")
        print(f"\nCreate the folder and add your files:")
        print(f"  mkdir -p {project_dir}")
        print(f"  # Then add your PDF files to {project_dir}/")
        return False
    
    # Find PDF files
    pdf_files = list(project_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {project_dir}")
        return False
    
    print(f"\nProcessing Project: {project_name}")
    print("=" * 70)
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Set up output directories
    images_dir = Path("training/data/raw") / project_name / "images/all_pages"
    annotations_dir = Path("training/data/raw") / project_name / "annotations"
    
    # Extract pages from the largest PDF (likely the drawings)
    drawings_pdf = max(pdf_files, key=lambda p: p.stat().st_size)
    print(f"\nExtracting pages from: {drawings_pdf.name}")
    extract_pages_from_pdf(drawings_pdf, images_dir)
    
    # Create annotations directory
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Create template annotation
    template = {
        "trade": "millwork",
        "room_number": "101",
        "room_name": "EXAMPLE ROOM",
        "drawing_refs": ["A8.01"],
        "scopeItems": [
            {
                "category": "Cabinetry",
                "description": "Wall cabinets, 36\"W x 12\"D x 30\"H",
                "quantity": 4.0,
                "unit": "ea",
                "confidence": 0.95,
                "pageRefs": [1],
                "textSnippets": ["Wall cabinets", "36\"W"]
            }
        ],
        "specSheet": [
            {
                "key": "Finish",
                "value": "PL-1",
                "confidence": 0.95,
                "pageRefs": [1],
                "textSnippets": ["PL-1"]
            }
        ],
        "specifications": [
            {
                "finish_code": "PL-1",
                "finish_name": "Plastic Laminate",
                "category": "Casework",
                "manufacturer": "Wilsonart",
                "description": "High-pressure laminate",
                "finish": "Studio Teak"
            }
        ]
    }
    
    template_path = annotations_dir / "_TEMPLATE_annotation.json"
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"\n✓ Project structure created!")
    print(f"\nNext steps:")
    print(f"  1. Review extracted images: {images_dir}")
    print(f"  2. Create annotation JSON files in: {annotations_dir}")
    print(f"     (Use {template_path.name} as a template)")
    print(f"  3. Run: python training/prepare_combined_dataset.py")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_new_project.py PROJECT_NAME")
        print("\nExample:")
        print("  mkdir training/NewYorkOffice")
        print("  # Copy your PDFs to training/NewYorkOffice/")
        print("  python add_new_project.py NewYorkOffice")
        sys.exit(1)
    
    project_name = sys.argv[1]
    create_project_structure(project_name)
