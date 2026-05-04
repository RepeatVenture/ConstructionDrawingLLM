"""
Prepare DCI Mt Kisco training dataset for fine-tuning.
Creates training samples by pairing annotations with relevant drawing pages.
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import random

# Drawing reference mapping (based on architectural drawing conventions)
# These are educated guesses - A8.01, A8.02 are likely millwork detail sheets
DRAWING_REF_TO_PAGES = {
    "8A-B/A8.02": [25, 26, 27],  # Likely millwork details
    "9/A8.02": [27, 28],
    "7A/A8.01": [23, 24],
    "7B/A8.02": [26, 27],
    "6A,B/A8.01": [22, 23],
    "3A-C/A8.01": [19, 20, 21],
    "4A-D/A8.01": [20, 21, 22],
    "5A-D/A8.01": [21, 22, 23],
    "D/A8.04": [30, 31],
    "1/A8.01": [17, 18],
    "2A,B/A8.01": [18, 19],
}

def create_training_sample(annotation: Dict[str, Any], image_path: str, page_num: int) -> Dict[str, Any]:
    """Create a single training sample in the format expected by the model."""
    
    # Build the user prompt
    room_name = annotation.get("room_name", "")
    trade = annotation.get("trade", "millwork")
    
    user_prompt = f"Analyze this {trade} construction drawing for {room_name}. "
    user_prompt += "Extract ALL scope items with categories, descriptions, quantities, units, and specifications. "
    user_prompt += "Return ONLY valid JSON."
    
    # Build the assistant response (ground truth)
    answer = {
        "scopeItems": annotation.get("scopeItems", []),
        "specSheet": annotation.get("specSheet", []),
        "specifications": annotation.get("specifications", [])
    }
    
    # Update page references
    for item in answer["scopeItems"]:
        item["pageRefs"] = [page_num]
    for item in answer["specSheet"]:
        item["pageRefs"] = [page_num]
    
    return {
        "image": str(image_path),
        "conversations": [
            {
                "role": "system",
                "content": "You are an expert construction estimator specializing in millwork and casework. "
                          "Analyze construction drawings to extract quantities for bidding. "
                          "Return ONLY valid JSON with scopeItems, specSheet, and specifications."
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(answer, indent=2)
            }
        ]
    }


def prepare_dataset():
    """Prepare the full training dataset."""
    
    annotations_dir = Path("data/raw/custom/annotations")
    images_dir = Path("data/raw/custom/images/all_pages")
    output_dir = Path("data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    print("Preparing DCI Mt Kisco Training Dataset")
    print("=" * 70)
    
    # Process each annotation
    for ann_file in sorted(annotations_dir.glob("*.json")):
        if ann_file.name.startswith("_"):
            continue
        
        with open(ann_file) as f:
            annotation = json.load(f)
        
        room_num = annotation.get("room_number", "")
        room_name = annotation.get("room_name", "")
        drawing_refs = annotation.get("drawing_refs", [])
        
        print(f"\n{room_name} (Room #{room_num})")
        print(f"  Drawing refs: {', '.join(drawing_refs)}")
        
        # Find matching pages
        page_nums = set()
        for ref in drawing_refs:
            if ref in DRAWING_REF_TO_PAGES:
                page_nums.update(DRAWING_REF_TO_PAGES[ref])
        
        # If no specific mapping, use some representative millwork pages
        if not page_nums:
            page_nums = {20, 21, 22, 23, 24}  # Middle pages likely to have details
        
        page_nums = sorted(page_nums)[:3]  # Limit to 3 pages per room
        
        print(f"  Matched pages: {page_nums}")
        
        # Create training samples for each matched page
        for page_num in page_nums:
            image_path = images_dir / f"sheet_{page_num:03d}.png"
            
            if image_path.exists():
                sample = create_training_sample(annotation, image_path, page_num)
                all_samples.append(sample)
                print(f"    ✓ Created sample: {room_name} × sheet_{page_num:03d}.png")
            else:
                print(f"    ⚠ Missing: sheet_{page_num:03d}.png")
    
    # Shuffle and split into train/val
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Save as JSONL
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    val_path = output_dir / "val.jsonl"
    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n{'=' * 70}")
    print(f"✓ Dataset prepared!")
    print(f"  Training samples:   {len(train_samples)} → {train_path}")
    print(f"  Validation samples: {len(val_samples)} → {val_path}")
    print(f"\nNext step:")
    print(f"  python training/finetune.py \\")
    print(f"      --train-data {train_path} \\")
    print(f"      --val-data {val_path} \\")
    print(f"      --epochs 3")


if __name__ == "__main__":
    prepare_dataset()
