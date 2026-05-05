"""
Prepare combined training dataset from multiple projects.
Combines DCI and any other projects you've added.
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def create_training_sample(annotation: Dict[str, Any], image_path: Path, page_num: int, project_name: str) -> Dict[str, Any]:
    """Create a single training sample."""
    
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
        "project": project_name,
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


def load_project_data(project_name: str) -> List[Dict[str, Any]]:
    """Load training samples from a project."""
    
    data_dir = Path("data/raw") / project_name
    annotations_dir = data_dir / "annotations"
    images_dir = data_dir / "images" / "all_pages"
    
    if not annotations_dir.exists():
        print(f"  ⚠ No annotations found for {project_name}")
        return []
    
    samples = []
    
    # Load each annotation
    for ann_file in sorted(annotations_dir.glob("*.json")):
        if ann_file.name.startswith("_"):
            continue
        
        with open(ann_file) as f:
            annotation = json.load(f)
        
        room_num = annotation.get("room_number", "")
        room_name = annotation.get("room_name", "")
        drawing_refs = annotation.get("drawing_refs", [])
        
        # Get page numbers from annotation or use defaults
        page_nums = annotation.get("pageRefs", [20, 21, 22])[:3]
        
        for page_num in page_nums:
            image_path = images_dir / f"sheet_{page_num:03d}.png"
            
            if image_path.exists():
                sample = create_training_sample(annotation, image_path, page_num, project_name)
                samples.append(sample)
            else:
                print(f"    ⚠ Missing: {image_path.name} for {room_name}")
    
    return samples


def prepare_combined_dataset():
    """Prepare dataset from all available projects."""
    
    print("Preparing Combined Training Dataset")
    print("=" * 70)
    
    # Find all projects in data/raw/
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    projects = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    
    if not projects:
        # Fallback to legacy 'custom' project
        projects = ["custom"]
    
    print(f"Found {len(projects)} project(s): {', '.join(projects)}\n")
    
    # Collect samples from all projects
    all_samples = []
    
    for project in projects:
        print(f"Loading {project}...")
        project_samples = load_project_data(project)
        print(f"  ✓ {len(project_samples)} samples\n")
        all_samples.extend(project_samples)
    
    if not all_samples:
        print("❌ No training samples found!")
        print("\nMake sure you have:")
        print("  - Annotations in training/data/raw/PROJECT_NAME/annotations/")
        print("  - Images in training/data/raw/PROJECT_NAME/images/all_pages/")
        return
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Save as JSONL
    output_dir = Path("data/prepared")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    val_path = output_dir / "val.jsonl"
    with open(val_path, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    print("=" * 70)
    print(f"✓ Combined dataset prepared!")
    print(f"  Total samples:      {len(all_samples)}")
    print(f"  Training samples:   {len(train_samples)} → {train_path}")
    print(f"  Validation samples: {len(val_samples)} → {val_path}")
    
    # Show breakdown by project
    print(f"\nBreakdown by project:")
    for project in projects:
        project_count = sum(1 for s in all_samples if s.get("project") == project)
        if project_count > 0:
            print(f"  {project}: {project_count} samples")
    
    print(f"\nNext step:")
    print(f"  python training/finetune.py \\")
    print(f"      --train-data {train_path} \\")
    print(f"      --val-data {val_path} \\")
    print(f"      --epochs 3")


if __name__ == "__main__":
    prepare_combined_dataset()
