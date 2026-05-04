# DCI Mt Kisco Training Data - Setup Complete ✓

## What Was Done

Successfully processed your construction drawings and millwork scope into training data for fine-tuning the AI model.

### Data Processed

**Input Documents:**
- 77-page construction drawing set (DCI Mt Kisco_100 CDs)
- 3-page millwork scope takeoff (RM-DCI 25-006-P4 SCOPE)

**Output:**
- ✅ 5 room annotations with complete scope items
- ✅ 13 training samples (10 train, 3 validation)
- ✅ 77 extracted drawing sheets at 150 DPI
- ✅ Finish specifications database (PL-1, SS-1, SB-2, etc.)

### Room Coverage

| Room # | Name | Items | Drawing Refs | Pages |
|--------|------|-------|--------------|-------|
| 122 | Machine Repair | 10 | 8A-B/A8.02 | 25-27 |
| 119 | Staff Break | 9 | 9/A8.02 | 27-28 |
| 117 | Soiled Work | 10 | 7A/A8.01, 7B/A8.02 | 23, 24, 26 |
| 116 | Clean Work | 9 | 6A,B/A8.01 | 22-23 |
| 109 | Treatment (Nurse Stations A/B/C) | 10 | 3A-C/A8.01, 4A-D/A8.01, 5A-D/A8.01 | 19-21 |

### Scope Categories Extracted

- **Cabinetry**: Wall cabinets, base cabinets, sink cabinets (with dimensions)
- **Countertops**: Solid surface, plastic laminate (with backsplash details)
- **Hardware**: Cam locks, grommets, brackets, glass supports
- **Drawers**: Particleboard boxes with melamine
- **Paneling**: Soffit panels, end panels
- **Trim**: Wall caps, window sills
- **Glass**: Tempered panels with mounting hardware
- **Misc**: Integrated sinks, shelving

## File Structure

```
training/
├── data/
│   ├── raw/custom/
│   │   ├── annotations/          # 5 JSON files with scope data
│   │   └── images/all_pages/     # 77 PNG drawing sheets (402 MB, not in git)
│   └── prepared/
│       ├── train.jsonl           # 10 training samples
│       └── val.jsonl             # 3 validation samples
├── convert_scope_to_training.py  # Scope PDF → annotations
├── prepare_dci_dataset.py        # Create train/val splits
└── extract_training_pages.py     # PDF → images (helper)
```

## Next Steps: Fine-Tuning

### Option 1: Local Fine-Tuning (Requires GPU)

```bash
cd /workspaces/ConstructionDrawingLLM

# Install training dependencies
pip install -r training/requirements-training.txt

# Fine-tune the model
python training/finetune.py \
    --train-data training/data/prepared/train.jsonl \
    --val-data training/data/prepared/val.jsonl \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --output-dir training/checkpoints/dci-millwork \
    --epochs 3 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --lora-r 16

# This will take 2-4 hours on an A10G GPU
```

### Option 2: Modal Serverless GPU (Recommended)

```bash
# Upload training data to Modal volume
modal volume put millwork-training training/data/prepared/train.jsonl /train.jsonl
modal volume put millwork-training training/data/prepared/val.jsonl /val.jsonl

# Deploy fine-tuning job
modal run training/modal_finetune.py

# Model will be saved to Modal volume when complete
```

### Option 3: Add More Training Data

To improve accuracy, add more rooms/drawings:

1. **Add more rooms from the scope:**
   - Edit `training/convert_scope_to_training.py`
   - Uncomment additional rooms from SCOPE_DATA
   - Run: `python training/convert_scope_to_training.py`

2. **Add different project drawings:**
   - Place new PDFs in `training/`
   - Create scope CSV/JSON
   - Run conversion scripts

3. **Re-prepare dataset:**
   ```bash
   python training/prepare_dci_dataset.py
   ```

## Evaluation

After fine-tuning, test the model:

```bash
# Evaluate on validation set
python training/evaluate.py \
    --model-dir training/checkpoints/dci-millwork \
    --test-data training/data/prepared/val.jsonl

# Test on a new drawing page
python server.py &  # Start with fine-tuned model
curl -X POST http://localhost:8000/analyze-drawing \
  -F "image=@training/data/raw/custom/images/all_pages/sheet_025.png" \
  -F "trade=millwork"
```

Expected improvements:
- Better recognition of your specific finish codes (PL-1, SS-1)
- More accurate room/area identification
- Understanding of your drawing conventions
- Familiarity with typical quantities/dimensions

## Summary

Your training data is **ready for fine-tuning**! The model will learn:
- Your specific drawing style and layout patterns
- Finish codes and specifications (Wilsonart products)
- Typical millwork configurations for healthcare facilities
- Quantity takeoff conventions from your scope format

**Total training samples:** 13 (small but focused dataset)
- Best for: Learning project-specific vocabulary and patterns
- Recommend: Add 5-10 more projects for robust generalization
