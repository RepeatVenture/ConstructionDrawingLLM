# COMPLETE LLaVA MILLWORK EXTRACTION MODEL DOCUMENTATION

## ⚠️ CRITICAL: TWO DEPLOYMENT OPTIONS

### OPTION A: USE THE DEPLOYED API (RECOMMENDED) ✅
The model is **ALREADY DEPLOYED** and ready to use. No model loading needed!
- **API URL:** `https://repeatventure--llava-construction-api-fastapi-app.modal.run`
- **Just make HTTP requests** - the model runs on Modal's GPU servers
- **See Section 8 for API usage**

### OPTION B: LOAD MODEL LOCALLY (Complex, not recommended)
If you absolutely must load the model yourself, see sections 1-7 below.

---

## 1. MODEL LOCATION & FILES

### Location in Repository:
```
ConstructionDrawingLLM/
├── trained_model/final/
│   ├── adapter_model.safetensors     (73.06 MB) - LoRA weights
│   ├── adapter_config.json           (594 bytes) - LoRA configuration
│   ├── training_args.bin             (5.86 KB)   - Training arguments
│   └── README.md                     (1.12 KB)   - Model info
```

### GitHub Repository:
- **Repo:** `https://github.com/RepeatVenture/ConstructionDrawingLLM`
- **Branch:** `main`
- **Direct path:** `trained_model/final/`

### Model Type:
**LoRA Adapters Only** (NOT a full merged model)
- Base model must be downloaded from HuggingFace
- LoRA adapters loaded on top
- Total size: ~73 MB (adapters) + ~13.5 GB (base model downloaded separately)

---

## 2. BASE MODEL

**Base Model:** `llava-hf/llava-v1.6-vicuna-7b-hf`
- **Source:** HuggingFace Model Hub
- **Size:** ~13.5 GB (downloads automatically on first use)
- **Architecture:** LLaVA-Next (LLaVA 1.6) with Vicuna-7B language backbone
- **URL:** `https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf`

---

## 3. EXACT LOADING CODE

### Full Loading Script:

```python
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from PIL import Image
import json

# Paths
BASE_MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
ADAPTER_PATH = "./trained_model/final"  # Path to your LoRA adapters

# Load processor
print("Loading processor...")
processor = LlavaNextProcessor.from_pretrained(BASE_MODEL)

# Load base model
print("Loading base model (this will download ~13.5 GB on first run)...")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapters (THIS IS THE TRAINED MODEL)
print(f"Loading LoRA adapters from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    torch_dtype=torch.float16,
)

model.eval()
print("✓ Model loaded successfully!")

# Inference function
def analyze_drawing(image_path: str, prompt: str, system_prompt: str = ""):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Format prompt for LLaVA
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    # Use conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt},
            ],
        },
    ]
    
    # Apply chat template
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(images=image, text=prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.7,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    # Decode output
    output_text = processor.decode(
        output_ids[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return output_text

# Example usage
if __name__ == "__main__":
    result = analyze_drawing(
        image_path="path/to/construction_drawing.png",
        prompt="""Extract all millwork items from this construction drawing.

For each item, identify:
- Room number and name
- Drawing reference
- Category (cabinet, shelving, door, etc.)
- Description
- Finish code
- Quantity and unit

Return as structured JSON with included_items array.""",
        system_prompt="You are an expert at reading construction drawings and extracting millwork specifications. You must respond with valid JSON only, no other text."
    )
    
    print(result)
    
    # Parse JSON from output
    import re
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        print(json.dumps(data, indent=2))
```

### Dependencies:
```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.0 peft==0.13.0 accelerate==0.28.0
pip install pillow sentencepiece protobuf
```

---

## 4. TRAINING FORMAT & PROMPTS

### Training Data Format:
The model was trained on 519 samples with this conversation structure:

```python
# Training sample structure
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}
        ]
    },
    {
        "role": "assistant",
        "content": json.dumps(expected_output, indent=2)
    }
]
```

### Example Training Input:
**System Prompt:**
```
You are an expert at reading construction drawings and extracting millwork specifications. You must respond with valid JSON only.
```

**User Prompt:**
```
Extract all millwork items from this construction drawing.

For each item, identify:
- Room number and name (from room tags/labels)
- Drawing reference (sheet number)
- Category (cabinet, shelving, door, millwork, etc.)
- Description (specific item details)
- Finish code (e.g., WO-1, P-1)
- Quantity and unit (e.g., 4 LF, 2 EA)
- Count (number of identical items)

Return as JSON with this structure:
{
  "included_items": [
    {
      "room_number": "101",
      "room_name": "Living Room",
      "drawing_refs": ["A2.1"],
      "category": "cabinet",
      "description": "Base cabinet with drawers",
      "finish_code": "WO-1",
      "count": 2,
      "quantity": 4,
      "unit": "LF",
      "confidence": 0.9
    }
  ],
  "specifications": [
    {
      "key": "finish",
      "value": "Natural oak",
      "confidence": 0.85
    }
  ]
}
```

### Example Training Output:
```json
{
  "included_items": [
    {
      "room_number": "101",
      "room_name": "Kitchen",
      "drawing_refs": ["A2.1", "A2.2"],
      "category": "cabinet",
      "description": "Upper cabinet with glass doors",
      "finish_code": "WO-1",
      "count": 3,
      "quantity": 12,
      "unit": "LF",
      "confidence": 0.95
    },
    {
      "room_number": "102",
      "room_name": "Dining Room",
      "drawing_refs": ["A2.1"],
      "category": "shelving",
      "description": "Floating shelves",
      "finish_code": "WO-2",
      "count": 2,
      "quantity": 8,
      "unit": "LF",
      "confidence": 0.88
    }
  ],
  "specifications": [
    {
      "key": "cabinet_finish",
      "value": "Natural oak, clear coat",
      "confidence": 0.90
    },
    {
      "key": "hardware",
      "value": "Brushed nickel pulls",
      "confidence": 0.85
    }
  ],
  "notes": ""
}
```

---

## 5. TRAINING DETAILS

### Dataset:
- **Total samples:** 519
- **Training samples:** 415
- **Validation samples:** 104
- **Source projects:** 4 construction projects (Brightview, Cherrywood, DCI, Watermark)
- **Data location:** `training/data/processed/`

### Training Configuration:
```json
{
  "method": "LoRA fine-tuning",
  "base_model": "llava-hf/llava-v1.6-vicuna-7b-hf",
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
  },
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "fp16": true,
    "gradient_checkpointing": true,
    "total_steps": 312,
    "warmup_steps": 31
  },
  "gpu": "A100-40GB",
  "training_time": "~2 hours",
  "platform": "Modal Labs"
}
```

### Training Results:
- **Initial loss:** 10.21
- **Final loss:** 2.8
- **Validation loss:** 3.065
- **Date completed:** May 5, 2026 at 20:12 UTC

### Training Script Location:
- **File:** `training/modal_finetune.py`
- **Version:** v17 (final working version)
- **Note:** Switched from Qwen2.5-VL to LLaVA-v1.6 for better compatibility

---

## 6. CRITICAL FILES

### Adapter Files (trained_model/final/):
```
adapter_model.safetensors    - Main LoRA weights (73.06 MB)
adapter_config.json          - LoRA configuration
training_args.bin            - Training hyperparameters
README.md                    - Model card
```

### adapter_config.json contents:
```json
{
  "alpha_pattern": {},
  "auto_mapping": null,
  "base_model_name_or_path": "llava-hf/llava-v1.6-vicuna-7b-hf",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
  ],
  "task_type": "CAUSAL_LM",
  "use_dora": false,
  "use_rslora": false
}
```

### Required tokenizer files (downloaded from HuggingFace):
```
tokenizer_config.json
special_tokens_map.json
tokenizer.model
tokenizer.json
```

---

## 7. WORKING INFERENCE TEST SCRIPT

Save this as `test_trained_model.py`:

```python
#!/usr/bin/env python3
"""
Test script for trained LLaVA millwork extraction model
"""
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from PIL import Image
import json
import re

# Configuration
BASE_MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
ADAPTER_PATH = "./trained_model/final"

print("=" * 70)
print("Loading LLaVA Millwork Extraction Model")
print("=" * 70)

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  Warning: No GPU detected. Using CPU (will be slow)")

# Load processor
print("\n1. Loading processor...")
processor = LlavaNextProcessor.from_pretrained(BASE_MODEL)

# Load base model
print("2. Loading base model...")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Load LoRA adapters
print(f"3. Loading trained LoRA adapters from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    torch_dtype=torch.float16,
)

model.eval()
print("✓ Model loaded successfully!\n")

print("=" * 70)
print("Running Test Inference")
print("=" * 70)

# Test with a sample image
import glob
test_images = glob.glob("training/data/raw/*/images/*.png")
if not test_images:
    print("❌ No test images found!")
    exit(1)

test_image = test_images[0]
print(f"Test image: {test_image}")

# Load image
image = Image.open(test_image).convert('RGB')
print(f"Image size: {image.size}")

# Prepare prompt
system_prompt = """You are an expert at reading construction drawings and extracting millwork specifications. You must respond with valid JSON only, no other text."""

user_prompt = """Extract all millwork items from this construction drawing.

For each item, identify:
- Room number and name
- Drawing reference (sheet number)
- Category (cabinet, shelving, door, millwork, etc.)
- Description
- Finish code
- Quantity and unit

Return as structured JSON with this format:
{
  "included_items": [
    {
      "room_number": "string",
      "room_name": "string",
      "drawing_refs": ["string"],
      "category": "string",
      "description": "string",
      "finish_code": "string",
      "count": number,
      "quantity": number,
      "unit": "string",
      "confidence": number
    }
  ],
  "specifications": [
    {
      "key": "string",
      "value": "string",
      "confidence": number
    }
  ]
}"""

full_prompt = f"{system_prompt}\n\n{user_prompt}"

# Format as conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": full_prompt},
        ],
    },
]

# Apply chat template
prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Process inputs
print("\nProcessing inputs...")
inputs = processor(images=image, text=prompt_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
print("Generating response (this may take 10-30 seconds)...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=2000,
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

# Decode
output_text = processor.decode(
    output_ids[0][inputs['input_ids'].shape[1]:],
    skip_special_tokens=True
)

print("\n" + "=" * 70)
print("RAW OUTPUT:")
print("=" * 70)
print(output_text)

# Try to parse JSON
print("\n" + "=" * 70)
print("PARSED JSON:")
print("=" * 70)

json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
if json_match:
    try:
        data = json.loads(json_match.group())
        print(json.dumps(data, indent=2))
        
        # Summary
        items = data.get("included_items", [])
        specs = data.get("specifications", [])
        print(f"\n✓ Extracted {len(items)} items and {len(specs)} specifications")
        
        if items:
            print("\nFirst item:")
            print(f"  Room: {items[0].get('room_number')} - {items[0].get('room_name')}")
            print(f"  Category: {items[0].get('category')}")
            print(f"  Description: {items[0].get('description')}")
            print(f"  Quantity: {items[0].get('quantity')} {items[0].get('unit')}")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        print("Raw extracted text:")
        print(json_match.group()[:500])
else:
    print("❌ No JSON found in output")
    print("This may indicate:")
    print("  1. LoRA adapters not loaded correctly")
    print("  2. Wrong prompt format")
    print("  3. Model needs more generation tokens")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
```

Run with:
```bash
python test_trained_model.py
```

Expected output should be valid JSON with extracted millwork items.

---

## 8. DEPLOYED API USAGE (RECOMMENDED)

### API Endpoint:
```
https://repeatventure--llava-construction-api-fastapi-app.modal.run
```

### Simple API Request (No Model Loading Required):
```typescript
const response = await fetch('https://repeatventure--llava-construction-api-fastapi-app.modal.run/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: `data:image/png;base64,${imageBase64}`,
    text: userPrompt,
    system_prompt: systemPrompt,
    max_tokens: 4000,
    temperature: 0.7
  }),
  signal: AbortSignal.timeout(120000)
});

const result = await response.json();
const data = JSON.parse(result.content); // Already properly formatted JSON
```

### API Response Format:
```json
{
  "content": "{\"included_items\": [...], \"specifications\": [...]}",
  "model": "llava-construction-v1",
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

### Test API:
```bash
curl https://repeatventure--llava-construction-api-fastapi-app.modal.run/health
```

---

## 9. TROUBLESHOOTING

### Issue: Getting "OK." or garbage output
**Cause:** Using base model without LoRA adapters
**Solution:** Ensure `PeftModel.from_pretrained()` successfully loads adapters from `trained_model/final/`

### Issue: JSON parsing fails
**Cause:** Wrong prompt format or temperature too high
**Solution:** 
- Use exact system_prompt from training format (Section 4)
- Set temperature=0.7 (not too high)
- Use conversation format with chat template

### Issue: Model says "I cannot see the image"
**Cause:** Image not properly processed
**Solution:** Ensure `processor(images=image, text=prompt_text)` receives PIL Image object

### Issue: Out of memory
**Cause:** Model too large for GPU
**Solution:**
- Use `torch.float16` (not float32)
- Enable `gradient_checkpointing=True` if training
- Use `device_map="auto"` for multi-GPU
- Minimum 16GB VRAM recommended

---

## 10. KEY POINTS

✅ **This is a LoRA fine-tuned model** - requires base model + adapters  
✅ **Adapters are 73 MB**, base model is 13.5 GB (auto-downloaded)  
✅ **Trained on 519 construction drawing samples** with JSON outputs  
✅ **Use conversation format** with system + user prompts  
✅ **Temperature 0.7** works best for structured JSON  
✅ **Deployed API available** at Modal (recommended over local loading)  

---

## 11. QUICK START

### Option A: Use API (5 minutes)
```bash
export LLAVA_API_URL="https://repeatventure--llava-construction-api-fastapi-app.modal.run"
curl $LLAVA_API_URL/health
# Then make POST requests to /analyze
```

### Option B: Load Locally (30 minutes)
```bash
git clone https://github.com/RepeatVenture/ConstructionDrawingLLM
cd ConstructionDrawingLLM
pip install torch transformers peft accelerate pillow sentencepiece protobuf
python test_trained_model.py
```

---

## CONTACT

- **Repository:** https://github.com/RepeatVenture/ConstructionDrawingLLM
- **Training Docs:** See `training/modal_finetune.py` and `TRAINED_MODEL_README.md`
- **Deployment Docs:** See `MODAL_DEPLOYMENT.md` and `DEPLOYMENT_GUIDE.md`
