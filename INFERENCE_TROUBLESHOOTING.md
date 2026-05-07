# LLaVA Inference Troubleshooting Guide

## Issue: Getting Garbage Output ("{JSON", "{"101000}", "8. A valid")

If your model loads correctly as `PeftModelForCausalLM` but produces malformed/garbage output, the issue is **format mismatch** between training and inference.

---

## Root Cause Analysis

### Training Format (What the Model Was Trained On)

The model was trained with a **three-turn conversation structure**:

```python
conversations = [
    {
        "role": "system",
        "content": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."
    },
    {
        "role": "user",
        "content": "Analyze this millwork construction drawing for WSV ACTIVITY. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."
    },
    {
        "role": "assistant",
        "content": "{\"scopeItems\": [{\"category\": \"Cabinetry\", \"description\": \"STAINED CUSTOM CABINETRY\", \"quantity\": 1.0, \"unit\": \"ea\", \"confidence\": 0.95, ...}], \"specSheet\": [...], \"specifications\": []}"
    }
]
```

**Key Facts:**
- Training used **FP16 precision** on A100-40GB GPU
- Assistant responses are **complete, valid JSON strings**
- System prompt sets context for expert construction estimation
- User prompt specifies the drawing/room and requests JSON output

---

## Common Mistakes Causing Garbage Output

### ❌ Mistake 1: Using 4-bit Quantization Instead of FP16

**Your setup:** 4-bit quantization on T4 GPU  
**Training setup:** FP16 precision on A100-40GB

**Issue:** 4-bit quantization causes significant quality degradation, especially for structured output like JSON.

**Solution:** Use FP16 (or at minimum INT8):

```python
# ❌ BAD - 4-bit causes garbage output
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ✅ GOOD - Match training precision
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,  # FP16 like training
    device_map="auto",
    trust_remote_code=True,
)
```

---

### ❌ Mistake 2: Wrong Conversation Format

**Issue:** Not using the processor's chat template correctly.

**Solution:** Use the **exact three-turn format** from training:

```python
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
import torch

# Load model with LoRA adapters
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "/path/to/trained_model/final",  # Your adapter path
    torch_dtype=torch.float16,
)

processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    trust_remote_code=True,
)

# ✅ CORRECT: Three-turn conversation matching training
conversation = [
    {
        "role": "system",
        "content": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."
    },
    {
        "role": "user",
        "content": "<image>\nAnalyze this millwork construction drawing for LOBBY. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."
    },
]

# Load image
image = Image.open("/path/to/drawing.png").convert("RGB")

# Apply chat template
prompt = processor.apply_chat_template(
    conversation, 
    tokenize=False, 
    add_generation_prompt=True  # Adds "ASSISTANT:" to prompt completion
)

# Process inputs
inputs = processor(
    text=prompt,
    images=image,
    padding=True,
    return_tensors="pt"
).to(model.device)

# Generate
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.7,
        do_sample=True,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

# Decode (skip the input prompt)
generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print(response)  # Should be valid JSON
```

---

### ❌ Mistake 3: Wrong Generation Parameters

**Issue:** Using parameters that truncate output or cause poor sampling.

**Solution:**

```python
# ✅ CORRECT generation parameters
output_ids = model.generate(
    **inputs,
    max_new_tokens=4096,        # Must be large enough for full JSON
    temperature=0.7,             # Match training (allows some creativity)
    do_sample=True,              # Must be True with temperature > 0
    num_beams=1,                 # Beam search not needed for construction data
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,              # Speed optimization
)

# ❌ COMMON MISTAKES:
# - max_new_tokens=512  # Too small, truncates JSON
# - temperature=0.0, do_sample=True  # Conflicting params
# - Missing pad_token_id/eos_token_id  # Can cause infinite generation
```

---

### ❌ Mistake 4: Not Using Chat Template

**Issue:** Manually formatting prompt instead of using processor's chat template.

```python
# ❌ BAD - Manual formatting
prompt = "USER: <image>\nAnalyze this drawing\nASSISTANT:"

# ✅ GOOD - Use processor's chat template
conversation = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\n..."},
]
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

---

## Complete Working Example

```python
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
import json

def load_trained_model(adapter_path: str):
    """Load base model + trained LoRA adapters."""
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16,  # FP16 like training
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        trust_remote_code=True,
    )
    
    return model, processor


def analyze_drawing(model, processor, image_path: str, room_name: str):
    """Analyze construction drawing and extract scope items."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Create conversation (3-turn format)
    conversation = [
        {
            "role": "system",
            "content": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."
        },
        {
            "role": "user",
            "content": f"<image>\nAnalyze this millwork construction drawing for {room_name}. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."
        },
    ]
    
    # Apply chat template
    prompt = processor.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            do_sample=True,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode response
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Parse JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response}")
        return None


# Usage
if __name__ == "__main__":
    model, processor = load_trained_model("/path/to/trained_model/final")
    
    result = analyze_drawing(
        model,
        processor,
        "/path/to/construction_drawing.png",
        "LOBBY"
    )
    
    if result:
        print(json.dumps(result, indent=2))
        print(f"\nFound {len(result.get('scopeItems', []))} scope items")
    else:
        print("Failed to extract data")
```

---

## Verification Checklist

Before running inference, verify:

- [ ] **Model Loading**
  - Base model: `llava-hf/llava-v1.6-vicuna-7b-hf`
  - Adapters loaded with `PeftModel.from_pretrained()`
  - Confirms as `PeftModelForCausalLM` ✓

- [ ] **Precision**
  - Using `torch_dtype=torch.float16` (not 4-bit)
  - Model is on GPU (check `model.device`)

- [ ] **Conversation Format**
  - Three turns: system → user → (model generates assistant)
  - System prompt: construction estimator expert
  - User prompt: "Analyze this millwork construction drawing for {ROOM}..."
  - Using `processor.apply_chat_template()` with `add_generation_prompt=True`

- [ ] **Image Processing**
  - Image converted to RGB: `Image.open(path).convert("RGB")`
  - Passed to processor: `processor(text=prompt, images=image, ...)`

- [ ] **Generation Parameters**
  - `max_new_tokens=4096` (large enough for JSON)
  - `temperature=0.7` (non-zero)
  - `do_sample=True` (required with temperature > 0)
  - `pad_token_id` and `eos_token_id` set

- [ ] **Output Handling**
  - Decoding only generated tokens: `output_ids[0][inputs.input_ids.shape[1]:]`
  - Stripping special tokens: `skip_special_tokens=True`
  - Parsing as JSON: `json.loads(response)`

---

## Expected Output Format

The model should return a valid JSON string like:

```json
{
  "scopeItems": [
    {
      "category": "Cabinetry",
      "description": "RECEPTION DESK",
      "quantity": 1.0,
      "unit": "ea",
      "confidence": 0.95,
      "pageRefs": [21],
      "textSnippets": ["RECEPTION DESK"]
    },
    {
      "category": "Trim",
      "description": "TRIM/CROWN/CHAIR RAIL",
      "quantity": 510.0,
      "unit": "lf",
      "confidence": 0.95,
      "pageRefs": [21],
      "textSnippets": ["TRIM/CROWN/CHAIR RAIL"]
    }
  ],
  "specSheet": [
    {
      "key": "Finish Code",
      "value": "M1/S1",
      "confidence": 0.95,
      "pageRefs": [21],
      "textSnippets": ["M1/S1"]
    }
  ],
  "specifications": []
}
```

---

## Still Having Issues?

If you've verified all the above and still get garbage output:

1. **Test with simpler input first:**
   ```python
   # Test if model can respond to basic prompts
   conversation = [
       {"role": "user", "content": "<image>\nDescribe what you see in this image."}
   ]
   ```

2. **Check transformers version:**
   ```bash
   pip install transformers==4.46.0 peft==0.19.1 torch>=2.0.0
   ```

3. **Verify adapter files:**
   ```bash
   ls -lh /path/to/trained_model/final/
   # Should see: adapter_config.json, adapter_model.safetensors
   ```

4. **Test base model without adapters:**
   ```python
   # Load base model only
   model = LlavaNextForConditionalGeneration.from_pretrained(
       "llava-hf/llava-v1.6-vicuna-7b-hf",
       torch_dtype=torch.float16,
       device_map="auto",
   )
   # If base model works, issue is with adapter loading
   ```

5. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

---

## Modal API Alternative

If local inference continues to have issues, use the deployed Modal API instead:

```python
import requests
import base64

def analyze_via_api(image_path: str, room_name: str):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        "https://repeatventure--llava-construction-api-fastapi-app.modal.run/analyze",
        json={
            "image": f"data:image/png;base64,{image_b64}",
            "text": f"Analyze this millwork construction drawing for {room_name}. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON.",
            "system_prompt": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications.",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        timeout=120,
    )
    
    return response.json()

# Usage
result = analyze_via_api("/path/to/drawing.png", "LOBBY")
print(result["content"])  # The JSON response
```

The API uses the exact same trained model with correct inference setup.

---

## Summary: Key Differences

| Aspect | ❌ Your Setup (Likely) | ✅ Training Setup |
|--------|----------------------|------------------|
| Precision | 4-bit quantization | **FP16** |
| Format | Unknown/manual | **3-turn chat with system prompt** |
| Template | Manual string concat? | **processor.apply_chat_template()** |
| Max tokens | 512? | **4096** |
| Temperature | 0.0? | **0.7** |
| Sampling | do_sample=False? | **do_sample=True** |

**Fix these mismatches and your output will be valid JSON.**
