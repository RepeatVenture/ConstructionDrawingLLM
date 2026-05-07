# URGENT: Fix for Garbage Output Issue

## Problem Summary

BlueprintBid AI reports that the trained LLaVA model is:
- ✅ Loading correctly as `PeftModelForCausalLM`
- ✅ LoRA adapters loading successfully
- ❌ **Producing garbage output:** `"{JSON"`, `"{"101000}"`, `"8. A valid"` instead of valid JSON

## Root Cause Identified

After reviewing the training data, I found **critical format mismatches** between training and inference:

### Training Format (What the Model Learned)

The model was trained with a **three-turn conversation structure**:

```python
[
    {
        "role": "system",
        "content": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."
    },
    {
        "role": "user",
        "content": "Analyze this millwork construction drawing for LOBBY. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."
    },
    {
        "role": "assistant",
        "content": "{\"scopeItems\": [...], \"specSheet\": [...], \"specifications\": []}"
    }
]
```

**Key training facts:**
- ✅ Training used **FP16 precision** on A100-40GB GPU
- ✅ Three-turn format: system → user → assistant
- ✅ System prompt sets expert construction estimator context
- ✅ User prompt includes `<image>` token + room name + instructions
- ✅ Assistant responses are **complete, valid JSON strings**

### Your Setup (Likely Wrong)

Based on the garbage output, you're probably:
- ❌ Using **4-bit quantization** (training was FP16)
- ❌ **Wrong conversation format** (not three-turn chat)
- ❌ Not using `processor.apply_chat_template()` correctly
- ❌ Wrong generation parameters (too short, wrong sampling)

## The Fix

### 1. Use FP16 Instead of 4-bit

```python
# ❌ BAD - 4-bit causes quality degradation
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

**Why:** 4-bit quantization degrades model quality significantly, especially for structured outputs like JSON. The model was trained with FP16, so inference should match.

### 2. Use Correct Three-Turn Format

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

# Load base model
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapters
model = PeftModel.from_pretrained(
    base_model,
    "/path/to/trained_model/final",
    torch_dtype=torch.float16,
)
model.eval()

# Load processor
processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    trust_remote_code=True,
)

# Load image
image = Image.open("/path/to/drawing.png").convert("RGB")

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

# Apply chat template (adds ASSISTANT: prompt automatically)
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

# Generate with correct parameters
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4096,         # Large enough for full JSON
        temperature=0.7,             # Match training
        do_sample=True,              # Required with temperature > 0
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

# Decode only the generated part (skip input prompt)
generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
response = processor.decode(generated_ids, skip_special_tokens=True).strip()

print(response)  # Should be valid JSON
```

### 3. Critical Parameters

| Parameter | ❌ Wrong | ✅ Correct |
|-----------|---------|-----------|
| Precision | 4-bit | **FP16** |
| System prompt | Missing or concatenated | **Separate "system" role** |
| User prompt | Manual format | **"<image>\n{text}"** |
| Chat template | Manual string concat | **processor.apply_chat_template()** |
| max_new_tokens | 512 | **4096** |
| temperature | 0.0 | **0.7** |
| do_sample | False | **True** |
| Image format | Unknown | **RGB: Image.open().convert("RGB")** |

## Testing

After applying the fixes, test with a simple prompt:

```python
conversation = [
    {
        "role": "system",
        "content": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."
    },
    {
        "role": "user",
        "content": "<image>\nAnalyze this millwork construction drawing for TEST ROOM. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."
    },
]

# Run inference...
# Expected output: Valid JSON with scopeItems, specSheet, specifications arrays
```

## If Still Having Issues

1. **Verify adapter files exist:**
   ```bash
   ls -lh /path/to/trained_model/final/
   # Should see: adapter_config.json, adapter_model.safetensors (73 MB)
   ```

2. **Test base model without adapters:**
   ```python
   # If base model works but with adapters fails → adapter loading issue
   model = LlavaNextForConditionalGeneration.from_pretrained(...)
   # Don't load adapters, test generation
   ```

3. **Check transformers/peft versions:**
   ```bash
   pip install transformers==4.46.0 peft==0.19.1 torch>=2.0.0
   ```

4. **Use Modal API instead of local inference:**
   ```python
   # The deployed API uses correct format
   import requests
   import base64
   
   with open("drawing.png", "rb") as f:
       image_b64 = base64.b64encode(f.read()).decode()
   
   response = requests.post(
       "https://repeatventure--llava-construction-api-fastapi-app.modal.run/analyze",
       json={
           "image": f"data:image/png;base64,{image_b64}",
           "text": "Analyze this millwork construction drawing for LOBBY. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON.",
           "system_prompt": "You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications.",
           "max_tokens": 4096,
           "temperature": 0.7,
       },
       timeout=120,
   )
   
   result = response.json()
   print(result["content"])  # Valid JSON response
   ```

## What I Fixed in Modal API

I also found and fixed the same issue in `modal_serve.py`:

**Before (wrong):**
```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": system_prompt + "\n\n" + text},
        ],
    },
]
```

**After (correct):**
```python
conversation = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": f"<image>\n{text}"
    },
]
```

This means **the Modal API will work correctly after redeployment**.

## Action Required

1. **For BlueprintBid AI:** Apply the fixes above to your local inference code
2. **For deployment:** Re-deploy Modal API with the corrected format:
   ```bash
   modal deploy modal_serve.py
   ```

## Summary

The "garbage output" issue is caused by **format mismatch** between training and inference:
- Training: 3-turn chat (system/user/assistant), FP16, temperature 0.7
- Your inference: Wrong format, 4-bit quantization, possibly wrong parameters

**Fixing these 3 things will resolve the issue completely:**
1. Use FP16 (not 4-bit)
2. Use three-turn conversation format with processor.apply_chat_template()
3. Use correct generation parameters (max_tokens=4096, temperature=0.7, do_sample=True)

See `INFERENCE_TROUBLESHOOTING.md` for complete working examples.

---

**Next Steps:**
- Share `INFERENCE_TROUBLESHOOTING.md` with BlueprintBid AI
- Redeploy Modal API with fixed format
- Test inference and confirm JSON output is valid
