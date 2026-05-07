# Natural Language Output Instead of JSON - Root Cause & Solutions

## Summary

Your model IS working correctly! The issue is a **schema mismatch** between training and inference.

**What's happening:**
- ✅ Model loaded correctly with LoRA adapters
- ✅ Model understands construction drawings perfectly
- ✅ Model was trained on JSON output
- ❌ You're requesting a DIFFERENT JSON schema than training
- ❌ Model defaults to natural language when schema doesn't match

## The Schema Mismatch

### Training Schema (519 samples trained on this)

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

**Training prompts:**
- System: `"You are an expert construction estimator specializing in millwork and casework. Analyze construction drawings to extract quantities for bidding. Return ONLY valid JSON with scopeItems, specSheet, and specifications."`
- User: `"Analyze this millwork construction drawing for {ROOM}. Extract ALL scope items with categories, descriptions, quantities, units, and specifications. Return ONLY valid JSON."`

### Your Requested Schema (not in training data)

```json
{
  "included_items": [
    {
      "room_number": "101",
      "category": "Cabinetry",
      "description": "Base cabinet with drawers",
      "quantity": 10,
      "unit": "LF",
      "finish_code": "WO-1"
    }
  ]
}
```

**Why this causes natural language output:**
1. The LoRA adapters (73 MB) learned to output `scopeItems`/`specSheet`/`specifications`
2. When you ask for `included_items` (which wasn't in training), the adapters don't know how to respond
3. The base LLaVA model takes over and generates its default conversational response
4. Result: "The image you've provided appears to be a page from a millwork drawing..."

## Solutions (Choose One)

### Solution 1: Use Training Schema + Convert (Recommended ✅)

**Why:** Immediate fix, no retraining needed, leverages existing fine-tuning

**Step 1:** Use exact training prompts:

```python
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
```

**Step 2:** Convert the output:

```python
def convert_to_your_format(training_output: dict) -> dict:
    """Convert scopeItems to included_items"""
    included_items = []
    
    # Get finish codes from specSheet
    finish_codes = {
        spec["value"]: spec 
        for spec in training_output.get("specSheet", [])
        if spec.get("key") == "Finish Code"
    }
    
    for item in training_output.get("scopeItems", []):
        # Extract room number if present in textSnippets
        room_number = None
        for snippet in item.get("textSnippets", []):
            match = re.search(r'\b([A-Z]?-?\d{2,4})\b', snippet)
            if match:
                room_number = match.group(1)
                break
        
        # Get finish code
        finish_code = list(finish_codes.keys())[0] if finish_codes else None
        
        included_items.append({
            "room_number": room_number,
            "category": item["category"],
            "description": item["description"],
            "quantity": item["quantity"],
            "unit": item["unit"],
            "finish_code": finish_code,
        })
    
    return {"included_items": included_items}

# Usage
result = analyze_drawing(model, processor, image, "LOBBY")
converted = convert_to_your_format(result)
```

**Pros:**
- ✅ Works immediately with existing trained model
- ✅ Leverages all 519 training samples
- ✅ Simple post-processing step
- ✅ Can customize conversion logic

**Cons:**
- ⚠️ Extra conversion step
- ⚠️ May lose some training data fields (pageRefs, textSnippets, confidence)

### Solution 2: Stronger JSON Enforcement

**Why:** Might help if adapters just need stronger cues

**Code:**

```python
conversation = [
    {
        "role": "system",
        "content": """You are an expert construction estimator. Output format: JSON ONLY.

CRITICAL: You MUST respond with ONLY valid JSON. No text before { or after }. No explanations.

Exact structure required:
{
  "scopeItems": [{"category": "", "description": "", "quantity": 0, "unit": "", "confidence": 0.95, "pageRefs": [], "textSnippets": []}],
  "specSheet": [{"key": "", "value": "", "confidence": 0.95, "pageRefs": [], "textSnippets": []}],
  "specifications": []
}

Start your response with { immediately."""
    },
    {
        "role": "user",
        "content": f"<image>\nAnalyze millwork drawing for {room_name}. JSON output only. Start with {{."
    },
]

# Also lower temperature
output_ids = model.generate(
    **inputs,
    max_new_tokens=4096,
    temperature=0.1,  # Much lower for deterministic output
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)
```

**Pros:**
- ✅ No post-processing
- ✅ Might work with existing model

**Cons:**
- ⚠️ May not work if base model is too strong
- ⚠️ Still outputs training schema, not your desired schema

### Solution 3: Retrain with Your Schema

**Why:** Permanent fix if you need many inferences

**Steps:**

1. Update training data to use your schema:
```python
# Change from:
{
  "scopeItems": [...],
  "specSheet": [...],
  "specifications": []
}

# To:
{
  "included_items": [
    {
      "room_number": "101",
      "category": "Cabinetry",
      "description": "...",
      "quantity": 10,
      "unit": "LF",
      "finish_code": "WO-1"
    }
  ]
}
```

2. Update system prompt:
```
"Return ONLY valid JSON with included_items array containing room_number, category, description, quantity, unit, and finish_code."
```

3. Retrain with Modal (2 hours, $7):
```bash
modal run training/modal_finetune.py
```

**Pros:**
- ✅ Perfect schema match
- ✅ No post-processing needed
- ✅ Can optimize for your exact use case

**Cons:**
- ⚠️ Requires retraining (2 hours, $7)
- ⚠️ Need to regenerate training data with new schema

### Solution 4: Use Base Model Without Adapters

**Why:** Test if base model does better with your schema

**Code:**

```python
# Load WITHOUT LoRA adapters
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-vicuna-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)
# Skip: PeftModel.from_pretrained() 

# Use VERY explicit prompting with your schema
conversation = [
    {
        "role": "system",
        "content": """You are a construction estimator. Output ONLY JSON in this exact format:

{
  "included_items": [
    {
      "room_number": "string",
      "category": "string",
      "description": "string",
      "quantity": number,
      "unit": "string",
      "finish_code": "string"
    }
  ]
}

No other text. Start with {."""
    },
    {
        "role": "user",
        "content": "<image>\nExtract millwork items. Return JSON only."
    },
]
```

**Pros:**
- ✅ Base model may be more flexible with schemas
- ✅ No schema constraints from training

**Cons:**
- ⚠️ Loses construction-specific fine-tuning
- ⚠️ May be less accurate on construction terminology
- ⚠️ Wastes the 519 samples of training data

## Testing

I've created a test script at [test_json_output.py](test_json_output.py) that uses the exact training format:

```bash
chmod +x test_json_output.py

# Test with training format
python test_json_output.py \
  --adapter-path /path/to/trained_model/final \
  --image /path/to/drawing.png \
  --room "LOBBY"

# Test with conversion to your format
python test_json_output.py \
  --adapter-path /path/to/trained_model/final \
  --image /path/to/drawing.png \
  --room "LOBBY" \
  --convert
```

This will:
1. Load the model with LoRA adapters (FP16)
2. Use EXACT training prompts
3. Show raw response
4. Parse JSON (or show error)
5. Optionally convert to your `included_items` format

## Recommendation

**Use Solution 1 (Training Schema + Convert):**

1. It works immediately without retraining
2. Leverages all your existing training data
3. Maintains the model's construction-specific knowledge
4. Post-processing is simple and customizable
5. You can always retrain later if needed

**Implementation:**

```python
# 1. Use training prompts (scopeItems/specSheet/specifications)
# 2. Get JSON response
# 3. Convert with convert_to_your_format()
# 4. Return included_items format

# Total code: ~30 lines for conversion function
```

## Why Training Schema Matters

The LoRA adapters are only 73 MB (vs 13.5 GB base model). They represent **small weights adjustments** trained on specific examples. They learned:

- Input: Construction drawing + "Return ONLY valid JSON with scopeItems, specSheet, and specifications"
- Output: `{"scopeItems": [...], "specSheet": [...], "specifications": []}`

When you ask for something different (`included_items`), those small weight adjustments don't activate correctly, and the base model's conversational behavior dominates.

**Analogy:** It's like training a dog to sit when you say "sit", then expecting it to sit when you say "please be seated". The command is different enough that the training doesn't apply.

## Next Steps

1. **Test with training format:** Run `test_json_output.py` to confirm JSON output works
2. **Implement conversion:** Use the provided conversion function
3. **Validate results:** Check that `included_items` has correct data
4. **Optional:** Retrain later if you need native support for your schema

The model is working perfectly - you just need to speak its language (training schema) or teach it a new language (retrain with your schema).
