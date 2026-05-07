#!/usr/bin/env python3
"""
Test script to verify LLaVA model outputs JSON in training format.

This uses the EXACT prompts and format from training to ensure JSON output.
"""

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
import json
import sys


def load_model(adapter_path: str):
    """Load base model + LoRA adapters with FP16"""
    print("Loading base model...")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    model.eval()
    
    print("Loading processor...")
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        trust_remote_code=True,
    )
    
    print("✓ Model loaded successfully\n")
    return model, processor


def analyze_with_training_format(model, processor, image_path: str, room_name: str):
    """
    Use EXACT prompts from training data to ensure JSON output.
    
    Training prompts were:
    - System: "You are an expert construction estimator... Return ONLY valid JSON with scopeItems, specSheet, and specifications."
    - User: "Analyze this millwork construction drawing for {ROOM}. Extract ALL scope items... Return ONLY valid JSON."
    """
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Analyzing: {image_path}")
    print(f"Room: {room_name}")
    print(f"Image size: {image.size}\n")
    
    # EXACT format from training data
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
    
    print("Prompt template applied:")
    print(prompt[:500] + "...\n")
    
    # Process inputs
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    print("Generating response...\n")
    
    # Generate with training parameters
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
    
    # Decode only generated tokens
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    print("=" * 80)
    print("RAW RESPONSE:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    print()
    
    # Try to parse as JSON
    try:
        result = json.loads(response)
        print("✓ Successfully parsed as JSON!")
        print(f"\nStructure:")
        print(f"  - scopeItems: {len(result.get('scopeItems', []))} items")
        print(f"  - specSheet: {len(result.get('specSheet', []))} specs")
        print(f"  - specifications: {len(result.get('specifications', []))} specifications")
        return result
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse as JSON: {e}")
        print("\nThis means the model is not outputting JSON format.")
        print("Possible causes:")
        print("1. LoRA adapters not loaded correctly")
        print("2. Base model behavior is overriding the fine-tuning")
        print("3. Need stronger JSON prompting or lower temperature")
        return None


def convert_to_included_items_format(training_output: dict) -> dict:
    """Convert training schema to included_items schema"""
    if not training_output:
        return {"included_items": []}
    
    included_items = []
    
    # Get all finish codes from specSheet
    finish_codes = {
        spec.get("value"): spec 
        for spec in training_output.get("specSheet", [])
        if spec.get("key") == "Finish Code"
    }
    
    # Convert scopeItems
    for item in training_output.get("scopeItems", []):
        # Try to extract room number from textSnippets if available
        room_number = None
        for snippet in item.get("textSnippets", []):
            # Look for room numbers like "101", "A-101", etc.
            import re
            match = re.search(r'\b([A-Z]?-?\d{2,4})\b', snippet)
            if match:
                room_number = match.group(1)
                break
        
        # Use first finish code if multiple (or could match by pageRef)
        finish_code = list(finish_codes.keys())[0] if finish_codes else None
        
        included_items.append({
            "room_number": room_number,
            "category": item.get("category"),
            "description": item.get("description"),
            "quantity": item.get("quantity"),
            "unit": item.get("unit"),
            "finish_code": finish_code,
        })
    
    return {"included_items": included_items}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLaVA model JSON output")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapters (e.g., ./trained_model/final)")
    parser.add_argument("--image", required=True, help="Path to construction drawing image")
    parser.add_argument("--room", default="TEST ROOM", help="Room name")
    parser.add_argument("--convert", action="store_true", help="Convert to included_items format")
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(args.adapter_path)
    
    # Analyze
    result = analyze_with_training_format(model, processor, args.image, args.room)
    
    if result:
        # Save training format
        with open("output_training_format.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n✓ Saved to output_training_format.json")
        
        # Convert if requested
        if args.convert:
            converted = convert_to_included_items_format(result)
            with open("output_included_items_format.json", "w") as f:
                json.dump(converted, f, indent=2)
            print("✓ Saved converted format to output_included_items_format.json")
            
            print("\nConverted Result:")
            print(json.dumps(converted, indent=2))
    else:
        print("\n❌ No valid JSON output - see troubleshooting steps above")
        sys.exit(1)
