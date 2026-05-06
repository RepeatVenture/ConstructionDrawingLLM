#!/usr/bin/env python3
"""
Test the trained Construction Drawing LLM.

Usage:
    python test_trained_model.py <image_path> "<question>"

Example:
    python test_trained_model.py data/raw/Brightview/images/all_pages/sheet_001.png "What quantities are shown on this drawing?"
"""
import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel

def load_model():
    """Load the base model and trained LoRA adapters."""
    print("Loading model...")
    
    # Base model
    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    
    # Load processor
    processor = LlavaNextProcessor.from_pretrained(model_id)
    
    # Load base model
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load trained LoRA adapters
    adapter_path = "trained_model/final"
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
    )
    
    print("✓ Model loaded successfully!")
    return processor, model


def analyze_drawing(image_path: str, question: str, processor, model):
    """Analyze a construction drawing with a specific question."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"\n📄 Analyzing: {image_path}")
    print(f"❓ Question: {question}\n")
    
    # Format prompt (LLaVA format)
    conversation = f"USER: <image>\n{question}\nASSISTANT:"
    
    # Process inputs
    inputs = processor(
        text=conversation,
        images=image,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate response
    print("🤖 Generating answer...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    # Decode response
    response = processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    print(f"\n💡 Answer:\n{response}\n")
    return response


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    image_path = sys.argv[1]
    question = sys.argv[2]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load model
    processor, model = load_model()
    
    # Analyze drawing
    analyze_drawing(image_path, question, processor, model)


if __name__ == "__main__":
    main()
