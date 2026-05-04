#!/usr/bin/env python3
"""
Download and cache the model weights before first inference.

Usage:
    python scripts/download_model.py                          # Qwen2.5-VL-7B (default)
    python scripts/download_model.py --model llava            # LLaVA-NeXT
    python scripts/download_model.py --model-id "Qwen/Qwen2.5-VL-3B"  # Specific repo
"""
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--model",
        choices=["qwen2-vl", "llava", "internvl2"],
        default="qwen2-vl",
        help="Model backend to download",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Specific HuggingFace model ID (overrides --model)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache weights",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or str(config.MODEL_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    if args.model_id:
        model_id = args.model_id
    else:
        backend = config.ModelBackend(args.model)
        model_id = config.MODEL_IDS[backend]

    print(f"Downloading model: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print()

    from transformers import AutoProcessor

    print("Downloading processor …")
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    print("Downloading model weights …")
    if "qwen" in model_id.lower():
        from transformers import Qwen2VLForConditionalGeneration
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    elif "llava" in model_id.lower():
        from transformers import LlavaNextForConditionalGeneration
        LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
        )
    else:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
        )

    print()
    print(f"✓ Model downloaded successfully to {cache_dir}")
    print(f"  You can now start the server with:")
    print(f"    python server.py")


if __name__ == "__main__":
    main()
