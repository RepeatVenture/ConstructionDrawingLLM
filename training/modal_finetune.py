"""
Modal GPU training for Construction Drawing LLM.

Model: LLaVA-v1.6-Vicuna-7B (vision-language model)

Usage:
    modal run training/modal_finetune.py

This will:
- Spin up an A100-40GB GPU instance
- Upload your training data
- Run fine-tuning for 3 epochs
- Download checkpoints when complete
- Auto-shutdown when finished

Cost: ~$3.50/hour, ~$14-21 total for 3 epochs
"""
from __future__ import annotations
import os
from pathlib import Path

import modal

# ── Docker image with all dependencies ────────────────────────────────

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.2",
        "torchvision",
        "transformers>=4.40",
        "accelerate>=0.28",
        "bitsandbytes>=0.43",
        "peft>=0.10",
        "Pillow>=10.0",
        "numpy>=1.26",
        "datasets>=2.18",
        "tqdm",
    )
    .env({"FORCE_REBUILD": "v17"})  # Force rebuild - must come before add_local_dir
    .add_local_dir(
        local_path="/workspaces/ConstructionDrawingLLM/training/data",
        remote_path="/data"
    )
)

app = modal.App("construction-llm-training")

# ── Volume for model cache and checkpoints ────────────────────────────

model_volume = modal.Volume.from_name(
    "construction-model-cache",
    create_if_missing=True,
)

checkpoint_volume = modal.Volume.from_name(
    "construction-checkpoints",
    create_if_missing=True,
)

MODEL_CACHE_PATH = "/model_cache"
CHECKPOINT_PATH = "/checkpoints"

# ── Training function ──────────────────────────────────────────────────


@app.function(
    image=gpu_image,
    gpu="A100-40GB",  # Upgraded from A10G for more VRAM
    timeout=6 * 3600,  # 6 hours max
    volumes={
        MODEL_CACHE_PATH: model_volume,
        CHECKPOINT_PATH: checkpoint_volume,
    },
    secrets=[],  # No secrets needed for public models
)
def train():
    """Run fine-tuning on Modal GPU."""
    import json
    import torch
    from transformers import (
        LlavaNextProcessor,
        LlavaNextForConditionalGeneration,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from PIL import Image
    import io
    
    print("=" * 70)
    print("  Construction Drawing LLM - Fine-tuning")
    print("=" * 70)
    print()
    
    # ── Load training data ─────────────────────────────────────────────
    
    print("📊 Loading training data from mounted directory...")
    
    # Data is mounted at /data
    train_path = "/data/prepared/train.jsonl"
    val_path = "/data/prepared/val.jsonl"
    
    # Parse JSONL
    train_samples = []
    with open(train_path, "r") as f:
        for line in f:
            train_samples.append(json.loads(line))
    
    val_samples = []
    with open(val_path, "r") as f:
        for line in f:
            val_samples.append(json.loads(line))
    
    print(f"✓ Loaded {len(train_samples)} training samples")
    print(f"✓ Loaded {len(val_samples)} validation samples")
    print()
    
    # ── Load model and processor ───────────────────────────────────────
    
    MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
    
    print(f"🤖 Loading model: {MODEL_ID}")
    print(f"   Cache: {MODEL_CACHE_PATH}")
    
    processor = LlavaNextProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_PATH,
    )
    
    # Load model in FP16 (A100-40GB has enough VRAM)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=MODEL_CACHE_PATH,
    )
    
    print(f"✓ Model loaded")
    print(f"   Device: {model.device}")
    print(f"   Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()
    
    # ── Apply LoRA ─────────────────────────────────────────────────────
    
    print("🔧 Applying LoRA adapters...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing to reduce memory usage
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")
    print()
    
    # ── Prepare dataset ────────────────────────────────────────────────
    
    def load_image(image_path_str):
        """Load image from file path."""
        # Paths in JSONL are like "data/raw/PROJECT/images/..."
        # Convert to absolute path in mounted directory: "/data/raw/..."
        if image_path_str.startswith("data/"):
            image_path_str = "/" + image_path_str
        return Image.open(image_path_str).convert("RGB")
    
    class LLaVADataCollator:
        """Custom data collator for LLaVA-v1.6 fine-tuning."""
        
        def __init__(self, processor):
            self.processor = processor
        
        def __call__(self, batch):
            """Process a batch of samples."""
            # Collect all texts and images for batch processing
            texts = []
            images = []
            
            for sample in batch:
                # Load image from file path
                image = load_image(sample["image"])
                images.append(image)
                
                # Extract conversation parts
                messages = sample["conversations"]
                user_msg = ""
                assistant_msg = ""
                
                for turn in messages:
                    role = turn.get("role", "")
                    if role == "user":
                        user_msg = turn["content"]
                    elif role == "assistant":
                        assistant_msg = turn["content"]
                
                # Format as LLaVA conversation
                # LLaVA format: "USER: <image>\n{question}\nASSISTANT: {answer}"
                conversation = f"USER: <image>\n{user_msg}\nASSISTANT: {assistant_msg}"
                texts.append(conversation)
            
            # Process entire batch at once with automatic padding  
            inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            
            # Create labels (mask padding tokens)
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            inputs["labels"] = labels
            return inputs
    
    print("🔄 Preparing datasets...")
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    data_collator = LLaVADataCollator(processor)
    print()
    
    # ── Training configuration ─────────────────────────────────────────
    
    import time
    run_id = f"train-{int(time.time())}"
    output_dir = f"{CHECKPOINT_PATH}/{run_id}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        remove_unused_columns=False,  # Keep all columns for custom collator
    )
    
    print("⚙️  Training configuration:")
    print(f"   Epochs: 3")
    print(f"   Batch size: 1 (reduced for memory)")
    print(f"   Gradient accumulation: 4")
    print(f"   Effective batch size: 4")
    print(f"   Learning rate: 2e-4")
    print(f"   Output: {output_dir}")
    print()
    
    # ── Start training ─────────────────────────────────────────────────
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("=" * 70)
    print("🚀 Starting training...")
    print("=" * 70)
    print()
    
    trainer.train()
    
    print()
    print("=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print()
    
    # ── Save final model ───────────────────────────────────────────────
    
    final_dir = f"{output_dir}/final"
    print(f"💾 Saving final model to: {final_dir}")
    trainer.save_model(final_dir)
    checkpoint_volume.commit()
    
    print()
    print("✓ Model saved successfully")
    print(f"   Download with: modal volume get construction-checkpoints {output_dir}")
    print()
    
    return output_dir


# ── Entry point ────────────────────────────────────────────────────────


@app.local_entrypoint()
def main():
    """Start GPU training on Modal."""
    import sys
    from pathlib import Path
    
    print()
    print("=" * 70)
    print("  Modal GPU Training - Construction Drawing LLM")
    print("=" * 70)
    print()
    
    # Verify training data exists locally
    data_dir = Path(__file__).parent / "data" / "prepared"
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"❌ Validation data not found: {val_file}")
        sys.exit(1)
    
    print(f"📂 Training data found:")
    print(f"   Train: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"   Val:   {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print()
    print("🚀 Launching GPU training on Modal...")
    print("   Instance: A100-40GB")
    print("   Cost: ~$3.50/hour (upgraded from A10G for VRAM)")
    print("   Duration: ~4-6 hours (3 epochs)")
    print("   Total cost: ~$14-21")
    print()
    
    # Start training (data is mounted automatically)
    checkpoint_path = train.remote()
    
    print()
    print("=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    print()
    print(f"📦 Checkpoints saved to: {checkpoint_path}")
    print()
    print("To download:")
    print(f"  modal volume get construction-checkpoints {checkpoint_path} ./checkpoints")
    print()
