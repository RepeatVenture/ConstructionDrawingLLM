"""
LoRA fine-tuning script for Qwen2.5-VL on construction drawing extraction.

Uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA to adapt the model
to construction-specific drawing analysis while keeping the base model frozen.

Hardware requirements:
  - Minimum: 1x A10G (24GB) for 7B model with INT4 + LoRA
  - Recommended: 1x A100 (40/80GB) for faster training
  - Budget: 1x T4 (16GB) for 3B model only

Usage:
  # Prepare data first
  python training/prepare_dataset.py all

  # Fine-tune
  python training/finetune.py \\
      --train-data training/data/prepared/train.jsonl \\
      --val-data training/data/prepared/val.jsonl \\
      --output-dir training/checkpoints \\
      --epochs 3

  # Resume from checkpoint
  python training/finetune.py \\
      --train-data training/data/prepared/train.jsonl \\
      --resume-from training/checkpoints/checkpoint-500
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════

class ConstructionDrawingDataset(Dataset):
    """Loads JSONL training data with image + conversation pairs."""

    def __init__(
        self,
        jsonl_path: str,
        processor: Any,
        max_length: int = 2048,
        max_image_size: int = 1280,
    ):
        self.samples: List[Dict[str, Any]] = []
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("Loaded %d training samples from %s", len(self.samples), jsonl_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image"]
        conversations = sample["conversations"]

        # Load and resize image
        try:
            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            if max(w, h) > self.max_image_size:
                ratio = self.max_image_size / max(w, h)
                image = image.resize(
                    (int(w * ratio), int(h * ratio)), Image.LANCZOS
                )
        except Exception as e:
            logger.warning("Failed to load image %s: %s", image_path, e)
            # Return a small blank image as fallback
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Build chat messages for the processor
        system_msg = ""
        user_msg = ""
        assistant_msg = ""

        for turn in conversations:
            role = turn["role"]
            if role == "system":
                system_msg = turn["content"]
            elif role == "user":
                user_msg = turn["content"]
            elif role == "assistant":
                assistant_msg = turn["content"]

        # Format as Qwen2.5-VL chat template
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_msg},
            ],
        })
        messages.append({"role": "assistant", "content": assistant_msg})

        # Process with the model's processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Create labels: mask everything before the assistant response
        labels = input_ids.clone()

        # Find where the assistant response starts and mask everything before
        # We mask the system + user tokens so the loss only applies to the answer
        assistant_marker = self.processor.tokenizer.encode(
            "assistant", add_special_tokens=False
        )
        if len(assistant_marker) > 0:
            marker_id = assistant_marker[-1]
            # Find last occurrence of assistant role marker
            positions = (input_ids == marker_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                # Mask everything up to and including the last assistant marker
                mask_end = positions[-1].item() + 1
                labels[:mask_end] = -100

        # Also mask padding
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # Include pixel values if present
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)

        return result


# ═══════════════════════════════════════════════════════════════════════
#  Training configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    # Model
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: str = "int4"  # int4 for LoRA training on 24GB

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"

    # Data
    max_length: int = 2048
    max_image_size: int = 1280

    # Saving
    output_dir: str = "training/checkpoints"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3

    # Misc
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2


# ═══════════════════════════════════════════════════════════════════════
#  Fine-tuning
# ═══════════════════════════════════════════════════════════════════════

def setup_model_and_lora(cfg: TrainingConfig):
    """Load base model with quantization and attach LoRA adapters."""
    from transformers import (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logger.info("Loading base model: %s (quant=%s)", cfg.model_id, cfg.quantization)

    # Quantization config
    bnb_config = None
    if cfg.quantization == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.quantization == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

    # Prepare for k-bit training
    if cfg.quantization in ("int4", "int8"):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    # LoRA config
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def run_finetuning(
    cfg: TrainingConfig,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """Run the full fine-tuning pipeline. Returns path to best checkpoint."""
    from transformers import TrainingArguments, Trainer

    # Setup
    model, processor = setup_model_and_lora(cfg)

    # Datasets
    train_dataset = ConstructionDrawingDataset(
        train_data_path, processor, cfg.max_length, cfg.max_image_size
    )

    val_dataset = None
    if val_data_path and Path(val_data_path).exists():
        val_dataset = ConstructionDrawingDataset(
            val_data_path, processor, cfg.max_length, cfg.max_image_size
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        bf16=cfg.bf16 and torch.cuda.is_bf16_supported(),
        fp16=not cfg.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,
        seed=cfg.seed,
        report_to="none",  # Set to "wandb" if you want W&B logging
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    logger.info("Starting fine-tuning: %d samples, %d epochs", len(train_dataset), cfg.num_epochs)
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final LoRA adapter
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("Final LoRA adapter saved → %s", final_dir)

    return str(final_dir)


# ═══════════════════════════════════════════════════════════════════════
#  Merge LoRA weights into base model (for deployment)
# ═══════════════════════════════════════════════════════════════════════

def merge_lora(
    base_model_id: str,
    lora_path: str,
    output_dir: str,
    quantization: str = "none",
) -> None:
    """Merge LoRA adapter into the base model for standalone deployment."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    logger.info("Loading base model: %s", base_model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter: %s", lora_path)
    model = PeftModel.from_pretrained(model, lora_path)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(out)

    logger.info("Merged model saved → %s", out)
    logger.info(
        "To use in production:\n"
        "  export QWEN2_VL_MODEL_ID=%s\n"
        "  python server.py",
        out,
    )


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for construction drawings")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --
    train_parser = sub.add_parser("train", help="Run LoRA fine-tuning")
    train_parser.add_argument("--train-data", required=True, help="Path to train.jsonl")
    train_parser.add_argument("--val-data", default=None, help="Path to val.jsonl")
    train_parser.add_argument("--output-dir", default="training/checkpoints")
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    train_parser.add_argument("--quantization", default="int4", choices=["none", "int4", "int8"])
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accum", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--lora-rank", type=int, default=64)
    train_parser.add_argument("--lora-alpha", type=int, default=128)
    train_parser.add_argument("--max-length", type=int, default=2048)
    train_parser.add_argument("--resume-from", default=None, help="Resume from checkpoint")
    train_parser.add_argument("--seed", type=int, default=42)

    # -- merge --
    merge_parser = sub.add_parser("merge", help="Merge LoRA into base model for deployment")
    merge_parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    merge_parser.add_argument("--lora-path", required=True, help="Path to LoRA checkpoint")
    merge_parser.add_argument("--output-dir", required=True, help="Output merged model dir")

    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainingConfig(
            model_id=args.model,
            quantization=args.quantization,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            max_length=args.max_length,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        final_path = run_finetuning(
            cfg,
            train_data_path=args.train_data,
            val_data_path=args.val_data,
            resume_from=args.resume_from,
        )
        print(f"\nTraining complete! LoRA adapter: {final_path}")
        print(f"\nTo deploy:")
        print(f"  # Option A: Use LoRA adapter directly (slower, smaller files)")
        print(f"  export LORA_PATH={final_path}")
        print(f"  python server.py")
        print(f"\n  # Option B: Merge into base model (faster inference)")
        print(f"  python training/finetune.py merge \\")
        print(f"      --lora-path {final_path} \\")
        print(f"      --output-dir training/merged_model")

    elif args.command == "merge":
        merge_lora(
            base_model_id=args.base_model,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
        )
