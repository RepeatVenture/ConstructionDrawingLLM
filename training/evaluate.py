"""
Evaluation pipeline for the fine-tuned construction drawing model.

Measures:
  - Per-trade extraction accuracy (precision, recall, F1)
  - Quantity accuracy (within tolerance)
  - Category classification accuracy
  - JSON schema compliance rate
  - Confidence calibration
  - End-to-end latency

Usage:
  python training/evaluate.py \
      --val-data training/data/prepared/val.jsonl \
      --model-path training/checkpoints/final \
      --output training/eval_results.json
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_item_matching(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    quantity_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """
    Match predicted scope items to ground truth using category + description.
    Returns precision, recall, F1, and per-item metrics.
    """
    matched_gt = set()
    matched_pred = set()
    matches = []

    for pi, pred in enumerate(predicted):
        best_score = 0.0
        best_gi = -1

        for gi, gt in enumerate(ground_truth):
            if gi in matched_gt:
                continue

            score = _item_similarity(pred, gt)
            if score > best_score:
                best_score = score
                best_gi = gi

        if best_score >= 0.4 and best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

            gt_item = ground_truth[best_gi]
            qty_pred = pred.get("quantity", 0)
            qty_gt = gt_item.get("quantity", 0)
            qty_error = abs(qty_pred - qty_gt) / max(qty_gt, 1e-6)
            qty_correct = qty_error <= quantity_tolerance

            matches.append({
                "predicted": pred,
                "ground_truth": gt_item,
                "similarity": best_score,
                "quantity_error": qty_error,
                "quantity_correct": qty_correct,
                "category_match": (
                    pred.get("category", "").lower().strip()
                    == gt_item.get("category", "").lower().strip()
                ),
                "unit_match": (
                    pred.get("unit", "").lower().strip()
                    == gt_item.get("unit", "").lower().strip()
                ),
            })

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    qty_correct = sum(1 for m in matches if m["quantity_correct"])
    cat_correct = sum(1 for m in matches if m["category_match"])
    unit_correct = sum(1 for m in matches if m["unit_match"])

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "quantity_accuracy": round(qty_correct / max(tp, 1), 4),
        "category_accuracy": round(cat_correct / max(tp, 1), 4),
        "unit_accuracy": round(unit_correct / max(tp, 1), 4),
        "avg_quantity_error": round(
            sum(m["quantity_error"] for m in matches) / max(len(matches), 1), 4
        ),
        "matches": matches,
    }


def _item_similarity(pred: Dict[str, Any], gt: Dict[str, Any]) -> float:
    """Compute similarity between two scope items (0-1)."""
    score = 0.0

    # Category match (0.4 weight)
    pred_cat = pred.get("category", "").lower().strip()
    gt_cat = gt.get("category", "").lower().strip()
    if pred_cat == gt_cat:
        score += 0.4
    elif pred_cat in gt_cat or gt_cat in pred_cat:
        score += 0.2

    # Description overlap (0.4 weight)
    pred_desc = set(pred.get("description", "").lower().split())
    gt_desc = set(gt.get("description", "").lower().split())
    if pred_desc and gt_desc:
        overlap = len(pred_desc & gt_desc) / max(len(pred_desc | gt_desc), 1)
        score += 0.4 * overlap

    # Unit match (0.2 weight)
    if pred.get("unit", "").lower() == gt.get("unit", "").lower():
        score += 0.2

    return score


def check_schema_compliance(result: Dict[str, Any]) -> Dict[str, Any]:
    """Check if the output conforms to the BlueprintBid schema."""
    issues = []

    # Top-level keys
    if "scopeItems" not in result:
        issues.append("Missing 'scopeItems' key")
    if "specSheet" not in result:
        issues.append("Missing 'specSheet' key")
    if "specifications" not in result:
        issues.append("Missing 'specifications' key")

    # Validate each scope item
    for i, item in enumerate(result.get("scopeItems", [])):
        required = ["category", "description", "quantity", "unit"]
        for key in required:
            if key not in item:
                issues.append(f"scopeItems[{i}] missing '{key}'")

        if "unit" in item and item["unit"] not in ("ea", "lf", "sf", "lot"):
            issues.append(f"scopeItems[{i}] invalid unit: {item['unit']}")

        if "confidence" in item:
            conf = item["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                issues.append(f"scopeItems[{i}] confidence out of range: {conf}")

        if "quantity" in item:
            qty = item["quantity"]
            if not isinstance(qty, (int, float)) or qty < 0:
                issues.append(f"scopeItems[{i}] invalid quantity: {qty}")

    return {
        "compliant": len(issues) == 0,
        "issue_count": len(issues),
        "issues": issues,
    }


def compute_confidence_calibration(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Check if confidence scores correlate with actual accuracy."""
    bins = defaultdict(lambda: {"correct": 0, "total": 0, "sum_conf": 0.0})

    for pred_result, gt_result in zip(predictions, ground_truths):
        matching = compute_item_matching(
            pred_result.get("scopeItems", []),
            gt_result.get("scopeItems", []),
        )
        for match in matching["matches"]:
            conf = match["predicted"].get("confidence", 0.5)
            bin_idx = min(int(conf * n_bins), n_bins - 1)
            is_correct = match["category_match"] and match["quantity_correct"]
            bins[bin_idx]["correct"] += int(is_correct)
            bins[bin_idx]["total"] += 1
            bins[bin_idx]["sum_conf"] += conf

    calibration = {}
    ece = 0.0  # Expected Calibration Error
    total = sum(b["total"] for b in bins.values())

    for i in range(n_bins):
        b = bins[i]
        if b["total"] > 0:
            accuracy = b["correct"] / b["total"]
            avg_conf = b["sum_conf"] / b["total"]
            calibration[f"bin_{i}"] = {
                "range": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                "count": b["total"],
                "accuracy": round(accuracy, 4),
                "avg_confidence": round(avg_conf, 4),
            }
            ece += abs(accuracy - avg_conf) * (b["total"] / max(total, 1))

    return {
        "ece": round(ece, 4),
        "bins": calibration,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model(
    val_data_path: str,
    model_path: Optional[str] = None,
    base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run evaluation on the validation set.

    Args:
        val_data_path: Path to val.jsonl
        model_path: Path to LoRA checkpoint (None = evaluate base model)
        base_model_id: Base model ID
        max_samples: Limit evaluation to N samples

    Returns:
        Full evaluation results dict.
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Load model
    logger.info("Loading model for evaluation...")
    processor = AutoProcessor.from_pretrained(
        model_path or base_model_id, trust_remote_code=True
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if model_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        logger.info("Loaded LoRA adapter from %s", model_path)

    model.eval()

    # Load val data
    samples = []
    with open(val_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    logger.info("Evaluating on %d samples...", len(samples))

    # Run inference
    predictions = []
    ground_truths = []
    latencies = []
    schema_results = []
    trade_metrics = defaultdict(lambda: {"preds": [], "gts": []})

    for i, sample in enumerate(samples):
        image_path = sample["image"]
        conversations = sample["conversations"]

        # Extract ground truth from assistant message
        gt_text = ""
        user_text = ""
        system_text = ""
        for turn in conversations:
            if turn["role"] == "assistant":
                gt_text = turn["content"]
            elif turn["role"] == "user":
                user_text = turn["content"]
            elif turn["role"] == "system":
                system_text = turn["content"]

        try:
            gt_result = json.loads(gt_text)
        except json.JSONDecodeError:
            continue

        # Determine trade from user prompt
        trade = "general"
        for t in ["millwork", "plumbing", "electrical", "flooring"]:
            if t in user_text.lower():
                trade = t
                break

        # Run inference
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        })

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
            )
        latency = time.time() - start_time
        latencies.append(latency)

        # Decode
        generated = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        # Parse prediction
        try:
            pred_result = json.loads(generated)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', generated)
            if json_match:
                try:
                    pred_result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pred_result = {"scopeItems": [], "specSheet": [], "specifications": []}
            else:
                pred_result = {"scopeItems": [], "specSheet": [], "specifications": []}

        predictions.append(pred_result)
        ground_truths.append(gt_result)

        # Schema compliance
        schema_results.append(check_schema_compliance(pred_result))

        # Per-trade tracking
        trade_metrics[trade]["preds"].append(pred_result)
        trade_metrics[trade]["gts"].append(gt_result)

        if (i + 1) % 10 == 0:
            logger.info("Evaluated %d/%d samples", i + 1, len(samples))

    # ── Aggregate metrics ──
    all_pred_items = [p.get("scopeItems", []) for p in predictions]
    all_gt_items = [g.get("scopeItems", []) for g in ground_truths]

    # Overall item matching
    all_preds_flat = [item for items in all_pred_items for item in items]
    all_gts_flat = [item for items in all_gt_items for item in items]
    overall = compute_item_matching(all_preds_flat, all_gts_flat)
    # Remove per-match details for summary
    overall_summary = {k: v for k, v in overall.items() if k != "matches"}

    # Per-trade metrics
    per_trade = {}
    for trade, data in trade_metrics.items():
        t_preds = [item for p in data["preds"] for item in p.get("scopeItems", [])]
        t_gts = [item for g in data["gts"] for item in g.get("scopeItems", [])]
        t_metrics = compute_item_matching(t_preds, t_gts)
        per_trade[trade] = {k: v for k, v in t_metrics.items() if k != "matches"}
        per_trade[trade]["sample_count"] = len(data["preds"])

    # Schema compliance
    schema_compliant = sum(1 for s in schema_results if s["compliant"])

    # Confidence calibration
    calibration = compute_confidence_calibration(predictions, ground_truths)

    results = {
        "summary": {
            "total_samples": len(predictions),
            "overall_f1": overall_summary["f1"],
            "overall_precision": overall_summary["precision"],
            "overall_recall": overall_summary["recall"],
            "quantity_accuracy": overall_summary["quantity_accuracy"],
            "category_accuracy": overall_summary["category_accuracy"],
            "unit_accuracy": overall_summary["unit_accuracy"],
            "schema_compliance": round(schema_compliant / max(len(schema_results), 1), 4),
            "avg_latency_sec": round(sum(latencies) / max(len(latencies), 1), 2),
            "p95_latency_sec": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
        },
        "per_trade": per_trade,
        "calibration": calibration,
        "overall_metrics": overall_summary,
        "model": model_path or base_model_id,
    }

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    s = results["summary"]
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:              {results['model']}")
    print(f"  Samples:            {s['total_samples']}")
    print(f"  Overall F1:         {s['overall_f1']:.1%}")
    print(f"  Precision:          {s['overall_precision']:.1%}")
    print(f"  Recall:             {s['overall_recall']:.1%}")
    print(f"  Quantity accuracy:  {s['quantity_accuracy']:.1%}")
    print(f"  Category accuracy:  {s['category_accuracy']:.1%}")
    print(f"  Unit accuracy:      {s['unit_accuracy']:.1%}")
    print(f"  Schema compliance:  {s['schema_compliance']:.1%}")
    print(f"  Avg latency:        {s['avg_latency_sec']:.1f}s")
    print(f"  P95 latency:        {s['p95_latency_sec']:.1f}s")

    if results["per_trade"]:
        print("\n  Per-Trade Breakdown:")
        print(f"  {'Trade':<15} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Qty%':>6} {'N':>5}")
        print("  " + "-" * 44)
        for trade, m in sorted(results["per_trade"].items()):
            print(
                f"  {trade:<15} {m['f1']:>5.1%} {m['precision']:>5.1%} "
                f"{m['recall']:>5.1%} {m['quantity_accuracy']:>5.1%} {m['sample_count']:>5}"
            )

    cal = results["calibration"]
    print(f"\n  Confidence calibration (ECE): {cal['ece']:.4f}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned construction drawing model")
    parser.add_argument("--val-data", required=True, help="Path to val.jsonl")
    parser.add_argument("--model-path", default=None, help="Path to LoRA checkpoint (omit for base model)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval to N samples")
    parser.add_argument("--output", default=None, help="Save results JSON to file")

    args = parser.parse_args()

    results = evaluate_model(
        val_data_path=args.val_data,
        model_path=args.model_path,
        base_model_id=args.base_model,
        max_samples=args.max_samples,
    )

    print_results(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved → {args.output}")
