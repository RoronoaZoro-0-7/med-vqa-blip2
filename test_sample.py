"""
test_sample.py — Search for any image+question in the downloaded dataset and run
inference on it.  Works with or without a trained checkpoint.

Usage:
  # Search dataset for a keyword and test (requires checkpoint)
  python test_sample.py --checkpoint checkpoints/best_model.pt --search "fracture"

  # Test on a specific dataset sample by index
  python test_sample.py --checkpoint checkpoints/best_model.pt --index 0

  # List all available samples (no checkpoint needed)
  python test_sample.py --list

  # List only VQA or report samples
  python test_sample.py --list --filter vqa
  python test_sample.py --list --filter report

  # Test on your own image + question
  python test_sample.py --checkpoint checkpoints/best_model.pt --image path/to/img.png --question "Is there a fracture?"

  # Run on multiple random samples
  python test_sample.py --checkpoint checkpoints/best_model.pt --random 5
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
from PIL import Image

from utils import Config, set_seed
from dataset import (
    download_vqa_rad,
    download_slake,
    download_iu_xray,
    generate_synthetic_qa_from_reports,
)


def load_all_samples(data_dir: str) -> list:
    """Download all datasets and return a flat list of samples."""
    all_samples = []

    print("Downloading / loading datasets …")

    try:
        vqa_rad = download_vqa_rad(data_dir)
        for split in vqa_rad.values():
            all_samples.extend(split)
        print(f"  VQA-RAD: {sum(len(v) for v in vqa_rad.values())} samples")
    except Exception as e:
        print(f"  VQA-RAD: failed ({e})")

    try:
        slake = download_slake(data_dir)
        for split in slake.values():
            all_samples.extend(split)
        print(f"  Slake:   {sum(len(v) for v in slake.values())} samples")
    except Exception as e:
        print(f"  Slake:   failed ({e})")

    try:
        iu = download_iu_xray(data_dir)
        for split in iu.values():
            all_samples.extend(split)
        print(f"  IU-XRay: {sum(len(v) for v in iu.values())} samples")
    except Exception as e:
        print(f"  IU-XRay: failed ({e})")

    # Filter out samples with missing images
    valid = [s for s in all_samples if os.path.exists(s.get("image_path", ""))]
    print(f"\nTotal valid samples: {len(valid)}  "
          f"(VQA: {sum(1 for s in valid if s['task']=='vqa')}, "
          f"Report: {sum(1 for s in valid if s['task']=='report')})")
    return valid


def search_samples(samples: list, query: str) -> list:
    """Search samples by keyword in question, answer, input_text, or image path."""
    query_lower = query.lower()
    results = []
    for s in samples:
        searchable = " ".join([
            s.get("question", ""),
            s.get("answer", ""),
            s.get("input_text", ""),
            s.get("image_path", ""),
            s.get("target_text", ""),
        ]).lower()
        if query_lower in searchable:
            results.append(s)
    return results


def print_sample(idx: int, sample: dict, compact: bool = False):
    """Pretty-print a dataset sample."""
    task = sample.get("task", "?")
    img = sample.get("image_path", "?")
    img_short = os.path.basename(img) if img else "?"

    if compact:
        if task == "vqa":
            q = sample.get("question", "")[:60]
            a = sample.get("answer", "")[:30]
            print(f"  [{idx:4d}] VQA    | Q: {q:<60s} | A: {a}")
        else:
            t = sample.get("target_text", "")[:80]
            print(f"  [{idx:4d}] Report | {t}")
    else:
        print(f"  Index      : {idx}")
        print(f"  Task       : {task}")
        print(f"  Image      : {img_short}")
        print(f"  Image Path : {img}")
        if task == "vqa":
            print(f"  Question   : {sample.get('question', '')}")
            print(f"  Answer(GT) : {sample.get('answer', '')}")
            qt = sample.get("question_type", -1)
            qt_str = "closed" if qt == 0 else ("open" if qt == 1 else "unknown")
            print(f"  Q-Type     : {qt_str}")
        else:
            print(f"  Report(GT) : {sample.get('target_text', '')[:200]}")
        print()


def run_inference(sample: dict, engine, task_override: str = None):
    """Run the inference engine on a single sample and display results."""
    from inference import InferenceEngine

    img_path = sample["image_path"]
    task = task_override or sample.get("task", "auto")
    question = sample.get("question") if task in ("vqa", "both") else None

    if task == "vqa" and question:
        result = engine.predict(img_path, question=question, task="vqa")
    elif task == "report":
        result = engine.predict(img_path, task="report")
    else:
        result = engine.predict(img_path, question=question, task="both" if question else "report")

    print("  --- Model Output ---")
    if result.get("answer"):
        print(f"  Answer       : {result['answer']}")
    if result.get("question_type"):
        print(f"  Q-Type(pred) : {result['question_type']}")
    if result.get("confidence") is not None:
        print(f"  Confidence   : {result['confidence']}")
    if result.get("report"):
        rpt = result["report"][:300]
        print(f"  Report       : {rpt}")
    if result.get("consistency") is not None:
        print(f"  Consistency  : {result['consistency']}")
    if result.get("explanation"):
        print(f"  Explanation  : {result['explanation']}")

    # Compare with ground truth
    gt_answer = sample.get("answer", "")
    pred_answer = result.get("answer", "")
    if gt_answer and pred_answer:
        match = gt_answer.strip().lower() == pred_answer.strip().lower()
        print(f"  Match(GT)    : {'✓ CORRECT' if match else '✗ WRONG'}")
        print(f"  GT Answer    : {gt_answer}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Search dataset samples and run inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_sample.py --list                                          # List all samples
  python test_sample.py --list --filter vqa --search "fracture"         # Search VQA for "fracture"
  python test_sample.py --checkpoint checkpoints/best_model.pt --index 42  # Test sample #42
  python test_sample.py --checkpoint checkpoints/best_model.pt --search "cardiomegaly"
  python test_sample.py --checkpoint checkpoints/best_model.pt --random 5  # Test 5 random samples
  python test_sample.py --checkpoint checkpoints/best_model.pt --image chest.png --question "Is there effusion?"
        """,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--list", action="store_true", help="List available dataset samples")
    parser.add_argument("--search", type=str, default=None, help="Search keyword in question/answer/path")
    parser.add_argument("--filter", type=str, choices=["vqa", "report"], default=None, help="Filter by task type")
    parser.add_argument("--index", type=int, default=None, help="Test a specific sample by index")
    parser.add_argument("--random", type=int, default=None, help="Test N random samples")
    parser.add_argument("--image", type=str, default=None, help="Custom image path for inference")
    parser.add_argument("--question", type=str, default=None, help="Custom question for inference")
    parser.add_argument("--task", type=str, default="auto", choices=["vqa", "report", "both", "auto"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config()
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir

    # ---- Custom image mode (no dataset needed) ----
    if args.image:
        if not args.checkpoint:
            parser.error("--checkpoint required for inference")
        if not os.path.exists(args.image):
            parser.error(f"Image not found: {args.image}")

        from inference import InferenceEngine
        print("Loading model …")
        engine = InferenceEngine(args.checkpoint, args.config, args.device)
        print("Model loaded.\n")

        sample = {
            "image_path": args.image,
            "task": args.task if args.task != "auto" else ("vqa" if args.question else "report"),
            "question": args.question or "",
        }
        print(f"Image: {args.image}")
        if args.question:
            print(f"Question: {args.question}")
        print()
        run_inference(sample, engine, task_override=args.task)
        return

    # ---- Load dataset ----
    samples = load_all_samples(config.data_dir)
    if not samples:
        print("No dataset samples found. Run 'python train.py' first to download datasets.")
        return

    # Apply task filter
    if args.filter:
        samples = [s for s in samples if s.get("task") == args.filter]
        print(f"Filtered to {len(samples)} {args.filter} samples\n")

    # Apply search
    if args.search:
        samples = search_samples(samples, args.search)
        print(f"Search '{args.search}' → {len(samples)} matches\n")

    # ---- List mode ----
    if args.list:
        if not samples:
            print("No matching samples found.")
            return
        print(f"\n{'='*80}")
        print(f"  Dataset Samples ({len(samples)} total)")
        print(f"{'='*80}")
        # Show at most 100
        show = samples[:100]
        for i, s in enumerate(show):
            # Use original index lookup
            print_sample(i, s, compact=True)
        if len(samples) > 100:
            print(f"\n  ... and {len(samples) - 100} more. Use --search to narrow.")
        return

    # ---- Inference modes (require checkpoint) ----
    if not args.checkpoint:
        parser.error("--checkpoint required for inference (or use --list to browse)")
    if not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    from inference import InferenceEngine
    print("Loading model …")
    engine = InferenceEngine(args.checkpoint, args.config, args.device)
    print("Model loaded.\n")

    # Test by index
    if args.index is not None:
        if args.index < 0 or args.index >= len(samples):
            parser.error(f"Index {args.index} out of range [0, {len(samples)-1}]")
        s = samples[args.index]
        print(f"{'='*60}")
        print(f"  Testing Sample #{args.index}")
        print(f"{'='*60}")
        print_sample(args.index, s)
        run_inference(s, engine)
        return

    # Test random samples
    if args.random:
        n = min(args.random, len(samples))
        chosen = random.sample(range(len(samples)), n)
        results = []
        correct = 0
        total_vqa = 0
        for i, idx in enumerate(chosen):
            s = samples[idx]
            print(f"\n{'='*60}")
            print(f"  Sample {i+1}/{n}  (dataset index #{idx})")
            print(f"{'='*60}")
            print_sample(idx, s)
            result = run_inference(s, engine)
            results.append(result)
            # Track accuracy for VQA
            gt = s.get("answer", "").strip().lower()
            pred = result.get("answer", "").strip().lower()
            if s.get("task") == "vqa" and gt:
                total_vqa += 1
                if gt == pred:
                    correct += 1

        print(f"\n{'='*60}")
        print(f"  Summary: Tested {n} samples")
        if total_vqa:
            print(f"  VQA Accuracy: {correct}/{total_vqa} = {correct/total_vqa:.1%}")
        print(f"{'='*60}")
        return

    # Default: search results inference
    if args.search and samples:
        n = min(5, len(samples))
        print(f"Testing top {n} search results:\n")
        for i in range(n):
            s = samples[i]
            print(f"\n{'='*60}")
            print(f"  Search Result {i+1}/{n}")
            print(f"{'='*60}")
            print_sample(i, s)
            run_inference(s, engine)
        return

    # No specific action
    print("Use --list, --search, --index, --random, or --image. See --help for details.")


if __name__ == "__main__":
    main()
