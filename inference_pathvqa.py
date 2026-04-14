"""
inference_pathvqa.py - Run inference on PathVQA dataset

Usage:
    python inference_pathvqa.py --checkpoint checkpoint_epoch_1.pt
    python inference_pathvqa.py --hf_weights checkpoint_epoch_1.pt --hf_repo roronoazoro07/med-vqa-weights
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from utils import (
    Config,
    setup_logging,
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    parse_json_output,
    visualize_predictions,
    visualize_results_comparison,
)
from model import MedicalBLIP2


def download_pathvqa():
    """Download PathVQA dataset from HuggingFace."""
    try:
        # PathVQA available datasets
        hf_names = [
            "flaviagiammarino/path-vqa",
            "JadeCheng/PathVQA",
        ]
        
        for hf_name in hf_names:
            try:
                print(f"Trying to load PathVQA from {hf_name}...")
                ds = load_dataset(hf_name)
                print(f"✓ Loaded PathVQA from {hf_name}")
                return ds
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Fallback: use VQA-RAD test set as demo
        print("PathVQA not available, using VQA-RAD test set as demo...")
        ds = load_dataset("flaviagiammarino/vqa-rad")
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def prepare_samples(dataset, split="test", max_samples=100):
    """Prepare samples for inference."""
    samples = []
    
    if split not in dataset:
        split = list(dataset.keys())[0]
        print(f"Split '{split}' not found, using '{split}'")
    
    ds = dataset[split]
    
    for idx, item in enumerate(tqdm(ds, desc=f"Preparing {split} samples")):
        if idx >= max_samples:
            break
        
        try:
            # Find image key
            img_key = None
            for k in item.keys():
                if k.lower() in ['image', 'img', 'picture']:
                    img_key = k
                    break
            
            if img_key is None:
                continue
            
            # Find question and answer keys
            question_key = next((k for k in item.keys() if 'question' in k.lower()), None)
            answer_key = next((k for k in item.keys() if 'answer' in k.lower()), None)
            
            if question_key is None or answer_key is None:
                continue
            
            image = item[img_key]
            if not isinstance(image, Image.Image):
                continue
            
            question = str(item[question_key]).strip()
            answer = str(item[answer_key]).strip()
            
            samples.append({
                'image': image,
                'question': question,
                'answer': answer,
                'input_text': f"Task: VQA Question: {question}",
                'target_text': json.dumps({"answer": answer, "explanation": ""}),
            })
            
        except Exception as e:
            continue
    
    print(f"Prepared {len(samples)} samples from {split} split")
    return samples


def run_inference(model, samples, device, config, logger):
    """Run inference on samples."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Running inference"):
            try:
                # Process image
                pixel_values = model.image_processor(
                    images=sample['image'],
                    return_tensors="pt"
                )["pixel_values"].to(device)
                
                # Process input text
                input_encoding = model.tokenizer(
                    sample['input_text'],
                    max_length=config.max_input_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = input_encoding["input_ids"].to(device)
                attention_mask = input_encoding["attention_mask"].to(device)
                
                # Generate prediction
                gen_ids = model.generate(
                    pixel_values,
                    input_ids,
                    attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                )
                
                pred_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                predictions.append(pred_text)
                targets.append(sample['target_text'])
                
            except Exception as e:
                logger.warning(f"Inference failed for sample: {e}")
                predictions.append('{"answer": "", "explanation": ""}')
                targets.append(sample['target_text'])
    
    return predictions, targets


def evaluate_predictions(predictions, targets, samples):
    """Evaluate predictions and compute metrics."""
    correct = 0
    total = 0
    bleu_scores = []
    rouge_scores = []
    
    results = []
    
    for pred, target, sample in zip(predictions, targets, samples):
        try:
            pred_parsed = parse_json_output(pred)
            target_parsed = parse_json_output(target)
            
            pred_answer = pred_parsed.get("answer", "").strip().lower()
            true_answer = target_parsed.get("answer", "").strip().lower()
            
            # Exact match accuracy
            is_correct = pred_answer == true_answer
            if is_correct:
                correct += 1
            total += 1
            
            # BLEU/ROUGE
            if pred_answer and true_answer:
                bleu = compute_bleu([true_answer], pred_answer)
                rouge = compute_rouge([true_answer], pred_answer)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge['rougeL'])
            
            results.append({
                'question': sample['question'],
                'ground_truth': true_answer,
                'prediction': pred_answer,
                'correct': is_correct,
                'image_path': f"sample_{len(results)}",
            })
            
        except Exception as e:
            total += 1
            results.append({
                'question': sample.get('question', ''),
                'ground_truth': sample.get('answer', ''),
                'prediction': '',
                'correct': False,
                'image_path': '',
            })
    
    metrics = {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
        'bleu': np.mean(bleu_scores) if bleu_scores else 0,
        'rougeL': np.mean(rouge_scores) if rouge_scores else 0,
    }
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Inference on PathVQA dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint path")
    parser.add_argument("--hf_weights", type=str, default=None, help="HuggingFace checkpoint filename")
    parser.add_argument("--hf_repo", type=str, default="roronoazoro07/med-vqa-weights", help="HuggingFace repo ID")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to process")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    config = Config()
    logger = setup_logging(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info("Medical BLIP-2: PathVQA Inference")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model = MedicalBLIP2(config)
    model.to(device)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if args.hf_weights:
        logger.info(f"Downloading checkpoint from HuggingFace: {args.hf_repo}/{args.hf_weights}")
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_weights,
            cache_dir="hf_cache"
        )
        logger.info(f"Downloaded to {checkpoint_path}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        logger.warning("No checkpoint loaded, using randomly initialized weights")
    
    model.eval()
    
    # Download dataset
    logger.info("Downloading PathVQA dataset...")
    dataset = download_pathvqa()
    
    # Prepare samples
    samples = prepare_samples(dataset, split="test", max_samples=args.max_samples)
    
    if not samples:
        logger.error("No samples prepared!")
        return
    
    # Run inference
    logger.info(f"Running inference on {len(samples)} samples...")
    predictions, targets = run_inference(model, samples, device, config, logger)
    
    # Evaluate
    logger.info("Evaluating predictions...")
    metrics, results = evaluate_predictions(predictions, targets, samples)
    
    # Print results
    print("\n" + "=" * 70)
    print("         PathVQA INFERENCE RESULTS")
    print("=" * 70)
    print(f"  Total Samples:    {metrics['total']}")
    print(f"  Correct:          {metrics['correct']}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  BLEU:             {metrics['bleu']:.4f}")
    print(f"  ROUGE-L:          {metrics['rougeL']:.4f}")
    print("=" * 70)
    
    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 70)
    for i, r in enumerate(results[:10]):
        print(f"[{i+1}] Q: {r['question'][:60]}")
        print(f"    GT:   {r['ground_truth'][:50]}")
        print(f"    Pred: {r['prediction'][:50]} {'✓' if r['correct'] else '✗'}")
        print()
    print("-" * 70)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, "pathvqa_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "pathvqa_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Visualize
    try:
        vis_samples = []
        for i, (sample, result) in enumerate(zip(samples[:8], results[:8])):
            vis_samples.append({
                'image': sample['image'],
                'question': result['question'],
                'ground_truth': result['ground_truth'],
                'prediction': result['prediction'],
            })
        
        if vis_samples:
            vis_path = os.path.join(args.output_dir, "pathvqa_predictions.png")
            visualize_predictions(vis_samples, vis_path)
            logger.info(f"Visualization saved to {vis_path}")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    # Create comparison chart
    try:
        comparison_metrics = {
            'vqa_accuracy': metrics['accuracy'],
            'bleu': metrics['bleu'],
            'rougeL': metrics['rougeL'],
        }
        comparison_path = os.path.join(args.output_dir, "pathvqa_comparison.png")
        visualize_results_comparison(comparison_metrics, comparison_path)
        logger.info(f"Comparison chart saved to {comparison_path}")
    except Exception as e:
        logger.warning(f"Comparison chart failed: {e}")
    
    logger.info("=" * 60)
    logger.info("PathVQA inference complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
