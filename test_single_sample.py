"""
test_single_sample.py - Run inference on a single sample from test dataset

Usage:
    python test_single_sample.py --checkpoint checkpoints/checkpoint_epoch_5.pt --sample_idx 0
    python test_single_sample.py --checkpoint checkpoints/checkpoint_epoch_5.pt --dataset vqa_rad --sample_idx 5
"""

import os
import sys
import json
import argparse
import torch
from PIL import Image

from utils import Config, set_seed
from model import MedicalBLIP2
from dataset import download_vqa_rad, download_slake


def main():
    parser = argparse.ArgumentParser(description="Test single sample inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="vqa_rad", 
                        choices=["vqa_rad", "slake"], help="Dataset to use")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index in test set")
    parser.add_argument("--config", type=str, default=None, help="Config JSON path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("=" * 70)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "vqa_rad":
        data = download_vqa_rad(config.data_dir)
        test_samples = data.get("test", [])
    elif args.dataset == "slake":
        data = download_slake(config.data_dir)
        test_samples = data.get("test", [])
    else:
        print("Dataset not supported")
        return

    if not test_samples:
        print("No test samples found!")
        return

    if args.sample_idx >= len(test_samples):
        print(f"Sample index {args.sample_idx} out of range. Dataset has {len(test_samples)} samples.")
        return

    # Get sample
    sample = test_samples[args.sample_idx]
    
    # Load model
    print("Loading model...")
    model = MedicalBLIP2(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    tokenizer = model.tokenizer
    image_processor = model.image_processor

    # Load image
    image_path = sample.get("image_path", "")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Extract info
    question = sample.get("question", "")
    ground_truth = sample.get("answer", sample.get("target_text", ""))
    task = sample.get("task", "vqa")

    # Prepare inputs
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    input_text = f"Task: VQA Question: {question}" if task == "vqa" else "Task: Report Generate a detailed medical report."
    
    enc = tokenizer(
        input_text,
        padding=True,
        truncation=True,
        max_length=config.max_input_length,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        gen_ids, confidence = model.generate(
            pixel_values,
            input_ids,
            attention_mask,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            return_confidence=True,
        )
    
    prediction = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    confidence_score = round(float(confidence[0]), 4)

    # Print results
    print("\n" + "=" * 70)
    print(f"SAMPLE #{args.sample_idx} - {args.dataset.upper()}")
    print("=" * 70)
    print(f"\n📷 Image: {os.path.basename(image_path)}")
    print(f"\n❓ Question:")
    print(f"   {question}")
    print(f"\n🎯 Ground Truth:")
    print(f"   {ground_truth}")
    print(f"\n🤖 Prediction:")
    print(f"   {prediction}")
    print(f"\n📊 Confidence: {confidence_score}")
    
    # Check if correct
    pred_clean = prediction.strip().lower()
    gt_clean = ground_truth.strip().lower()
    
    if pred_clean == gt_clean:
        print(f"\n✅ CORRECT!")
    else:
        print(f"\n❌ INCORRECT")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
