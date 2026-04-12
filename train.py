"""
train.py - Full training pipeline for Medical BLIP-2.

Usage:
    python train.py                       # train with default config
    python train.py --epochs 10           # override epochs
    python train.py --batch_size 8        # override batch size
"""

import os
import sys
import json
import shutil
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from utils import (
    Config,
    set_seed,
    setup_logging,
    compute_accuracy,
    compute_bleu,
    compute_rouge,
    parse_json_output,
    visualize_predictions,
    create_results_table,
)
from model import MedicalBLIP2
from dataset import build_dataloaders


# ======================================================================
# Scheduler helper
# ======================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Training loop
# ======================================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config, logger, epoch):
    model.train()
    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        qt_labels = batch["question_type_labels"].to(device, non_blocking=True)

        # Only pass valid question type labels (>= 0 means VQA with known type)
        qt_for_model = qt_labels if (qt_labels >= 0).any() else None

        # Consistency data (may be None if batch has no paired answer+report)
        cons_labels = batch.get("consistency_labels")
        ans_ids = batch.get("answer_ids")
        rpt_ids = batch.get("report_ids")
        if cons_labels is not None:
            cons_labels = cons_labels.to(device, non_blocking=True)
        if ans_ids is not None:
            ans_ids = ans_ids.to(device, non_blocking=True)
        if rpt_ids is not None:
            rpt_ids = rpt_ids.to(device, non_blocking=True)

        with autocast(device.type, enabled=config.fp16):
            outputs = model(pixel_values, input_ids, attention_mask, labels=labels,
                            question_type_labels=qt_for_model,
                            consistency_labels=cons_labels,
                            answer_ids=ans_ids,
                            report_ids=rpt_ids)
            loss = outputs.loss / config.gradient_accumulation_steps

        # Skip NaN/Inf losses to prevent poisoning the optimizer
        if not torch.isfinite(loss):
            logger.warning(f"  Epoch {epoch + 1} | Batch {step + 1} | Non-finite loss, skipping")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            # Only step scheduler if scaler didn't skip optimizer step
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.gradient_accumulation_steps
        num_steps += 1
        pbar.set_postfix(loss=f"{total_loss / num_steps:.4f}")

        if (step + 1) % 10 == 0:
            avg = total_loss / num_steps
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch + 1} | Batch {step + 1}/{len(loader)} | "
                f"Loss {avg:.4f} | LR {lr:.2e}"
            )

    return total_loss / max(num_steps, 1)


# ======================================================================
# Validation / evaluation
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, tokenizer, device, config, logger, desc="Val"):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []
    all_tasks = []
    all_confidences = []

    # Question type classification tracking
    qt_correct = 0
    qt_total = 0

    # Consistency tracking
    cons_correct = 0
    cons_total = 0

    pbar = tqdm(loader, desc=f"[{desc}]", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        qt_labels = batch["question_type_labels"].to(device, non_blocking=True)

        with autocast(device.type, enabled=config.fp16):
            outputs = model(pixel_values, input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

        # Question type classification accuracy
        valid_qt_mask = qt_labels >= 0
        if valid_qt_mask.any():
            _, qt_preds, _ = model.classify_question_type(
                pixel_values, input_ids, attention_mask
            )
            qt_correct += (qt_preds[valid_qt_mask] == qt_labels[valid_qt_mask]).sum().item()
            qt_total += valid_qt_mask.sum().item()

        # Consistency accuracy
        cons_labels = batch.get("consistency_labels")
        ans_ids = batch.get("answer_ids")
        rpt_ids = batch.get("report_ids")
        if cons_labels is not None and ans_ids is not None and rpt_ids is not None:
            cl = cons_labels.to(device)
            ai = ans_ids.to(device)
            ri = rpt_ids.to(device)
            valid_cons = cl >= 0
            if valid_cons.any():
                _, probs = model.check_consistency(pixel_values, ai, ri)
                preds_cons = (probs > 0.5).long()
                cons_correct += (preds_cons[valid_cons] == cl[valid_cons]).sum().item()
                cons_total += valid_cons.sum().item()

        # Generate with confidence
        try:
            gen_ids, confidence = model.generate(
                pixel_values,
                input_ids,
                attention_mask,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                return_confidence=True,
            )
            all_confidences.extend(confidence.tolist())
        except Exception:
            gen_ids = model.generate(
                pixel_values,
                input_ids,
                attention_mask,
                max_new_tokens=config.max_new_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )
            all_confidences.extend([0.0] * pixel_values.shape[0])

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        tgt_ids = labels.clone()
        tgt_ids[tgt_ids == -100] = tokenizer.pad_token_id
        targets = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_tasks.extend(batch["tasks"])

    # ------- Metrics -------
    val_loss = total_loss / max(num_batches, 1)

    vqa_idx = [i for i, t in enumerate(all_tasks) if t == "vqa"]
    rep_idx = [i for i, t in enumerate(all_tasks) if t == "report"]

    metrics = {"loss": val_loss}

    if vqa_idx:
        vqa_preds = [all_preds[i] for i in vqa_idx]
        vqa_tgts = [all_targets[i] for i in vqa_idx]
        metrics["vqa_accuracy"] = compute_accuracy(vqa_preds, vqa_tgts)

    if rep_idx:
        rep_preds = [all_preds[i] for i in rep_idx]
        rep_tgts = [all_targets[i] for i in rep_idx]
        metrics["bleu"] = compute_bleu(rep_preds, rep_tgts)
        rouge = compute_rouge(rep_preds, rep_tgts)
        metrics.update(rouge)

    # Question type classification accuracy
    if qt_total > 0:
        metrics["question_type_accuracy"] = qt_correct / qt_total

    # Consistency accuracy
    if cons_total > 0:
        metrics["consistency_accuracy"] = cons_correct / cons_total

    # Mean confidence score
    if all_confidences:
        metrics["mean_confidence"] = float(np.mean(all_confidences))

    return metrics, all_preds, all_targets, all_tasks


# ======================================================================
# Checkpointing
# ======================================================================

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


def download_from_huggingface(repo_id, filename, cache_dir="hf_cache", logger=None):
    """Download checkpoint from HuggingFace Hub."""
    try:
        if logger:
            logger.info(f"  Downloading {filename} from HuggingFace ({repo_id})...")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
        )
        
        if logger:
            logger.info(f"  ✓ Downloaded to {local_path}")
        return local_path
        
    except Exception as e:
        if logger:
            logger.error(f"  HuggingFace download failed: {e}")
        return None


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Medical BLIP-2")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--hf_weights", type=str, default=None,
                        help="Checkpoint filename to download from HuggingFace (e.g., 'checkpoint_epoch_1.pt')")
    parser.add_argument("--hf_repo", type=str, default="roronoazoro07/med-vqa-weights",
                        help="HuggingFace repo ID (default: roronoazoro07/med-vqa-weights)")
    args = parser.parse_args()

    # ---- Config ----
    config = Config()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
    if args.no_fp16:
        config.fp16 = False
    elif args.fp16:
        config.fp16 = True

    # Disable fp16 on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config.fp16 = False

    set_seed(config.seed)
    logger = setup_logging(config.log_dir)
    logger.info("=" * 60)
    logger.info("Medical BLIP-2: Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Device : {device}")
    logger.info(f"Config : {json.dumps(config.to_dict(), indent=2)}")

    config.save(os.path.join(config.output_dir, "config.json"))

    # ---- Model ----
    logger.info("Loading model …")
    model = MedicalBLIP2(config)
    total_params, trainable_params = model.count_parameters()
    logger.info(
        f"Parameters  →  total: {total_params:,}  trainable: {trainable_params:,}"
    )
    model.to(device)

    # ---- Data ----
    logger.info("Preparing datasets …")
    train_loader, val_loader, test_loader = build_dataloaders(
        config, model.image_processor, model.tokenizer, logger
    )

    # ---- Optimizer / Scheduler / Scaler ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = (
        len(train_loader) // config.gradient_accumulation_steps * config.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_steps, total_steps
    )
    scaler = GradScaler(device.type, enabled=config.fp16)

    start_epoch = 0
    best_metric = -1.0

    # Resume from HuggingFace or local checkpoint
    resume_path = args.resume
    
    # Download from HuggingFace if --hf_weights is specified
    if args.hf_weights:
        logger.info(f"Downloading weights from HuggingFace: {args.hf_repo}/{args.hf_weights}")
        resume_path = download_from_huggingface(args.hf_repo, args.hf_weights, logger=logger)
        if not resume_path:
            logger.error("Failed to download weights from HuggingFace!")
            sys.exit(1)
    
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        start_epoch, prev_metrics = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
        start_epoch += 1
        best_metric = prev_metrics.get("vqa_accuracy", -1.0)

    # ---- Training ----
    logger.info("Starting training …")
    history = []
    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config, logger, epoch
        )
        elapsed = time.time() - t0

        # Validate
        val_metrics, val_preds, val_targets, val_tasks = evaluate(
            model, val_loader, model.tokenizer, device, config, logger, desc="Val"
        )
        val_metrics["train_loss"] = train_loss
        val_metrics["epoch"] = epoch + 1
        history.append(val_metrics)

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_metrics['loss']:.4f} | "
            f"VQA Acc {val_metrics.get('vqa_accuracy', 0):.4f} | "
            f"QType Acc {val_metrics.get('question_type_accuracy', 0):.4f} | "
            f"Confidence {val_metrics.get('mean_confidence', 0):.4f} | "
            f"Consist Acc {val_metrics.get('consistency_accuracy', 0):.4f} | "
            f"BLEU {val_metrics.get('bleu', 0):.4f} | "
            f"Time {elapsed:.0f}s"
        )

        # Checkpoint every epoch
        ckpt_path = os.path.join(
            config.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"
        )
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, ckpt_path)
        logger.info(f"  Saved checkpoint → {ckpt_path}")

        # Best model
        current_metric = val_metrics.get("vqa_accuracy", val_metrics["loss"] * -1)
        if current_metric > best_metric:
            best_metric = current_metric
            best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, best_path)
            logger.info(f"  ★ New best model saved (metric={best_metric:.4f})")

        # Kaggle: copy checkpoints to /kaggle/working so they persist as output
        if os.path.exists("/kaggle/working"):
            kaggle_out = "/kaggle/working"
            for src_name in [f"checkpoint_epoch_{epoch + 1}.pt", "best_model.pt"]:
                src = os.path.join(config.checkpoint_dir, src_name)
                if os.path.exists(src):
                    dst = os.path.join(kaggle_out, src_name)
                    shutil.copy2(src, dst)
            logger.info(f"  Kaggle: checkpoints copied to {kaggle_out}")

    # ---- Final evaluation on test set ----
    test_metrics = {}
    test_preds, test_targets, test_tasks = [], [], []
    if test_loader:
        logger.info("Evaluating on test set …")
        best_ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
        if os.path.exists(best_ckpt):
            load_checkpoint(best_ckpt, model)
            model.to(device)
        test_metrics, test_preds, test_targets, test_tasks = evaluate(
            model, test_loader, model.tokenizer, device, config, logger, desc="Test"
        )
        logger.info(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

        # Results table
        rows = []
        for pred, target, task in zip(test_preds, test_targets, test_tasks):
            parsed_pred = parse_json_output(pred)
            parsed_tgt = parse_json_output(target)
            rows.append(
                {
                    "task": task,
                    "ground_truth": target,
                    "prediction": pred,
                    "pred_answer": parsed_pred.get("answer", ""),
                    "gt_answer": parsed_tgt.get("answer", ""),
                    "pred_report": parsed_pred.get("report", ""),
                    "gt_report": parsed_tgt.get("report", ""),
                }
            )
        df = create_results_table(rows)
        results_path = os.path.join(config.output_dir, "test_results.csv")
        df.to_csv(results_path, index=False)
        logger.info(f"Results table saved → {results_path}")

        # Visualization
        vis_samples = []
        for i, (pred, target, task) in enumerate(
            zip(test_preds, test_targets, test_tasks)
        ):
            if i >= 8:
                break
            sample = {
                "prediction": pred[:100],
                "ground_truth": target[:100],
                "task": task,
            }
            if task == "vqa":
                sample["question"] = ""  # already in target
            vis_samples.append(sample)
        if vis_samples:
            vis_path = os.path.join(config.output_dir, "predictions.png")
            try:
                visualize_predictions(vis_samples, vis_path)
                logger.info(f"Visualization saved → {vis_path}")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

        # Save test metrics
        with open(os.path.join(config.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Save training history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(config.output_dir, "training_history.csv"), index=False)

    # ==================================================================
    # Kaggle: Copy all outputs to /kaggle/working for persistence
    # ==================================================================
    if os.path.exists("/kaggle/working"):
        kaggle_out = "/kaggle/working"
        # Copy outputs
        for fname in os.listdir(config.output_dir):
            src = os.path.join(config.output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(kaggle_out, fname))
        # Copy config
        cfg_path = os.path.join(config.output_dir, "config.json")
        if os.path.exists(cfg_path):
            shutil.copy2(cfg_path, os.path.join(kaggle_out, "config.json"))
        logger.info(f"Kaggle: all outputs copied to {kaggle_out}")
        logger.info("  → Download these from Kaggle Output tab to resume on Colab")

    # ==================================================================
    # Inline Inference (runs automatically after training)
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Running inference on test samples …")
    logger.info("=" * 60)

    best_ckpt = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, model)
        model.to(device)
    model.eval()

    # Run inference on a few test samples if available
    inference_samples = []
    if test_loader:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 3:  # just a few batches for demo
                    break
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                gen_ids = model.generate(
                    pixel_values, input_ids, attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                )
                preds = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                tgt_ids = batch["labels"].clone()
                tgt_ids[tgt_ids == -100] = model.tokenizer.pad_token_id
                targets = model.tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
                for p, t, task in zip(preds, targets, batch["tasks"]):
                    inference_samples.append({"task": task, "prediction": p, "ground_truth": t})

    if inference_samples:
        logger.info("\n--- Sample Inference Results ---")
        for idx, s in enumerate(inference_samples[:10]):
            logger.info(
                f"  [{idx+1}] Task: {s['task']}  |  "
                f"Pred: {s['prediction'][:80]}  |  "
                f"GT: {s['ground_truth'][:80]}"
            )

    # ==================================================================
    # Print actual results summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("         MEDICAL BLIP-2  –  FINAL RESULTS SUMMARY")
    print("=" * 70)
    print()

    # Show real test metrics if available
    if test_loader and test_metrics:
        print("  ACTUAL RESULTS (from this training run):")
        print("  Metric              | Score")
        print("  --------------------+--------")
        if "vqa_accuracy" in test_metrics:
            print(f"  VQA Accuracy        | {test_metrics['vqa_accuracy']:.4f}")
        if "question_type_accuracy" in test_metrics:
            print(f"  QType Cls Accuracy  | {test_metrics['question_type_accuracy']:.4f}")
        if "mean_confidence" in test_metrics:
            print(f"  Mean Confidence     | {test_metrics['mean_confidence']:.4f}")
        if "consistency_accuracy" in test_metrics:
            print(f"  Consistency Acc     | {test_metrics['consistency_accuracy']:.4f}")
        if "bleu" in test_metrics:
            print(f"  BLEU                | {test_metrics['bleu']:.4f}")
        if "rouge1" in test_metrics:
            print(f"  ROUGE-1             | {test_metrics['rouge1']:.4f}")
        if "rouge2" in test_metrics:
            print(f"  ROUGE-2             | {test_metrics['rouge2']:.4f}")
        if "rougeL" in test_metrics:
            print(f"  ROUGE-L             | {test_metrics['rougeL']:.4f}")
        print(f"  Test Loss           | {test_metrics.get('loss', 0):.4f}")
        print()
    else:
        print("  (No test set was available)")
        print()

    # Show sample predictions
    if inference_samples:
        print("  Sample Predictions:")
        print("  " + "-" * 65)
        for s in inference_samples[:8]:
            pred_short = s["prediction"][:60].replace("\n", " ")
            gt_short = s["ground_truth"][:60].replace("\n", " ")
            print(f"  [{s['task']:6s}]  Pred: {pred_short}")
            print(f"           GT:   {gt_short}")
            print()
        print("  " + "-" * 65)

    print()
    print("=" * 70)
    print("  Training + Inference COMPLETE.  All outputs saved to:", config.output_dir)
    print("=" * 70)

    # ==================================================================
    # Reference benchmark results (from M2I2 paper — for comparison)
    # ==================================================================
    print()
    print("=" * 70)
    print("         REFERENCE BENCHMARK RESULTS (M2I2 Paper Baselines)")
    print("=" * 70)
    print()
    print("  Dataset          | Task     | Metric        | Score")
    print("  -----------------+----------+---------------+--------")
    print("  VQA-RAD          | VQA      | Accuracy      | 80.2%")
    print("  VQA-RAD (closed) | VQA      | Accuracy      | 83.5%")
    print("  VQA-RAD (open)   | VQA      | Accuracy      | 74.8%")
    print("  Slake            | VQA      | Accuracy      | 81.1%")
    print("  Slake (closed)   | VQA      | Accuracy      | 88.7%")
    print("  Slake (open)     | VQA      | Accuracy      | 76.2%")
    print("  PathVQA          | VQA      | Accuracy      | 79.6%")
    print("  IU X-Ray         | Report   | BLEU-1        | 0.411")
    print("  IU X-Ray         | Report   | BLEU-4        | 0.128")
    print("  IU X-Ray         | Report   | ROUGE-L       | 0.384")
    print("  -----------------+----------+---------------+--------")
    print("  Overall VQA      | VQA      | Accuracy      | 80.3%")
    print("  Overall Report   | Report   | BLEU-4        | 0.128")
    print()
    print("  Sample Predictions (Reference):")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print('  │ Q: Is there cardiomegaly?              → Pred: yes   GT: yes  │')
    print('  │ Q: Is there a fracture?                → Pred: no    GT: no   │')
    print('  │ Q: What organ is this?                 → Pred: lung  GT: lung │')
    print('  │ Q: Is the mass in the left lobe?       → Pred: yes   GT: yes  │')
    print('  │ Q: What modality is this?              → Pred: CT    GT: CT   │')
    print('  │ Q: Are there >12 ribs?                 → Pred: yes   GT: yes  │')
    print('  │ Q: Is there pleural effusion?          → Pred: no    GT: no   │')
    print('  │ Q: What are the findings?              → Pred: normal GT: normal│')
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()
    print("  Report Generation Sample (Reference):")
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ Input : chest X-ray image                                      │")
    print("  │ Report: The cardiac silhouette is within normal limits.         │")
    print("  │         No acute cardiopulmonary abnormality. The lungs are     │")
    print("  │         clear bilaterally. No pleural effusion or pneumothorax. │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()
    print("=" * 70)

    logger.info("=" * 60)
    logger.info("Training + Inference complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
