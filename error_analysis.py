"""
error_analysis.py - Error Analysis Dashboard for Medical BLIP-2 VQA.

Runs the model on the validation/test set and generates a comprehensive
analysis of prediction errors, including:

  1. Overall accuracy breakdown by dataset and question type
  2. Confusion matrix for closed-ended (yes/no) answers
  3. Most common error patterns
  4. Confidence calibration analysis
  5. Per-category accuracy (answer groups)
  6. Sample correct/incorrect predictions with images

Usage:
    python error_analysis.py --checkpoint checkpoints/best_model.pt

    # Use test set instead of validation
    python error_analysis.py --checkpoint checkpoints/best_model.pt --split test

    # Limit number of samples (for quick testing)
    python error_analysis.py --checkpoint checkpoints/best_model.pt --max_samples 100
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MedicalBLIP2
from utils import Config, set_seed
from dataset import (
    download_vqa_rad, download_slake, MedicalVLDataset, create_collate_fn,
)


# ======================================================================
# Model loading
# ======================================================================

def load_model(checkpoint_path: str, device: Optional[str] = None):
    """Load trained model from checkpoint."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    config = Config()
    model = MedicalBLIP2(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev)
    model.eval()
    return model, config, dev


# ======================================================================
# Data loading
# ======================================================================

def load_evaluation_data(config: Config, split: str = "val"):
    """Load VQA evaluation data.

    Args:
        config: Model config with data_dir.
        split: "val" or "test". If "val", uses VQA-RAD test split
               (which is used as validation during training).

    Returns:
        List of sample dicts with image_path, question, answer, etc.
    """
    all_samples = []

    try:
        vqa_rad = download_vqa_rad(config.data_dir)
        if split == "val":
            # VQA-RAD test split = our validation set
            all_samples.extend(vqa_rad.get("test", []))
        else:
            all_samples.extend(vqa_rad.get("test", []))
    except Exception as e:
        print(f"  Warning: VQA-RAD load failed: {e}")

    if split == "test":
        try:
            slake = download_slake(config.data_dir)
            all_samples.extend(slake.get("test", []))
        except Exception as e:
            print(f"  Warning: Slake load failed: {e}")

    # Filter to VQA samples only (skip report generation)
    vqa_samples = [s for s in all_samples if s.get("task") == "vqa"]
    print(f"Loaded {len(vqa_samples)} VQA samples for evaluation")
    return vqa_samples


# ======================================================================
# Evaluation
# ======================================================================

def run_evaluation(
    model: MedicalBLIP2,
    config: Config,
    device: torch.device,
    samples: List[dict],
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """Run model inference on all samples and collect detailed results.

    Returns:
        List of dicts with: image_path, question, ground_truth, prediction,
        confidence, question_type_pred, question_type_label, correct,
        dataset name.
    """
    if max_samples and max_samples < len(samples):
        np.random.seed(42)
        indices = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    results = []
    from PIL import Image

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        image_path = sample.get("image_path", "")
        question = sample.get("question", "")
        gt_answer = sample.get("answer", "")
        qt_label = sample.get("question_type", -1)
        dataset_name = sample.get("dataset", "unknown")

        if not os.path.exists(image_path):
            continue

        try:
            # Prepare inputs
            image = Image.open(image_path).convert("RGB")
            pixel_values = model.image_processor(
                images=image, return_tensors="pt"
            )["pixel_values"].to(device)

            input_text = f"Task: VQA Question: {question}"
            enc = model.tokenizer(
                input_text, padding=True, truncation=True,
                max_length=config.max_input_length, return_tensors="pt",
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            with torch.no_grad():
                # Generate answer with confidence
                gen_ids, confidence = model.generate(
                    pixel_values, input_ids, attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    num_beams=config.num_beams,
                    return_confidence=True,
                )
                prediction = model.tokenizer.decode(
                    gen_ids[0], skip_special_tokens=True
                ).strip()
                conf = float(confidence[0])

                # Classify question type
                _, qt_pred, _ = model.classify_question_type(
                    pixel_values, input_ids, attention_mask
                )
                qt_pred_val = qt_pred[0].item()

        except Exception as e:
            prediction = ""
            conf = 0.0
            qt_pred_val = -1
            print(f"  Error on sample {i}: {e}")

        correct = prediction.strip().lower() == gt_answer.strip().lower()

        results.append({
            "image_path": image_path,
            "question": question,
            "ground_truth": gt_answer,
            "prediction": prediction,
            "confidence": conf,
            "correct": correct,
            "question_type_label": qt_label,  # 0=closed, 1=open
            "question_type_pred": qt_pred_val,
            "dataset": dataset_name,
        })

    return results


# ======================================================================
# Analysis functions
# ======================================================================

def overall_accuracy(results: List[Dict]) -> Dict:
    """Compute overall and per-dataset accuracy."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    by_dataset = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        ds = r["dataset"]
        by_dataset[ds]["total"] += 1
        if r["correct"]:
            by_dataset[ds]["correct"] += 1

    dataset_acc = {}
    for ds, counts in by_dataset.items():
        dataset_acc[ds] = {
            "accuracy": counts["correct"] / max(counts["total"], 1),
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return {
        "overall_accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "by_dataset": dataset_acc,
    }


def accuracy_by_question_type(results: List[Dict]) -> Dict:
    """Break down accuracy by question type (closed vs open)."""
    types = {0: "closed", 1: "open"}
    breakdown = {}

    for qt_val, qt_name in types.items():
        subset = [r for r in results if r["question_type_label"] == qt_val]
        if subset:
            correct = sum(1 for r in subset if r["correct"])
            breakdown[qt_name] = {
                "accuracy": correct / len(subset),
                "correct": correct,
                "total": len(subset),
                "avg_confidence": np.mean([r["confidence"] for r in subset]),
            }

    return breakdown


def closed_answer_confusion(results: List[Dict]) -> Dict:
    """Build confusion matrix for closed-ended (yes/no) questions."""
    closed = [r for r in results if r["question_type_label"] == 0]
    if not closed:
        return {}

    # Normalize answers
    def normalize(ans):
        a = ans.strip().lower()
        if a in ("yes", "true", "1"):
            return "yes"
        elif a in ("no", "false", "0"):
            return "no"
        return a

    gt_vals = [normalize(r["ground_truth"]) for r in closed]
    pred_vals = [normalize(r["prediction"]) for r in closed]

    # Get unique labels
    labels = sorted(set(gt_vals + pred_vals))

    # Build confusion matrix
    matrix = {}
    for gt_label in labels:
        matrix[gt_label] = {}
        for pred_label in labels:
            count = sum(
                1 for g, p in zip(gt_vals, pred_vals)
                if g == gt_label and p == pred_label
            )
            matrix[gt_label][pred_label] = count

    return {"labels": labels, "matrix": matrix, "total": len(closed)}


def common_error_patterns(results: List[Dict], top_k: int = 15) -> List[Dict]:
    """Find the most common (ground_truth → prediction) error pairs."""
    errors = [r for r in results if not r["correct"]]
    pair_counts = Counter()
    for r in errors:
        gt = r["ground_truth"].strip().lower()
        pred = r["prediction"].strip().lower()
        pair_counts[(gt, pred)] += 1

    patterns = []
    for (gt, pred), count in pair_counts.most_common(top_k):
        patterns.append({
            "ground_truth": gt,
            "prediction": pred,
            "count": count,
        })
    return patterns


def accuracy_by_answer_category(results: List[Dict]) -> Dict:
    """Group answers into categories and report per-category accuracy."""
    categories = {
        "yes/no": {"yes", "no"},
        "anatomy": {
            "lung", "heart", "liver", "brain", "chest", "abdomen",
            "kidney", "bone", "spine", "skull", "pelvis", "rib",
            "left", "right", "bilateral",
        },
        "modality": {
            "ct", "mri", "x-ray", "xray", "ultrasound", "pet",
            "mammography", "fluoroscopy",
        },
        "condition": {
            "fracture", "pneumonia", "effusion", "cardiomegaly",
            "edema", "atelectasis", "consolidation", "mass", "nodule",
            "tumor", "lesion", "normal", "abnormal",
        },
    }

    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        gt_lower = r["ground_truth"].strip().lower()
        assigned = False
        for cat_name, keywords in categories.items():
            if gt_lower in keywords or any(kw in gt_lower for kw in keywords):
                cat_results[cat_name]["total"] += 1
                if r["correct"]:
                    cat_results[cat_name]["correct"] += 1
                assigned = True
                break
        if not assigned:
            cat_results["other"]["total"] += 1
            if r["correct"]:
                cat_results["other"]["correct"] += 1

    output = {}
    for cat, counts in sorted(cat_results.items()):
        output[cat] = {
            "accuracy": counts["correct"] / max(counts["total"], 1),
            "correct": counts["correct"],
            "total": counts["total"],
        }
    return output


def confidence_calibration(results: List[Dict], n_bins: int = 10) -> Dict:
    """Bin predictions by confidence and compute accuracy per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [
            r for r in results
            if lo <= r["confidence"] < hi or (i == n_bins - 1 and r["confidence"] == hi)
        ]
        if in_bin:
            acc = sum(1 for r in in_bin if r["correct"]) / len(in_bin)
            avg_conf = np.mean([r["confidence"] for r in in_bin])
            bin_data.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "accuracy": acc,
                "avg_confidence": float(avg_conf),
                "count": len(in_bin),
            })

    return {"bins": bin_data, "n_bins": n_bins}


# ======================================================================
# Visualization / Dashboard generation
# ======================================================================

def plot_accuracy_by_type(breakdown: Dict, save_path: str):
    """Bar chart: accuracy by question type."""
    types = list(breakdown.keys())
    accs = [breakdown[t]["accuracy"] * 100 for t in types]
    counts = [breakdown[t]["total"] for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, accs, color=["#2196F3", "#FF9800"], edgecolor="black")

    for bar, acc, n in zip(bars, accs, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%\n(n={n})", ha="center", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("VQA Accuracy by Question Type", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(confusion: Dict, save_path: str):
    """Heatmap confusion matrix for closed-ended answers."""
    labels = confusion["labels"]
    matrix = confusion["matrix"]

    n = len(labels)
    data = np.zeros((n, n))
    for i, gt in enumerate(labels):
        for j, pred in enumerate(labels):
            data[i, j] = matrix.get(gt, {}).get(pred, 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Ground Truth", fontsize=13)
    ax.set_title("Confusion Matrix (Closed-Ended Questions)", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = int(data[i, j])
            color = "white" if data[i, j] > data.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_category_accuracy(cat_acc: Dict, save_path: str):
    """Horizontal bar chart: accuracy by answer category."""
    cats = list(cat_acc.keys())
    accs = [cat_acc[c]["accuracy"] * 100 for c in cats]
    counts = [cat_acc[c]["total"] for c in cats]

    # Sort by accuracy
    order = np.argsort(accs)
    cats = [cats[i] for i in order]
    accs = [accs[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(cats) * 0.6)))
    colors = plt.cm.RdYlGn(np.array(accs) / 100)
    bars = ax.barh(cats, accs, color=colors, edgecolor="black")

    for bar, acc, n in zip(bars, accs, counts):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1f}% (n={n})", va="center", fontsize=11,
        )

    ax.set_xlabel("Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy by Answer Category", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_calibration(cal_data: Dict, save_path: str):
    """Reliability diagram: confidence vs accuracy."""
    bins = cal_data["bins"]
    if not bins:
        return

    avg_confs = [b["avg_confidence"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])

    # Top: Calibration curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(avg_confs, accs, width=0.08, alpha=0.7, color="#2196F3",
            edgecolor="black", label="Model")
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=13)
    ax1.set_ylabel("Fraction Correct", fontsize=13)
    ax1.set_title("Confidence Calibration", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Bottom: Histogram of predictions per bin
    ax2.bar(avg_confs, counts, width=0.08, color="#FF9800", edgecolor="black")
    ax2.set_xlabel("Confidence", fontsize=13)
    ax2.set_ylabel("Count", fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_patterns(patterns: List[Dict], save_path: str):
    """Bar chart of most common error pairs."""
    if not patterns:
        return

    labels = [f"{p['ground_truth']} → {p['prediction']}" for p in patterns[:10]]
    counts = [p["count"] for p in patterns[:10]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    bars = ax.barh(labels, counts, color="#F44336", edgecolor="black", alpha=0.8)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Count", fontsize=13)
    ax.set_title("Most Common Errors (GT → Prediction)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_html_report(
    acc: Dict, qt_breakdown: Dict, confusion: Dict,
    cat_acc: Dict, cal_data: Dict, patterns: List[Dict],
    results: List[Dict], save_dir: str,
):
    """Generate an HTML dashboard summarizing all analysis."""
    # Collect sample predictions (5 correct, 5 incorrect)
    correct_samples = [r for r in results if r["correct"]][:5]
    wrong_samples = [r for r in results if not r["correct"]][:5]

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Medical BLIP-2 VQA - Error Analysis Dashboard</title>
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }
    h2 { color: #333; margin-top: 30px; }
    .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .metric { display: inline-block; text-align: center; padding: 15px 30px;
              margin: 5px; background: #E3F2FD; border-radius: 8px; }
    .metric .value { font-size: 32px; font-weight: bold; color: #1565C0; }
    .metric .label { font-size: 14px; color: #666; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #1565C0; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .correct { color: #4CAF50; font-weight: bold; }
    .wrong { color: #F44336; font-weight: bold; }
    img.chart { max-width: 100%; border-radius: 4px; margin: 10px 0; }
    .sample { padding: 10px; margin: 5px 0; border-left: 4px solid; }
    .sample.correct-sample { border-color: #4CAF50; background: #E8F5E9; }
    .sample.wrong-sample { border-color: #F44336; background: #FFEBEE; }
</style>
</head>
<body>
<div class="container">
<h1>Medical BLIP-2 VQA — Error Analysis Dashboard</h1>
"""

    # Overall metrics
    html += '<div class="card">\n<h2>Overall Performance</h2>\n'
    html += '<div class="metric"><div class="value">'
    html += f'{acc["overall_accuracy"]:.1%}</div><div class="label">Overall Accuracy</div></div>\n'
    html += '<div class="metric"><div class="value">'
    html += f'{acc["correct"]}/{acc["total"]}</div><div class="label">Correct / Total</div></div>\n'

    for ds, ds_acc in acc.get("by_dataset", {}).items():
        html += f'<div class="metric"><div class="value">{ds_acc["accuracy"]:.1%}</div>'
        html += f'<div class="label">{ds} ({ds_acc["correct"]}/{ds_acc["total"]})</div></div>\n'
    html += '</div>\n'

    # Question type breakdown
    if qt_breakdown:
        html += '<div class="card">\n<h2>Accuracy by Question Type</h2>\n'
        html += '<img class="chart" src="accuracy_by_type.png">\n'
        html += '<table><tr><th>Type</th><th>Accuracy</th><th>Correct</th><th>Total</th>'
        html += '<th>Avg Confidence</th></tr>\n'
        for qt_name, data in qt_breakdown.items():
            html += f'<tr><td>{qt_name}</td><td>{data["accuracy"]:.1%}</td>'
            html += f'<td>{data["correct"]}</td><td>{data["total"]}</td>'
            html += f'<td>{data.get("avg_confidence", 0):.3f}</td></tr>\n'
        html += '</table>\n</div>\n'

    # Confusion matrix
    if confusion:
        html += '<div class="card">\n<h2>Confusion Matrix (Closed-Ended)</h2>\n'
        html += '<img class="chart" src="confusion_matrix.png">\n'
        html += '</div>\n'

    # Category accuracy
    if cat_acc:
        html += '<div class="card">\n<h2>Accuracy by Answer Category</h2>\n'
        html += '<img class="chart" src="category_accuracy.png">\n'
        html += '</div>\n'

    # Confidence calibration
    html += '<div class="card">\n<h2>Confidence Calibration</h2>\n'
    html += '<img class="chart" src="confidence_calibration.png">\n'
    html += '</div>\n'

    # Error patterns
    if patterns:
        html += '<div class="card">\n<h2>Most Common Error Patterns</h2>\n'
        html += '<img class="chart" src="error_patterns.png">\n'
        html += '<table><tr><th>#</th><th>Ground Truth</th><th>Prediction</th><th>Count</th></tr>\n'
        for i, p in enumerate(patterns[:15], 1):
            html += f'<tr><td>{i}</td><td>{p["ground_truth"]}</td>'
            html += f'<td>{p["prediction"]}</td><td>{p["count"]}</td></tr>\n'
        html += '</table>\n</div>\n'

    # Sample predictions
    html += '<div class="card">\n<h2>Sample Predictions</h2>\n'
    html += '<h3 class="correct">Correct Predictions</h3>\n'
    for r in correct_samples:
        html += '<div class="sample correct-sample">\n'
        html += f'<strong>Q:</strong> {r["question"]}<br>\n'
        html += f'<strong>Answer:</strong> {r["prediction"]} '
        html += f'<strong>(Confidence: {r["confidence"]:.2%})</strong><br>\n'
        html += f'<strong>Ground Truth:</strong> {r["ground_truth"]}\n'
        html += '</div>\n'

    html += '<h3 class="wrong">Incorrect Predictions</h3>\n'
    for r in wrong_samples:
        html += '<div class="sample wrong-sample">\n'
        html += f'<strong>Q:</strong> {r["question"]}<br>\n'
        html += f'<strong>Predicted:</strong> {r["prediction"]} '
        html += f'<strong>(Confidence: {r["confidence"]:.2%})</strong><br>\n'
        html += f'<strong>Ground Truth:</strong> {r["ground_truth"]}\n'
        html += '</div>\n'
    html += '</div>\n'

    html += '</div>\n</body>\n</html>'

    html_path = os.path.join(save_dir, "dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML dashboard saved to {html_path}")


# ======================================================================
# Main pipeline
# ======================================================================

def run_analysis(
    checkpoint_path: str,
    save_dir: str = "outputs/error_analysis",
    split: str = "val",
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
):
    """Full error analysis pipeline."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load model
    print("Loading model...")
    model, config, dev = load_model(checkpoint_path, device)
    print(f"Model loaded on {dev}")

    # 2. Load data
    print(f"\nLoading {split} data...")
    samples = load_evaluation_data(config, split=split)
    if not samples:
        print("No evaluation samples found!")
        return

    # 3. Run evaluation
    print(f"\nRunning evaluation on {len(samples)} samples...")
    results = run_evaluation(model, config, dev, samples, max_samples=max_samples)

    # Save raw results
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Raw results saved to {results_path}")

    # 4. Compute all analyses
    print("\nComputing analysis...")

    acc = overall_accuracy(results)
    print(f"  Overall accuracy: {acc['overall_accuracy']:.1%} ({acc['correct']}/{acc['total']})")

    qt_breakdown = accuracy_by_question_type(results)
    for qt_name, data in qt_breakdown.items():
        print(f"  {qt_name}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    confusion = closed_answer_confusion(results)
    cat_acc = accuracy_by_answer_category(results)
    cal_data = confidence_calibration(results)
    patterns = common_error_patterns(results)

    # 5. Generate plots
    print("\nGenerating visualizations...")

    if qt_breakdown:
        plot_accuracy_by_type(qt_breakdown, os.path.join(save_dir, "accuracy_by_type.png"))
        print("  Saved: accuracy_by_type.png")

    if confusion:
        plot_confusion_matrix(confusion, os.path.join(save_dir, "confusion_matrix.png"))
        print("  Saved: confusion_matrix.png")

    if cat_acc:
        plot_category_accuracy(cat_acc, os.path.join(save_dir, "category_accuracy.png"))
        print("  Saved: category_accuracy.png")

    if cal_data.get("bins"):
        plot_confidence_calibration(cal_data, os.path.join(save_dir, "confidence_calibration.png"))
        print("  Saved: confidence_calibration.png")

    if patterns:
        plot_error_patterns(patterns, os.path.join(save_dir, "error_patterns.png"))
        print("  Saved: error_patterns.png")

    # 6. Generate HTML dashboard
    generate_html_report(
        acc, qt_breakdown, confusion, cat_acc, cal_data, patterns,
        results, save_dir,
    )

    # 7. Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {acc['overall_accuracy']:.1%}")
    for qt_name, data in qt_breakdown.items():
        print(f"  {qt_name}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")
    print(f"\nTotal errors: {acc['total'] - acc['correct']}")
    if patterns:
        print(f"Top error: '{patterns[0]['ground_truth']}' → '{patterns[0]['prediction']}' ({patterns[0]['count']}x)")
    print(f"\nAll outputs saved to: {save_dir}/")
    print(f"  - dashboard.html (open in browser)")
    print(f"  - evaluation_results.json")
    print(f"  - *.png charts")
    print("=" * 60)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medical BLIP-2 VQA Error Analysis Dashboard"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--save_dir", type=str, default="outputs/error_analysis",
                        help="Directory to save analysis outputs")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                        help="Dataset split to analyze")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for quick testing)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    run_analysis(
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        split=args.split,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
