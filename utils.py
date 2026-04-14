"""
utils.py - Configuration, logging, metrics, and visualization utilities.
"""

import os
import json
import random
import logging
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class Config:
    """Central configuration for the Medical VLP system."""

    # ---- Model ----
    vision_model_name: str = "openai/clip-vit-base-patch16"
    t5_model_name: str = "google/flan-t5-base"
    blip2_model_name: str = "Salesforce/blip2-opt-2.7b"  # Actually exists on HuggingFace
    qt_bert_model_name: str = "google-bert/bert-base-uncased"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    num_query_tokens: int = 32
    qformer_hidden_size: int = 768
    qformer_num_heads: int = 12
    qformer_num_layers: int = 6
    qformer_intermediate_size: int = 3072
    qformer_cross_attention_every: int = 2
    qformer_dropout: float = 0.1

    # ---- Training ----
    epochs: int = 1  # Run 1 epoch then upload to HuggingFace
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_input_length: int = 128
    max_target_length: int = 256
    gradient_accumulation_steps: int = 4
    fp16: bool = False  # Disabled for stability with random Q-Former init
    seed: int = 42
    num_workers: int = 2
    max_grad_norm: float = 1.0

    # ---- Generation ----
    max_new_tokens: int = 256
    num_beams: int = 4
    no_repeat_ngram_size: int = 3

    # ---- Paths ----
    base_dir: str = ""
    data_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""

    def __post_init__(self):
        if not self.base_dir:
            self.base_dir = (
                "/kaggle/working"
                if os.path.exists("/kaggle/working")
                else os.getcwd()
            )
        if not self.data_dir:
            self.data_dir = os.path.join(self.base_dir, "data")
        if not self.output_dir:
            self.output_dir = os.path.join(self.base_dir, "outputs")
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        if not self.log_dir:
            self.log_dir = os.path.join(self.base_dir, "logs")

        for d in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("MedVLP")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"training_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_json_output(text: str) -> Dict[str, str]:
    """Attempt to parse a structured JSON output from the model."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"answer": text, "explanation": "", "report": ""}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """Exact-match accuracy after extracting the answer field."""
    if not predictions:
        return 0.0
    correct = 0
    for pred, ref in zip(predictions, references):
        parsed = parse_json_output(pred)
        pred_ans = parsed.get("answer", pred).strip().lower()
        ref_parsed = parse_json_output(ref)
        ref_ans = ref_parsed.get("answer", ref).strip().lower()
        if pred_ans == ref_ans:
            correct += 1
    return correct / len(predictions)


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Sentence-level BLEU with smoothing."""
    try:
        import nltk

        for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
            try:
                nltk.data.find(res)
            except LookupError:
                nltk.download(res.split("/")[-1], quiet=True)

        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smooth = SmoothingFunction().method1
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tok = nltk.word_tokenize(pred.lower())
            ref_tok = nltk.word_tokenize(ref.lower())
            if not pred_tok or not ref_tok:
                scores.append(0.0)
                continue
            scores.append(sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth))
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L f-measure."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        agg: Dict[str, list] = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for k in agg:
                agg[k].append(result[k].fmeasure)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_predictions(samples: List[Dict], save_path: str):
    """Save a grid of sample images with predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(samples), 8)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        s = samples[i]
        ax = axes[i]
        if "image" in s and s["image"] is not None:
            ax.imshow(s["image"], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        title_parts = []
        if "question" in s:
            title_parts.append(f"Q: {s['question'][:60]}")
        if "ground_truth" in s:
            title_parts.append(f"GT: {s['ground_truth'][:60]}")
        if "prediction" in s:
            title_parts.append(f"Pred: {s['prediction'][:60]}")
        ax.set_title("\n".join(title_parts), fontsize=7, wrap=True)
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_attention(image, attention_weights, save_path: str, title: str = ""):
    """Overlay an attention heatmap on the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    if attention_weights is not None:
        attn = attention_weights.mean(dim=0) if attention_weights.dim() > 2 else attention_weights
        attn = attn.mean(dim=0)
        side = int(attn.shape[0] ** 0.5)
        if side * side == attn.shape[0]:
            attn = attn.reshape(side, side)
        else:
            attn = attn.unsqueeze(0)
        attn = attn.float().cpu().numpy()
        axes[1].imshow(image, cmap="gray")
        axes[1].imshow(attn, cmap="jet", alpha=0.5, interpolation="bilinear",
                       extent=[0, image.size[0] if hasattr(image, "size") else image.shape[1],
                               image.size[1] if hasattr(image, "size") else image.shape[0], 0])
        axes[1].set_title("Attention")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_results_comparison(actual_metrics: Dict, save_path: str):
    """Create bar chart comparing actual results vs reference benchmarks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Reference benchmarks (M2I2 paper baselines)
    reference = {
        "VQA Accuracy": 0.802,
        "BLEU": 0.128,
        "ROUGE-L": 0.384,
    }
    
    # Extract actual metrics
    actual = {
        "VQA Accuracy": actual_metrics.get("vqa_accuracy", 0),
        "BLEU": actual_metrics.get("bleu", 0),
        "ROUGE-L": actual_metrics.get("rougeL", 0),
    }
    
    metrics = list(reference.keys())
    ref_values = [reference[m] for m in metrics]
    act_values = [actual[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, act_values, width, label='Our Model (Actual)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, ref_values, width, label='M2I2 Benchmark (Reference)', color='#3498db')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Medical BLIP-2: Actual Results vs Reference Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, val in zip(bars1, act_values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, ref_values):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_detailed_metrics(actual_metrics: Dict, save_path: str):
    """Create detailed metrics visualization with all available metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Reference benchmarks (detailed)
    reference = {
        "VQA-RAD": 0.802,
        "VQA-RAD (closed)": 0.835,
        "VQA-RAD (open)": 0.748,
        "Slake": 0.811,
        "Slake (closed)": 0.887,
        "Slake (open)": 0.762,
        "BLEU-4": 0.128,
        "ROUGE-L": 0.384,
    }
    
    # Actual metrics
    actual = {
        "VQA Accuracy": actual_metrics.get("vqa_accuracy", 0),
        "QType Accuracy": actual_metrics.get("question_type_accuracy", 0),
        "Mean Confidence": actual_metrics.get("mean_confidence", 0),
        "Consistency Acc": actual_metrics.get("consistency_accuracy", 0),
        "BLEU": actual_metrics.get("bleu", 0),
        "ROUGE-1": actual_metrics.get("rouge1", 0),
        "ROUGE-2": actual_metrics.get("rouge2", 0),
        "ROUGE-L": actual_metrics.get("rougeL", 0),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Actual Results
    ax1 = axes[0]
    metrics = [k for k, v in actual.items() if v > 0]
    values = [actual[k] for k in metrics]
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(metrics)))
    bars = ax1.barh(metrics, values, color=colors)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel('Score', fontsize=11)
    ax1.set_title('OUR MODEL - Actual Results', fontsize=13, fontweight='bold', color='#27ae60')
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.4f}', xy=(val + 0.02, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)
    
    # Right: Reference Benchmarks
    ax2 = axes[1]
    ref_metrics = list(reference.keys())
    ref_values = list(reference.values())
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ref_metrics)))
    bars = ax2.barh(ref_metrics, ref_values, color=colors)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_title('M2I2 BENCHMARK - Reference Results', fontsize=13, fontweight='bold', color='#2980b9')
    for bar, val in zip(bars, ref_values):
        ax2.annotate(f'{val:.3f}', xy=(val + 0.02, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=10)
    
    plt.suptitle('Medical BLIP-2: Comprehensive Results Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_training_history(history_df, save_path: str):
    """Create training curves visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history_df['epoch'].values if 'epoch' in history_df.columns else range(1, len(history_df)+1)
    
    # Loss curves
    ax = axes[0, 0]
    if 'train_loss' in history_df.columns:
        ax.plot(epochs, history_df['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    if 'loss' in history_df.columns:
        ax.plot(epochs, history_df['loss'], 'r-o', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[0, 1]
    if 'vqa_accuracy' in history_df.columns:
        ax.plot(epochs, history_df['vqa_accuracy'], 'g-o', label='VQA Accuracy', linewidth=2)
    if 'question_type_accuracy' in history_df.columns:
        ax.plot(epochs, history_df['question_type_accuracy'], 'm-o', label='QType Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # BLEU/ROUGE curves
    ax = axes[1, 0]
    if 'bleu' in history_df.columns:
        ax.plot(epochs, history_df['bleu'], 'c-o', label='BLEU', linewidth=2)
    if 'rougeL' in history_df.columns:
        ax.plot(epochs, history_df['rougeL'], 'y-o', label='ROUGE-L', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Generation Metrics (BLEU/ROUGE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Confidence & Consistency
    ax = axes[1, 1]
    if 'mean_confidence' in history_df.columns:
        ax.plot(epochs, history_df['mean_confidence'], 'orange', marker='o', label='Mean Confidence', linewidth=2)
    if 'consistency_accuracy' in history_df.columns:
        ax.plot(epochs, history_df['consistency_accuracy'], 'purple', marker='o', label='Consistency Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Confidence & Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.suptitle('Medical BLIP-2: Training Progress', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_results_table(results: List[Dict]):
    """Return a pandas DataFrame summarising predictions."""
    import pandas as pd

    return pd.DataFrame(results)
