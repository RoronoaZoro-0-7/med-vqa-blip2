"""
attention_viz.py - Attention Visualization for Medical BLIP-2 VQA.

Extracts cross-attention maps from the Q-Former (where 32 learned queries
attend to 196 CLIP image patches) and overlays them as heatmaps on the
original medical image.

Usage:
    # Single image
    python attention_viz.py --checkpoint checkpoints/best_model.pt \
        --image data/vqa_rad/images/test_0.png \
        --question "Is there a fracture?"

    # Batch from validation set (auto-loads dataset)
    python attention_viz.py --checkpoint checkpoints/best_model.pt \
        --from_dataset --num_samples 10

    # Batch from JSON file
    python attention_viz.py --checkpoint checkpoints/best_model.pt \
        --batch samples.json
"""

import os
import json
import argparse
from typing import Optional, Tuple, List, Dict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from model import MedicalBLIP2
from utils import Config, set_seed


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
# Attention extraction
# ======================================================================

def extract_attention_maps(
    model: MedicalBLIP2, pixel_values: torch.Tensor
) -> Tuple[Optional[np.ndarray], Optional[List[torch.Tensor]]]:
    """Extract cross-attention maps from Q-Former layers.

    The Q-Former's cross-attention layers let 32 learnable query tokens
    attend to the 197 image tokens (196 patches + 1 CLS) from CLIP ViT.
    We average over layers, heads, and queries to get a single [14, 14]
    spatial attention map over image patches.

    Args:
        model: Trained MedicalBLIP2 model (eval mode).
        pixel_values: [1, 3, 224, 224] preprocessed image tensor.

    Returns:
        attn_grid: [14, 14] numpy array of averaged attention weights.
        raw_attentions: list of [num_heads, 32, 197] tensors per cross-attn layer.
    """
    with torch.no_grad():
        # Step 1: Frozen CLIP image encoding
        image_out = model.image_encoder(pixel_values=pixel_values)
        image_features = image_out.last_hidden_state  # [1, 197, 768]

        if model._using_pretrained_qformer:
            # Step 2: Project CLIP hidden (768) → BLIP-2 expected dim (1408)
            image_features_proj = model.qformer_encoder_proj(image_features)
            batch_size = image_features.shape[0]
            query_tokens = model.query_tokens.expand(batch_size, -1, -1)
            image_attn_mask = torch.ones(
                batch_size, image_features_proj.shape[1],
                dtype=torch.long, device=pixel_values.device,
            )

            # Step 3: Run Q-Former with output_attentions=True
            qformer_out = model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features_proj,
                encoder_attention_mask=image_attn_mask,
                output_attentions=True,
                return_dict=True,
            )

            # Step 4: Extract cross-attention weights
            cross_attentions = getattr(qformer_out, "cross_attentions", None)

            if cross_attentions and len(cross_attentions) > 0:
                # Each element: [B, num_heads, 32_queries, 197_patches]
                attn_stack = torch.stack(list(cross_attentions))
                # Average over layers, heads, queries → [B, 197]
                attn_avg = attn_stack.mean(dim=(0, 2, 3))
                attn_map = attn_avg[0].cpu()  # [197]
                # Remove CLS token (index 0) → [196]
                attn_patches = attn_map[1:]
                # Reshape to 14×14 spatial grid
                attn_grid = attn_patches.reshape(14, 14).numpy()
                raw = [ca[0].cpu() for ca in cross_attentions]
                return attn_grid, raw

            # Fallback: use hook-based extraction
            print("  ⚠ output_attentions returned None, using hook fallback")
            return _extract_with_hooks(model, pixel_values, image_features)
        else:
            return _extract_with_hooks(model, pixel_values, image_features)


def _extract_with_hooks(
    model: MedicalBLIP2,
    pixel_values: torch.Tensor,
    image_features: Optional[torch.Tensor] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[torch.Tensor]]]:
    """Fallback: extract cross-attention via forward hooks.

    Works for both pretrained Blip2QFormerModel (hooks on crossattention.self)
    and custom QFormer (hooks on cross_attn MultiheadAttention).
    """
    attn_maps = []

    def _hook(module, inp, out):
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            attn_maps.append(out[1].detach().cpu())

    hooks = []
    if model._using_pretrained_qformer:
        # Blip2QFormerModel: crossattention is at encoder.layer[i].crossattention
        for layer in model.qformer.encoder.layer:
            if hasattr(layer, "crossattention"):
                # The actual attention module inside
                ca_module = layer.crossattention.self
                hooks.append(ca_module.register_forward_hook(_hook))
    else:
        # Custom QFormer
        for layer in model.qformer.layers:
            if hasattr(layer, "has_cross_attention") and layer.has_cross_attention:
                hooks.append(layer.cross_attn.register_forward_hook(_hook))

    with torch.no_grad():
        if image_features is None:
            image_out = model.image_encoder(pixel_values=pixel_values)
            image_features = image_out.last_hidden_state

        if model._using_pretrained_qformer:
            image_features_proj = model.qformer_encoder_proj(image_features)
            batch_size = image_features.shape[0]
            query_tokens = model.query_tokens.expand(batch_size, -1, -1)
            image_attn_mask = torch.ones(
                batch_size, image_features_proj.shape[1],
                dtype=torch.long, device=pixel_values.device,
            )
            model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features_proj,
                encoder_attention_mask=image_attn_mask,
            )
        else:
            model.qformer(image_features)

    for h in hooks:
        h.remove()

    if attn_maps:
        attn_stack = torch.stack(attn_maps)  # [layers, B, heads, Q, K]
        attn_avg = attn_stack.mean(dim=(0, 2, 3))  # [B, K]
        attn_map = attn_avg[0]  # [K]
        # Remove CLS if present (197 → 196)
        if attn_map.shape[0] == 197:
            attn_map = attn_map[1:]
        side = int(attn_map.shape[0] ** 0.5)
        attn_grid = attn_map.reshape(side, side).numpy()
        raw = [am[0] for am in attn_maps]
        return attn_grid, raw

    return None, None


# ======================================================================
# Visualization
# ======================================================================

def create_heatmap_overlay(
    image: Image.Image, attn_grid: np.ndarray, sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a smooth attention heatmap overlay.

    Args:
        image: Original PIL Image.
        attn_grid: [14, 14] attention weights.
        sigma: Gaussian blur sigma for smoothing.

    Returns:
        heatmap_rgba: [H, W, 4] RGBA heatmap array.
        smooth_attn: [H, W] smoothed attention (0–1).
    """
    from scipy.ndimage import gaussian_filter

    w, h = image.size

    # Normalize to [0, 1]
    attn_min, attn_max = attn_grid.min(), attn_grid.max()
    attn_norm = (attn_grid - attn_min) / (attn_max - attn_min + 1e-8)

    # Upsample to image size via PIL bilinear interpolation
    attn_uint8 = (attn_norm * 255).astype(np.uint8)
    attn_pil = Image.fromarray(attn_uint8, mode="L")
    attn_resized = np.array(attn_pil.resize((w, h), Image.BILINEAR)) / 255.0

    # Gaussian smoothing for presentation-quality heatmap
    smooth_attn = gaussian_filter(attn_resized, sigma=sigma)
    s_min, s_max = smooth_attn.min(), smooth_attn.max()
    smooth_attn = (smooth_attn - s_min) / (s_max - s_min + 1e-8)

    # Apply jet colormap
    heatmap_rgba = cm.jet(smooth_attn)
    return heatmap_rgba, smooth_attn


def visualize_attention(
    image_path: str,
    question: str,
    answer: str,
    attn_grid: np.ndarray,
    save_path: str,
    ground_truth: str = "",
    confidence: Optional[float] = None,
    question_type: Optional[str] = None,
    sigma: float = 2.0,
    alpha: float = 0.4,
):
    """Create a 3-panel attention visualization and save to disk.

    Panels: [Original Image] | [Attention Overlay] | [14×14 Attention Grid]

    Args:
        image_path: Path to the original medical image.
        question: The clinical question.
        answer: Model's predicted answer.
        attn_grid: [14, 14] averaged attention weights.
        save_path: Where to save the figure.
        ground_truth: Optional ground-truth answer for comparison.
        confidence: Optional model confidence score.
        question_type: Optional "closed" or "open".
        sigma: Gaussian blur sigma.
        alpha: Heatmap overlay opacity.
    """
    image = Image.open(image_path).convert("RGB")
    heatmap, _ = create_heatmap_overlay(image, attn_grid, sigma=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Attention overlay
    axes[1].imshow(image)
    axes[1].imshow(heatmap, alpha=alpha)
    axes[1].set_title("Attention Overlay", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Raw 14×14 attention grid
    im = axes[2].imshow(attn_grid, cmap="jet", interpolation="nearest")
    axes[2].set_title(
        "Q-Former Cross-Attention\n(14×14 patch grid)",
        fontsize=12, fontweight="bold",
    )
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Build title text
    lines = [f"Q: {question}"]
    if ground_truth:
        correct = answer.strip().lower() == ground_truth.strip().lower()
        mark = "✓" if correct else "✗"
        lines.append(f"Predicted: {answer}  |  Ground Truth: {ground_truth}  [{mark}]")
    else:
        lines.append(f"A: {answer}")
    meta_parts = []
    if confidence is not None:
        meta_parts.append(f"Confidence: {confidence:.2%}")
    if question_type:
        meta_parts.append(f"Type: {question_type}")
    if meta_parts:
        lines.append("  |  ".join(meta_parts))

    fig.suptitle("\n".join(lines), fontsize=12, y=1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_per_layer(
    image_path: str,
    attn_raw: List[torch.Tensor],
    save_path: str,
    sigma: float = 2.0,
    alpha: float = 0.4,
):
    """Visualize attention maps for each Q-Former cross-attention layer.

    Args:
        image_path: Path to the original image.
        attn_raw: List of [num_heads, 32, key_len] tensors per layer.
        save_path: Output path.
    """
    num_layers = len(attn_raw)
    image = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for i, layer_attn in enumerate(attn_raw):
        # layer_attn: [num_heads, 32, key_len]
        avg = layer_attn.mean(dim=(0, 1))  # [key_len]
        if avg.shape[0] == 197:
            avg = avg[1:]  # remove CLS
        grid = avg.reshape(14, 14).numpy()

        heatmap, _ = create_heatmap_overlay(image, grid, sigma=sigma)
        axes[i].imshow(image)
        axes[i].imshow(heatmap, alpha=alpha)
        axes[i].set_title(f"Layer {i + 1}", fontsize=13, fontweight="bold")
        axes[i].axis("off")

    fig.suptitle(
        "Per-Layer Q-Former Cross-Attention Maps",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-layer: {save_path}")


# ======================================================================
# Inference + visualization combined
# ======================================================================

def run_single(
    model: MedicalBLIP2,
    config: Config,
    device: torch.device,
    image_path: str,
    question: str,
    save_dir: str,
    ground_truth: str = "",
    sigma: float = 2.0,
    alpha: float = 0.4,
) -> Dict:
    """Run VQA inference + attention visualization for one image-question pair."""
    os.makedirs(save_dir, exist_ok=True)

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

    # Generate answer with confidence
    with torch.no_grad():
        gen_ids, confidence = model.generate(
            pixel_values, input_ids, attention_mask,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            return_confidence=True,
        )
        answer = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        conf = float(confidence[0])

        # Question type
        _, qt_pred, _ = model.classify_question_type(
            pixel_values, input_ids, attention_mask
        )
        q_type = "closed" if qt_pred[0].item() == 0 else "open"

    # Extract attention maps
    attn_grid, attn_raw = extract_attention_maps(model, pixel_values)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if attn_grid is not None:
        # Main 3-panel visualization
        viz_path = os.path.join(save_dir, f"{base_name}_attention.png")
        visualize_attention(
            image_path, question, answer, attn_grid, viz_path,
            ground_truth=ground_truth,
            confidence=conf, question_type=q_type,
            sigma=sigma, alpha=alpha,
        )
        # Per-layer visualization
        if attn_raw and len(attn_raw) > 1:
            layer_path = os.path.join(save_dir, f"{base_name}_per_layer.png")
            visualize_per_layer(
                image_path, attn_raw, layer_path, sigma=sigma, alpha=alpha,
            )
    else:
        print("  ⚠ Could not extract attention maps for this sample")

    return {
        "image": image_path,
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "confidence": conf,
        "question_type": q_type,
        "correct": (answer.strip().lower() == ground_truth.strip().lower())
        if ground_truth else None,
    }


def run_on_dataset(
    model: MedicalBLIP2,
    config: Config,
    device: torch.device,
    save_dir: str,
    num_samples: int = 10,
    split: str = "val",
):
    """Run attention visualization on samples from the validation/test set.

    Automatically loads the dataset, picks diverse samples (mix of correct/
    incorrect, closed/open), and generates visualizations.
    """
    from dataset import download_vqa_rad

    print(f"\nLoading VQA-RAD dataset for {split} split...")
    vqa_data = download_vqa_rad(config.data_dir)

    # Use test split from VQA-RAD (the split used for validation in training)
    samples = vqa_data.get("test", vqa_data.get("train", []))
    vqa_samples = [s for s in samples if s.get("task") == "vqa"]

    if not vqa_samples:
        print("No VQA samples found in dataset!")
        return []

    # Pick a diverse subset: some closed, some open
    closed = [s for s in vqa_samples if s.get("question_type", -1) == 0]
    open_q = [s for s in vqa_samples if s.get("question_type", -1) == 1]

    # Select roughly half closed, half open
    np.random.seed(42)
    n_closed = min(len(closed), num_samples // 2)
    n_open = min(len(open_q), num_samples - n_closed)
    selected = []
    if closed:
        idx_c = np.random.choice(len(closed), n_closed, replace=False)
        selected.extend([closed[i] for i in idx_c])
    if open_q:
        idx_o = np.random.choice(len(open_q), n_open, replace=False)
        selected.extend([open_q[i] for i in idx_o])

    print(f"Selected {len(selected)} samples ({n_closed} closed, {n_open} open)")

    results = []
    for i, sample in enumerate(selected):
        image_path = sample["image_path"]
        question = sample["question"]
        gt = sample.get("answer", "")
        q_type_str = "closed" if sample.get("question_type") == 0 else "open"

        print(f"\n[{i+1}/{len(selected)}] ({q_type_str}) {question}")

        result = run_single(
            model, config, device,
            image_path, question, save_dir,
            ground_truth=gt,
        )
        results.append(result)

        mark = ""
        if result["correct"] is not None:
            mark = " ✓" if result["correct"] else " ✗"
        print(f"  Pred: {result['answer']}  |  GT: {gt}{mark}")

    # Save results summary
    summary_path = os.path.join(save_dir, "attention_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {summary_path}")

    # Print summary
    correct = sum(1 for r in results if r.get("correct") is True)
    total = sum(1 for r in results if r.get("correct") is not None)
    if total > 0:
        print(f"Accuracy on selected samples: {correct}/{total} = {correct/total:.1%}")

    return results


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Medical BLIP-2 Attention Visualization"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--question", type=str, default=None,
                        help="Question for VQA")
    parser.add_argument("--ground_truth", type=str, default="",
                        help="Ground truth answer for comparison")
    parser.add_argument("--batch", type=str, default=None,
                        help="JSON file with samples [{image, question, ground_truth}]")
    parser.add_argument("--from_dataset", action="store_true",
                        help="Auto-load validation samples from VQA-RAD")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples for --from_dataset mode")
    parser.add_argument("--save_dir", type=str, default="outputs/attention_viz",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian blur sigma for smoothing")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Heatmap overlay opacity")
    args = parser.parse_args()

    set_seed(42)

    print("Loading model...")
    model, config, device = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {device}")

    if args.from_dataset:
        run_on_dataset(
            model, config, device, args.save_dir,
            num_samples=args.num_samples,
        )
    elif args.batch:
        with open(args.batch) as f:
            samples = json.load(f)
        results = []
        for i, s in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] {s.get('question', '')}")
            r = run_single(
                model, config, device,
                s["image"], s["question"], args.save_dir,
                ground_truth=s.get("ground_truth", ""),
                sigma=args.sigma, alpha=args.alpha,
            )
            results.append(r)
        with open(os.path.join(args.save_dir, "batch_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    elif args.image and args.question:
        run_single(
            model, config, device,
            args.image, args.question, args.save_dir,
            ground_truth=args.ground_truth,
            sigma=args.sigma, alpha=args.alpha,
        )
    else:
        parser.error("Provide --image + --question, --batch, or --from_dataset")


if __name__ == "__main__":
    main()
