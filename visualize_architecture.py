"""
visualize_architecture.py - Generate architecture diagram for Medical BLIP-2.
Run: python visualize_architecture.py
Output: outputs/architecture_diagram.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(22, 16))
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 17)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # ================================================================
    # Color scheme
    # ================================================================
    FROZEN_COLOR = "#B0BEC5"       # grey — frozen
    PRETRAINED_COLOR = "#81D4FA"   # light blue — pretrained + finetuned
    SCRATCH_COLOR = "#A5D6A7"      # light green — trained from scratch
    HEAD_COLOR = "#FFE082"         # amber — extension heads
    NLI_COLOR = "#CE93D8"          # purple — NLI model
    TITLE_COLOR = "#1565C0"        # dark blue

    # ================================================================
    # Helper functions
    # ================================================================
    def draw_box(x, y, w, h, color, label, sublabel="", fontsize=11, bold=True):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#37474F", linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color="#212121")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.3,
                    sublabel, ha="center", va="center",
                    fontsize=8, color="#616161", style="italic")

    def draw_arrow(x1, y1, x2, y2, color="#37474F", style="-|>", lw=2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=15,
            linewidth=lw,
            color=color,
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow)

    def draw_curved_arrow(x1, y1, x2, y2, color="#37474F", rad=0.3):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=color,
            connectionstyle=f"arc3,rad={rad}"
        )
        ax.add_patch(arrow)

    # ================================================================
    # Title
    # ================================================================
    ax.text(10, 16.2, "Medical BLIP-2 Architecture",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=TITLE_COLOR)
    ax.text(10, 15.7, "VQA + Report Generation + Consistency Checking",
            ha="center", va="center", fontsize=12, color="#616161")

    # ================================================================
    # MAIN PIPELINE (row at y=11)
    # ================================================================

    # Input Image
    draw_box(0, 10.5, 2.5, 1.5, "#E3F2FD", "Input Image", "224 × 224 × 3", fontsize=10)

    # Arrow: Image → CLIP
    draw_arrow(2.5, 11.25, 3.5, 11.25)

    # CLIP ViT (Frozen)
    draw_box(3.5, 10.2, 3, 2, FROZEN_COLOR, "CLIP ViT-B/16", "FROZEN  |  86M params")
    ax.text(5, 10.0, "openai/clip-vit-base-patch16", ha="center", fontsize=6.5, color="#9E9E9E")
    ax.text(5, 12.4, "Output: [B, 197, 768]", ha="center", fontsize=7, color="#455A64")

    # Arrow: CLIP → Q-Former
    draw_arrow(6.5, 11.25, 7.5, 11.25)

    # Q-Former (Pretrained)
    draw_box(7.5, 10.2, 3, 2, PRETRAINED_COLOR, "Q-Former", "Pretrained  |  107M params")
    ax.text(9, 10.0, "Salesforce/blip2-flan-t5-base", ha="center", fontsize=6.5, color="#9E9E9E")
    ax.text(9, 12.4, "32 query tokens × 768", ha="center", fontsize=7, color="#455A64")

    # Arrow: Q-Former → Projection
    draw_arrow(10.5, 11.25, 11.3, 11.25)

    # Projection
    draw_box(11.3, 10.5, 2, 1.3, SCRATCH_COLOR, "Projection", "Linear + LNorm", fontsize=10)

    # Arrow: Projection → T5
    draw_arrow(13.3, 11.25, 14.2, 11.25)

    # T5 (Pretrained)
    draw_box(14.2, 10.0, 3.3, 2.5, PRETRAINED_COLOR, "FLAN-T5", "Pretrained  |  248M params")
    ax.text(15.85, 9.78, "google/flan-t5-base", ha="center", fontsize=6.5, color="#9E9E9E")
    ax.text(15.85, 12.7, "Encoder-Decoder (12+12 layers)", ha="center", fontsize=7, color="#455A64")

    # Arrow: T5 → Output
    draw_arrow(17.5, 11.25, 18.5, 11.25)

    # Output
    draw_box(18.5, 10.2, 2, 2, "#E8F5E9", "Generated\nText", "Answer / Report", fontsize=10)

    # ================================================================
    # TEXT INPUT (row at y=7.5) — merges into T5
    # ================================================================
    draw_box(7.5, 7.2, 3.5, 1.3, "#FFF3E0", "Text Prompt", '"Task: VQA Question: ..."', fontsize=10)
    draw_arrow(11, 7.85, 12.3, 7.85)
    draw_box(12.3, 7.2, 2.5, 1.3, "#FFF3E0", "T5 Embed", "Text tokens [L, 768]", fontsize=9)

    # Arrow: text embed → concat point → T5
    # Draw a merge arrow upward to T5 input
    draw_arrow(13.55, 8.5, 15.1, 10.0, color="#E65100")
    ax.text(13.7, 9.3, "concat", ha="center", fontsize=8, color="#E65100", rotation=55)

    # Visual tokens also go to concat
    draw_arrow(12.3, 11.15, 14.2, 11.15, color="#1565C0", lw=0)  # invisible — already have arrow
    ax.text(12.6, 10.3, "[visual | text]", ha="center", fontsize=7.5, color="#1565C0")

    # ================================================================
    # QUESTION TYPE HEAD (bottom-left, y=4)
    # ================================================================
    ax.text(3, 6.0, "Extension Heads", ha="center", fontsize=14,
            fontweight="bold", color="#37474F")

    draw_box(0.5, 3.5, 3, 2, HEAD_COLOR, "BERT Encoder", "Pretrained  |  110M params", fontsize=10)
    ax.text(2, 3.3, "google-bert/bert-base-uncased", ha="center", fontsize=6.5, color="#9E9E9E")

    draw_arrow(3.5, 4.5, 4.5, 4.5)

    draw_box(4.5, 3.8, 2.5, 1.3, HEAD_COLOR, "QType Head", "MLP → 2 classes", fontsize=9)

    draw_arrow(7, 4.45, 7.8, 4.45)

    draw_box(7.8, 3.8, 2.5, 1.3, "#FFF9C4", "Question\nType", "closed / open", fontsize=9)

    # Label
    ax.text(5.3, 5.5, "Question Type Classifier", ha="center", fontsize=10,
            fontweight="bold", color="#F57F17")

    # Input arrow from text
    draw_curved_arrow(9.25, 7.2, 2, 5.5, color="#F57F17", rad=-0.3)
    ax.text(4.5, 6.5, "question text", ha="center", fontsize=7, color="#F57F17")

    # ================================================================
    # CONSISTENCY HEAD (bottom-right, y=4)
    # ================================================================
    draw_box(11.5, 3.5, 3.2, 2, NLI_COLOR, "DeBERTa-v3\nNLI Model", "Pretrained  |  86M params", fontsize=10)
    ax.text(13.1, 3.3, "cross-encoder/nli-deberta-v3-base", ha="center", fontsize=6.5, color="#9E9E9E")

    draw_arrow(14.7, 4.5, 15.7, 4.5)

    draw_box(15.7, 3.8, 2, 1.3, NLI_COLOR, "Adapter", "Linear(3→1)", fontsize=9)

    draw_arrow(17.7, 4.45, 18.5, 4.45)

    draw_box(18.5, 3.8, 2, 1.3, "#F3E5F5", "Consistency\nScore", "[0, 1]", fontsize=9)

    # Label
    ax.text(15.5, 5.8, "Answer-Report Consistency Checker", ha="center", fontsize=10,
            fontweight="bold", color="#7B1FA2")

    # Input arrows from generated text
    draw_curved_arrow(19.5, 10.2, 19.5, 5.1, color="#7B1FA2", rad=-0.4)
    ax.text(20.2, 7.5, "answer +\nreport", ha="center", fontsize=7, color="#7B1FA2")

    # ================================================================
    # CONFIDENCE (right side)
    # ================================================================
    draw_box(18.5, 7.5, 2, 1.2, "#E0F7FA", "Confidence", "mean token prob", fontsize=9)
    draw_curved_arrow(19, 10.2, 19.5, 8.7, color="#00695C", rad=0.2)
    ax.text(18.3, 9.5, "gen\nscores", ha="center", fontsize=7, color="#00695C")

    # ================================================================
    # LOSS DIAGRAM (bottom, y=1)
    # ================================================================
    ax.text(10, 2.2, "Training Loss", ha="center", fontsize=13,
            fontweight="bold", color="#37474F")

    # Loss formula
    loss_box = FancyBboxPatch(
        (3, 0.3), 14, 1.3,
        boxstyle="round,pad=0.2",
        facecolor="#ECEFF1", edgecolor="#37474F", linewidth=1.5
    )
    ax.add_patch(loss_box)
    ax.text(10, 0.95,
            "Total Loss  =  LM Loss (×1.0)  +  QType Loss (×0.1)  +  Consistency Loss (×0.1)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#37474F", family="monospace")
    ax.text(10, 0.55,
            "Cross-Entropy              Cross-Entropy (2-class)         BCE (binary)",
            ha="center", va="center", fontsize=8, color="#757575", family="monospace")

    # ================================================================
    # LEGEND
    # ================================================================
    legend_items = [
        (FROZEN_COLOR, "Frozen (not trained)"),
        (PRETRAINED_COLOR, "Pretrained + Finetuned"),
        (SCRATCH_COLOR, "Trained from Scratch"),
        (HEAD_COLOR, "Question Type (BERT)"),
        (NLI_COLOR, "Consistency (DeBERTa NLI)"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 0.3 + i * 3.6
        box = FancyBboxPatch(
            (x, 15.0), 0.4, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#37474F", linewidth=1
        )
        ax.add_patch(box)
        ax.text(x + 0.6, 15.2, label, va="center", fontsize=8.5, color="#424242")

    # ================================================================
    # Save
    # ================================================================
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "architecture_diagram.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Architecture diagram saved → {out_path}")
    return out_path


if __name__ == "__main__":
    draw_architecture()
