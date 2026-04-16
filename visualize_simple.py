"""
visualize_simple.py - Simple architecture diagram for Medical BLIP-2.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

def draw_simple_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Colors
    INPUT_COLOR = "#E3F2FD"      # light blue
    FROZEN_COLOR = "#B0BEC5"     # grey
    TRAINABLE_COLOR = "#81D4FA"  # blue
    PROJECTION_COLOR = "#A5D6A7" # green
    OUTPUT_COLOR = "#FFE082"     # amber
    TEXT_COLOR = "#FFF3E0"       # light orange

    def draw_box(x, y, w, h, color, label, sublabel="", fontsize=12):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#37474F", linewidth=2
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#212121")
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.25,
                    sublabel, ha="center", va="center",
                    fontsize=9, color="#616161")

    def draw_arrow(x1, y1, x2, y2, color="#37474F", lw=2.5):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=lw,
            color=color,
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow)

    def draw_curved_arrow(x1, y1, x2, y2, color="#37474F", rad=0.2, lw=2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}"
        )
        ax.add_patch(arrow)

    # ================================================================
    # Title
    # ================================================================
    ax.text(5, 13.3, "Medical BLIP-2 Architecture", ha="center", fontsize=18, 
            fontweight="bold", color="#1565C0")

    # ================================================================
    # MAIN PIPELINE (vertical: top to bottom)
    # ================================================================
    
    # 1. Input Image (top)
    draw_box(3.5, 11.5, 3, 1.3, INPUT_COLOR, "Image", "224×224")
    
    # Arrow: Image → CLIP
    draw_arrow(5, 11.5, 5, 10.5)
    
    # 2. CLIP ViT
    draw_box(3.5, 9.2, 3, 1.3, FROZEN_COLOR, "CLIP ViT", "(Frozen)")
    
    # Arrow: CLIP → Q-Former
    draw_arrow(5, 9.2, 5, 8.2)
    
    # 3. Q-Former
    draw_box(3.5, 6.9, 3, 1.3, TRAINABLE_COLOR, "Q-Former", "(Trainable)")
    
    # Arrow: Q-Former → Projection
    draw_arrow(5, 6.9, 5, 5.9)
    
    # 4. Projection Layer
    draw_box(3.5, 4.6, 3, 1.3, PROJECTION_COLOR, "Projection", "Linear+LN")

    # ================================================================
    # TEXT PROMPT (side branch)
    # ================================================================
    draw_box(0.3, 4.6, 2.5, 1.3, TEXT_COLOR, "Text Prompt", '"Question: ..."')
    
    # Arrow: Text → merge point
    draw_curved_arrow(2.8, 5.25, 3.5, 5.25, color="#E65100", rad=0, lw=2)
    
    # concat indicator
    ax.text(3.2, 5.6, "concat", ha="center", fontsize=9, color="#E65100", 
            style="italic", rotation=0)

    # ================================================================
    # DECODER OUTPUT (continue down)
    # ================================================================
    
    # Arrow: Projection → T5/Decoder
    draw_arrow(5, 4.6, 5, 3.6)
    
    # 5. Answer Decoder (T5)
    draw_box(3.5, 2.3, 3, 1.3, OUTPUT_COLOR, "T5 Decoder", "Answer/Report")

    # ================================================================
    # AUXILIARY HEADS (side branches)
    # ================================================================
    
    # Q/A Type Classifier - branches from Q-Former 
    draw_box(7.2, 6.9, 2.5, 1.3, "#CE93D8", "QType Head", "closed/open")
    draw_arrow(6.5, 7.55, 7.2, 7.55, color="#7B1FA2", lw=1.8)
    
    # Confidence - branches from T5 output
    draw_box(7.2, 2.3, 2.5, 1.3, "#B2EBF2", "Confidence", "token probs")
    draw_arrow(6.5, 2.95, 7.2, 2.95, color="#00838F", lw=1.8)

    # ================================================================
    # TENSOR SHAPES (annotations on arrows)
    # ================================================================
    ax.text(5.5, 10.8, "[B,197,768]", fontsize=8, color="#455A64", ha="left")
    ax.text(5.5, 8.5, "[B,32,768]", fontsize=8, color="#455A64", ha="left")
    ax.text(5.5, 6.2, "[B,32,768]", fontsize=8, color="#455A64", ha="left")

    # ================================================================
    # LEGEND (bottom)
    # ================================================================
    legend_y = 0.6
    legend_items = [
        (FROZEN_COLOR, "Frozen"),
        (TRAINABLE_COLOR, "Trainable"),
        (PROJECTION_COLOR, "From Scratch"),
        (OUTPUT_COLOR, "Decoder"),
    ]
    for i, (color, label) in enumerate(legend_items):
        x = 1.5 + i * 2.0
        box = FancyBboxPatch((x, legend_y), 0.3, 0.3, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor="#37474F", linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.45, legend_y + 0.15, label, va="center", fontsize=9, color="#424242")

    # ================================================================
    # Save
    # ================================================================
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "architecture_simple.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Simple architecture diagram saved → {out_path}")
    return out_path


if __name__ == "__main__":
    draw_simple_architecture()
