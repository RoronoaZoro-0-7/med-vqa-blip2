# Medical BLIP-2: Architecture Documentation

## Overview

A medical Vision-Language model that performs **Visual Question Answering (VQA)** and **Medical Report Generation** using a BLIP-2 style architecture. Every major component is initialized from pretrained weights and fully finetuned on medical data.

---

## Architecture Pipeline

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    MEDICAL BLIP-2                            │
                    └─────────────────────────────────────────────────────────────┘

     ┌──────────────┐      ┌──────────────┐      ┌──────┐      ┌────────────────┐
     │  CLIP ViT     │      │   Q-Former    │      │Linear│      │  FLAN-T5       │
     │  (Frozen)     │─────▶│  (Pretrained) │─────▶│Proj +│─────▶│  Encoder-      │──▶ Output
     │               │      │              │      │LNorm │      │  Decoder       │
     └──────────────┘      └──────────────┘      └──────┘      └────────────────┘
     openai/clip-vit-       Salesforce/blip2-                    google/flan-t5-
     base-patch16           flan-t5-base                         base

                         ┌────────────────────┐
     Text Prompt ──────▶ │  T5 Text Embeddings │──────────────────┘
     "Task: VQA           └────────────────────┘
      Question: ..."           (concatenated with visual embeddings)
```

---

## Component Details

### 1. Image Encoder — CLIP ViT (`openai/clip-vit-base-patch16`)

| Property | Value |
|----------|-------|
| **Architecture** | Vision Transformer (ViT-B/16) |
| **Parameters** | ~86M |
| **Pretrained on** | 400M image-text pairs (WebImageText) |
| **Finetuned?** | **No — completely frozen** |
| **Output** | `[B, 197, 768]` — 196 patch tokens + 1 CLS token |

The image is split into 16×16 patches, each embedded into 768-dim vectors. These 197 tokens are the visual features the Q-Former reads.

### 2. Q-Former — Querying Transformer (`Salesforce/blip2-flan-t5-base`)

| Property | Value |
|----------|-------|
| **Architecture** | 6-layer Transformer with cross-attention |
| **Parameters** | ~107M |
| **Pretrained on** | 129M image-text pairs (BLIP-2 Stage 1) |
| **Finetuned?** | **Yes — all weights finetuned** |
| **Input** | 197 image tokens from CLIP |
| **Output** | `[B, 32, 768]` — 32 learned query tokens |

**What it does**: The Q-Former has 32 learnable query tokens that cross-attend to the 197 CLIP image tokens. This compresses 197 visual features into 32 compact tokens that the language model can process. Think of it as asking 32 questions about the image and getting 32 focused visual summaries.

**Why pretrained?** Salesforce pretrained this Q-Former with three objectives:
1. **Image-Text Contrastive (ITC)** — align image and text representations
2. **Image-Text Matching (ITM)** — predict if image-text pair matches
3. **Image-grounded Text Generation (ITG)** — generate text from image

These pre-learned visual-linguistic mappings transfer well to medical images.

**Cross-attention pattern**: Layers 0, 2, 4 have cross-attention (every 2 layers). Layers 1, 3, 5 only have self-attention. This alternating pattern lets queries first attend to image features, then refine amongst themselves.

### 3. Projection Layer (trained from scratch)

| Property | Value |
|----------|-------|
| **Architecture** | Linear(768→768) + LayerNorm |
| **Parameters** | ~590K |
| **Finetuned?** | **Yes — trained from scratch** |

Maps Q-Former output space to T5's input embedding space. Small but critical bridge.

### 4. Language Model — FLAN-T5 (`google/flan-t5-base`)

| Property | Value |
|----------|-------|
| **Architecture** | T5 Encoder-Decoder (12+12 layers) |
| **Parameters** | ~248M |
| **Pretrained on** | C4 corpus + instruction-tuned on 1.8K tasks |
| **Finetuned?** | **Yes — all weights finetuned** |
| **Input** | [32 visual tokens | text prompt tokens] |
| **Output** | Generated text (answer or report) |

This is the **only decoder** in the system. It handles both tasks:
- **VQA**: Input = `"Task: VQA Question: Is there a fracture?"` → Output = `"yes"`
- **Report**: Input = `"Task: Report Generate a detailed medical report"` → Output = full report

FLAN-T5 was specifically instruction-tuned on 1.8K tasks, making it better at following task-specific prompts than vanilla T5.

---

## Extension Heads

### 5. Question Type Classifier — BERT (`google-bert/bert-base-uncased`)

| Property | Value |
|----------|-------|
| **Architecture** | BERT encoder (12 layers) + 2-layer MLP head |
| **Parameters** | ~110M (BERT) + ~295K (head) |
| **Pretrained on** | BookCorpus + English Wikipedia (3.3B words) |
| **Finetuned?** | **Yes — all weights finetuned** |
| **Input** | Question text |
| **Output** | 2 classes: `0 = closed` (yes/no), `1 = open` (free-form) |

**Why BERT?** The question type depends purely on text (not the image). BERT's [CLS] token provides rich sentence-level representations. When a question is classified as "closed", the model uses **constrained decoding** — restricting generation to only yes/no vocabulary tokens.

**Fallback**: If BERT can't be loaded, falls back to a lightweight 2-layer MLP pooling Q-Former features.

### 6. Consistency Checker — DeBERTa NLI (`cross-encoder/nli-deberta-v3-base`)

| Property | Value |
|----------|-------|
| **Architecture** | DeBERTa-v3 (12 layers) + 3-class head → Linear(3→1) adapter |
| **Parameters** | ~86M (DeBERTa) + ~4 (adapter) |
| **Pretrained on** | 943K NLI pairs (MNLI + SNLI) |
| **Finetuned?** | **Yes — all weights finetuned** |
| **Input** | (VQA answer, generated report) as sentence pair |
| **Output** | Consistency probability `[0, 1]` |

**What it does**: After the model generates both a VQA answer and a report, this head checks if they agree. For example:
- Answer: `"yes, fracture present"` + Report: `"no fracture detected"` → **inconsistent (0.1)**
- Answer: `"yes, fracture present"` + Report: `"fracture seen in left femur"` → **consistent (0.9)**

**Why NLI?** Natural Language Inference is exactly the task of determining if one sentence entails/contradicts another. The DeBERTa model was pretrained on:
- **MNLI**: 433K sentence pairs from 10 genres
- **SNLI**: 570K sentence pairs from image captions
- Maps: `entailment → consistent`, `contradiction → inconsistent`, `neutral → somewhere between`

**Fallback**: If NLI model can't be loaded, uses T5 encoder CLS-token pooling + 2-layer MLP (trained from scratch).

### 7. Confidence Estimation (no extra parameters)

| Property | Value |
|----------|-------|
| **Architecture** | Mean token probability from beam search scores |
| **Parameters** | 0 (computed from generation scores) |

During generation, the model returns the average probability across all generated tokens. Low confidence flags uncertain predictions.

---

## Training Strategy

### Multi-Task Loss

```
Total Loss = LM Loss + 0.1 × QType Loss + 0.1 × Consistency Loss
```

| Loss | Type | Weight | Purpose |
|------|------|--------|---------|
| LM Loss | Cross-entropy | 1.0 | Main text generation (VQA + Report) |
| QType Loss | Cross-entropy (2-class) | 0.1 | Question type classification |
| Consistency Loss | BCE (binary) | 0.1 | Answer-report agreement |

### What's Frozen vs Finetuned

| Component | Source | Training Status |
|-----------|--------|----------------|
| CLIP ViT | OpenAI (400M pairs) | **Frozen** — 0 weights updated |
| Q-Former | Salesforce BLIP-2 (129M pairs) | **All weights finetuned** |
| Projection + LayerNorm | Random init | **Trained from scratch** |
| FLAN-T5 | Google (1.8K tasks) | **All weights finetuned** |
| BERT (QType) | Google (3.3B words) | **All weights finetuned** |
| DeBERTa (NLI) | cross-encoder (943K NLI) | **All weights finetuned** |
| Consistency adapter | Random init | **Trained from scratch** |

### Optimizer & Schedule

- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Schedule**: Linear warmup (200 steps) → cosine decay to 0
- **Mixed precision**: FP16 on GPU, FP32 on CPU
- **Gradient accumulation**: 4 steps (effective batch = 16)
- **Gradient clipping**: max_norm = 1.0

---

## Data Flow (Step by Step)

### VQA Inference

```
1. Image (224×224×3) → CLIP ViT ─→ 197 visual tokens [197, 768]
2. Visual tokens → Q-Former cross-attention → 32 query tokens [32, 768]
3. Query tokens → Linear + LayerNorm → 32 projected tokens [32, 768]
4. "Task: VQA Question: Is there a fracture?" → T5 tokenizer → text tokens [L, 768]
5. [32 visual | L text] → T5 Encoder → contextualized representations
6. T5 Decoder → autoregressive generation → "yes"
7. Question text → BERT [CLS] → QType head → "closed"
8. Since closed → constrained decoding (only yes/no tokens allowed)
9. Generation scores → mean token probability → confidence = 0.87
```

### Report Inference

```
1-5. Same as VQA (with report prompt)
6.   T5 Decoder → "The cardiac silhouette is within normal limits..."
9.   confidence = 0.72
```

### Consistency Check (when task="both")

```
1-6. Generate VQA answer: "yes, fracture present"
1-6. Generate report: "fracture seen in left femur..."
7.   (answer, report) → DeBERTa NLI → [contradiction=0.02, neutral=0.08, entail=0.90]
8.   NLI logits → Linear(3→1) → sigmoid → consistency = 0.92
```

---

## Parameter Count

| Component | Total Params | Trainable Params |
|-----------|-------------|-----------------|
| CLIP ViT | ~86M | 0 (frozen) |
| Q-Former | ~107M | ~107M |
| Projection | ~590K | ~590K |
| FLAN-T5 | ~248M | ~248M |
| BERT (QType head) | ~110M | ~110M |
| DeBERTa (NLI) | ~86M | ~86M |
| QType MLP head | ~295K | ~295K |
| Consistency adapter | ~4 | ~4 |
| **Total** | **~638M** | **~552M** |

---

## File Structure

```
model.py      — Model architecture (all components above)
dataset.py    — Dataset download, preprocessing, collation
train.py      — Training loop, evaluation, checkpointing
inference.py  — Inference engine (VQA, Report, Consistency)
utils.py      — Config, metrics (accuracy, BLEU, ROUGE), visualization
```

---

## Config Reference

```python
# Model
vision_model_name    = "openai/clip-vit-base-patch16"
t5_model_name        = "google/flan-t5-base"
blip2_model_name     = "Salesforce/blip2-flan-t5-base"
qt_bert_model_name   = "google-bert/bert-base-uncased"
nli_model_name       = "cross-encoder/nli-deberta-v3-base"
num_query_tokens     = 32

# Q-Former (used as fallback only)
qformer_hidden_size  = 768
qformer_num_heads    = 12
qformer_num_layers   = 6

# Training
epochs               = 20
batch_size           = 4
learning_rate        = 2e-5
gradient_accumulation= 4
fp16                 = True (auto-disabled on CPU)
```
