# Medical BLIP-2 Architecture - Complete Technical Guide

## 📋 Overview

This project implements a **BLIP-2 (Bootstrapping Language-Image Pre-training 2) style architecture** specifically adapted for **Medical Visual Question Answering (VQA)** and **Medical Report Generation**. The architecture combines multiple pretrained components that are fine-tuned on medical image datasets.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MEDICAL BLIP-2 ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   Medical Image  │
                              │   (224×224×3)    │
                              └────────┬─────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: VISUAL FEATURE EXTRACTION (FROZEN)                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         CLIP Vision Transformer                                 │  │
│  │                      (openai/clip-vit-base-patch16)                            │  │
│  │                                                                                 │  │
│  │   Image → 16×16 Patches → Patch Embeddings + Position Embeddings → ViT Layers │  │
│  │                                                                                 │  │
│  │   Output: 197 tokens × 768 dimensions (196 patch + 1 CLS token)                │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ [B, 197, 768]
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: VISUAL-LANGUAGE ALIGNMENT (TRAINABLE)                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Q-Former                                           │  │
│  │                     (Salesforce/blip2-opt-2.7b)                                │  │
│  │                                                                                 │  │
│  │   32 Learnable Query Tokens ──┐                                                │  │
│  │                               ├── Cross-Attention → Compressed Visual Features │  │
│  │   197 Visual Tokens ──────────┘                                                │  │
│  │                                                                                 │  │
│  │   6 Transformer Layers (Cross-Attention on layers 0, 2, 4)                     │  │
│  │   Output: 32 tokens × 768 dimensions                                           │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                              │
│                                       ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                      Projection Layer (Linear + LayerNorm)                      │  │
│  │                        768 → 768 dimensions (T5 space)                         │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ [B, 32, 768]
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: TEXT INPUT PROCESSING                                                      │
│                                                                                      │
│   Task Prompt: "Task: VQA Question: Is there a fracture?"                           │
│                              │                                                       │
│                              ▼                                                       │
│                    T5 Tokenizer + Embedding Layer                                   │
│                              │                                                       │
│                              ▼ [B, L, 768]                                          │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  CONCATENATION: [32 Visual Tokens] + [L Text Tokens] = [32+L, 768]          │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ [B, 32+L, 768]
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: LANGUAGE UNDERSTANDING & GENERATION (TRAINABLE)                           │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                            FLAN-T5 Base                                        │  │
│  │                        (google/flan-t5-base)                                   │  │
│  │                                                                                 │  │
│  │   ┌─────────────────┐                    ┌─────────────────┐                   │  │
│  │   │    T5 Encoder   │                    │    T5 Decoder   │                   │  │
│  │   │   (12 Layers)   │ ──────────────────▶│   (12 Layers)   │ ───▶ Output      │  │
│  │   │                 │  Cross-Attention   │                 │                   │  │
│  │   └─────────────────┘                    └─────────────────┘                   │  │
│  │                                                                                 │  │
│  │   Encoder receives: [Visual Tokens | Text Prompt Tokens]                       │  │
│  │   Decoder generates: Answer or Medical Report (autoregressive)                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                            Generated Text Output
                         (VQA Answer or Medical Report)
```

---

## 🔧 Component Details

### 1. Image Encoder - CLIP Vision Transformer

| Specification | Value |
|---------------|-------|
| **Model** | `openai/clip-vit-base-patch16` |
| **Architecture** | Vision Transformer (ViT-B/16) |
| **Parameters** | ~86 Million |
| **Pretrained On** | 400M image-text pairs (WebImageText dataset) |
| **Status** | **FROZEN** (no gradient updates) |
| **Input** | Image tensor `[B, 3, 224, 224]` |
| **Output** | `[B, 197, 768]` - 196 patch tokens + 1 CLS token |

**How it works:**
1. The input image (224×224) is divided into **16×16 pixel patches** (14×14 = 196 patches)
2. Each patch is linearly projected to a 768-dimensional embedding
3. A learnable [CLS] token is prepended (total 197 tokens)
4. Positional embeddings are added
5. Tokens pass through 12 Transformer encoder layers
6. Output: Rich visual features for each image region

**Why Frozen?**
- CLIP was pretrained on 400M diverse image-text pairs
- Already learned robust visual-semantic representations
- Freezing prevents catastrophic forgetting and reduces compute

---

### 2. Q-Former (Querying Transformer)

| Specification | Value |
|---------------|-------|
| **Model** | Extracted from `Salesforce/blip2-opt-2.7b` |
| **Architecture** | 6-layer Transformer with Cross-Attention |
| **Parameters** | ~107 Million |
| **Pretrained On** | 129M image-text pairs (BLIP-2 Stage 1) |
| **Status** | **FULLY TRAINABLE** |
| **Input** | 197 visual tokens from CLIP |
| **Output** | `[B, 32, 768]` - 32 learned query tokens |

**Architecture Details:**
```
Q-Former Layer Structure (6 layers total):
├── Layer 0: Self-Attention + Cross-Attention + FFN
├── Layer 1: Self-Attention + FFN (no cross-attention)
├── Layer 2: Self-Attention + Cross-Attention + FFN
├── Layer 3: Self-Attention + FFN
├── Layer 4: Self-Attention + Cross-Attention + FFN
└── Layer 5: Self-Attention + FFN

Cross-Attention Pattern: Every 2nd layer (0, 2, 4)
```

**Key Innovation:**
- **32 Learnable Query Tokens**: These are trainable parameters that "ask questions" to the image
- **Cross-Attention**: Queries attend to all 197 image tokens to extract relevant information
- **Compression**: 197 image tokens → 32 query tokens (6x compression)

**Why Q-Former?**
- Bridges the "modality gap" between vision and language
- The original BLIP-2 paper showed Q-Former's pretraining learns:
  - Image-Text Contrastive (ITC) alignment
  - Image-Text Matching (ITM) discrimination
  - Image-grounded Text Generation (ITG)
- These pretrained visual-linguistic mappings transfer well to medical images

---

### 3. Projection Layer

| Specification | Value |
|---------------|-------|
| **Architecture** | `Linear(768→768)` + `LayerNorm(768)` |
| **Parameters** | ~590K |
| **Status** | **TRAINED FROM SCRATCH** |

**Purpose:**
- Maps Q-Former output space to T5's input embedding space
- LayerNorm stabilizes the representations
- Critical bridge between visual and language modalities

---

### 4. Language Model - FLAN-T5

| Specification | Value |
|---------------|-------|
| **Model** | `google/flan-t5-base` |
| **Architecture** | Encoder-Decoder Transformer (12+12 layers) |
| **Parameters** | ~248 Million |
| **Pretrained On** | C4 corpus + instruction-tuned on 1,800+ tasks |
| **Status** | **FULLY TRAINABLE** |
| **Vocabulary** | 32,128 tokens (SentencePiece) |

**Why FLAN-T5?**
- **Instruction-tuned**: Better at following task-specific prompts like `"Task: VQA Question: ..."`
- **Encoder-Decoder**: Superior for generative tasks compared to decoder-only models
- **Efficient**: ~250M params (not billions) - suitable for fine-tuning

**Generation Strategy:**
- Beam Search (4 beams by default)
- No-repeat n-gram blocking (n=3)
- Early stopping

---

## 🧩 Extension Heads

### 5. Question Type Classifier (BERT)

| Specification | Value |
|---------------|-------|
| **Model** | `google-bert/bert-base-uncased` |
| **Parameters** | ~110M (BERT) + ~295K (MLP head) |
| **Task** | Binary classification: Closed (yes/no) vs Open (free-form) |
| **Status** | **FULLY TRAINABLE** |

```
Question Text → BERT Encoder → [CLS] Token → MLP Head → [Closed, Open]
                                              ↓
                                    MLP: 768→384→2
```

**Purpose:**
- If question is "closed-ended" (yes/no), apply **constrained decoding**
- Only allow "yes"/"no" tokens during generation
- Improves accuracy on binary questions

---

### 6. Consistency Checker (DeBERTa NLI)

| Specification | Value |
|---------------|-------|
| **Model** | `cross-encoder/nli-deberta-v3-base` |
| **Parameters** | ~86M (DeBERTa) + Linear(3→1) adapter |
| **Pretrained On** | 943K NLI pairs (MNLI + SNLI) |
| **Status** | **FULLY TRAINABLE** |

```
(VQA Answer, Generated Report) → DeBERTa NLI → [Contradiction, Neutral, Entailment]
                                                          ↓
                                              Linear(3→1) → Sigmoid → Consistency Score
```

**Purpose:**
- Checks if VQA answer and generated report are semantically consistent
- Uses Natural Language Inference which is designed for entailment detection
- Helps ensure model doesn't contradict itself

---

### 7. Confidence Estimation

| Specification | Value |
|---------------|-------|
| **Architecture** | Mean token probability from beam search |
| **Parameters** | 0 (no learnable parameters) |

**How it works:**
1. During generation, T5 outputs probability distribution at each step
2. Extract probability of the token that was actually selected
3. Average probabilities across all generated tokens
4. Result: Confidence score in [0, 1]

---

## 📊 Training Details

### Datasets Used

#### Training Data

| Dataset | Task | Split Used | Description |
|---------|------|------------|-------------|
| **VQA-RAD** | VQA | `train` | ~2,500 radiology image-question-answer triplets |
| **SLAKE** | VQA | `train` | ~10,000 medical VQA samples (sememe-enhanced) |
| **IU X-Ray** | Report Generation | `train` | ~6,000 chest X-ray images with radiology reports |
| **Synthetic QA** | VQA | Generated | Rule-based QA pairs extracted from IU X-Ray reports |

#### Validation Data

| Dataset | Split Used | Notes |
|---------|------------|-------|
| **SLAKE** | `val` | Primary validation set |
| **Fallback** | 10% of train | Used if SLAKE val not available |

#### Testing Data ✅ (All datasets used for evaluation)

| Dataset | Task | Split Used |
|---------|------|------------|
| **VQA-RAD** | VQA | `test` (~1,000 samples) |
| **SLAKE** | VQA | `test` (~4,000 samples) |
| **IU X-Ray** | Report Generation | `test` (~1,000 samples) |

---

### Multi-Task Loss Function

The model is trained with a **combined loss** from three objectives:

```
𝓛_total = 𝓛_LM + 0.1 × 𝓛_QType + 0.1 × 𝓛_Consistency
```

---

### Loss 1: Language Model Loss (Main Loss)

**Purpose:** Train the T5 model to generate correct VQA answers and medical reports.

**Formula (Cross-Entropy Loss):**

$$\mathcal{L}_{LM} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t}, X)$$

Where:
- $T$ = number of tokens in target sequence
- $y_t$ = ground truth token at position $t$
- $y_{<t}$ = all previous tokens
- $X$ = input (visual embeddings + text prompt)
- $P(y_t | y_{<t}, X)$ = model's predicted probability for token $y_t$

**Implementation:**
```python
# Built into T5ForConditionalGeneration
outputs = self.t5_model(inputs_embeds=..., labels=labels)
lm_loss = outputs.loss  # Cross-entropy over vocabulary
```

| Property | Value |
|----------|-------|
| **Weight** | 1.0 |
| **Applied To** | All samples (VQA + Report) |
| **Ignore Index** | -100 (padding tokens) |

---

### Loss 2: Question Type Classification Loss

**Purpose:** Classify questions as closed-ended (yes/no) or open-ended (free-form).

**Formula (Cross-Entropy Loss):**

$$\mathcal{L}_{QType} = -\sum_{c=0}^{1} y_c \log(\hat{y}_c)$$

Where:
- $c \in \{0, 1\}$ = class (0=closed, 1=open)
- $y_c$ = ground truth one-hot label
- $\hat{y}_c$ = predicted probability after softmax

**Expanded:**

$$\mathcal{L}_{QType} = -\log\left(\frac{e^{z_{y}}}{\sum_{j=0}^{1} e^{z_j}}\right)$$

Where $z$ = logits from BERT + MLP head, $y$ = true class.

**Implementation:**
```python
qt_logits, _, _ = self.classify_question_type(pixel_values, input_ids, attention_mask)
qt_loss = F.cross_entropy(qt_logits, question_type_labels, ignore_index=-1)
```

| Property | Value |
|----------|-------|
| **Weight** | 0.1 |
| **Classes** | 2 (closed=0, open=1) |
| **Applied To** | VQA samples only |
| **Ignore Index** | -1 (report samples skipped) |

---

### Loss 3: Answer-Report Consistency Loss

**Purpose:** Ensure the VQA answer and generated report are semantically consistent (not contradictory).

**Formula (Binary Cross-Entropy with Logits):**

$$\mathcal{L}_{Consistency} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \cdot \log(\sigma(z_i)) + (1-y_i) \cdot \log(1-\sigma(z_i)) \right]$$

Where:
- $N$ = batch size
- $y_i \in \{0, 1\}$ = ground truth (0=inconsistent, 1=consistent)
- $z_i$ = raw logit from consistency head
- $\sigma(z) = \frac{1}{1+e^{-z}}$ = sigmoid function

**Pipeline:**
1. Answer text + Report text → DeBERTa NLI → 3 logits [contradiction, neutral, entailment]
2. 3 logits → Linear(3→1) → 1 consistency logit
3. Apply BCE loss

**Implementation:**
```python
cons_logits, _ = self.check_consistency(pixel_values, answer_ids, report_ids)
cons_loss = F.binary_cross_entropy_with_logits(
    cons_logits.squeeze(-1),
    consistency_labels.float()
)
```

| Property | Value |
|----------|-------|
| **Weight** | 0.1 |
| **Output** | Single logit → sigmoid → probability |
| **Applied To** | Samples with both answer AND report |
| **Skip Condition** | `consistency_labels < 0` |

---

### Combined Loss Summary

| Loss | Formula | Weight | When Applied |
|------|---------|--------|--------------|
| **LM Loss** | $-\frac{1}{T}\sum_t \log P(y_t\|y_{<t}, X)$ | **1.0** | All samples |
| **QType Loss** | $-\sum_c y_c \log(\hat{y}_c)$ | **0.1** | VQA only (label ≥ 0) |
| **Consistency Loss** | $-[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ | **0.1** | Answer+Report pairs only |

**Final Total:**
$$\boxed{\mathcal{L}_{total} = \mathcal{L}_{LM} + 0.1 \cdot \mathcal{L}_{QType} + 0.1 \cdot \mathcal{L}_{Consistency}}$$

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 2e-5 |
| **Weight Decay** | 0.01 |
| **Warmup Steps** | 500 |
| **LR Schedule** | Linear warmup → Cosine decay |
| **Batch Size** | 4 (effective 16 with gradient accumulation) |
| **Gradient Accumulation** | 4 steps |
| **Mixed Precision** | FP16 (on GPU) |
| **Gradient Clipping** | max_norm = 1.0 |

### What's Frozen vs Trainable

| Component | Training Status | Why? |
|-----------|----------------|------|
| CLIP ViT | ❄️ Frozen | Already learned robust visual features |
| Q-Former | 🔥 Trainable | Adapt visual-language alignment to medical domain |
| Projection | 🔥 Trainable | Bridge to T5 space |
| FLAN-T5 | 🔥 Trainable | Adapt language generation to medical vocabulary |
| BERT (QType) | 🔥 Trainable | Learn medical question patterns |
| DeBERTa (NLI) | 🔥 Trainable | Adapt consistency checking to medical domain |

---

## 🔄 Data Flow Examples

### VQA Inference Flow

```
INPUT: Medical X-ray image + Question: "Is there a fracture?"

Step 1: Image Encoding
   └─→ Image [224,224,3] → CLIP ViT → [197, 768] visual tokens

Step 2: Visual Compression  
   └─→ Visual tokens → Q-Former (cross-attention) → [32, 768] query tokens

Step 3: Projection
   └─→ Query tokens → Linear+LayerNorm → [32, 768] projected tokens

Step 4: Text Encoding
   └─→ "Task: VQA Question: Is there a fracture?" → T5 Tokenizer → [L, 768]

Step 5: Concatenation
   └─→ [32 visual | L text] = [32+L, 768] combined input

Step 6: T5 Generation
   └─→ T5 Encoder → T5 Decoder → "yes"

Step 7: Question Type Check
   └─→ Question text → BERT → "closed-ended" → Constrained decoding

Step 8: Confidence
   └─→ Generation scores → Mean probability → 0.87

OUTPUT: "yes" (confidence: 87%)
```

### Report Generation Flow

```
INPUT: Chest X-ray image + Prompt: "Task: Report Generate a detailed medical report"

Steps 1-5: Same as VQA

Step 6: T5 Generation (longer sequence)
   └─→ T5 Decoder → "The cardiac silhouette is within normal limits. 
                     The lungs are clear without infiltrates or effusions.
                     No pneumothorax is identified..."

OUTPUT: Complete medical report (confidence: 72%)
```

---

## 📁 Dataset Support

The system supports multiple medical VQA datasets:

| Dataset | Task | Size | Description |
|---------|------|------|-------------|
| **VQA-RAD** | VQA | ~3,500 Q&A pairs | Radiology images with questions |
| **SLAKE** | VQA | ~14,000 Q&A pairs | Sememe knowledge-enhanced VQA |
| **IU X-Ray** | Report | ~7,470 reports | Indiana University chest X-rays |

**Input Format:**
```
VQA:    "Task: VQA Question: {question}"
Report: "Task: Report Generate a detailed medical report"
```

---

## 📈 Parameter Summary

| Component | Total Params | Trainable Params |
|-----------|-------------|-----------------|
| CLIP ViT | ~86M | 0 (frozen) |
| Q-Former | ~107M | ~107M |
| Projection | ~590K | ~590K |
| FLAN-T5 | ~248M | ~248M |
| BERT (QType) | ~110M | ~110M |
| DeBERTa (NLI) | ~86M | ~86M |
| Heads & Adapters | ~295K | ~295K |
| **TOTAL** | **~638M** | **~552M** |

---

## 🎯 Key Design Decisions

1. **Frozen CLIP**: Preserves universal visual representations, reduces training cost
2. **Pretrained Q-Former**: Leverages 129M image-text pair pretraining from BLIP-2
3. **FLAN-T5 over GPT**: Encoder-decoder is better for conditioned generation
4. **Multi-head Architecture**: Separate heads for classification, consistency, and generation
5. **Constrained Decoding**: Restricts yes/no questions to valid responses only
6. **Confidence Estimation**: Built-in uncertainty quantification without extra parameters

---

## 📚 References

- **BLIP-2**: [Li et al., 2023 - "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"](https://arxiv.org/abs/2301.12597)
- **CLIP**: [Radford et al., 2021 - "Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/abs/2103.00020)
- **FLAN-T5**: [Chung et al., 2022 - "Scaling Instruction-Finetuned Language Models"](https://arxiv.org/abs/2210.11416)
- **DeBERTa**: [He et al., 2020 - "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"](https://arxiv.org/abs/2006.03654)
