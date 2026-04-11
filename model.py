"""
model.py - BLIP-2 style Medical Vision-Language model.

Architecture:
  Frozen Image Encoder (CLIP ViT) ──▶ Q-Former (pretrained, finetuned) ──▶ Projection ──▶ T5 Decoder

Components (all pretrained, all finetuned):
  - Q-Former: Salesforce/blip2-flan-t5-base (pretrained on 129M image-text pairs)
  - Question Type Classifier: google-bert/bert-base-uncased backbone + 2-class head
  - Consistency Checker: cross-encoder/nli-deberta-v3-base (pretrained on 943K NLI pairs)
    Maps entailment/contradiction/neutral → binary consistency score
  - Confidence-Calibrated Generation: per-sample confidence from token probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Blip2QFormerModel,
    Blip2QFormerConfig,
    BertModel,
    BertConfig,
)


# ---------------------------------------------------------------------------
# Q-Former components
# ---------------------------------------------------------------------------

class QFormerLayer(nn.Module):
    """Single Q-Former transformer layer with optional cross-attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        has_cross_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn_drop = nn.Dropout(dropout)

        # Cross-attention (only on selected layers)
        self.has_cross_attention = has_cross_attention
        if has_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_size)
            self.cross_attn_drop = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, encoder_hidden_states=None):
        # Self-attention (post-norm)
        residual = hidden_states
        attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.self_attn_norm(residual + self.self_attn_drop(attn_out))

        # Cross-attention
        if self.has_cross_attention and encoder_hidden_states is not None:
            residual = hidden_states
            attn_out, _ = self.cross_attn(
                hidden_states, encoder_hidden_states, encoder_hidden_states
            )
            hidden_states = self.cross_attn_norm(
                residual + self.cross_attn_drop(attn_out)
            )

        # FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(residual + self.ffn(hidden_states))
        return hidden_states


class QFormer(nn.Module):
    """
    Querying Transformer that bridges a frozen image encoder and a language model.
    Learnable query tokens cross-attend to image encoder outputs.
    """

    def __init__(
        self,
        num_query_tokens: int = 32,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        intermediate_size: int = 3072,
        cross_attention_every: int = 2,
        dropout: float = 0.1,
        encoder_hidden_size: int = 768,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)

        # Optional projection if vision encoder hidden size differs
        self.encoder_proj = (
            nn.Linear(encoder_hidden_size, hidden_size)
            if encoder_hidden_size != hidden_size
            else nn.Identity()
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            has_cross = (i % cross_attention_every == 0)
            self.layers.append(
                QFormerLayer(
                    hidden_size, num_heads, intermediate_size, has_cross, dropout
                )
            )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states: [B, seq_len, encoder_hidden_size] from image encoder
        Returns:
            [B, num_query_tokens, hidden_size]
        """
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        batch_size = encoder_hidden_states.shape[0]
        hidden_states = self.query_tokens.expand(batch_size, -1, -1)

        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states)

        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MedicalBLIP2(nn.Module):
    """
    BLIP-2 style model for medical VQA and report generation.

    Pipeline:
        image → frozen CLIP ViT → Q-Former cross-attention → projection → T5 encoder-decoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ----- Frozen image encoder -----
        self.image_encoder = CLIPVisionModel.from_pretrained(config.vision_model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_model_name
        )
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        self.image_encoder.eval()

        vision_hidden = self.image_encoder.config.hidden_size  # e.g. 768

        # ----- T5 decoder -----
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.t5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.t5_model_name)
        t5_hidden = self.t5_model.config.d_model  # e.g. 768

        # ----- Q-Former (pretrained from BLIP-2) -----
        # Load pretrained Q-Former weights from Salesforce BLIP-2.
        # This Q-Former was pretrained on 129M image-text pairs with
        # image-text contrastive, matching, and generation objectives.
        # We finetune ALL its weights for our medical domain.
        blip2_model_name = getattr(config, "blip2_model_name",
                                   "Salesforce/blip2-flan-t5-base")
        try:
            from transformers import Blip2Model
            blip2_full = Blip2Model.from_pretrained(blip2_model_name)
            self.qformer = blip2_full.qformer
            # Copy pretrained query tokens
            self.query_tokens = nn.Parameter(
                blip2_full.query_tokens.data.clone()
            )
            # The BLIP-2 Q-Former may have a different hidden size than our config
            qformer_hidden = self.qformer.config.hidden_size  # typically 768
            # Encoder projection: vision_hidden → qformer_hidden
            self.qformer_encoder_proj = (
                nn.Linear(vision_hidden, qformer_hidden)
                if vision_hidden != qformer_hidden
                else nn.Identity()
            )
            del blip2_full  # free memory
            self._using_pretrained_qformer = True
            print(f"  ✓ Loaded pretrained Q-Former from {blip2_model_name}")
        except Exception as e:
            print(f"  ⚠ Could not load pretrained Q-Former ({e}), using random init")
            self.qformer = QFormer(
                num_query_tokens=config.num_query_tokens,
                hidden_size=config.qformer_hidden_size,
                num_heads=config.qformer_num_heads,
                num_layers=config.qformer_num_layers,
                intermediate_size=config.qformer_intermediate_size,
                cross_attention_every=config.qformer_cross_attention_every,
                dropout=config.qformer_dropout,
                encoder_hidden_size=vision_hidden,
            )
            self.query_tokens = None  # custom QFormer has its own
            self.qformer_encoder_proj = nn.Identity()
            qformer_hidden = config.qformer_hidden_size
            self._using_pretrained_qformer = False

        self._qformer_hidden = qformer_hidden

        # ----- Projection: Q-Former → T5 input space -----
        self.projection = nn.Linear(qformer_hidden, t5_hidden)
        self.proj_norm = nn.LayerNorm(t5_hidden)

        # ----- Question Type Classifier (pretrained BERT backbone) -----
        # Instead of a random 2-layer MLP, we use a pretrained BERT encoder
        # as backbone, with a classification head on top.
        # This gives better feature extraction for distinguishing closed vs open questions.
        # All BERT weights are finetuned during training.
        qt_model_name = getattr(config, "qt_bert_model_name",
                                "google-bert/bert-base-uncased")
        try:
            self.qt_bert = BertModel.from_pretrained(qt_model_name)
            self.qt_tokenizer = AutoTokenizer.from_pretrained(qt_model_name)
            qt_hidden = self.qt_bert.config.hidden_size  # 768
            self._using_pretrained_qt = True
            print(f"  ✓ Loaded pretrained BERT for question type from {qt_model_name}")
        except Exception as e:
            print(f"  ⚠ Could not load pretrained BERT ({e}), using lightweight head")
            self.qt_bert = None
            self.qt_tokenizer = None
            qt_hidden = qformer_hidden
            self._using_pretrained_qt = False

        self.question_type_head = nn.Sequential(
            nn.Linear(qt_hidden, qt_hidden // 2),
            nn.GELU(),
            nn.Dropout(config.qformer_dropout),
            nn.Linear(qt_hidden // 2, 2),
        )

        # Build constrained token IDs for closed-ended decoding
        self._closed_answer_ids = None  # lazily built

        # ----- Answer–Report Consistency Head (pretrained NLI backbone) -----
        # Uses a pretrained Natural Language Inference model (DeBERTa-v3 trained
        # on 943K MNLI+SNLI pairs) that already understands entailment vs
        # contradiction between sentence pairs. We finetune all its weights.
        # Output: 3 classes → [contradiction, neutral, entailment]
        # We map entailment → consistent (1), contradiction → inconsistent (0).
        nli_model_name = getattr(config, "nli_model_name",
                                 "cross-encoder/nli-deberta-v3-base")
        try:
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name
            )
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self._using_pretrained_nli = True
            # Keep old consistency_head as a simple adapter on top of NLI logits
            # NLI outputs 3 classes; we project to 1 logit for binary consistency
            self.consistency_head = nn.Linear(3, 1)
            print(f"  ✓ Loaded pretrained NLI model for consistency from {nli_model_name}")
        except Exception as e:
            print(f"  ⚠ Could not load pretrained NLI model ({e}), using T5-CLS fallback")
            self.nli_model = None
            self.nli_tokenizer = None
            self._using_pretrained_nli = False
            self.consistency_head = nn.Sequential(
                nn.Linear(t5_hidden * 2, t5_hidden),
                nn.GELU(),
                nn.Dropout(config.qformer_dropout),
                nn.Linear(t5_hidden, 1),
            )

    # ------------------------------------------------------------------
    # Ensure frozen encoder stays in eval mode
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract visual token embeddings via frozen encoder + Q-Former."""
        with torch.no_grad():
            image_out = self.image_encoder(pixel_values=pixel_values)
            image_features = image_out.last_hidden_state  # [B, 197, 768]

        if self._using_pretrained_qformer:
            # Pretrained BLIP-2 Q-Former: expects encoder_hidden_states and query_embeds
            image_features_proj = self.qformer_encoder_proj(image_features)
            batch_size = image_features.shape[0]
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            # BLIP-2 Q-Former attention mask: all ones for encoder states
            image_attn_mask = torch.ones(
                batch_size, image_features_proj.shape[1],
                dtype=torch.long, device=pixel_values.device,
            )
            qformer_out = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features_proj,
                encoder_attention_mask=image_attn_mask,
                return_dict=True,
            )
            visual_tokens = qformer_out.last_hidden_state  # [B, num_queries, qf_hidden]
        else:
            # Custom Q-Former (fallback)
            visual_tokens = self.qformer(image_features)  # [B, num_queries, qf_hidden]

        visual_embeds = self.proj_norm(self.projection(visual_tokens))  # [B, nq, t5_h]
        return visual_embeds

    def _build_encoder_inputs(self, pixel_values, input_ids, attention_mask):
        """Concatenate visual embeddings with text embeddings for T5 encoder."""
        visual_embeds = self.get_visual_embeds(pixel_values)  # [B, nq, h]
        text_embeds = self.t5_model.encoder.embed_tokens(input_ids)  # [B, tl, h]

        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        batch_size = pixel_values.shape[0]
        vis_mask = torch.ones(
            batch_size,
            visual_embeds.shape[1],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        return inputs_embeds, full_mask

    # ------------------------------------------------------------------
    # Answer–Report Consistency Checking
    # ------------------------------------------------------------------
    def _get_text_cls_embedding(self, text_ids, pixel_values):
        """Encode text through T5 encoder with visual context, return CLS-style pooling.

        Uses the first token position (index 0) as CLS representation,
        similar to BERT's [CLS] token pooling.
        """
        # Build full encoder inputs with visual prefix
        text_embeds = self.t5_model.encoder.embed_tokens(text_ids)
        # Use visual context from the image
        visual_embeds = self.get_visual_embeds(pixel_values)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        batch_size = pixel_values.shape[0]
        vis_mask = torch.ones(batch_size, visual_embeds.shape[1],
                              dtype=torch.long, device=pixel_values.device)
        text_mask = torch.ones(batch_size, text_embeds.shape[1],
                               dtype=torch.long, device=pixel_values.device)
        full_mask = torch.cat([vis_mask, text_mask], dim=1)

        encoder_out = self.t5_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            return_dict=True,
        )
        # CLS pooling: take first token (visual prefix position 0)
        cls_embed = encoder_out.last_hidden_state[:, 0, :]  # [B, h]
        return cls_embed

    def check_consistency(self, pixel_values, answer_ids, report_ids):
        """Check if a VQA answer and a generated report are consistent.

        When pretrained NLI model is available:
            Decodes answer/report tokens to text → feeds as sentence pair to
            DeBERTa NLI → maps entailment probability to consistency score.

        Fallback:
            Uses T5 encoder CLS-token pooling + MLP binary head.

        Args:
            pixel_values: [B, 3, H, W]
            answer_ids: [B, seq_len] tokenized answer text
            report_ids: [B, seq_len] tokenized report text

        Returns:
            logits: [B, 1] raw logits (>0 = consistent)
            probability: [B] probability of consistency
        """
        if self._using_pretrained_nli and self.nli_model is not None:
            # Decode back to text using T5 tokenizer
            answer_texts = self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
            report_texts = self.tokenizer.batch_decode(report_ids, skip_special_tokens=True)

            # Tokenize as sentence pairs for NLI model
            nli_enc = self.nli_tokenizer(
                answer_texts, report_texts,
                return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(pixel_values.device)

            nli_out = self.nli_model(**nli_enc, return_dict=True)
            nli_logits = nli_out.logits  # [B, 3] — contradiction, neutral, entailment

            # Project 3-class NLI logits → 1 binary consistency logit
            logits = self.consistency_head(nli_logits)  # [B, 1]
            prob = torch.sigmoid(logits).squeeze(-1)  # [B]
            return logits, prob
        else:
            # Fallback: T5 CLS-token based approach
            answer_cls = self._get_text_cls_embedding(answer_ids, pixel_values)
            report_cls = self._get_text_cls_embedding(report_ids, pixel_values)

            combined = torch.cat([answer_cls, report_cls], dim=-1)  # [B, 2*h]
            logits = self.consistency_head(combined)  # [B, 1]
            prob = torch.sigmoid(logits).squeeze(-1)  # [B]
            return logits, prob

    # ------------------------------------------------------------------
    # Question Type Classification
    # ------------------------------------------------------------------
    def classify_question_type(self, pixel_values, input_ids, attention_mask):
        """Predict whether the question is closed-ended (0) or open-ended (1).

        Uses pretrained BERT encoder if available, otherwise falls back to
        pooled Q-Former features.

        Returns:
            logits: [B, 2] raw logits
            predicted: [B] — 0=closed, 1=open
            probabilities: [B, 2] — softmax probabilities
        """
        if self._using_pretrained_qt and self.qt_bert is not None:
            # Re-tokenize using BERT tokenizer for proper embeddings.
            # Decode T5 tokens back to text, then encode with BERT tokenizer.
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            bert_enc = self.qt_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=128,
            ).to(pixel_values.device)
            bert_out = self.qt_bert(
                input_ids=bert_enc.input_ids,
                attention_mask=bert_enc.attention_mask,
                return_dict=True,
            )
            # Use BERT [CLS] token as question representation
            pooled = bert_out.last_hidden_state[:, 0, :]  # [B, 768]
        else:
            # Fallback: pool Q-Former output
            with torch.no_grad():
                image_features = self.image_encoder(
                    pixel_values=pixel_values
                ).last_hidden_state
            # Use get_visual_embeds which handles both pretrained and fallback
            visual_embeds = self.get_visual_embeds(pixel_values)
            pooled = visual_embeds.mean(dim=1)

        logits = self.question_type_head(pooled)  # [B, 2]
        probs = F.softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        return logits, predicted, probs

    def _get_closed_answer_ids(self):
        """Build and cache the set of token IDs for closed-ended answers."""
        if self._closed_answer_ids is not None:
            return self._closed_answer_ids
        closed_answers = [
            "yes", "no", "Yes", "No", "YES", "NO",
            '{"answer": "yes"', '{"answer": "no"',
            '{"answer": "Yes"', '{"answer": "No"',
        ]
        all_ids = set()
        for ans in closed_answers:
            ids = self.tokenizer.encode(ans, add_special_tokens=False)
            all_ids.update(ids)
        # Always allow special tokens, punctuation, and JSON structural tokens
        for tok in ['{"', '"', ':', '}', ',', ' ', '\n',
                     'answer', 'explanation', 'true', 'false']:
            ids = self.tokenizer.encode(tok, add_special_tokens=False)
            all_ids.update(ids)
        # Add pad/eos/bos
        for special_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
            if special_id is not None:
                all_ids.add(special_id)
        self._closed_answer_ids = sorted(all_ids)
        return self._closed_answer_ids

    # ------------------------------------------------------------------
    # Forward (training) — with optional question type labels
    # ------------------------------------------------------------------
    def forward(self, pixel_values, input_ids, attention_mask, labels=None,
                question_type_labels=None, consistency_labels=None,
                answer_ids=None, report_ids=None):
        """
        Args:
            pixel_values:   [B, 3, H, W]
            input_ids:      [B, seq_len]   (tokenized task prompt)
            attention_mask: [B, seq_len]
            labels:         [B, tgt_len]   (tokenized target, pad → -100)
            question_type_labels: [B] optional — 0=closed, 1=open
            consistency_labels: [B] optional — 1=answer/report consistent, 0=contradictory
            answer_ids: [B, seq_len] optional — tokenized answer for consistency check
            report_ids: [B, seq_len] optional — tokenized report for consistency check
        Returns:
            Seq2SeqLMOutput (with .loss when labels provided).
            .loss includes auxiliary losses when auxiliary labels are given.
        """
        inputs_embeds, full_mask = self._build_encoder_inputs(
            pixel_values, input_ids, attention_mask
        )
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )

        # Add question type classification loss (multi-task)
        if question_type_labels is not None:
            qt_logits, _, _ = self.classify_question_type(
                pixel_values, input_ids, attention_mask
            )
            # ignore_index=-1 skips report/unknown samples
            qt_loss = F.cross_entropy(qt_logits, question_type_labels, ignore_index=-1)
            # Weight the auxiliary loss
            outputs.loss = outputs.loss + 0.1 * qt_loss

        # Add answer–report consistency loss
        if (consistency_labels is not None and answer_ids is not None
                and report_ids is not None):
            # Only compute for samples with valid labels (>= 0)
            valid_mask = consistency_labels >= 0
            if valid_mask.any():
                cons_logits, _ = self.check_consistency(
                    pixel_values[valid_mask], answer_ids[valid_mask], report_ids[valid_mask]
                )
                cons_loss = F.binary_cross_entropy_with_logits(
                    cons_logits.squeeze(-1),
                    consistency_labels[valid_mask].float(),
                )
                outputs.loss = outputs.loss + 0.1 * cons_loss

        return outputs

    # ------------------------------------------------------------------
    # Generation (inference) — with question type routing & confidence
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, **kwargs):
        """Generate text with optional question-type-aware constrained decoding.

        Extra kwargs:
            constrain_closed (bool): If True, auto-classify question type and
                apply constrained vocabulary for closed-ended questions.
            return_confidence (bool): If True, return (token_ids, confidence) tuple.
        """
        constrain_closed = kwargs.pop("constrain_closed", False)
        return_confidence = kwargs.pop("return_confidence", False)

        inputs_embeds, full_mask = self._build_encoder_inputs(
            pixel_values, input_ids, attention_mask
        )

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            num_beams=kwargs.get("num_beams", 4),
            early_stopping=kwargs.get("early_stopping", True),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 3),
        )

        # If constrained decoding requested, classify question types
        if constrain_closed:
            with torch.no_grad():
                qt_logits, qt_preds, _ = self.classify_question_type(
                    pixel_values, input_ids, attention_mask
                )

            # If ALL samples in batch are closed-ended, apply constrained decoding
            if (qt_preds == 0).all():
                closed_ids = self._get_closed_answer_ids()

                def _prefix_allowed(batch_id, sent):
                    return closed_ids

                gen_kwargs["prefix_allowed_tokens_fn"] = _prefix_allowed
                gen_kwargs["max_new_tokens"] = min(gen_kwargs["max_new_tokens"], 32)

        # Add output_scores for confidence computation
        if return_confidence:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True

        try:
            generated = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                **gen_kwargs,
            )
        except Exception:
            # Fallback: manually compute encoder outputs
            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                return_dict=True,
            )
            generated = self.t5_model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=full_mask,
                **gen_kwargs,
            )

        if return_confidence:
            # Compute confidence from transition scores
            sequences = generated.sequences
            scores = generated.scores  # tuple of [B, vocab] logits per step
            confidence = self._compute_confidence(sequences, scores)
            return sequences, confidence

        return generated

    @torch.no_grad()
    def _compute_confidence(self, sequences, scores):
        """Compute per-sample confidence as mean token probability.

        Args:
            sequences: [B, seq_len] generated token IDs
            scores: tuple of [B, vocab_size] logits, one per generation step

        Returns:
            [B] confidence scores in [0, 1]
        """
        if not scores:
            return torch.ones(sequences.shape[0], device=sequences.device)

        batch_size = sequences.shape[0]
        all_probs = []

        for step_idx, step_scores in enumerate(scores):
            # step_scores: [B * num_beams, vocab] or [B, vocab]
            step_probs = F.softmax(step_scores, dim=-1)
            # Get the token that was actually selected at this step
            token_idx = step_idx + 1  # sequences include the start token
            if token_idx < sequences.shape[1]:
                selected_tokens = sequences[:, token_idx]
                # Gather probabilities of selected tokens
                if step_probs.shape[0] == batch_size:
                    token_probs = step_probs.gather(
                        1, selected_tokens.unsqueeze(1)
                    ).squeeze(1)
                else:
                    # Beam search: take first beam per sample
                    num_beams = step_probs.shape[0] // batch_size
                    beam_probs = step_probs[::num_beams]  # [B, vocab]
                    token_probs = beam_probs.gather(
                        1, selected_tokens.unsqueeze(1)
                    ).squeeze(1)
                all_probs.append(token_probs)

        if not all_probs:
            return torch.ones(batch_size, device=sequences.device)

        # Mean probability across all generated tokens
        prob_stack = torch.stack(all_probs, dim=1)  # [B, num_steps]
        # Mask out padding (prob ~= 0 for pad tokens)
        mask = prob_stack > 1e-8
        confidence = (prob_stack * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return confidence.cpu()

    # ------------------------------------------------------------------
    # Attention map extraction (for visualization)
    # ------------------------------------------------------------------
    def get_cross_attention_map(self, pixel_values):
        """Return averaged cross-attention weights from Q-Former."""
        attn_maps = []

        def _hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                attn_maps.append(out[1].detach().cpu())

        hooks = []
        for layer in self.qformer.layers:
            if layer.has_cross_attention:
                hooks.append(layer.cross_attn.register_forward_hook(_hook))

        with torch.no_grad():
            image_out = self.image_encoder(pixel_values=pixel_values)
            _ = self.qformer(image_out.last_hidden_state)

        for h in hooks:
            h.remove()

        if attn_maps:
            # Average over layers → [B, num_queries, num_patches+1]
            return torch.stack(attn_maps).mean(dim=0)
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
