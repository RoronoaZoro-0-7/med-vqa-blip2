"""
inference.py - Inference engine for Medical BLIP-2.

Usage:
    # VQA inference
    python inference.py --image path/to/image.png --question "Is there a fracture?"

    # Report generation
    python inference.py --image path/to/image.png --task report

    # Batch inference from a JSON file
    python inference.py --batch inputs.json --output results.json
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import torch
from PIL import Image

from utils import Config, parse_json_output, set_seed
from model import MedicalBLIP2


class InferenceEngine:
    """Load a trained Medical BLIP-2 model and run VQA / report generation."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load config
        if config_path and os.path.exists(config_path):
            self.config = Config.load(config_path)
        else:
            self.config = Config()

        # Build model & load weights
        self.model = MedicalBLIP2(self.config)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.image_processor = self.model.image_processor

    # ------------------------------------------------------------------

    def _prepare_image(self, image_input) -> torch.Tensor:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError(f"Unsupported image input: {type(image_input)}")

        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = self.image_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values.to(self.device)

    def _prepare_text(self, text: str):
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        )
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_vqa(
        self, image, question: str, return_raw: bool = False
    ) -> Dict[str, str]:
        """Answer a visual question with confidence score and question type.

        Args:
            image: file path or PIL Image.
            question: clinical question about the image.

        Returns:
            {"answer": "...", "explanation": "...", "confidence": float,
             "question_type": "closed"|"open"}
        """
        pixel_values = self._prepare_image(image)
        input_text = f"Task: VQA Question: {question}"
        input_ids, attention_mask = self._prepare_text(input_text)

        # Classify question type
        with torch.no_grad():
            _, qt_pred, qt_probs = self.model.classify_question_type(
                pixel_values, input_ids, attention_mask
            )
        q_type = "closed" if qt_pred[0].item() == 0 else "open"

        # Generate with confidence and optional constrained decoding
        with torch.no_grad():
            gen_ids, confidence = self.model.generate(
                pixel_values,
                input_ids,
                attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                constrain_closed=(q_type == "closed"),
                return_confidence=True,
            )
        raw = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        conf_score = round(float(confidence[0]), 4)

        result = parse_json_output(raw)
        result["confidence"] = conf_score
        result["question_type"] = q_type

        if return_raw:
            result["raw"] = raw
        return result

    def predict_report(self, image, return_raw: bool = False) -> Dict[str, str]:
        """Generate a medical report for the image with confidence score.

        Args:
            image: file path or PIL Image.

        Returns:
            {"report": "...", "confidence": float}
        """
        pixel_values = self._prepare_image(image)
        input_text = "Task: Report Generate a detailed medical report for this image."
        input_ids, attention_mask = self._prepare_text(input_text)

        with torch.no_grad():
            gen_ids, confidence = self.model.generate(
                pixel_values,
                input_ids,
                attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                return_confidence=True,
            )
        raw = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        conf_score = round(float(confidence[0]), 4)

        result = parse_json_output(raw)
        result["confidence"] = conf_score

        if return_raw:
            result["raw"] = raw
        return result

    def predict(
        self,
        image,
        question: Optional[str] = None,
        task: str = "auto",
    ) -> Dict[str, str]:
        """Unified prediction interface.

        Args:
            image: file path or PIL Image.
            question: question text (only for VQA).
            task: "vqa", "report", or "auto" (auto picks VQA if question given).

        Returns:
            {"answer": "...", "explanation": "...", "report": "..."}
        """
        if task == "auto":
            task = "vqa" if question else "report"

        result = {"answer": "", "explanation": "", "report": "",
                  "confidence": 0.0, "question_type": "", "consistency": None}

        if task == "vqa" and question:
            vqa_result = self.predict_vqa(image, question)
            result.update(vqa_result)

        elif task == "report":
            report_result = self.predict_report(image)
            result.update(report_result)

        elif task == "both" and question:
            vqa_result = self.predict_vqa(image, question)
            result.update(vqa_result)
            report_result = self.predict_report(image)
            result["report"] = report_result.get("report", "")
            # Average confidence across both tasks
            result["confidence"] = round(
                (vqa_result.get("confidence", 0) + report_result.get("confidence", 0)) / 2, 4
            )

            # Answer-Report consistency check
            answer_text = vqa_result.get("answer", result.get("answer", ""))
            report_text = report_result.get("report", result.get("report", ""))
            if answer_text and report_text:
                try:
                    pixel_values = self._prepare_image(image)
                    ans_enc = self.tokenizer(
                        answer_text, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=self.config.max_input_length,
                    )
                    rpt_enc = self.tokenizer(
                        report_text, return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=self.config.max_input_length,
                    )
                    with torch.no_grad():
                        _, cons_prob = self.model.check_consistency(
                            pixel_values,
                            ans_enc.input_ids.to(self.device),
                            rpt_enc.input_ids.to(self.device),
                        )
                    result["consistency"] = round(float(cons_prob[0]), 4)
                except Exception:
                    result["consistency"] = None

        return result

    def batch_predict(self, samples: List[dict]) -> List[dict]:
        """Run inference on a batch of samples.

        Each sample dict should have:
            - "image": path to image
            - "question" (optional): question text
            - "task" (optional): "vqa" | "report" (default: "auto")
        """
        results = []
        for sample in samples:
            image = sample["image"]
            question = sample.get("question")
            task = sample.get("task", "auto")
            try:
                result = self.predict(image, question=question, task=task)
            except Exception as e:
                result = {"answer": "", "explanation": "", "report": "", "error": str(e)}
            result["image"] = image if isinstance(image, str) else "<PIL>"
            result["question"] = question or ""
            results.append(result)
        return results

    def get_attention_map(self, image):
        """Return cross-attention weights for visualization."""
        pixel_values = self._prepare_image(image)
        return self.model.get_cross_attention_map(pixel_values)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Medical BLIP-2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--question", type=str, default=None, help="Question for VQA")
    parser.add_argument("--task", type=str, default="auto", choices=["vqa", "report", "auto", "both"])
    parser.add_argument("--batch", type=str, default=None, help="Path to JSON file with batch inputs")
    parser.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading model …")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    print("Model loaded.")

    # --- Batch mode ---
    if args.batch:
        with open(args.batch) as f:
            samples = json.load(f)
        results = engine.batch_predict(samples)
        output_path = args.output or "inference_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
        return

    # --- Single-image mode ---
    if not args.image:
        parser.error("Provide --image or --batch")

    result = engine.predict(
        image=args.image,
        question=args.question,
        task=args.task,
    )

    print("\n" + "=" * 50)
    print("Inference Result")
    print("=" * 50)
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
