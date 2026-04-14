"""
dataset.py - Dataset downloading, preprocessing, and DataLoader construction.

Supports:
  * VQA-RAD   (HuggingFace datasets)
  * Slake     (HuggingFace datasets / direct download)
  * IU X-Ray  (HuggingFace datasets / synthetic fallback)
"""

import os
import io
import json
import logging
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger("MedVLP")


# ======================================================================
# Image helpers
# ======================================================================

def load_image(source) -> Image.Image:
    """Load an image from a path or PIL Image and convert to RGB."""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (str, Path)):
        img = Image.open(str(source))
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ======================================================================
# Download helpers
# ======================================================================

def _safe_install(package: str):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _save_image(image: Image.Image, path: str):
    image.save(path, "PNG")


# ======================================================================
# Question type detection helper
# ======================================================================

_CLOSED_ANSWERS = {
    "yes", "no", "true", "false", "1", "0",
    "left", "right", "both", "neither",
    "normal", "abnormal",
}

def classify_question_type(answer: str) -> int:
    """Classify a VQA pair as closed-ended (0) or open-ended (1).

    Heuristic: if the answer is in a known closed set or very short (<=3 words),
    it's likely closed-ended.
    """
    ans_lower = answer.strip().lower()
    if ans_lower in _CLOSED_ANSWERS:
        return 0  # closed
    if len(ans_lower.split()) <= 2 and len(ans_lower) <= 10:
        return 0  # likely closed (short answer like "lung", "CT")
    return 1  # open


# ======================================================================
# VQA-RAD
# ======================================================================

def download_vqa_rad(data_dir: str) -> Dict[str, List[dict]]:
    """Download VQA-RAD via HuggingFace ``datasets``."""
    save_dir = os.path.join(data_dir, "vqa_rad")
    cache_file = os.path.join(save_dir, "processed.json")

    if os.path.exists(cache_file):
        logger.info("VQA-RAD: loading from cache")
        with open(cache_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    logger.info("Downloading VQA-RAD from HuggingFace …")
    from datasets import load_dataset

    ds = load_dataset("flaviagiammarino/vqa-rad")

    processed: Dict[str, list] = {"train": [], "test": []}
    for split in ["train", "test"]:
        for idx, sample in enumerate(ds[split]):
            try:
                img = load_image(sample["image"])
                img_path = os.path.join(img_dir, f"{split}_{idx}.png")
                if not os.path.exists(img_path):
                    _save_image(img, img_path)

                answer = str(sample["answer"]).strip()
                question = str(sample["question"]).strip()
                target = json.dumps({"answer": answer, "explanation": ""})
                q_type = classify_question_type(answer)

                processed[split].append(
                    {
                        "image_path": img_path,
                        "input_text": f"Task: VQA Question: {question}",
                        "target_text": target,
                        "task": "vqa",
                        "dataset": "vqa_rad",
                        "question": question,
                        "answer": answer,
                        "question_type": q_type,
                    }
                )
            except Exception as e:
                logger.warning(f"VQA-RAD: skipping sample {split}/{idx}: {e}")

    with open(cache_file, "w") as f:
        json.dump(processed, f)
    logger.info(
        f"VQA-RAD: {len(processed['train'])} train, {len(processed['test'])} test"
    )
    return processed


# ======================================================================
# Slake
# ======================================================================

def download_slake(data_dir: str) -> Dict[str, List[dict]]:
    """Download Slake dataset.  Tries HuggingFace then falls back to a
    lightweight synthetic placeholder so training never crashes."""

    save_dir = os.path.join(data_dir, "slake")
    cache_file = os.path.join(save_dir, "processed.json")

    if os.path.exists(cache_file):
        logger.info("Slake: loading from cache")
        with open(cache_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    processed: Dict[str, list] = {"train": [], "val": [], "test": []}

    # Attempt 1: HuggingFace
    hf_names = [
        "BoKelvin/SLAKE",
        "mdwiratathya/SLAKE",
        "lincoln/SLAKE",
        "Harshil7/slake-vqa",
    ]
    for hf_name in hf_names:
        try:
            logger.info(f"Slake: trying HuggingFace dataset '{hf_name}' …")
            from datasets import load_dataset

            ds = load_dataset(hf_name)
            split_map = {}
            for s in ds.keys():
                sl = s.lower()
                if "train" in sl:
                    split_map[s] = "train"
                elif "val" in sl or "valid" in sl:
                    split_map[s] = "val"
                elif "test" in sl:
                    split_map[s] = "test"
                else:
                    split_map[s] = "train"

            for src_split, tgt_split in split_map.items():
                for idx, sample in enumerate(ds[src_split]):
                    try:
                        # Case-insensitive image key lookup
                        img_key = next(
                            (k for k in sample if k.lower() == "image" or "img" in k.lower()), None
                        )
                        if img_key is None:
                            continue
                        img = load_image(sample[img_key])
                        img_path = os.path.join(
                            img_dir, f"{tgt_split}_{idx}.png"
                        )
                        if not os.path.exists(img_path):
                            _save_image(img, img_path)

                        q_key = next(
                            (k for k in sample if "question" in k.lower()), None
                        )
                        a_key = next(
                            (k for k in sample if "answer" in k.lower()), None
                        )
                        if q_key is None or a_key is None:
                            continue

                        question = str(sample[q_key]).strip()
                        answer = str(sample[a_key]).strip()
                        target = json.dumps(
                            {"answer": answer, "explanation": ""}
                        )
                        q_type = classify_question_type(answer)

                        processed[tgt_split].append(
                            {
                                "image_path": img_path,
                                "input_text": f"Task: VQA Question: {question}",
                                "target_text": target,
                                "task": "vqa",
                                "dataset": "slake",
                                "question": question,
                                "answer": answer,
                                "question_type": q_type,
                            }
                        )
                    except Exception:
                        continue

            if processed["train"]:
                break  # success
        except Exception as e:
            logger.warning(f"Slake: HuggingFace '{hf_name}' failed – {e}")

    if not processed["train"]:
        logger.warning(
            "Slake: all download attempts failed; dataset will be skipped."
        )
        return {"train": [], "val": [], "test": []}

    # Promote val→test if test is empty
    if not processed["test"] and processed["val"]:
        processed["test"] = processed["val"]
        processed["val"] = []

    with open(cache_file, "w") as f:
        json.dump(processed, f)
    total = sum(len(v) for v in processed.values())
    logger.info(f"Slake: {total} samples loaded")
    return processed


# ======================================================================
# IU X-Ray
# ======================================================================

def download_iu_xray(data_dir: str) -> Dict[str, List[dict]]:
    """Download IU X-Ray.  If direct download fails, fall back to
    generating synthetic report-generation data from VQA images."""

    save_dir = os.path.join(data_dir, "iu_xray")
    cache_file = os.path.join(save_dir, "processed.json")

    if os.path.exists(cache_file):
        logger.info("IU X-Ray: loading from cache")
        with open(cache_file) as f:
            return json.load(f)

    os.makedirs(save_dir, exist_ok=True)
    processed: Dict[str, list] = {"train": [], "test": []}

    # Attempt HuggingFace datasets - try actual IU X-Ray and chest X-ray datasets
    hf_names = [
        ("kenza-ily/indiana-university-chest-xray-dicom", None),
        ("hongrui/mimic_chest_xray_v_1", None),
        ("alkzar90/NIH-Chest-X-ray-dataset", None),
    ]
    for hf_name, hf_split in hf_names:
        try:
            logger.info(f"IU X-Ray: trying HuggingFace '{hf_name}' …")
            from datasets import load_dataset

            ds = load_dataset(hf_name, split=hf_split or "train", streaming=True)
            img_dir = os.path.join(save_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            count = 0
            max_samples = 2000  # limit for memory
            for sample in ds:
                if count >= max_samples:
                    break
                try:
                    img_key = next(
                        (k for k in sample if "image" in k.lower()), None
                    )
                    if img_key is None:
                        continue
                    img = load_image(sample[img_key])
                    img_path = os.path.join(img_dir, f"train_{count}.png")
                    if not os.path.exists(img_path):
                        _save_image(img, img_path)

                    label_key = next(
                        (
                            k
                            for k in sample
                            if any(
                                w in k.lower()
                                for w in ["label", "finding", "report", "text"]
                            )
                        ),
                        None,
                    )
                    report = (
                        str(sample[label_key]).strip()
                        if label_key and sample[label_key]
                        else "Normal chest X-ray. No acute cardiopulmonary abnormality."
                    )
                    target = json.dumps({"report": report})

                    split = "train" if count < int(max_samples * 0.85) else "test"
                    processed[split].append(
                        {
                            "image_path": img_path,
                            "input_text": "Task: Report Generate a detailed medical report for this image.",
                            "target_text": target,
                            "task": "report",
                            "dataset": "iu_xray",
                            "report": report,
                            "question_type": -1,
                        }
                    )
                    count += 1
                except Exception:
                    continue

            if processed["train"]:
                break
        except Exception as e:
            logger.warning(f"IU X-Ray: HuggingFace '{hf_name}' failed – {e}")

    # Fallback: generate synthetic report data from existing VQA images
    if not processed["train"]:
        logger.info(
            "IU X-Ray: download failed; generating synthetic report data from VQA images"
        )
        processed = _generate_synthetic_report_data(data_dir)

    if processed["train"]:
        with open(cache_file, "w") as f:
            json.dump(processed, f)
        logger.info(
            f"IU X-Ray / Reports: {len(processed['train'])} train, "
            f"{len(processed['test'])} test"
        )
    return processed


def _generate_synthetic_report_data(data_dir: str) -> Dict[str, list]:
    """Create report-generation training samples from VQA images."""
    processed: Dict[str, list] = {"train": [], "test": []}

    # Gather VQA images already downloaded
    vqa_cache = os.path.join(data_dir, "vqa_rad", "processed.json")
    slake_cache = os.path.join(data_dir, "slake", "processed.json")

    image_qa: Dict[str, list] = {}  # image_path → list of (question, answer)
    for cache_path in [vqa_cache, slake_cache]:
        if not os.path.exists(cache_path):
            continue
        with open(cache_path) as f:
            data = json.load(f)
        for split_data in data.values():
            for s in split_data:
                ip = s.get("image_path", "")
                if ip and os.path.exists(ip):
                    image_qa.setdefault(ip, []).append(
                        (s.get("question", ""), s.get("answer", ""))
                    )

    template_reports = [
        "The medical image examination reveals {findings}. Overall impression: {impression}.",
        "Radiological findings: {findings}. Clinical impression: {impression}.",
        "Upon review, the image demonstrates {findings}. Conclusion: {impression}.",
    ]

    items = list(image_qa.items())
    np.random.shuffle(items)
    for idx, (img_path, qa_pairs) in enumerate(items):
        findings_parts = []
        for q, a in qa_pairs[:5]:
            findings_parts.append(f"{q.rstrip('?')} - {a}")
        findings = "; ".join(findings_parts) if findings_parts else "no significant abnormality"
        impression = (
            "No acute abnormality detected"
            if any(a.lower() in ("no", "normal", "none") for _, a in qa_pairs)
            else "Further clinical correlation recommended"
        )
        report = template_reports[idx % len(template_reports)].format(
            findings=findings, impression=impression
        )
        target = json.dumps({"report": report})
        split = "train" if idx < len(items) * 0.85 else "test"
        processed[split].append(
            {
                "image_path": img_path,
                "input_text": "Task: Report Generate a detailed medical report for this image.",
                "target_text": target,
                "task": "report",
                "dataset": "synthetic_report",
                "report": report,
                "question_type": -1,
            }
        )

    logger.info(f"Synthetic report data: {len(processed['train'])} train, {len(processed['test'])} test")
    return processed


# ======================================================================
# Synthetic QA from reports
# ======================================================================

def generate_synthetic_qa_from_reports(
    report_samples: List[dict],
) -> List[dict]:
    """Rule-based QA pair generation from report text."""
    conditions = {
        "cardiomegaly": "Is there cardiomegaly?",
        "effusion": "Is there pleural effusion?",
        "pneumothorax": "Is there pneumothorax?",
        "mass": "Is there a mass?",
        "nodule": "Is there a nodule?",
        "fracture": "Is there a fracture?",
        "opacity": "Is there an opacity?",
        "consolidation": "Is there consolidation?",
        "edema": "Is there pulmonary edema?",
        "atelectasis": "Is there atelectasis?",
    }
    negation_prefixes = ["no ", "without ", "absence of ", "negative for ", "no evidence of "]

    qa_samples: List[dict] = []
    for sample in report_samples:
        report = sample.get("report", "")
        if not report:
            continue
        report_lower = report.lower()
        img_path = sample.get("image_path", "")

        for cond, question in conditions.items():
            present = cond in report_lower
            negated = any((neg + cond) in report_lower for neg in negation_prefixes)
            if present and not negated:
                answer = "yes"
            else:
                answer = "no"
            target = json.dumps({"answer": answer, "explanation": ""})
            qa_samples.append(
                {
                    "image_path": img_path,
                    "input_text": f"Task: VQA Question: {question}",
                    "target_text": target,
                    "task": "vqa",
                    "dataset": "synthetic_qa",
                    "question": question,
                    "answer": answer,
                    "question_type": 0,
                }
            )

        # Open-ended question
        snippet = report[:200].strip()
        target = json.dumps({"answer": snippet, "explanation": ""})
        qa_samples.append(
            {
                "image_path": img_path,
                "input_text": "Task: VQA Question: What are the findings in this image?",
                "target_text": target,
                "task": "vqa",
                "dataset": "synthetic_qa",
                "question": "What are the findings in this image?",
                "answer": snippet,
                "question_type": 1,
            }
        )
    return qa_samples


# ======================================================================
# Unified Dataset
# ======================================================================

class MedicalVLDataset(Dataset):
    """Unified dataset for medical VQA and report generation."""

    def __init__(self, samples: List[dict], image_processor, max_retries: int = 3):
        self.samples = samples
        self.image_processor = image_processor
        self.max_retries = max_retries

        # Filter out samples with missing images
        valid = []
        for s in self.samples:
            ip = s.get("image_path", "")
            if ip and os.path.exists(ip):
                valid.append(s)
        if len(valid) < len(self.samples):
            logger.warning(
                f"Dataset: dropped {len(self.samples) - len(valid)} samples with missing images"
            )
        self.samples = valid

        # Build image_path → report lookup for consistency checking
        self._image_to_report: Dict[str, str] = {}
        for s in self.samples:
            if s.get("task") == "report":
                rpt = s.get("report", "")
                if rpt:
                    self._image_to_report[s["image_path"]] = rpt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]

        try:
            image = load_image(img_path)
        except Exception:
            # Return a black image as fallback
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_text": sample["input_text"],
            "target_text": sample["target_text"],
            "task": sample["task"],
            "question_type": sample.get("question_type", -1),
            "answer_text": sample.get("answer", ""),
            "report_text": self._image_to_report.get(img_path, ""),
        }


# ======================================================================
# Consistency label computation
# ======================================================================

_POSITIVE_WORDS = {"yes", "present", "positive", "seen", "noted", "identified",
                   "detected", "found", "observed", "bilateral", "abnormal"}
_NEGATIVE_WORDS = {"no", "absent", "negative", "normal", "none", "clear",
                   "unremarkable", "without", "not"}

def _compute_consistency_label(answer: str, report: str) -> int:
    """Determine if a VQA answer is consistent with the report.

    Returns 1 if consistent, 0 if contradictory.

    Logic:
    - Detect positive/negative sentiment of the answer
    - Scan the report for contradictory sentiment
    - E.g., answer="yes" (positive) + report contains "no fracture" → contradictory (0)
    """
    ans_lower = answer.strip().lower()
    rpt_lower = report.strip().lower()
    rpt_words = set(rpt_lower.split())

    # Determine answer sentiment
    ans_positive = any(w in ans_lower.split() for w in _POSITIVE_WORDS)
    ans_negative = any(w in ans_lower.split() for w in _NEGATIVE_WORDS)

    # If answer is too ambiguous, assume consistent
    if not ans_positive and not ans_negative:
        return 1

    # Extract key medical terms from the answer to search in report
    answer_terms = set(ans_lower.split()) - _POSITIVE_WORDS - _NEGATIVE_WORDS - {
        "is", "are", "there", "a", "the", "of", "in", "it", "this", "that",
    }

    if not answer_terms:
        # Simple yes/no answer — check if report is generally positive/negative
        # Use word-level matching to avoid false positives
        rpt_has_positive = bool(rpt_words & _POSITIVE_WORDS)
        rpt_has_negative = bool(rpt_words & _NEGATIVE_WORDS)

        # Also check for common negation phrases
        negation_phrases = ["no ", "without ", "absence of ", "negative for ",
                            "no evidence of ", "not ", "unremarkable"]
        has_negation_phrase = any(neg in rpt_lower for neg in negation_phrases)

        if ans_positive and has_negation_phrase and not (rpt_has_positive and not rpt_has_negative):
            return 0  # contradictory
        if ans_negative and rpt_has_positive and not has_negation_phrase:
            return 0  # contradictory
        return 1  # consistent or neutral

    # Check consistency for each term
    for term in answer_terms:
        if len(term) < 3:
            continue
        if term not in rpt_lower:
            continue
        # Find if the term appears in a negated context in the report
        negation_prefixes = ["no ", "without ", "absence of ", "negative for ",
                             "no evidence of ", "not "]
        term_negated_in_report = any(
            (neg + term) in rpt_lower for neg in negation_prefixes
        )
        term_affirmed_in_report = (
            term in rpt_lower and not term_negated_in_report
        )

        if ans_positive and term_negated_in_report:
            return 0  # answer says yes but report says no
        if ans_negative and term_affirmed_in_report:
            return 0  # answer says no but report says yes

    return 1  # consistent


# ======================================================================
# Collate
# ======================================================================

def create_collate_fn(tokenizer, max_input_length: int = 128, max_target_length: int = 256):
    """Return a collate function closed over the tokenizer."""

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        input_texts = [b["input_text"] for b in batch]
        target_texts = [b["target_text"] for b in batch]
        tasks = [b["task"] for b in batch]
        question_types = [b["question_type"] for b in batch]
        answer_texts = [b["answer_text"] for b in batch]
        report_texts = [b["report_text"] for b in batch]

        input_enc = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        target_enc = tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = target_enc.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        # Question type labels: 0=closed, 1=open, -1=unknown/report
        qt_labels = torch.tensor(question_types, dtype=torch.long)

        # Answer-Report consistency: tokenize answer and report texts,
        # compute consistency labels using sentiment heuristics.
        # Only valid for VQA samples that have a paired report.
        has_consistency = [
            (t == "vqa" and len(a) > 0 and len(r) > 0)
            for t, a, r in zip(tasks, answer_texts, report_texts)
        ]

        if any(has_consistency):
            # Use answer text or a placeholder for non-VQA samples
            ans_for_tok = [a if h else "none" for a, h in zip(answer_texts, has_consistency)]
            rpt_for_tok = [r if h else "none" for r, h in zip(report_texts, has_consistency)]

            answer_enc = tokenizer(
                ans_for_tok, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            )
            report_enc = tokenizer(
                rpt_for_tok, padding=True, truncation=True,
                max_length=max_target_length, return_tensors="pt",
            )

            # Compute consistency labels:
            # 1 = answer sentiment matches report, 0 = contradictory
            # -1 = not applicable (no paired report or not VQA)
            cons_labels = []
            for i, (ans, rpt, valid) in enumerate(
                zip(answer_texts, report_texts, has_consistency)
            ):
                if not valid:
                    cons_labels.append(-1)
                else:
                    cons_labels.append(
                        _compute_consistency_label(ans, rpt)
                    )
            cons_labels_t = torch.tensor(cons_labels, dtype=torch.float)
        else:
            answer_enc = None
            report_enc = None
            cons_labels_t = torch.full((len(batch),), -1, dtype=torch.float)

        result = {
            "pixel_values": pixel_values,
            "input_ids": input_enc.input_ids,
            "attention_mask": input_enc.attention_mask,
            "labels": labels,
            "tasks": tasks,
            "question_type_labels": qt_labels,
            "consistency_labels": cons_labels_t,
        }
        if answer_enc is not None:
            result["answer_ids"] = answer_enc.input_ids
            result["report_ids"] = report_enc.input_ids

        return result

    return collate_fn


# ======================================================================
# DataLoader builder
# ======================================================================

def build_dataloaders(config, image_processor, tokenizer, _logger=None):
    """Download all datasets and return train / val / test DataLoaders."""
    global logger
    if _logger is not None:
        logger = _logger

    all_train: list = []
    all_val: list = []
    all_test: list = []

    # ------ VQA-RAD ------
    try:
        vqa_rad = download_vqa_rad(config.data_dir)
        all_train.extend(vqa_rad.get("train", []))
        all_test.extend(vqa_rad.get("test", []))
    except Exception as e:
        logger.warning(f"VQA-RAD download failed: {e}")

    # ------ Slake ------
    try:
        slake = download_slake(config.data_dir)
        all_train.extend(slake.get("train", []))
        all_val.extend(slake.get("val", []))
        all_test.extend(slake.get("test", []))
    except Exception as e:
        logger.warning(f"Slake download failed: {e}")

    # ------ IU X-Ray (report generation) ------
    try:
        iu = download_iu_xray(config.data_dir)
        all_train.extend(iu.get("train", []))
        all_test.extend(iu.get("test", []))
        # Generate synthetic QA from report data
        report_train = [s for s in iu.get("train", []) if s["task"] == "report"]
        if report_train:
            synth_qa = generate_synthetic_qa_from_reports(report_train)
            all_train.extend(synth_qa)
    except Exception as e:
        logger.warning(f"IU X-Ray download failed: {e}")

    if not all_train:
        raise RuntimeError(
            "No training data available. Check your internet connection."
        )

    # Create val split from train if empty
    if not all_val:
        np.random.shuffle(all_train)
        split_idx = max(1, int(len(all_train) * 0.1))
        all_val = all_train[:split_idx]
        all_train = all_train[split_idx:]

    logger.info(
        f"Dataset sizes  →  train: {len(all_train)}  val: {len(all_val)}  test: {len(all_test)}"
    )

    collate = create_collate_fn(
        tokenizer, config.max_input_length, config.max_target_length
    )

    train_ds = MedicalVLDataset(all_train, image_processor)
    val_ds = MedicalVLDataset(all_val, image_processor)
    test_ds = MedicalVLDataset(all_test, image_processor) if all_test else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    test_loader = None
    if test_ds and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
