"""
wbs_core.lora
=============

Offline-friendly LoRA training + merge utilities built for your HPC:

• No HuggingFace Hub calls (all paths are local).
• Avoids bitsandbytes (full/bfloat16 training).
• Works with Python 3.14, CUDA 12.x, L4 24GB GPUs.
• Uses your SFT data format (JSONL with {"prompt": ..., "response": ...}).
• Provides a simple CLI with two subcommands: `train` and `merge`.

Data expectations
-----------------
Your SFT JSONL must have lines like:
    {"prompt": "Design X for Y", "response": "{...JSON WBS...}"}

Recommended training call (example)
-----------------------------------
python -m wbs_core.lora train \
  --base base_files/models/Qwen2.5-3B-Instruct \
  --train base_files/model/data/train.jsonl \
  --val base_files/model/data/val.jsonl \
  --out base_files/model/checkpoints/wbs-lora \
  --epochs 3 --bsz 1 --grad-accum 4 --lr 2e-4 --max-length 2048

Merging adapter into a full model
---------------------------------
python -m wbs_core.lora merge \
  --base base_files/models/Qwen2.5-3B-Instruct \
  --adapter base_files/model/checkpoints/wbs-lora \
  --out base_files/model/merged/qwen2.5-3b-wbs

Notes
-----
• If VRAM is tight, lower --max-length, or use --bsz 1 and higher --grad-accum.
• We default to bfloat16 on CUDA if available; else float32.
• Gradient checkpointing is enabled; model.use_cache is disabled.
• We target Qwen2-style modules by default; override with --lora-targets if needed.
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Iterator, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, PeftModel


# -------------------------
# Utilities
# -------------------------

def _bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


# -------------------------
# Dataset
# -------------------------

class SFTJsonlDataset(Dataset):
    """
    Pre-tokenizes each sample to fixed length so the DataLoader can stack
    tensors trivially (no dynamic pad needed).
    """
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        add_bos: bool = True,
        add_eos: bool = True,
        system_prefix: Optional[str] = None,
        user_prefix: str = "User:",
        assistant_prefix: str = "Assistant:",
    ):
        self.samples: List[Dict] = []
        self.tok = tokenizer
        self.max_length = int(max_length)
        bos = self.tok.bos_token or ""
        eos = self.tok.eos_token or ""

        sp = (system_prefix.strip() + "\n\n") if system_prefix else ""
        up = user_prefix
        ap = assistant_prefix

        # Ensure pad token exists
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        for row in _read_jsonl(jsonl_path):
            prompt = (row.get("prompt") or "").strip()
            response = (row.get("response") or "").strip()
            if not prompt or not response:
                continue

            text = ""
            if add_bos and bos:
                text += bos
            if sp:
                text += sp
            text += f"{up} {prompt}\n{ap} {response}"
            if add_eos and eos:
                text += eos

            enc = self.tok(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]

            # Labels = input_ids, but ignore pad tokens
            labels = []
            for tid, mask in zip(input_ids, attn):
                labels.append(tid if mask == 1 else -100)

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# -------------------------
# Config
# -------------------------

QWEN2_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

@dataclass
class LoraTrainConfig:
    base_model_dir: Path
    out_dir: Path
    train_jsonl: Path
    val_jsonl: Optional[Path] = None

    # Tokenization / sequence
    max_length: int = 2048

    # Optim / schedule
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # LoRA hyperparams
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: QWEN2_LORA_TARGETS)

    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_total_limit: int = 2

    # Dtype & device
    use_bf16: bool = True  # if supported
    gradient_checkpointing: bool = True

    def to_training_args(self) -> TrainingArguments:
        # Conservative TrainingArguments that work on older Transformers as well.
        return TrainingArguments(
            output_dir=str(self.out_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_steps=self.logging_steps,
            save_steps=10_000_000,  # effectively only save at end
            save_total_limit=self.save_total_limit,
            fp16=False,
            bf16=(self.use_bf16 and _bf16_supported()),
            dataloader_num_workers=0,
            report_to=[],
        )


# -------------------------
# Training / Merging
# -------------------------

def train_lora(cfg: LoraTrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    # Load base model/tokenizer strictly from local dir
    tok = AutoTokenizer.from_pretrained(str(cfg.base_model_dir), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if (cfg.use_bf16 and _bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(cfg.base_model_dir),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    # Train-mode knobs
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable") and cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Datasets
    train_ds = SFTJsonlDataset(cfg.train_jsonl, tok, max_length=cfg.max_length)
    eval_ds = SFTJsonlDataset(cfg.val_jsonl, tok, max_length=cfg.max_length) if cfg.val_jsonl and cfg.val_jsonl.exists() else None

    # Since items are fixed-length tensors, default collate (stack) is fine.
    targs = cfg.to_training_args()
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=None,  # dataset already returns fixed-size tensors
    )

    trainer.train()
    # Save LoRA adapter + tokenizer artifacts to cfg.out_dir
    model.save_pretrained(str(cfg.out_dir))
    tok.save_pretrained(str(cfg.out_dir))
    # Also persist training args for reproducibility
    (cfg.out_dir / "lora_train_config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[OK] Saved LoRA adapter to {cfg.out_dir}")

def merge_lora(base_model_dir: Path, adapter_dir: Path, out_dir: Path, dtype: str = "bf16") -> None:
    """
    Merge a LoRA adapter into the base model and save a *standalone* merged model.

    dtype: one of {"bf16","fp16","fp32"} (fp16 is cast to float16 on save)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if dtype == "bf16" and not _bf16_supported():
        print("[WARN] bf16 not supported on this GPU; falling back to fp32")
        dtype = "fp32"

    _dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype, torch.float32)

    tok = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    merged = peft_model.merge_and_unload()  # returns a vanilla CausalLM with weights merged

    # Ensure deterministic dtype on save
    merged.to(dtype=_dtype)
    merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[OK] Saved merged model to {out_dir}")


# -------------------------
# CLI
# -------------------------

def _add_train_subparser(sp: argparse._SubParsersAction):
    ap = sp.add_parser("train", help="Train a LoRA adapter offline from local base model + JSONL SFT data.")
    ap.add_argument("--base", required=True, type=str, help="Path to local base model dir (e.g., base_files/models/Qwen2.5-3B-Instruct)")
    ap.add_argument("--train", required=True, type=str, help="Path to train.jsonl")
    ap.add_argument("--val", type=str, default="", help="Optional path to val.jsonl")
    ap.add_argument("--out", required=True, type=str, help="Output dir for LoRA adapter")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1, help="Per-device train batch size")
    ap.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--logging-steps", type=int, default=10)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", type=str, default=",".join(QWEN2_LORA_TARGETS),
                    help="Comma-separated module names to target (default for Qwen2)")

    ap.add_argument("--no-bf16", action="store_true", help="Force float32 training even if bf16 is supported")
    ap.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    ap.add_argument("--seed", type=int, default=42)

def _add_merge_subparser(sp: argparse._SubParsersAction):
    ap = sp.add_parser("merge", help="Merge a trained LoRA adapter into the base model and save a standalone model.")
    ap.add_argument("--base", required=True, type=str, help="Path to local base model dir")
    ap.add_argument("--adapter", required=True, type=str, help="Path to LoRA adapter dir")
    ap.add_argument("--out", required=True, type=str, help="Output dir for merged model")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16","fp32"], help="Save dtype for merged model")

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline LoRA train/merge utilities")
    sp = ap.add_subparsers(dest="cmd", required=True)
    _add_train_subparser(sp)
    _add_merge_subparser(sp)
    return ap.parse_args()

def _cmd_train(args: argparse.Namespace):
    cfg = LoraTrainConfig(
        base_model_dir=Path(args.base),
        out_dir=Path(args.out),
        train_jsonl=Path(args.train),
        val_jsonl=Path(args.val) if args.val else None,
        max_length=args.max_length,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_ratio=args.warmup_ratio,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=[s.strip() for s in args.lora_targets.split(",") if s.strip()],
        use_bf16=not args.no_bf16,
        gradient_checkpointing=not args.no_grad_checkpoint,
        logging_steps=args.logging_steps,
        seed=args.seed,
    )
    print(f"[INFO] train_examples={sum(1 for _ in _read_jsonl(cfg.train_jsonl))} "
          f"epochs={cfg.num_epochs} bsz={cfg.per_device_train_batch_size} grad_accum={cfg.gradient_accumulation_steps} "
          f"lr={cfg.learning_rate} max_len={cfg.max_length}")
    train_lora(cfg)

def _cmd_merge(args: argparse.Namespace):
    merge_lora(
        base_model_dir=Path(args.base),
        adapter_dir=Path(args.adapter),
        out_dir=Path(args.out),
        dtype=args.dtype,
    )

def main():
    # Prevent any accidental bitsandbytes use (robust on quirky installs)
    os.environ.setdefault("PEFT_DISABLE_BNB", "1")
    args = _parse_args()
    if args.cmd == "train":
        _cmd_train(args)
    elif args.cmd == "merge":
        _cmd_merge(args)
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
