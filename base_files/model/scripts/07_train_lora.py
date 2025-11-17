#!/usr/bin/env python3
import argparse, json, os, sys, math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# ---- load PEFT, then force it to think bnb is NOT available ----
import peft
try:
    import peft.import_utils as _peft_import_utils
    _peft_import_utils.is_bnb_available = lambda: False
    import peft.tuners.lora.model as _lora_model
    _lora_model.is_bnb_available = lambda: False
except Exception:
    pass

from peft import LoraConfig, get_peft_model
# ----------------------------------------------------------------

def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def concat_segments(tokenizer, messages, cutoff_len: int):
    input_ids, attn_mask, labels = [], [], []

    def _append_segment(text, label_all: bool):
        enc = tokenizer(text, add_special_tokens=True, return_tensors=None)
        ids = enc["input_ids"]; am = enc["attention_mask"]
        lab = ids[:] if label_all else [-100] * len(ids)
        return ids, am, lab

    for m in messages:
        role = m["role"]; content = m["content"]
        if role == "user":
            seg = f"<|user|>\n{content}\n<|assistant|>\n"
            ids, am, lab = _append_segment(seg, label_all=False)
        elif role == "assistant":
            seg = f"{content}</s>\n"
            ids, am, lab = _append_segment(seg, label_all=True)
        else:
            seg = f"{role.upper()}:\n{content}\n"
            ids, am, lab = _append_segment(seg, label_all=False)
        input_ids.extend(ids); attn_mask.extend(am); labels.extend(lab)

    if len(input_ids) > cutoff_len:
        input_ids = input_ids[-cutoff_len:]
        attn_mask = attn_mask[-cutoff_len:]
        labels    = labels[-cutoff_len:]

    if all(x == -100 for x in labels):
        last = next((i for i in range(len(labels)-1, -1, -1) if labels[i] != -100), -1)
        if last != -1:
            start = max(0, last - min(256, cutoff_len) + 1)
            input_ids = input_ids[start:last+1]
            attn_mask = attn_mask[start:last+1]
            labels    = labels[start:last+1]

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

class JsonlChatDataset(Dataset):
    def __init__(self, path, tokenizer, cutoff_len):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
        self.rows = [r for r in self.rows if any(m["role"]=="assistant" for m in r.get("messages", []))]
        self.tok = tokenizer
        self.cut = cutoff_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        return concat_segments(self.tok, self.rows[i]["messages"], self.cut)

def make_collator(tokenizer):
    def collate(features):
        inps = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        ams  = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labs = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        batch = {
            "input_ids":      torch.nn.utils.rnn.pad_sequence(inps, batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(ams,  batch_first=True, padding_value=0),
            "labels":         torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100),
        }
        return batch
    return collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data-train", required=True)
    ap.add_argument("--data-val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="model/configs/lora.yaml")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] CUDA GPU not found.")

    alloc = os.environ.get("PYTORCH_ALLOC_CONF", "")
    if "expandable_segments:true" in alloc:
        os.environ["PYTORCH_ALLOC_CONF"] = alloc.replace("true", "True")
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    cfg = load_yaml(args.cfg)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16 if cfg.get("use_bf16", True) else torch.float16,
    )
    if hasattr(model.config, "use_cache"): model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = JsonlChatDataset(args.data_train, tok, cfg["cutoff_len"])
    val_ds   = JsonlChatDataset(args.data_val, tok, cfg["cutoff_len"])
    collate  = make_collator(tok)

    eff_bsz = cfg["batch_size"] * cfg["grad_accum"]
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bsz))
    use_max_steps = cfg.get("max_steps", -1)
    total_steps = (steps_per_epoch * cfg["epochs"]) if (use_max_steps is None or use_max_steps < 0) else use_max_steps
    print(f"[INFO] train_examples={len(train_ds)} eff_batch_size={eff_bsz} "
          f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
          f"epochs={cfg['epochs']} max_steps={use_max_steps}")

    # IMPORTANT: in v5, -1 means "no cap"; 0 means "0 steps"
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=max(1, cfg["batch_size"]//2),
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"] if (use_max_steps is None or use_max_steps < 0) else 1,
        max_steps=(-1 if (use_max_steps is None or use_max_steps < 0) else use_max_steps),
        logging_steps=1,           # log every step
        logging_first_step=True,
        eval_strategy="no",
        save_strategy="no",        # final save only (less I/O)
        bf16=cfg.get("use_bf16", True),
        tf32=cfg.get("use_tf32", True),
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[OK] Saved LoRA adapter to {args.out}")

if __name__ == "__main__":
    main()
