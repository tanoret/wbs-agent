#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Avoid bitsandbytes import by forcing PEFT to think bnb is unavailable
import peft
try:
    import peft.import_utils as _peft_import_utils
    _peft_import_utils.is_bnb_available = lambda: False
    import peft.tuners.lora.model as _lora_model
    _lora_model.is_bnb_available = lambda: False
except Exception:
    pass
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="LOCAL base model dir (same used for training)")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (output of training)")
    ap.add_argument("--out", required=True, help="Output directory for merged model")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    merged = model.merge_and_unload()  # fold LoRA weights into base weights

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] Merged model saved to {out_dir}")


if __name__ == "__main__":
    main()
