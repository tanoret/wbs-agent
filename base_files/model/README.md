# Step C — Model (LoRA fine-tune + RAG inference)

This folder contains:
- LoRA fine-tuning (7–13B class instruct models, offline-friendly)
- Data prep from `supervision/data/instructions.jsonl`
- RAG inference that always retrieves top-k chunks from `corpus/index` and **forces citations**

## Quick start

### 0) Choose a base instruct model (offline if possible)
Recommended: `Qwen2.5-7B-Instruct` (open, strong).
Download via git-lfs to `models/Qwen2.5-7B-Instruct` or similar, or use your cluster's mirror.
Then:
```bash
export BASE_MODEL_DIR=models/Qwen2.5-7B-Instruct   # adjust path if elsewhere
```

Or:
```bash
export BASE_MODEL_DIR="$(pwd)/models/Qwen2.5-3B-Instruct"   # adjust path if elsewhere
```

### 1) Prepare SFT data
```bash
python model/scripts/06_prepare_sft.py   --in supervision/data/instructions.jsonl   --out-dir model/data   --val-ratio 0.06
```

### 2) Train LoRA (QLoRA if bitsandbytes is available)
```bash
# Optional: edit configs/lora.yaml first
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
python model/scripts/07_train_lora.py   --base "${BASE_MODEL_DIR}"   --data-train model/data/train.jsonl   --data-val model/data/val.jsonl   --out model/checkpoints/wbs-lora   --cfg model/configs/lora.yaml
```

python model/scripts/07_train_lora.py --base "${BASE_MODEL_DIR}" --data-train model/data/train.jsonl --data-val model/data/val.jsonl --out model/checkpoints/wbs-lora

### 3) RAG inference (uses FAISS if present else TF-IDF)
```bash
# EMBED_LOCAL_PATH is only needed for FAISS retrieval; TF-IDF works offline.
export EMBED_LOCAL_PATH=models/bge-small-en-v1.5  # if you built FAISS
python model/scripts/rag_infer.py   --adapter model/checkpoints/wbs-lora   --base "${BASE_MODEL_DIR}"   --q "Design seismic for SMR"   --k 6
```

Outputs JSON with a WBS and citations to your local chunks.

## Notes
- Training is GPU-oriented. If no GPU is available, the script exits with a clear message.
- Everything is offline-first: no downloads at train/infer time.
