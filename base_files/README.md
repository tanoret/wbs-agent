# WBS-Agent: Regulatory-Grounded WBS Builder for Nuclear Systems

End-to-end pipeline to:

1) **Build a local regulatory corpus** (CFR, RG, SRP) with dense & sparse retrieval  
2) **Generate supervision pairs** (instruction → WBS JSON w/ citations)  
3) **Fine-tune an open-weights LLM via LoRA** to learn the JSON schema & citation style  
4) **Run RAG inference** so answers are grounded in the corpus  
5) **Batch generate & validate** outputs, and optionally **merge** LoRA into a standalone model

Designed to work **offline** (HPC/air-gapped) using locally cloned models and a local sentence-transformer.

---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Environment & Requirements](#environment--requirements)
- [A. Corpus Pipeline](#a-corpus-pipeline)
  - [1) Download raw sources](#1-download-raw-sources)
  - [2) Parse & normalize](#2-parse--normalize)
  - [3) Chunk & embed](#3-chunk--embed)
  - [4) Build indices](#4-build-indices)
  - [5) Ad-hoc retrieval test](#5-ad-hoc-retrieval-test)
- [B. Supervision Signals](#b-supervision-signals)
  - [6) Make instruction pairs](#6-make-instruction-pairs)
  - [7) Prepare SFT datasets](#7-prepare-sft-datasets)
- [C. Model: LoRA Fine-Tuning + RAG](#c-model-lora-fine-tuning--rag)
  - [8) Train LoRA](#8-train-lora)
  - [9) Single-query RAG inference](#9-single-query-rag-inference)
  - [10) Batch RAG inference](#10-batch-rag-inference)
  - [11) Validate outputs](#11-validate-outputs)
  - [12) Merge LoRA (optional)](#12-merge-lora-optional)
- [Configuration & Tuning](#configuration--tuning)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License & Notes](#license--notes)
- [Quick Start](#quick-start)

---

## Repository Layout

```
.
├── corpus
│   ├── manifests/                 # what to download + metadata
│   ├── raw/                       # HTML/PDF originals
│   ├── parsed/                    # normalized JSON per source
│   ├── chunks/                    # small JSONL chunks (+ embeddings)
│   ├── index/                     # FAISS + TF-IDF artifacts
│   ├── models/                    # local sentence-transformer (bge-small-en-v1.5)
│   ├── scripts/                   # corpus pipeline
│   └── README.md
├── model
│   ├── configs/                   # LoRA config (YAML)
│   ├── data/                      # train/val SFT data
│   ├── checkpoints/               # LoRA adapters
│   ├── merged/                    # merged base+LoRA (optional)
│   ├── prompts/                   # example prompts
│   ├── scripts/                   # training + inference + validation
│   └── README.md
├── models/                        # base LLMs (Qwen2.5-7B/7B) cloned locally
└── supervision
    ├── configs/                   # WBS JSON schema + weak label map
    ├── data/                      # generated instruction pairs
    ├── scripts/                   # pair generation
    └── logs/
```

---

## Environment & Requirements

- **Python**: 3.10–3.13 recommended (works on 3.14 with our scripts avoiding `torch.compile` & `bitsandbytes`).  
- **CUDA**: Optional (GPU recommended). Example GPU: NVIDIA L4 (24 GB).  
- **PyTorch**: GPU build matching your CUDA.  
- **Key Python libs**:  
  `transformers`, `accelerate`, `peft`, `datasets`, `sentence-transformers`,  
  `scikit-learn`, `faiss-cpu`/`faiss-gpu`, `numpy`, `pandas`, `tqdm`, `pyyaml`,  
  `joblib`, `markdownify`, `pdfminer.six` (or `pypdf`).

> **Offline/air-gapped notes**
>
> - Clone HF models via **git-LFS**:
>   - Base LLMs → `models/Qwen2.5-7B-Instruct` and/or `models/Qwen2.5-7B-Instruct`
>   - Sentence-transformer → `corpus/models/bge-small-en-v1.5`
> - Set offline flags when needed:
>   ```bash
>   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
>   export TOKENIZERS_PARALLELISM=false
>   ```
> - If your network injects a corporate TLS proxy, prefer **git clone** over Python downloads.
> - We do **not** require `bitsandbytes`. Training runs in fp32/bfloat16.

---

## A. Corpus Pipeline

### 1) Download raw sources

Edit `corpus/manifests/corpus_manifest.yaml` to add/remove sources (CFR, RG, SRP). Mirrors are included to bypass FR/eCFR bot walls.

```bash
python corpus/scripts/01_download.py
```

**Output**: files under `corpus/raw/…`  
If some hosts return **“Request Access”**, mirrors will be used. See stderr lines `[BLOCKED]` and `[FAIL]`.

---

### 2) Parse & normalize

Converts raw HTML/PDF into normalized JSON (title, section ids, text, anchors, etc.).

```bash
python corpus/scripts/02_parse_normalize.py
```

**Output**: one JSON per source in `corpus/parsed/`.  
If manifest entries have no corresponding raw file, you’ll see a summary like:

```
[ERROR] Missing raw files for these manifest entries:
  - id=rg-1-26-rev6 looked for: RG-1.26-Rev6.pdf, rg-1-26-rev6.pdf, ...
```

Fix by adjusting the manifest or re-running download.

---

### 3) Chunk & embed

Splits parsed docs into retrievable chunks and builds embeddings with a **local** sentence-transformer.

```bash
# ensure your local ST model is here:
# corpus/models/bge-small-en-v1.5/
export EMBED_LOCAL_PATH="$(pwd)/corpus/models/bge-small-en-v1.5"

python corpus/scripts/03_chunk_and_embed.py
```

**Output**:
- `corpus/chunks/*.jsonl` – chunk text & metadata
- (optional) embedding cache files, depending on your script version

> If you see a HuggingFace SSL error, the script tried to fetch the model remotely. Ensure `EMBED_LOCAL_PATH` is set and points to your local folder.

---

### 4) Build indices

Creates **FAISS** dense index + **TF-IDF** sparse fallback.

```bash
python corpus/scripts/04_build_faiss.py
```

**Output**:
- `corpus/index/dense_open_bge_small_ip.faiss`  (dense)
- `corpus/index/tfidf_vectorizer.joblib` and `tfidf_matrix.npz`  (sparse)
- `corpus/index/meta.jsonl` (chunk metadata catalog)

---

### 5) Ad-hoc retrieval test

Given a query, print retrieved chunks + citations:

```bash
python corpus/scripts/retrieve.py   --q "seismic design classification for pumps"   --k 5   --corpus-root corpus
```

---

## B. Supervision Signals

### 6) Make instruction pairs

**Your script’s CLI** expects `--chunks-dir`, `--map`, `--out`:

```bash
python supervision/scripts/05_make_supervision.py   --chunks-dir corpus/chunks   --map supervision/configs/weak_label_map.yml   --out supervision/data/instructions.jsonl
```

This synthesizes pairs like:

- **prompt**: “Design seismic classification & qualification for reactor coolant pumps in a PWR”  
- **target**: WBS JSON with tasks, **citations** to CFR/RG/SRP chunks  
- **retrieved_subset**: chunk ids used to build the target

> The weak label map seeds citations for topics (e.g., “seismic” → *10 CFR 50 App S; RG 1.29*), which the generator then expands/validates using the chunked corpus.

---

### 7) Prepare SFT datasets

**Your script’s CLI** expects `--in`, `--out-dir`, `--val-ratio`:

```bash
python model/scripts/06_prepare_sft.py   --in supervision/data/instructions.jsonl   --out-dir model/data   --val-ratio 0.2
```

**Output**:  
- `model/data/train.jsonl`  
- `model/data/val.jsonl`

Each line is a chat-style record matching the base model’s chat template, targeting your **WBS JSON schema**.

---

## C. Model: LoRA Fine-Tuning + RAG

### 8) Train LoRA

**Your training script** accepts: `--base`, `--data-train`, `--data-val`, `--out`, optional `--cfg`.  
Put hyper-parameters in `model/configs/lora.yaml` and pass `--cfg`.

```bash
# pick a local base (7B is much lighter than 7B)
export BASE_MODEL_DIR="$(pwd)/models/Qwen2.5-7B-Instruct"

python model/scripts/07_train_lora.py   --base "${BASE_MODEL_DIR}"   --data-train model/data/train.jsonl   --data-val model/data/val.jsonl   --out model/checkpoints/wbs-lora   --cfg model/configs/lora.yaml
```

Example `model/configs/lora.yaml`:

```yaml
# LoRA adapter config
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]
bias: none
task_type: CAUSAL_LM

# Training loop
epochs: 3               # or set max_steps: 200 to override
max_steps: 0
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4   # eff batch = per_device * grad_accum
learning_rate: 2.0e-4
weight_decay: 0.0
warmup_steps: 0
lr_scheduler_type: cosine

# Sequence & precision
max_seq_len: 2048
dtype: bfloat16          # or float32 if needed
gradient_checkpointing: true
use_cache: false

# Logging / saving
logging_steps: 1
save_every: 0            # 0 = only save at the end
```

**Train longer / faster**  
- Increase `epochs`, or set `max_steps` (takes precedence).  
- Raise `gradient_accumulation_steps` if you’re VRAM-limited.  
- Lower `max_seq_len` to reduce memory.

---

### 9) Single-query RAG inference

Loads base + LoRA and retrieves from `corpus/index/` (FAISS, then TF-IDF fallback).

```bash
python model/scripts/rag_infer.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --q "Design the seismic classification and qualification for the reactor coolant pumps in a PWR"   --k 8   --corpus-root corpus   --max-new-tokens 700   --temperature 0.2
```

**Output**: WBS JSON (with citations) printed to stdout and saved to `model/last_response.json`.

> If you see `[FATAL] No parsed corpus found; cannot run TF-IDF fallback`, ensure steps **A.2–A.4** completed and `corpus/index/` contains artifacts.

---

### 10) Batch RAG inference

Generate for many prompts at once:

```bash
python model/scripts/08_batch_rag_infer.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --input model/prompts/example_prompt.txt   --out model/batch_outputs.jsonl   --corpus-root corpus   --k 8   --max-new-tokens 700
```

Each line of `batch_outputs.jsonl` corresponds to an input prompt; it records the **raw model text** and the **parsed JSON** (when possible).

---

### 11) Validate outputs

Sanity-check schema compliance and retrieved-subset alignment.

```bash
python model/scripts/10_validate_outputs.py   --file model/batch_outputs.jsonl
```

**Output**: a summary object with counts and issues (bad JSON, schema mismatch, empty citations, etc.).

---

### 12) Merge LoRA (optional)

Create a standalone FP32/BF16 model (base ⊕ LoRA) for export/inference without the adapter:

```bash
python model/scripts/09_merge_lora.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --out-dir model/merged/qwen2.5-7b-wbs
```

> Use `--dtype float32` if BF16 isn’t available.

You can then point inference scripts at `model/merged/...` (no `--adapter` needed).

---

## Configuration & Tuning

- **Weak label map**: `supervision/configs/weak_label_map.yml`  
  Maps tokens → seed citations (e.g., *seismic* → 10 CFR 50 App S; RG 1.29). Extend as needed.

- **WBS schema**: `supervision/configs/schema.json`  
  Defines fields (e.g., `tasks[*].name`, `disciplines`, `citations[*]`, `acceptance_criteria`, `dependencies`, etc.).  
  The supervision script populates this shape; LoRA learns to emit it.

- **LoRA config**: `model/configs/lora.yaml`  
  Controls adapter rank/dropout, target modules, and **all** training hyper-params.

- **Retrieval artifacts**:  
  - Dense: `corpus/index/dense_open_bge_small_ip.faiss`  
  - Sparse fallback: `corpus/index/tfidf_vectorizer.joblib`, `tfidf_matrix.npz`  
  - Chunk catalog: `corpus/index/meta.jsonl`

---

## Troubleshooting

**“Request Access” HTML from FederalRegister/eCFR**  
Use the mirrors in the manifest. The downloader auto-skips known FR/eCFR bot walls.

**HuggingFace SSL or rate-limit errors**  
Work completely **offline**:
- Clone models with `git lfs clone ...` into `models/` and `corpus/models/`.
- Set:
  ```bash
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  ```
- For sentence-transformers, ensure `EMBED_LOCAL_PATH` points to `corpus/models/bge-small-en-v1.5`.

**bitsandbytes / torch.compile / Python 3.14**  
The scripts avoid `bitsandbytes` and `torch.compile`. If you still see bnb imports, ensure you’re on the updated scripts and there’s no lingering installation pulling it in at runtime.

**CUDA OOM**  
Use the **7B** base, lower `max_seq_len`, raise `gradient_accumulation_steps`, or train on CPU (slower, but safe).

**FAISS not found / TF-IDF fatal**  
Run **03** and **04** to build embeddings and indices. Verify `corpus/index/` exists.

**Only a few training steps**  
Increase `epochs` or set `max_steps` in `model/configs/lora.yaml`.

---

## FAQ

**Q: Can I add more RGs/NUREGs/IAEA docs?**  
Yes—add to `corpus/manifests/corpus_manifest.yaml`, then re-run steps **A.1–A.4**.

**Q: How are citations enforced?**  
RAG retrieves top-k chunks; prompts require the model to cite chunk ids/anchors. Validation checks the output JSON and referenced ids.

**Q: Can I switch base models?**  
Yes—clone another instruct-tuned 7B–7B model locally and set `BASE_MODEL_DIR` to its path.

**Q: How do I control JSON fidelity?**  
Schema is in `supervision/configs/schema.json`. Keep supervision targets strict; increase training steps; set temperature ~0.2 at inference.

---

## License & Notes

- U.S. Government works (CFR, NUREGs, SRP) are generally public domain; always verify the status of any third-party material before redistribution.
- This repository is for research/prototyping only. It does **not** substitute for licensed engineering judgment or formal regulatory submissions.

---

## Quick Start

```bash
# 0) Offline env (optional but recommended)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 1) Corpus
python corpus/scripts/01_download.py
python corpus/scripts/02_parse_normalize.py
export EMBED_LOCAL_PATH="$(pwd)/corpus/models/bge-small-en-v1.5"
python corpus/scripts/03_chunk_and_embed.py
python corpus/scripts/04_build_faiss.py

# 2) Supervision
python supervision/scripts/05_make_supervision.py   --chunks-dir corpus/chunks   --map supervision/configs/weak_label_map.yml   --out supervision/data/instructions.jsonl

python model/scripts/06_prepare_sft.py   --in supervision/data/instructions.jsonl   --out-dir model/data   --val-ratio 0.2

# 3) Train LoRA
export BASE_MODEL_DIR="$(pwd)/models/Qwen2.5-7B-Instruct"
python model/scripts/07_train_lora.py   --base "${BASE_MODEL_DIR}"   --data-train model/data/train.jsonl   --data-val model/data/val.jsonl   --out model/checkpoints/wbs-lora   --cfg model/configs/lora.yaml

# 4) Single-query RAG
python model/scripts/rag_infer.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --q "Design the seismic classification and qualification for the reactor coolant pumps in a PWR"   --k 8 --corpus-root corpus --max-new-tokens 700 --temperature 0.2

# 5) Batch + validate
python model/scripts/08_batch_rag_infer.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --input model/prompts/example_prompt.txt   --out model/batch_outputs.jsonl   --corpus-root corpus --k 8 --max-new-tokens 700

python model/scripts/10_validate_outputs.py   --file model/batch_outputs.jsonl

# 6) (Optional) Merge LoRA → standalone
python model/scripts/09_merge_lora.py   --base "${BASE_MODEL_DIR}"   --adapter model/checkpoints/wbs-lora   --out-dir model/merged/qwen2.5-7b-wbs
```
