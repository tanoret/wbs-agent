# WBS-Agent — Reactor Engineering WBS Builder (RAG + LoRA)

End-to-end toolkit and Streamlit app to:

1) curate a regulatory/technical **corpus**,  
2) build **TF‑IDF/FAISS** indices (with local sentence embeddings),  
3) generate **weak supervision** and **SFT** splits,  
4) train a **LoRA** adapter on a local base model (Qwen2.5‑xB‑Instruct),  
5) run **RAG inference** to produce a **Work Breakdown Structure (WBS)** with citations, and  
6) visualize tasks as **boxes** in a web app.

> ✅ Designed to work **offline** with local models and embeddings.  
> ✅ Falls back to **TF‑IDF** if FAISS isn’t present.  
> ✅ No `bitsandbytes` required (Python 3.11+ friendly).


---

## Repository Layout

```
.
├── base_files
│   ├── corpus
│   │   ├── raw/            # downloaded PDFs/HTML
│   │   ├── parsed/         # normalized JSON from raw
│   │   ├── chunks/         # chunked + embedded JSONL
│   │   ├── index/          # FAISS + TF‑IDF artifacts
│   │   ├── models/         # local sentence-embedding model (e.g., bge-small-en-v1.5)
│   │   ├── manifests/      # manifest + metadata
│   │   └── scripts/        # 01..04 corpus pipeline
│   ├── model
│   │   ├── scripts/        # 06..10 model pipeline + rag_infer.py (single-prompt)
│   │   ├── checkpoints/    # LoRA adapters (e.g., wbs-lora)
│   │   ├── merged/         # merged base+LoRA full models
│   │   ├── data/           # SFT train/val splits
│   │   ├── prompts/        # batch prompts
│   │   ├── configs/        # LoRA config
│   │   ├── exports/        # WBS JSON/DOT exports
│   │   └── last_response.json
│   ├── models/             # local base models (Qwen2.5‑3B/7B‑Instruct)
│   ├── supervision/
│   │   ├── scripts/        # 05_make_supervision.py
│   │   ├── configs/        # schema + weak label map
│   │   └── data/           # generated instruction pairs
│   └── README.md
├── wbs_core/               # Python library (pipeline helpers)
├── wbs_app/                # Streamlit app (pages 1–5)
│   ├── streamlit_app.py    # landing page + status/shortcuts
│   └── pages/              # step-by-step UI pages
└── model/last_response.json# (optional) most-recent single-prompt snapshot
```


---

## Quickstart

### 1) Environment (recommended: Python **3.11**)

Create a fresh venv/conda and install:

```bash
# e.g., conda:
conda create -n wbs-agent python=3.11 -y
conda activate wbs-agent

# minimal core deps
pip install --upgrade pip

# Retrieval & utils
pip install scikit-learn joblib numpy scipy pandas faiss-cpu==1.8.0.post1

# Transformers stack (CPU ok; CUDA if available)
pip install "torch==2.3.*" --index-url https://download.pytorch.org/whl/cpu
pip install "transformers==4.44.*" "accelerate==0.33.*" "peft==0.12.*" "sentence-transformers==2.7.*"

# Web app
pip install streamlit==1.31.0 altair==5.2.0 pyarrow>=14.0.0 watchdog

# Graphviz rendering in app (optional but recommended)
# RHEL/CentOS:
sudo yum -y install graphviz || true
# Ubuntu/Debian:
# sudo apt-get update && sudo apt-get install -y graphviz
```

> **Notes**
> - We avoid `bitsandbytes`. The included `rag_infer.py` forces PEFT to behave as if bnb is unavailable.  
> - If you’re fully **offline**, set:
>   ```bash
>   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
>   ```


### 2) Local Models & Embeddings

Place local base LLMs under `base_files/models/` (e.g., `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`).  
Place a sentence-embedding model under `base_files/corpus/models/` (e.g., `bge-small-en-v1.5`).


---

## Pipeline — Command Line (canonical route)

All scripts run **locally** and take paths relative to the repo root.

### A) Corpus

```bash
# 01 Download raw docs → base_files/corpus/raw/
python base_files/corpus/scripts/01_download.py

# 02 Parse & normalize → base_files/corpus/parsed/
python base_files/corpus/scripts/02_parse_normalize.py \
  --raw base_files/corpus/raw \
  --out base_files/corpus/parsed

# 03 Chunk + embed → base_files/corpus/chunks/
export EMBED_LOCAL_PATH="$(pwd)/base_files/corpus/models/bge-small-en-v1.5"
python base_files/corpus/scripts/03_chunk_and_embed.py \
  --parsed base_files/corpus/parsed \
  --out base_files/corpus/chunks

# 04 Build FAISS / TF‑IDF → base_files/corpus/index/
python base_files/corpus/scripts/04_build_faiss.py \
  --chunks base_files/corpus/chunks \
  --index-dir base_files/corpus/index
```

> If FAISS files are missing, RAG falls back to **TF‑IDF** automatically.


### B) Supervision + SFT

```bash
# 05 Generate supervision instruction pairs → base_files/supervision/data/instructions.jsonl
python base_files/supervision/scripts/05_make_supervision.py \
  --chunks-dir base_files/corpus/chunks \
  --map base_files/supervision/configs/weak_label_map.yml \
  --out base_files/supervision/data/instructions.jsonl \
  --num 500

# 06 Prepare SFT train/val → base_files/model/data/train.jsonl + val.jsonl
python base_files/model/scripts/06_prepare_sft.py \
  --in base_files/supervision/data/instructions.jsonl \
  --out-dir base_files/model/data \
  --val-ratio 0.2
```


### C) LoRA Training

```bash
# 07 Train LoRA (config-driven)
python base_files/model/scripts/07_train_lora.py \
  --base base_files/models/Qwen2.5-7B-Instruct \
  --data-train base_files/model/data/train.jsonl \
  --data-val base_files/model/data/val.jsonl \
  --out base_files/model/checkpoints/wbs-lora \
  --cfg base_files/model/configs/lora.yaml
```

> `07_train_lora.py` reads **epochs**, **batch size**, **gradient accumulation**, **learning rate**, etc. from the YAML.  
> Adjust values in `base_files/model/configs/lora.yaml`.


### D) Batch RAG & Validation

```bash
# 08 Batch RAG inference (prompts → outputs.jsonl)
python base_files/model/scripts/08_batch_rag_infer.py \
  --base base_files/models/Qwen2.5-7B-Instruct \
  --adapter base_files/model/checkpoints/wbs-lora \
  --input base_files/model/prompts/example_prompt.txt \
  --out base_files/model/batch_outputs.jsonl \
  --corpus-root base_files/corpus \
  --k 8 --max-new-tokens 700 --temperature 0.2

# 10 Validate JSONL outputs (schema / JSON sanity)
python base_files/model/scripts/10_validate_outputs.py \
  --file base_files/model/batch_outputs.jsonl
```


### E) Merge LoRA → Full Model (optional)

```bash
# 09 Merge into a standalone model → base_files/model/merged/<name>
python base_files/model/scripts/09_merge_lora.py \
  --base base_files/models/Qwen2.5-7B-Instruct \
  --adapter base_files/model/checkpoints/wbs-lora \
  --out base_files/model/merged/qwen2.5-7b-wbs
```

You can then use the merged model for inference (no adapter needed).


---

## Single‑Prompt RAG Inference (CLI)

```bash
# Uses TF‑IDF or FAISS if available; writes model/last_response.json
python base_files/model/scripts/rag_infer.py \
  --base base_files/models/Qwen2.5-7B-Instruct \
  --adapter base_files/model/checkpoints/wbs-lora \
  --q "Design the seismic classification and qualification for the reactor coolant pumps in a PWR" \
  --k 8 \
  --corpus-root base_files/corpus \
  --max-new-tokens 700 \
  --temperature 0.2
```

- Output snapshot: `base_files/model/last_response.json` (and/or `model/last_response.json`) with fields:
  - `query`: your prompt
  - `retrieved`: list of chunks/citations used
  - `output`: raw text that generally ends with a JSON WBS object (app also supports `output_text`)

**WBS JSON schema (typical)**

```json
{
  "wbs_title": "Seismic Qualification of Reactor Coolant Pumps",
  "tasks": [
    {
      "name": "Seismic Classification",
      "description": "Classify pumps per RG 1.29 / GDC 2 / §50.55a.",
      "applicable_codes": ["10 CFR 50 Appendix A, GDC 2", "10 CFR 50.55a"],
      "guidance": "Assign safety classes; document design basis SSE input.",
      "citations": [{"id": "cfr-10-50-app-a#14"}, {"id": "cfr-10-50-55a#83"}]
    }
  ]
}
```

The app will parse this JSON and draw **task boxes** automatically.


---

## Streamlit App

Launch:

```bash
streamlit run wbs_app/streamlit_app.py
```

- **Page 1 – Manifest & Data:** optionally redefine/upload corpus manifest.  
- **Page 2 – Build Corpus:** run 01→04 (download, parse, chunk+embed, build index).  
- **Page 3 – Supervision & SFT:** run 05→06 (instruction pairs, SFT splits).  
- **Page 4 – Train & Evaluate:** train LoRA (07), batch infer (08), validate (10), merge (09).  
- **Page 5 – WBS Generator:** choose model (merged or base+adapter), run **single‑prompt RAG**, render **WBS graph** + **citations**, and **export** JSON/DOT.

### Known‑good versions

- `streamlit==1.31.0` + `altair==5.2.0`  
  (Older `streamlit` like 1.17 expects `altair.v4`, newer `altair>=6` can break built‑ins.)

If you see an Altair schema error, pin `altair==5.2.0`.


---

## Troubleshooting

- **Transformers/PEFT on Python 3.14:** install Python 3.11 for compatibility.  
- **`bitsandbytes` / `torch.compile` errors:** this project does **not** use `bitsandbytes`. The provided `rag_infer.py` disables bnb paths in PEFT.
- **FAISS not found:** ensure `faiss-cpu==1.8.0.post1` is installed. If index files are missing, we automatically fall back to TF‑IDF.
- **CUDA/CPU:** by default we install CPU‑only torch. For CUDA, install a matching `torch` wheel from PyTorch.
- **Graphviz not installed:** app still runs, but Graphviz export is nicer with the system package installed.
- **Duplicate widget IDs (Streamlit):** each input on a page needs a unique `key=` if you duplicate labels.

---

## Git & Large Files

A suggested `.gitignore`:

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env

# Streamlit
.streamlit/

# Data & model artifacts
base_files/corpus/raw/
base_files/corpus/parsed/
base_files/corpus/chunks/
base_files/corpus/index/
base_files/model/checkpoints/
base_files/model/merged/
base_files/model/data/
base_files/model/exports/
base_files/model/last_response.json
model/last_response.json
```

If you prefer **Git LFS** for model binaries:

```bash
git lfs install
# track large model shards:
git lfs track "*.safetensors"
git lfs track "*.bin"
git lfs track "base_files/corpus/index/*.faiss"
git add .gitattributes
git commit -m "Track large artifacts via LFS"
```

> Tip: Keep the repo small; don’t commit large models unless required.


---

## Repro Tips

- Always verify `EMBED_LOCAL_PATH` is set before chunking/FAISS build and during RAG:  
  ```bash
  export EMBED_LOCAL_PATH="$(pwd)/base_files/corpus/models/bge-small-en-v1.5"
  ```
- Ensure **base LLM** path is correct and readable by Transformers.
- If your kernel is older than 5.5 you may see warnings; training still proceeds on many distros.

---

## License / Attribution

This repository wires together standard open-source components (Transformers, PEFT, Sentence‑Transformers, FAISS, Streamlit).  
Respect licenses of your local models and any ingested documents.

Author: Mauricio E. Tano

–––

Happy building! Generate grounded WBS task structures for complex reactor engineering prompts, with traceable citations from your regulatory corpus.
