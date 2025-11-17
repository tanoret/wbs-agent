WBS-Agent (Core + Base Files) — Full Documentation
==================================================

This repository contains a local-first pipeline that builds a regulatory corpus, creates weakly supervised instruction pairs, fine-tunes an open-weights LLM with LoRA, and performs RAG-grounded inference to generate a Work Breakdown Structure (WBS) with citations for nuclear engineering/design prompts.

It is designed to operate fully offline after you’ve cloned your base models and built your local corpus. No paid APIs are required.

----------------------------------------------------------------

Table of Contents
-----------------

- Repository Layout
- Quickstart (Two Ways)
  - Programmatic with wbs_core
  - CLI with base_files scripts
- wbs_core Module Reference
- Worked Examples
- Bring Your Own Documents & Manifest
- Training Advice & Tuning
- Troubleshooting
- Security & Regulatory Notice
- License

----------------------------------------------------------------

Repository Layout
-----------------

.
├── base_files/
│   ├── corpus/        # raw/parsed/chunks/index + manifest + local embed model
│   ├── model/         # LoRA checkpoints, merged model, data, scripts
│   ├── models/        # Base models (Qwen*), cloned locally
│   └── supervision/   # Weak label configs + generated supervision data
├── wbs_core/          # Library-style core modules
└── wbs_app/           # (Next step) Streamlit UI

Keep using base_files/*/scripts/*.py as CLIs, or call the equivalent features via the new wbs_core package.

----------------------------------------------------------------

Quickstart (Two Ways)
---------------------

A) Programmatic with wbs_core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal example using your existing corpus under base_files/corpus.

Python example:

    from pathlib import Path
    from wbs_core.supervision import SupervisionBuilder
    from wbs_core.sft import SFTPreparer
    from wbs_core.lora import LoraTrainer
    from wbs_core.rag import RAGInfer

    ROOT = Path.cwd()
    CORPUS = ROOT / "base_files" / "corpus"
    MODELS = ROOT / "base_files" / "models"
    MODEL_DIR = MODELS / "Qwen2.5-3B-Instruct"  # prefer 3B for a single L4
    SUP_MAP = ROOT / "base_files" / "supervision" / "configs" / "weak_label_map.yml"

    # 1) Supervision (weakly-labeled instruction pairs)
    sup = SupervisionBuilder(CORPUS / "chunks", CORPUS / "index", SUP_MAP, seed=42, k=6)
    recs = sup.build(num=60, disciplines=["seismic","qa","ep"], reactor_types=["PWR","BWR"])
    sup_out = ROOT / "base_files" / "supervision" / "data" / "instructions.jsonl"
    sup.write_jsonl(sup_out, recs)

    # 2) Prepare SFT
    SFTPreparer().prepare(
        inp=sup_out,
        out_dir=ROOT / "base_files" / "model" / "data",
        val_ratio=0.2, seed=42
    )

    # 3) LoRA training (offline-friendly)
    trainer = LoraTrainer(
        base_model_dir=MODEL_DIR,
        out_dir=ROOT / "base_files" / "model" / "checkpoints" / "wbs-lora",
        dtype="bfloat16",
        gradient_checkpointing=True
    )
    trainer.fit(
        train_jsonl=ROOT / "base_files" / "model" / "data" / "train.jsonl",
        val_jsonl=ROOT / "base_files" / "model" / "data" / "val.jsonl",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        epochs=3,
        logging_steps=1
    )

    # 4) RAG-grounded inference
    infer = RAGInfer(
        base_model_dir=MODEL_DIR,
        adapter_dir=ROOT / "base_files" / "model" / "checkpoints" / "wbs-lora",
        corpus_root=CORPUS,
        embed_model_dir=CORPUS / "models" / "bge-small-en-v1.5",
        use_faiss=True
    )
    resp = infer.generate(
        query="Design the seismic classification and qualification for the reactor coolant pumps in a PWR",
        k=8, max_new_tokens=700, temperature=0.2
    )
    print(resp["text"])
    print(resp["citations"])

B) CLI with base_files scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # 1) Build/refresh corpus (skip if already built)
    python base_files/corpus/scripts/01_download.py
    python base_files/corpus/scripts/02_parse_normalize.py
    python base_files/corpus/scripts/03_chunk_and_embed.py
    python base_files/corpus/scripts/04_build_faiss.py

    # 2) Supervision
    python base_files/supervision/scripts/05_make_supervision.py \
      --chunks-dir base_files/corpus/chunks \
      --index-dir base_files/corpus/index \
      --weak-map base_files/supervision/configs/weak_label_map.yml \
      --num 120 \
      --out base_files/supervision/data/instructions.jsonl

    # 3) Prepare SFT
    python base_files/model/scripts/06_prepare_sft.py \
      --in base_files/supervision/data/instructions.jsonl \
      --out-dir base_files/model/data \
      --val-ratio 0.2

    # 4) Train LoRA (offline)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    python base_files/model/scripts/07_train_lora.py \
      --base "$(pwd)/base_files/models/Qwen2.5-3B-Instruct" \
      --data-train base_files/model/data/train.jsonl \
      --data-val base_files/model/data/val.jsonl \
      --out base_files/model/checkpoints/wbs-lora

    # 5) Batch RAG + validate
    python base_files/model/scripts/08_batch_rag_infer.py \
      --base "$(pwd)/base_files/models/Qwen2.5-3B-Instruct" \
      --adapter base_files/model/checkpoints/wbs-lora \
      --input base_files/model/prompts/example_prompt.txt \
      --out base_files/model/batch_outputs.jsonl \
      --corpus-root base_files/corpus \
      --k 8 --max-new-tokens 700

    python base_files/model/scripts/10_validate_outputs.py \
      --file base_files/model/batch_outputs.jsonl

    # 6) (Optional) Merge LoRA → full model
    python base_files/model/scripts/09_merge_lora.py \
      --base "$(pwd)/base_files/models/Qwen2.5-7B-Instruct" \
      --adapter base_files/model/checkpoints/wbs-lora \
      --out base_files/model/merged/qwen2.5-7b-wbs

----------------------------------------------------------------

wbs_core Module Reference
-------------------------

- __init__.py — small helpers: logger, ModelPaths, ensure_dir, timestamp.
- downloader.py — fetches raw docs per manifest (skips eCFR/FR block pages).
- parser.py — HTML/PDF → normalized JSON (parsed/).
- chunker.py — normalized JSON → chunks (+ optional embeddings) in chunks/.
- indexer.py — builds TF-IDF + FAISS (writes to index/).
- supervision.py — weakly labeled instruction pairs with retrieved context.
- sft.py — formats instruction pairs to SFT records; splits train/val.
- lora.py — LoRA finetune wrapper (no bitsandbytes requirement).
- rag.py — retrieval-augmented generation with citations.
- manifest.py — helpers to read/validate corpus manifests.
- pipeline.py — orchestrates ETL → index, idempotently.
- retrievy.py — tiny retrieval utilities (name is intentional).

----------------------------------------------------------------

Worked Examples
---------------

A) Generate 60 supervision pairs (Seismic/QA/EP; PWR, BWR)

    python -m wbs_core.supervision \
      --chunks-dir base_files/corpus/chunks \
      --index-dir base_files/corpus/index \
      --weak-map base_files/supervision/configs/weak_label_map.yml \
      --disciplines seismic,qa,ep \
      --reactor-types PWR,BWR \
      --k 6 \
      --num 60 \
      --out base_files/supervision/data/instructions.jsonl

B) Prepare SFT

    python - <<'PY'
    from pathlib import Path
    from wbs_core.sft import SFTPreparer

    inp = Path("base_files/supervision/data/instructions.jsonl")
    out_dir = Path("base_files/model/data")
    SFTPreparer().prepare(inp, out_dir, val_ratio=0.2, seed=42)
    print("Wrote train/val JSONL into", out_dir)
    PY

C) LoRA fine-tune on Qwen 3B

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false

    python - <<'PY'
    from pathlib import Path
    from wbs_core.lora import LoraTrainer

    trainer = LoraTrainer(
        base_model_dir=Path("base_files/models/Qwen2.5-3B-Instruct"),
        out_dir=Path("base_files/model/checkpoints/wbs-lora"),
        dtype="bfloat16",
        gradient_checkpointing=True,
    )
    trainer.fit(
        train_jsonl=Path("base_files/model/data/train.jsonl"),
        val_jsonl=Path("base_files/model/data/val.jsonl"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        epochs=3,
        logging_steps=1,
    )
    print("Done.")
    PY

D) RAG-grounded single prompt

    python - <<'PY'
    from pathlib import Path
    from wbs_core.rag import RAGInfer

    infer = RAGInfer(
        base_model_dir=Path("base_files/models/Qwen2.5-3B-Instruct"),
        adapter_dir=Path("base_files/model/checkpoints/wbs-lora"),
        corpus_root=Path("base_files/corpus"),
        embed_model_dir=Path("base_files/corpus/models/bge-small-en-v1.5"),
        use_faiss=True
    )

    resp = infer.generate(
        query="Design the seismic classification and qualification for the reactor coolant pumps in a PWR",
        k=8, max_new_tokens=700, temperature=0.2
    )
    print(resp["text"])
    print("Citations:", resp["citations"])
    PY

----------------------------------------------------------------

Bring Your Own Documents & Manifest
-----------------------------------

Example YAML manifest:

    documents:
      - id: cfr-10-50-app-s
        title: 10 CFR Part 50, Appendix S
        url: https://example.gov/cfr_10_part50_appendixS.html
        mirrors:
          - https://mirror1.example/cfr_10_part50_appendixS.html
          - https://mirror2.example/cfr_10_part50_appendixS.html

- Use wbs_core.pipeline.Pipeline to run download → parse → chunk → index.
- Ensure weak_label_map.yml references doc IDs present in your chunks.

----------------------------------------------------------------

Training Advice & Tuning
------------------------

- Prefer Qwen2.5-3B for single-GPU (L4) workflows; 7B may need smaller batches.
- Increase effective batch via gradient accumulation.
- We avoid torch.compile / bitsandbytes to stay Python 3.14-compatible.
- Steps ≈ epochs * ceil(num_examples / (per_device_bs * grad_accum * num_devices)).
- Grow data: add more supervision samples and disciplines that exist in the corpus.

----------------------------------------------------------------

Troubleshooting
---------------

- SSL / proxy: clone models via git and set HF_HUB_OFFLINE=1.
- Fallback: if FAISS missing, RAG uses TF-IDF (when artifacts exist).
- CUDA OOM: lower batch size, increase grad-accum, or switch to 3B.
- Kernel < 5.5: you may see a warning; upgrade recommended but not required.

----------------------------------------------------------------

Security & Regulatory Notice
----------------------------

This tool helps draft WBS structures and cite relevant sections from publicly available regulations and guidance. It does not replace expert engineering judgment or formal nuclear licensing processes. Always verify the generated content against the authoritative sources and your project’s licensing basis.

----------------------------------------------------------------

License
-------

Your local code remains under your chosen license. Embedded models and regulatory texts retain their original terms.

----------------------------------------------------------------

Appendix — Common Paths
-----------------------

- Corpus
  - base_files/corpus/raw/ — downloaded HTML/PDF
  - base_files/corpus/parsed/ — normalized JSON
  - base_files/corpus/chunks/ — chunked JSONL
  - base_files/corpus/index/ — TF-IDF & FAISS
  - base_files/corpus/models/bge-small-en-v1.5/ — local embedding model
- Supervision
  - base_files/supervision/configs/weak_label_map.yml
  - base_files/supervision/data/instructions.jsonl
- Model
  - base_files/models/Qwen2.5-3B-Instruct/
  - base_files/model/data/{train,val}.jsonl
  - base_files/model/checkpoints/wbs-lora/
  - base_files/model/merged/
