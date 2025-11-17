# wbs_app/pages/4_Train_and_Evaluate.py
# Streamlit UI to Train LoRA, Batch RAG Inference, Validate Outputs, and Merge LoRA
# Works offline with your verified CLIs in base_files/.../scripts. Optionally uses wbs_core if available.

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ------------------------------------------------------------------------------------
# Project paths (relative to app root)
# ------------------------------------------------------------------------------------
APP_ROOT = Path.cwd()
BASE_FILES = APP_ROOT / "base_files"

# Scripts (CLI fallbacks)
MODEL_SCRIPTS = BASE_FILES / "model" / "scripts"
TRAIN_SCRIPT = MODEL_SCRIPTS / "07_train_lora.py"
BATCH_INFER_SCRIPT = MODEL_SCRIPTS / "08_batch_rag_infer.py"
MERGE_SCRIPT = MODEL_SCRIPTS / "09_merge_lora.py"
VALIDATE_SCRIPT = MODEL_SCRIPTS / "10_validate_outputs.py"

# Data / Artifacts
MODEL_ROOT = BASE_FILES / "model"
MODEL_DATA = MODEL_ROOT / "data"
CHECKPOINTS_DIR = MODEL_ROOT / "checkpoints"
DEFAULT_ADAPTER_DIR = CHECKPOINTS_DIR / "wbs-lora"
MERGED_DIR = MODEL_ROOT / "merged" / "qwen2.5-wbs-merged"

# Corpus
CORPUS_ROOT = BASE_FILES / "corpus"
EMBED_MODEL_DIR = CORPUS_ROOT / "models" / "bge-small-en-v1.5"  # used by RAG infer if available
INDEX_DIR = CORPUS_ROOT / "index"  # FAISS / TF-IDF

# Models (local, offline)
MODELS_DIR = BASE_FILES / "models"
DEFAULT_BASE_3B = MODELS_DIR / "Qwen2.5-3B-Instruct"
DEFAULT_BASE_7B = MODELS_DIR / "Qwen2.5-7B-Instruct"

# Prompts & outputs
PROMPTS_DIR = MODEL_ROOT / "prompts"
DEFAULT_PROMPTS = PROMPTS_DIR / "example_prompt.txt"
DEFAULT_BATCH_OUT = MODEL_ROOT / "batch_outputs.jsonl"
LAST_RESPONSE = MODEL_ROOT / "last_response.json"

# Ensure dirs exist
for p in [MODEL_DATA, CHECKPOINTS_DIR, MERGED_DIR.parent, PROMPTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Make local project importable (to try wbs_core in future)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def try_import_wbs_core():
    """Optionally import wbs_core lora utilities; return dict of callables or None."""
    try:
        from wbs_core.lora import (
            LoRATrainer as _LoRATrainer,
            BatchRAGInfer as _BatchRAGInfer,
            LoRAMerger as _LoRAMerger,
            OutputsValidator as _OutputsValidator,
        )
        return dict(
            trainer=_LoRATrainer,
            batch_infer=_BatchRAGInfer,
            merger=_LoRAMerger,
            validator=_OutputsValidator,
        )
    except Exception:
        return None


def run_cli(py_script: Path, args: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run a Python script via subprocess and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(py_script)] + list(args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def show_logs(stdout: str, stderr: str):
    if stdout.strip():
        with st.expander("stdout", expanded=False):
            st.code(stdout, language="text")
    if stderr.strip():
        with st.expander("stderr", expanded=False):
            st.code(stderr, language="text")


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def write_yaml(path: Path, data: Dict):
    """Minimal YAML writer (no external deps)."""
    def _dump(d, indent=0):
        lines = []
        sp = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                lines.append(f"{sp}{k}:")
                lines.extend(_dump(v, indent + 1))
            else:
                if isinstance(v, bool):
                    v_str = "true" if v else "false"
                else:
                    v_str = str(v)
                lines.append(f"{sp}{k}: {v_str}")
        return lines

    ensure_parent(path)
    path.write_text("\n".join(_dump(data)) + "\n", encoding="utf-8")


def file_head(path: Path, n: int = 5) -> List[str]:
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            out.append(line.rstrip("\n"))
            if i + 1 >= n:
                break
    return out


def jsonl_head(path: Path, n: int = 3):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_raw": line.strip(), "_error": "JSON decode failed"})
            if i + 1 >= n:
                break
    return rows


# ------------------------------------------------------------------------------------
# Page UI
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="4 · Train & Evaluate", layout="wide")
st.title("④ Train & Evaluate")

st.sidebar.header("Paths")
st.sidebar.caption("All paths are local to this app.")
st.sidebar.write("**Corpus root:**")
st.sidebar.code(str(CORPUS_ROOT))
st.sidebar.write("**Models dir:**")
st.sidebar.code(str(MODELS_DIR))
st.sidebar.write("**Model data (train/val):**")
st.sidebar.code(str(MODEL_DATA))

core_api = try_import_wbs_core()
st.info(f"Execution mode: {'wbs_core APIs' if core_api else 'base_files scripts'}")

st.divider()
# =========================================
# A) LoRA Training
# =========================================
st.subheader("A) LoRA Training")

colA, colB = st.columns(2)
with colA:
    base_model_choice = st.selectbox(
        "Base model (local)",
        options=[str(DEFAULT_BASE_3B), str(DEFAULT_BASE_7B), "Custom..."],
        index=0 if DEFAULT_BASE_3B.exists() else (1 if DEFAULT_BASE_7B.exists() else 2),
    )
with colB:
    if base_model_choice == "Custom...":
        base_model_dir = st.text_input("Custom base model path", value=str(DEFAULT_BASE_3B))
    else:
        base_model_dir = base_model_choice

train_path = st.text_input("Train file (JSONL)", value=str(MODEL_DATA / "train.jsonl"))
val_path = st.text_input("Val file (JSONL)", value=str(MODEL_DATA / "val.jsonl"))
adapter_out = st.text_input("LoRA output dir", value=str(DEFAULT_ADAPTER_DIR))

st.caption("If your train/val files are empty, go to Page ③ to build supervision & split SFT data.")

# Training params (used only if script supports --cfg; otherwise script's defaults apply)
adv = st.checkbox("Customize training hyperparameters (write a temporary lora.yaml and pass --cfg)", value=True)
tmp_cfg_path = MODEL_ROOT / "configs" / "tmp_lora.yaml"

if adv:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        epochs = st.number_input("epochs", min_value=1, max_value=50, value=3, step=1)
    with c2:
        lr = st.number_input("learning_rate", min_value=1e-6, max_value=1e-2, value=2e-4, step=1e-6, format="%.6f")
    with c3:
        batch_size = st.number_input("per_device_train_batch_size", min_value=1, max_value=8, value=1, step=1)
    with c4:
        grad_accum = st.number_input("gradient_accumulation_steps", min_value=1, max_value=64, value=4, step=1)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        max_seq_len = st.number_input("max_seq_len", min_value=256, max_value=8192, value=2048, step=128)
    with c6:
        warmup_ratio = st.number_input("warmup_ratio", min_value=0.0, max_value=0.5, value=0.03, step=0.01)
    with c7:
        weight_decay = st.number_input("weight_decay", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    with c8:
        bfloat16 = st.checkbox("bfloat16", value=True)

    st.caption("Tip: if your CLI ignores --cfg, it will still run with its internal defaults.")

train_btn = st.button("Start Training", type="primary", use_container_width=True)

if train_btn:
    if not Path(base_model_dir).exists():
        st.error("Base model path does not exist.")
    elif not Path(train_path).exists():
        st.error("Train file not found.")
    elif not Path(val_path).exists():
        st.error("Val file not found.")
    else:
        ensure_parent(Path(adapter_out))
        # Prefer core API if present
        if core_api:
            TrainerCls = core_api["trainer"]
            with st.spinner("Training LoRA via wbs_core..."):
                try:
                    cfg = None
                    if adv:
                        cfg = dict(
                            epochs=int(epochs),
                            learning_rate=float(lr),
                            per_device_train_batch_size=int(batch_size),
                            gradient_accumulation_steps=int(grad_accum),
                            max_seq_len=int(max_seq_len),
                            warmup_ratio=float(warmup_ratio),
                            weight_decay=float(weight_decay),
                            bfloat16=bool(bfloat16),
                        )
                    trainer = TrainerCls(
                        base=str(base_model_dir),
                        data_train=str(train_path),
                        data_val=str(val_path),
                        out_dir=str(adapter_out),
                        config=cfg,
                    )
                    info = trainer.run()
                    st.success("Training complete.")
                    if info:
                        st.json(info)
                except Exception as e:
                    st.error(f"LoRA training failed: {e}")
        else:
            # Fallback: CLI
            env = os.environ.copy()
            # Keep things offline & stable
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"

            args = ["--base", str(base_model_dir), "--data-train", str(train_path), "--data-val", str(val_path), "--out", str(adapter_out)]

            # If requested, write a temp lora.yaml and pass as --cfg
            if adv:
                cfg_payload = dict(
                    training=dict(
                        epochs=int(epochs),
                        learning_rate=float(lr),
                        per_device_train_batch_size=int(batch_size),
                        gradient_accumulation_steps=int(grad_accum),
                        max_seq_len=int(max_seq_len),
                        warmup_ratio=float(warmup_ratio),
                        weight_decay=float(weight_decay),
                        bfloat16=bool(bfloat16),
                    )
                )
                write_yaml(tmp_cfg_path, cfg_payload)
                args += ["--cfg", str(tmp_cfg_path)]

            with st.spinner("Training LoRA via base_files/model/scripts/07_train_lora.py ..."):
                rc, out, err = run_cli(TRAIN_SCRIPT, args, env=env)
            show_logs(out, err)
            if rc != 0:
                st.error("Training script failed.")
            else:
                st.success(f"Training complete. Adapter: {adapter_out}")
                # Show last few trainer logs if present
                ta = Path(adapter_out) / "training_args.bin"
                if ta.exists():
                    st.caption("training_args.bin present ✅")

st.divider()
# =========================================
# B) Batch RAG Inference
# =========================================
st.subheader("B) Batch RAG Inference")

col1, col2 = st.columns(2)
with col1:
    infer_base_model = st.text_input("Base model (same as trained on)", value=str(base_model_dir))
    infer_adapter_dir = st.text_input("Adapter dir", value=str(adapter_out))
with col2:
    prompts_path = st.text_input("Prompts file", value=str(DEFAULT_PROMPTS))
    batch_out = st.text_input("Batch outputs JSONL", value=str(DEFAULT_BATCH_OUT))

col3, col4, col5 = st.columns(3)
with col3:
    k_docs = st.number_input("Top-k retrieved", min_value=2, max_value=32, value=8, step=1)
with col4:
    max_new_tokens = st.number_input("Max new tokens", min_value=64, max_value=4096, value=700, step=16)
with col5:
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

run_batch = st.button("Run Batch Inference", type="primary", use_container_width=True)

if run_batch:
    if not Path(infer_base_model).exists():
        st.error("Base model path not found.")
    elif not Path(infer_adapter_dir).exists():
        st.error("Adapter dir not found.")
    elif not Path(prompts_path).exists():
        st.error("Prompts file not found.")
    else:
        ensure_parent(Path(batch_out))
        # Prefer core if available
        if core_api:
            Runner = core_api["batch_infer"]
            with st.spinner("Batch inference via wbs_core..."):
                try:
                    runner = Runner(
                        base=str(infer_base_model),
                        adapter=str(infer_adapter_dir),
                        prompts=str(prompts_path),
                        out_path=str(batch_out),
                        corpus_root=str(CORPUS_ROOT),
                        top_k=int(k_docs),
                        max_new_tokens=int(max_new_tokens),
                        temperature=float(temperature),
                        embed_local_path=str(EMBED_MODEL_DIR) if EMBED_MODEL_DIR.exists() else None,
                    )
                    info = runner.run()
                    st.success(f"Batch inference complete. {json.dumps(info, indent=2)}")
                except Exception as e:
                    st.error(f"Batch inference failed: {e}")
        else:
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"
            if EMBED_MODEL_DIR.exists():
                env["EMBED_LOCAL_PATH"] = str(EMBED_MODEL_DIR)

            args = [
                "--base", str(infer_base_model),
                "--adapter", str(infer_adapter_dir),
                "--input", str(prompts_path),
                "--out", str(batch_out),
                "--corpus-root", str(CORPUS_ROOT),
                "--k", str(int(k_docs)),
                "--max-new-tokens", str(int(max_new_tokens)),
                "--temperature", str(float(temperature)),
            ]
            with st.spinner("Running base_files/model/scripts/08_batch_rag_infer.py ..."):
                rc, out, err = run_cli(BATCH_INFER_SCRIPT, args, env=env)
            show_logs(out, err)
            if rc != 0:
                st.error("Batch inference failed.")
            else:
                st.success(f"Batch outputs written to {batch_out}")
                st.caption("First 3 records:")
                st.json(jsonl_head(Path(batch_out), n=3))

st.divider()
# =========================================
# C) Validate Outputs
# =========================================
st.subheader("C) Validate Outputs")

val_in_file = st.text_input("Batch outputs JSONL to validate", value=str(batch_out))
validate_btn = st.button("Validate", type="primary", use_container_width=True)

if validate_btn:
    if not Path(val_in_file).exists():
        st.error("Outputs file not found.")
    else:
        if core_api:
            Validator = core_api["validator"]
            with st.spinner("Validating via wbs_core..."):
                try:
                    v = Validator(path=str(val_in_file))
                    report = v.run()
                    st.success("Validation complete.")
                    st.json(report)
                except Exception as e:
                    st.error(f"Validation failed: {e}")
        else:
            rc, out, err = run_cli(VALIDATE_SCRIPT, ["--file", str(val_in_file)])
            show_logs(out, err)
            if rc != 0:
                st.error("Validation script failed.")
            else:
                # The validator prints a JSON dict to stdout
                try:
                    report = json.loads(out.strip().splitlines()[-1])
                    st.success("Validation complete.")
                    st.json(report)
                except Exception:
                    st.warning("Validator output wasn't valid JSON; see logs above.")

st.divider()
# =========================================
# D) Merge LoRA → Full Model
# =========================================
st.subheader("D) Merge LoRA → Full Model")

merge_base = st.text_input("Base model (same as trained on)", value=str(base_model_dir))
merge_adapter = st.text_input("Adapter dir", value=str(adapter_out))
merge_out_dir = st.text_input("Merged model out dir", value=str(MERGED_DIR))

merge_btn = st.button("Merge & Save", type="primary", use_container_width=True)

if merge_btn:
    if not Path(merge_base).exists():
        st.error("Base model path not found.")
    elif not Path(merge_adapter).exists():
        st.error("Adapter path not found.")
    else:
        ensure_parent(Path(merge_out_dir) / "dummy")
        if core_api:
            Merger = core_api["merger"]
            with st.spinner("Merging via wbs_core..."):
                try:
                    merger = Merger(base=str(merge_base), adapter=str(merge_adapter), out_dir=str(merge_out_dir))
                    info = merger.run()
                    st.success("Merge complete.")
                    if info:
                        st.json(info)
                except Exception as e:
                    st.error(f"Merge failed: {e}")
        else:
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"

            args = ["--base", str(merge_base), "--adapter", str(merge_adapter), "--out-dir", str(merge_out_dir)]
            with st.spinner("Running base_files/model/scripts/09_merge_lora.py ..."):
                rc, out, err = run_cli(MERGE_SCRIPT, args, env=env)
            show_logs(out, err)
            if rc != 0:
                st.error("Merge script failed.")
            else:
                st.success(f"Merged model saved to {merge_out_dir}")
                # Show file heads
                head_files = list(Path(merge_out_dir).glob("*.json"))[:3]
                if head_files:
                    st.caption("Preview of merged model config files:")
                    for fp in head_files:
                        st.write(f"**{fp.name}**")
                        st.code("\n".join(file_head(fp, 30)), language="json")

st.divider()
st.caption("This page is offline-friendly and uses your local corpus, models, and scripts.")
