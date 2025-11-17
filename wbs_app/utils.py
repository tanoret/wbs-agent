# wbs_app/utils.py
# Common helpers for Streamlit pages:
# - Path discovery (base_files, corpus, scripts, models, merged, adapters)
# - Subprocess helpers to run your verified CLIs
# - JSON/YAML IO + JSONL
# - HF offline env + EMBED_LOCAL_PATH wiring
# - Corpus/model discovery & status checks
# - last_response.json loader/parser
# - Small UI helpers for consistency

from __future__ import annotations

import os
import sys
import json
import yaml
import time
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import streamlit as st


# --------------------------------------------------------------------------------------
# Paths & configuration
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AppPaths:
    app_root: Path
    base_files: Path
    corpus_root: Path
    corpus_scripts: Path
    model_root: Path
    model_scripts: Path
    model_merged_dir: Path
    model_checkpoints_dir: Path
    supervision_root: Path
    supervision_scripts: Path
    models_dir: Path
    embed_model_dir: Path
    index_dir: Path
    last_response: Path
    exports_dir: Path

    @staticmethod
    def detect(app_root: Optional[Path] = None) -> "AppPaths":
        root = Path.cwd() if app_root is None else Path(app_root).resolve()
        base = root / "base_files"
        corpus = base / "corpus"
        model_root = base / "model"
        sup_root = base / "supervision"
        return AppPaths(
            app_root=root,
            base_files=base,
            corpus_root=corpus,
            corpus_scripts=corpus / "scripts",
            model_root=model_root,
            model_scripts=model_root / "scripts",
            model_merged_dir=model_root / "merged",
            model_checkpoints_dir=model_root / "checkpoints",
            supervision_root=sup_root,
            supervision_scripts=sup_root / "scripts",
            models_dir=base / "models",
            embed_model_dir=corpus / "models" / "bge-small-en-v1.5",
            index_dir=corpus / "index",
            last_response=model_root / "last_response.json",
            exports_dir=model_root / "exports",
        )


# --------------------------------------------------------------------------------------
# Filesystem & IO helpers
# --------------------------------------------------------------------------------------

def ensure_dirs(paths: Iterable[Union[str, Path]]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_yaml(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Union[str, Path], obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_jsonl(path: Union[str, Path]) -> List[Any]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Any] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # keep raw line on error for debugging
                out.append({"_raw": line, "_error": "parse"})
    return out


def write_jsonl(path: Union[str, Path], rows: Iterable[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1_of_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------------------

def build_offline_env(embed_model_dir: Optional[Path] = None, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create an env dict with HF offline flags and optional EMBED_LOCAL_PATH."""
    env = dict(os.environ if base_env is None else base_env)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if embed_model_dir and Path(embed_model_dir).exists():
        env["EMBED_LOCAL_PATH"] = str(embed_model_dir)
    return env


def run_cli(py_script: Union[str, Path], args: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[Union[str, Path]] = None) -> Tuple[int, str, str]:
    """
    Run a Python CLI script via subprocess; return (rc, stdout, stderr).
    """
    cmd = [sys.executable, str(py_script)] + list(map(str, args))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=None if cwd is None else str(cwd),
    )
    return proc.returncode, proc.stdout, proc.stderr


def show_logs(stdout: str, stderr: str, expanded: bool = False) -> None:
    """Streamlit-friendly log expanders."""
    if stdout and stdout.strip():
        with st.expander("stdout", expanded=expanded):
            st.code(stdout, language="text")
    if stderr and stderr.strip():
        with st.expander("stderr", expanded=expanded):
            st.code(stderr, language="text")


# --------------------------------------------------------------------------------------
# Discovery helpers (corpus, models, adapters)
# --------------------------------------------------------------------------------------

def corpus_status(paths: AppPaths) -> Dict[str, Any]:
    """
    Quick health-check of the corpus pipeline.
    Returns flags and counts for raw/parsed/chunks/index.
    """
    raw = list((paths.corpus_root / "raw").glob("*"))
    parsed = list((paths.corpus_root / "parsed").glob("*.json"))
    chunks = list((paths.corpus_root / "chunks").glob("*.jsonl"))

    faiss_dense = paths.index_dir / "dense_open_bge_small_ip.faiss"
    tfidf_vec = paths.index_dir / "tfidf_vectorizer.joblib"
    tfidf_mat = paths.index_dir / "tfidf_matrix.npz"

    return {
        "raw_count": len(raw),
        "parsed_count": len(parsed),
        "chunks_count": len(chunks),
        "has_faiss": faiss_dense.exists(),
        "has_tfidf": tfidf_vec.exists() and tfidf_mat.exists(),
        "has_embed_model": paths.embed_model_dir.exists(),
    }


def find_base_models(paths: AppPaths) -> List[Path]:
    """
    Candidate base models = dirs under base_files/models with a config.json.
    """
    out: List[Path] = []
    if not paths.models_dir.exists():
        return out
    for d in sorted(paths.models_dir.glob("*")):
        if d.is_dir() and (d / "config.json").exists():
            out.append(d)
    return out


def find_merged_models(paths: AppPaths) -> List[Path]:
    out: List[Path] = []
    if not paths.model_merged_dir.exists():
        return out
    for d in sorted(paths.model_merged_dir.glob("*")):
        if d.is_dir() and (d / "config.json").exists():
            out.append(d)
    return out


def find_adapters(paths: AppPaths) -> List[Path]:
    """
    LoRA adapters = dirs under base_files/model/checkpoints containing adapter_config.json
    """
    out: List[Path] = []
    if not paths.model_checkpoints_dir.exists():
        return out
    for d in sorted(paths.model_checkpoints_dir.glob("*")):
        if d.is_dir() and (d / "adapter_config.json").exists():
            out.append(d)
    return out


# --------------------------------------------------------------------------------------
# last_response.json handling (rag_infer / batch_infer outputs)
# --------------------------------------------------------------------------------------

def load_last_response(paths: AppPaths) -> Optional[Dict[str, Any]]:
    """
    Returns dict with keys like: {"query", "output_text", "output_json", "retrieved", ...}
    or None if not present/parseable.
    """
    if not paths.last_response.exists():
        return None
    try:
        return json.loads(paths.last_response.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Small UX helpers
# --------------------------------------------------------------------------------------

def ok_bad(flag: bool) -> str:
    return "✅" if flag else "❌"


def human_count(n: int) -> str:
    return f"{n:,}"


def spinner_msg(msg: str):
    """
    Context manager for a consistent spinner.
    Usage:
      with spinner_msg("Doing work..."):
          ...
    """
    return st.spinner(msg)


# --------------------------------------------------------------------------------------
# High-level wrappers around your verified CLIs (optional)
# (Keep names generic so pages can call them if they prefer not to import scripts directly)
# --------------------------------------------------------------------------------------

def cli_corpus_download(paths: AppPaths) -> Tuple[int, str, str]:
    """Run 01_download.py"""
    script = paths.corpus_scripts / "01_download.py"
    env = build_offline_env(paths.embed_model_dir)
    return run_cli(script, [], env=env)


def cli_corpus_parse(paths: AppPaths) -> Tuple[int, str, str]:
    """Run 02_parse_normalize.py"""
    script = paths.corpus_scripts / "02_parse_normalize.py"
    env = build_offline_env(paths.embed_model_dir)
    return run_cli(script, [], env=env)


def cli_corpus_chunk_embed(paths: AppPaths) -> Tuple[int, str, str]:
    """Run 03_chunk_and_embed.py"""
    script = paths.corpus_scripts / "03_chunk_and_embed.py"
    env = build_offline_env(paths.embed_model_dir)
    return run_cli(script, [], env=env)


def cli_corpus_build_index(paths: AppPaths) -> Tuple[int, str, str]:
    """Run 04_build_faiss.py"""
    script = paths.corpus_scripts / "04_build_faiss.py"
    env = build_offline_env(paths.embed_model_dir)
    return run_cli(script, [], env=env)


def cli_make_supervision(paths: AppPaths,
                         corpus_root: Optional[Path] = None,
                         schema_path: Optional[Path] = None,
                         weak_map: Optional[Path] = None,
                         out_path: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Run supervision/scripts/05_make_supervision.py with defaults if not provided.
    (Your working script uses --chunks-dir/--map/--out)
    """
    script = paths.supervision_scripts / "05_make_supervision.py"
    env = build_offline_env(paths.embed_model_dir)
    args: List[str] = []
    if corpus_root:
        args += ["--chunks-dir", str(corpus_root / "chunks")]
    if weak_map:
        args += ["--map", str(weak_map)]
    if out_path:
        args += ["--out", str(out_path)]
    return run_cli(script, args, env=env)


def cli_prepare_sft(paths: AppPaths,
                    instructions: Path,
                    out_dir: Path,
                    val_ratio: float = 0.2) -> Tuple[int, str, str]:
    """
    Run model/scripts/06_prepare_sft.py
    (Your working script uses --in / --out-dir / --val-ratio)
    """
    script = paths.model_scripts / "06_prepare_sft.py"
    env = build_offline_env(paths.embed_model_dir)
    args = [
        "--in", str(instructions),
        "--out-dir", str(out_dir),
        "--val-ratio", str(val_ratio),
    ]
    return run_cli(script, args, env=env)


def cli_train_lora(paths: AppPaths,
                   base_model_dir: Path,
                   data_train: Path,
                   data_val: Path,
                   out_dir: Path,
                   cfg: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Run model/scripts/07_train_lora.py with current compatible flags:
      --base --data-train --data-val --out [--cfg]
    (Other knobs can live in the YAML cfg)
    """
    script = paths.model_scripts / "07_train_lora.py"
    env = build_offline_env(paths.embed_model_dir)
    args = [
        "--base", str(base_model_dir),
        "--data-train", str(data_train),
        "--data-val", str(data_val),
        "--out", str(out_dir),
    ]
    if cfg:
        args += ["--cfg", str(cfg)]
    return run_cli(script, args, env=env)


def cli_batch_rag(paths: AppPaths,
                  base_or_merged: Path,
                  prompts_file: Path,
                  out_file: Path,
                  k: int = 8,
                  max_new_tokens: int = 700,
                  temperature: float = 0.2,
                  adapter_dir: Optional[Path] = None) -> Tuple[int, str, str]:
    """
    Run model/scripts/08_batch_rag_infer.py
      --base --input --out --corpus-root --k --max-new-tokens [--adapter --temperature]
    """
    script = paths.model_scripts / "08_batch_rag_infer.py"
    env = build_offline_env(paths.embed_model_dir)
    args = [
        "--base", str(base_or_merged),
        "--input", str(prompts_file),
        "--out", str(out_file),
        "--corpus-root", str(paths.corpus_root),
        "--k", str(int(k)),
        "--max-new-tokens", str(int(max_new_tokens)),
        "--temperature", str(float(temperature)),
    ]
    if adapter_dir:
        args += ["--adapter", str(adapter_dir)]
    return run_cli(script, args, env=env)


def cli_merge_lora(paths: AppPaths,
                   base_model_dir: Path,
                   adapter_dir: Path,
                   out_dir: Path) -> Tuple[int, str, str]:
    """
    Run model/scripts/09_merge_lora.py
      --base --adapter --out
    """
    script = paths.model_scripts / "09_merge_lora.py"
    env = build_offline_env(paths.embed_model_dir)
    args = ["--base", str(base_model_dir), "--adapter", str(adapter_dir), "--out", str(out_dir)]
    return run_cli(script, args, env=env)


def cli_validate_outputs(paths: AppPaths, jsonl_file: Path) -> Tuple[int, str, str]:
    """
    Run model/scripts/10_validate_outputs.py --file ...
    """
    script = paths.model_scripts / "10_validate_outputs.py"
    env = build_offline_env(paths.embed_model_dir)
    args = ["--file", str(jsonl_file)]
    return run_cli(script, args, env=env)


# --------------------------------------------------------------------------------------
# Small guards / validations used by pages
# --------------------------------------------------------------------------------------

def require_corpus(paths: AppPaths) -> bool:
    status = corpus_status(paths)
    ok = status["chunks_count"] > 0 and (status["has_faiss"] or status["has_tfidf"])
    if not ok:
        st.error("Corpus is not ready. Please run page ② Build Corpus.")
    return ok


def require_models_dir(paths: AppPaths) -> bool:
    if not paths.models_dir.exists():
        st.error("No local models directory found at: " + str(paths.models_dir))
        return False
    return True


def require_file(path: Union[str, Path], friendly: Optional[str] = None) -> bool:
    if not Path(path).exists():
        st.error(f"{friendly or 'Required file'} not found: {path}")
        return False
    return True


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 2)


def sizeof(path: Union[str, Path]) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return mb(p.stat().st_size)
