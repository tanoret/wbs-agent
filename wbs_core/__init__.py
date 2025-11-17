
"""
wbs_core
========
Lightweight orchestration helpers for the WBS reactor-design pipeline, adapted to the
new repo layout where *all working scripts and models live under* ``base_files/``.

This module provides:
  - Robust path discovery for a repo containing a ``base_files`` folder.
  - Offline/HPC-friendly environment defaults (HF offline, tokenizers, allocator).
  - A simple subprocess wrapper with optional streaming logs.
  - Thin convenience functions that call your existing scripts in ``base_files``.

Quickstart
----------
>>> from wbs_core import paths, run_download, run_parse, run_chunk_embed, run_build_faiss
>>> p = paths()               # discover repo + base_files
>>> run_download(p)           # calls base_files/corpus/scripts/01_download.py
>>> run_parse(p)              # calls 02_parse_normalize.py
>>> run_chunk_embed(p)        # calls 03_chunk_and_embed.py (uses EMBED_LOCAL_PATH)
>>> run_build_faiss(p)        # calls 04_build_faiss.py

You can override autodetected locations by exporting WBS_REPO_ROOT, or by passing
a root path into ``paths(repo_root=...)``.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

DEFAULT_BASE_FILES_DIRNAME = "base_files"

# -----------------------------
# Path discovery & data model
# -----------------------------

@dataclass
class WbsPaths:
    repo_root: Path
    base_files: Path

    # top-level working trees (inside base_files)
    corpus_root: Path
    model_root: Path
    models_root: Path
    supervision_root: Path

    # convenience: script folders
    corpus_scripts: Path
    model_scripts: Path
    supervision_scripts: Path

    # frequently used artifacts
    embed_local_path: Path  # corpus/models/bge-small-en-v1.5

    def ensure_dirs(self) -> None:
        self.corpus_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.supervision_root.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "base_files": str(self.base_files),
            "corpus_root": str(self.corpus_root),
            "model_root": str(self.model_root),
            "models_root": str(self.models_root),
            "supervision_root": str(self.supervision_root),
            "corpus_scripts": str(self.corpus_scripts),
            "model_scripts": str(self.model_scripts),
            "supervision_scripts": str(self.supervision_scripts),
            "embed_local_path": str(self.embed_local_path),
        }


def _discover_repo_root(start: Optional[Path] = None) -> Path:
    """
    Find the repo root by:
      1) WBS_REPO_ROOT env var (if set), else
      2) Crawl up from CWD (or 'start') until a folder containing 'base_files' is found.
    """
    env_root = os.environ.get("WBS_REPO_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / DEFAULT_BASE_FILES_DIRNAME).exists():
            return p
    here = (start or Path.cwd()).resolve()
    for cur in [here, *here.parents]:
        if (cur / DEFAULT_BASE_FILES_DIRNAME).exists():
            return cur
    raise FileNotFoundError(
        f"Could not find repo root containing '{DEFAULT_BASE_FILES_DIRNAME}'. "
        "Set WBS_REPO_ROOT or run from within the repository."
    )


def paths(repo_root: Optional[Path | str] = None) -> WbsPaths:
    root = Path(repo_root).resolve() if repo_root else _discover_repo_root()
    base = root / DEFAULT_BASE_FILES_DIRNAME

    corpus_root = base / "corpus"
    model_root = base / "model"
    models_root = base / "models"
    supervision_root = base / "supervision"

    corpus_scripts = corpus_root / "scripts"
    model_scripts = model_root / "scripts"
    supervision_scripts = supervision_root / "scripts"

    # common local embedding path default
    embed_local_path = corpus_root / "models" / "bge-small-en-v1.5"

    return WbsPaths(
        repo_root=root,
        base_files=base,
        corpus_root=corpus_root,
        model_root=model_root,
        models_root=models_root,
        supervision_root=supervision_root,
        corpus_scripts=corpus_scripts,
        model_scripts=model_scripts,
        supervision_scripts=supervision_scripts,
        embed_local_path=embed_local_path,
    )


# -----------------------------
# Runtime environment helpers
# -----------------------------

def offline_env(p: Optional[WbsPaths] = None, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Builds an environment with reasonable offline defaults for HF/Transformers,
    sets EMBED_LOCAL_PATH to the local sentence-transformer directory, and uses a
    safe PyTorch allocator config to reduce fragmentation.
    """
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    if p:
        env.setdefault("EMBED_LOCAL_PATH", str(p.embed_local_path))
    if extra:
        env.update(extra)
    return env


def run_script(
    args: Iterable[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    Execute a script with optional real-time streaming of stdout.
    Returns (returncode, combined_stdout).
    """
    proc = subprocess.Popen(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=str(cwd or Path.cwd()),
        env=env or offline_env(),
    )
    out_lines = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            out_lines.append(line)
            if stream:
                stream(line)
    finally:
        proc.wait()
    return proc.returncode, "\\n".join(out_lines)


# -----------------------------
# Thin wrappers around scripts
# -----------------------------

# CORPUS: 01..04
def run_download(p: Optional[WbsPaths] = None, stream: Optional[Callable[[str], None]] = None) -> Tuple[int, str]:
    p = p or paths()
    script = p.corpus_scripts / "01_download.py"
    return run_script(["python", str(script)], cwd=p.repo_root, env=offline_env(p), stream=stream)

def run_parse(p: Optional[WbsPaths] = None, stream: Optional[Callable[[str], None]] = None) -> Tuple[int, str]:
    p = p or paths()
    script = p.corpus_scripts / "02_parse_normalize.py"
    return run_script(["python", str(script)], cwd=p.repo_root, env=offline_env(p), stream=stream)

def run_chunk_embed(
    p: Optional[WbsPaths] = None, stream: Optional[Callable[[str], None]] = None
) -> Tuple[int, str]:
    p = p or paths()
    script = p.corpus_scripts / "03_chunk_and_embed.py"
    env = offline_env(p, extra={"EMBED_LOCAL_PATH": str(p.embed_local_path)})
    return run_script(["python", str(script)], cwd=p.repo_root, env=env, stream=stream)

def run_build_faiss(p: Optional[WbsPaths] = None, stream: Optional[Callable[[str], None]] = None) -> Tuple[int, str]:
    p = p or paths()
    script = p.corpus_scripts / "04_build_faiss.py"
    return run_script(["python", str(script)], cwd=p.repo_root, env=offline_env(p), stream=stream)


# SUPERVISION: 05, SFT SPLIT: 06
def run_make_supervision(
    p: Optional[WbsPaths] = None,
    weak_map: Optional[str] = None,
    num: int = 100,
    out: Optional[str] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    05_make_supervision.py interface (updated):
      --map WEAK_MAP --num N --out OUT_FILE
    """
    p = p or paths()
    script = p.supervision_scripts / "05_make_supervision.py"
    weak_map = weak_map or str(p.supervision_root / "configs" / "weak_label_map.yml")
    out = out or str(p.supervision_root / "data" / "instructions.jsonl")
    args = ["python", str(script), "--map", weak_map, "--num", str(num), "--out", out]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)

def run_prepare_sft(
    p: Optional[WbsPaths] = None,
    instr: Optional[str] = None,
    out_dir: Optional[str] = None,
    val_ratio: float = 0.2,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    06_prepare_sft.py interface (updated):
      --in INSTRUCTIONS.jsonl --out-dir MODEL_DATA_DIR --val-ratio 0.2
    """
    p = p or paths()
    script = p.model_scripts / "06_prepare_sft.py"
    instr = instr or str(p.supervision_root / "data" / "instructions.jsonl")
    out_dir = out_dir or str(p.model_root / "data")
    args = ["python", str(script), "--in", instr, "--out-dir", out_dir, "--val-ratio", str(val_ratio)]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)


# TRAIN: 07
def run_train_lora(
    p: Optional[WbsPaths] = None,
    base: Optional[str] = None,
    train_jsonl: Optional[str] = None,
    val_jsonl: Optional[str] = None,
    out_dir: Optional[str] = None,
    cfg_yaml: Optional[str] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    07_train_lora.py interface:
      --base BASE_DIR --data-train TRAIN.jsonl --data-val VAL.jsonl --out CHECKPOINT_DIR [--cfg lora.yaml]
    """
    p = p or paths()
    script = p.model_scripts / "07_train_lora.py"
    base = base or str(p.models_root / "Qwen2.5-3B-Instruct")
    train_jsonl = train_jsonl or str(p.model_root / "data" / "train.jsonl")
    val_jsonl = val_jsonl or str(p.model_root / "data" / "val.jsonl")
    out_dir = out_dir or str(p.model_root / "checkpoints" / "wbs-lora")
    args = ["python", str(script), "--base", base, "--data-train", train_jsonl, "--data-val", val_jsonl, "--out", out_dir]
    if cfg_yaml:
        args += ["--cfg", cfg_yaml]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)


# INFER: 08 (batch)
def run_batch_rag_infer(
    p: Optional[WbsPaths] = None,
    base: Optional[str] = None,
    adapter: Optional[str] = None,
    prompts_file: Optional[str] = None,
    out_file: Optional[str] = None,
    corpus_root: Optional[str] = None,
    k: int = 8,
    max_new_tokens: int = 700,
    temperature: float = 0.2,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    08_batch_rag_infer.py interface:
      --base BASE_DIR --adapter ADAPTER_DIR --input PROMPTS.txt --out outputs.jsonl --corpus-root corpus --k 8 --max-new-tokens 700 [--temperature 0.2]
    """
    p = p or paths()
    script = p.model_scripts / "08_batch_rag_infer.py"
    base = base or str(p.models_root / "Qwen2.5-3B-Instruct")
    adapter = adapter or str(p.model_root / "checkpoints" / "wbs-lora")
    prompts_file = prompts_file or str(p.model_root / "prompts" / "example_prompt.txt")
    out_file = out_file or str(p.model_root / "batch_outputs.jsonl")
    corpus_root = corpus_root or str(p.corpus_root)
    args = [
        "python", str(script),
        "--base", base, "--adapter", adapter,
        "--input", prompts_file, "--out", out_file,
        "--corpus-root", corpus_root, "--k", str(k), "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
    ]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)


# VALIDATE: 10
def run_validate_outputs(
    p: Optional[WbsPaths] = None,
    jsonl_file: Optional[str] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    p = p or paths()
    script = p.model_scripts / "10_validate_outputs.py"
    jsonl_file = jsonl_file or str(p.model_root / "batch_outputs.jsonl")
    args = ["python", str(script), "--file", jsonl_file]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)


# MERGE: 09
def run_merge_lora(
    p: Optional[WbsPaths] = None,
    base: Optional[str] = None,
    adapter: Optional[str] = None,
    out_dir: Optional[str] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> Tuple[int, str]:
    """
    09_merge_lora.py interface:
      --base BASE_DIR --adapter ADAPTER_DIR --out MERGED_DIR
    """
    p = p or paths()
    script = p.model_scripts / "09_merge_lora.py"
    base = base or str(p.models_root / "Qwen2.5-3B-Instruct")
    adapter = adapter or str(p.model_root / "checkpoints" / "wbs-lora")
    out_dir = out_dir or str(p.model_root / "merged" / "qwen2.5-7b-wbs")
    args = ["python", str(script), "--base", base, "--adapter", adapter, "--out", out_dir]
    return run_script(args, cwd=p.repo_root, env=offline_env(p), stream=stream)


# -----------------------------
# Utilities
# -----------------------------

def list_local_base_models(p: Optional[WbsPaths] = None) -> list[str]:
    p = p or paths()
    out = []
    if p.models_root.exists():
        for d in p.models_root.iterdir():
            if d.is_dir() and (d / "config.json").exists():
                out.append(str(d))
    return out
