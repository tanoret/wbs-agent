# wbs_core/pipeline.py
# -*- coding: utf-8 -*-
"""
High-level orchestration for the WBS agent.

This module wires together all stages:
1) Download raw sources from a manifest
2) Parse & normalize to JSON
3) Chunk & embed
4) Build search indices (FAISS dense + TF-IDF sparse)
5) Generate weakly-supervised instruction pairs
6) Prepare SFT train/val splits
7) Fine-tune (LoRA)
8) Batch RAG inference over prompts
9) Validate model JSON outputs
10) (Optional) Merge LoRA into a full merged base

Design goals
------------
- No internet is required at runtime if your base model and embedder
  are already on disk.
- Best-effort use of your *working* scripts under `base_files/`.
  If you later migrate everything into `wbs_core.*` functions, you can
  swap the call sites here easily.
- Clear logging and explicit paths.

Quick starts
------------
# Run everything (end-to-end)
python -m wbs_core.pipeline run-all \
  --project-root . \
  --base-model models/Qwen2.5-3B-Instruct \
  --embed-model corpus/models/bge-small-en-v1.5 \
  --epochs 3

# Run selected steps only (download + parse + chunk + index)
python -m wbs_core.pipeline run \
  --project-root . \
  --steps download parse chunk index

# Run individual step (examples)
python -m wbs_core.pipeline download --project-root .
python -m wbs_core.pipeline parse --project-root .
python -m wbs_core.pipeline chunk --project-root . --embed-model corpus/models/bge-small-en-v1.5
python -m wbs_core.pipeline index --project-root .
python -m wbs_core.pipeline supervise --project-root .
python -m wbs_core.pipeline prepare-sft --project-root .
python -m wbs_core.pipeline train --project-root . --base-model models/Qwen2.5-3B-Instruct
python -m wbs_core.pipeline batch-infer --project-root .
python -m wbs_core.pipeline validate --project-root .
python -m wbs_core.pipeline merge --project-root . --base-model models/Qwen2.5-7B-Instruct

Environment toggles
-------------------
- Set offline-safe defaults automatically for HF/Transformers.
- You can override CUDA allocator advice via:
  export PYTORCH_ALLOC_CONF=expandable_segments:true

Notes
-----
This pipeline, by default, shells out to the *working* scripts you have in
`base_files/`. That keeps behavior identical to what you've already validated.
When you’re ready, you can redirect steps to call functions from the new
`wbs_core` modules (downloader, parser, chunker, indexer, lora) directly.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# Reuse the package logger setup
from . import get_logger

log = get_logger(__name__)


# ---------------------------
# Path config
# ---------------------------

@dataclass
class ProjectPaths:
    root: Path

    # base_files layout (your working scripts & artifacts)
    base_files: Path
    bf_corpus: Path
    bf_corpus_scripts: Path
    bf_model_scripts: Path
    bf_supervision_scripts: Path

    # corpus IO
    manifest: Path
    raw: Path
    parsed: Path
    chunks: Path
    index: Path
    embed_model_dir: Path  # set by user/CLI

    # supervision IO
    weak_map: Path
    supervision_out: Path

    # SFT data
    sft_out_dir: Path
    train_jsonl: Path
    val_jsonl: Path

    # training / adapters / merged
    checkpoints_dir: Path
    lora_out_dir: Path
    merged_out_dir: Path

    # batch prompts + outputs
    prompts_txt: Path
    batch_outputs: Path

    # base model path (folder with config.json etc) - set by user/CLI
    base_model_dir: Path

    @staticmethod
    def from_root(
        root: Path,
        base_model_dir: Optional[Path] = None,
        embed_model_dir: Optional[Path] = None
    ) -> "ProjectPaths":
        root = root.resolve()
        base_files = root / "base_files"

        bf_corpus = base_files / "corpus"
        bf_corpus_scripts = bf_corpus / "scripts"
        bf_model_scripts = base_files / "model" / "scripts"
        bf_supervision_scripts = base_files / "supervision" / "scripts"

        manifest = bf_corpus / "manifests" / "corpus_manifest.yaml"
        raw = bf_corpus / "raw"
        parsed = bf_corpus / "parsed"
        chunks = bf_corpus / "chunks"
        index = bf_corpus / "index"

        weak_map = base_files / "supervision" / "configs" / "weak_label_map.yml"
        supervision_out = base_files / "supervision" / "data" / "instructions.jsonl"

        sft_out_dir = base_files / "model" / "data"
        train_jsonl = sft_out_dir / "train.jsonl"
        val_jsonl = sft_out_dir / "val.jsonl"

        checkpoints_dir = base_files / "model" / "checkpoints"
        lora_out_dir = checkpoints_dir / "wbs-lora"
        merged_out_dir = base_files / "model" / "merged" / "qwen2.5-7b-wbs"

        prompts_txt = base_files / "model" / "prompts" / "example_prompt.txt"
        batch_outputs = base_files / "model" / "batch_outputs.jsonl"

        # Defaults if not provided
        base_model_dir = (base_model_dir or (root / "models" / "Qwen2.5-3B-Instruct")).resolve()
        embed_model_dir = (embed_model_dir or (bf_corpus / "models" / "bge-small-en-v1.5")).resolve()

        return ProjectPaths(
            root=root,
            base_files=base_files,
            bf_corpus=bf_corpus,
            bf_corpus_scripts=bf_corpus_scripts,
            bf_model_scripts=bf_model_scripts,
            bf_supervision_scripts=bf_supervision_scripts,
            manifest=manifest,
            raw=raw,
            parsed=parsed,
            chunks=chunks,
            index=index,
            embed_model_dir=embed_model_dir,
            weak_map=weak_map,
            supervision_out=supervision_out,
            sft_out_dir=sft_out_dir,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            checkpoints_dir=checkpoints_dir,
            lora_out_dir=lora_out_dir,
            merged_out_dir=merged_out_dir,
            prompts_txt=prompts_txt,
            batch_outputs=batch_outputs,
            base_model_dir=base_model_dir,
        )


# ---------------------------
# Util: run a Python script
# ---------------------------

def _env_offline(extra: Optional[dict] = None) -> dict:
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # avoid CUDA fragmentation warnings while staying explicit:
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:true")
    if extra:
        env.update(extra)
    return env


def run_py(script: Path, args: Sequence[str], cwd: Path, env: Optional[dict] = None) -> None:
    """Run a python script with logging, raising on non-zero exit."""
    cmd = [sys.executable, str(script), *args]
    log.info("▶ %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env or _env_offline())
    if proc.returncode != 0:
        raise RuntimeError(f"Script failed (rc={proc.returncode}): {script.name}")


# ---------------------------
# Pipeline step wrappers
# ---------------------------

def step_download(p: ProjectPaths) -> None:
    """01: Download all docs in manifest -> raw/"""
    script = p.bf_corpus_scripts / "01_download.py"
    args = []
    run_py(script, args, cwd=p.root)


def step_parse(p: ProjectPaths) -> None:
    """02: Parse raw -> parsed/"""
    script = p.bf_corpus_scripts / "02_parse_normalize.py"
    args = []
    run_py(script, args, cwd=p.root)


def step_chunk_and_embed(p: ProjectPaths) -> None:
    """03: Chunk parsed + embed with local SentenceTransformer -> chunks/"""
    script = p.bf_corpus_scripts / "03_chunk_and_embed.py"
    args = [
        "--embed-local-path", str(p.embed_model_dir),
    ]
    run_py(script, args, cwd=p.root)


def step_build_indices(p: ProjectPaths) -> None:
    """04: Build FAISS and TF-IDF -> index/"""
    script = p.bf_corpus_scripts / "04_build_faiss.py"
    args = []
    run_py(script, args, cwd=p.root)


def step_supervision(p: ProjectPaths,
                     reactor_types: str = "PWR,BWR,MSR,SFR,HTGR",
                     disciplines: str = "seismic,qa,emc,electrical,th,core,materials,ep,fire,iet") -> None:
    """05: Create weakly-labeled instruction pairs -> supervision/data/instructions.jsonl"""
    script = p.bf_supervision_scripts / "05_make_supervision.py"
    args = [
        "--chunks-dir", str(p.chunks),
        "--map", str(p.weak_map),
        "--reactor-types", reactor_types,
        "--disciplines", disciplines,
        "--num", "100",  # change as desired
        "--out", str(p.supervision_out),
    ]
    run_py(script, args, cwd=p.root)


def step_prepare_sft(p: ProjectPaths, val_ratio: float = 0.2) -> None:
    """06: Build train.jsonl / val.jsonl for SFT"""
    script = p.bf_model_scripts / "06_prepare_sft.py"
    args = [
        "--in", str(p.supervision_out),
        "--out-dir", str(p.sft_out_dir),
        "--val-ratio", str(val_ratio),
    ]
    run_py(script, args, cwd=p.root)


def step_train_lora(p: ProjectPaths,
                    base_model: Optional[Path] = None,
                    cfg: Optional[Path] = None) -> None:
    """07: Train LoRA adapter -> checkpoints/wbs-lora"""
    script = p.bf_model_scripts / "07_train_lora.py"
    base_dir = base_model.resolve() if base_model else p.base_model_dir
    args = [
        "--base", str(base_dir),
        "--data-train", str(p.train_jsonl),
        "--data-val", str(p.val_jsonl),
        "--out", str(p.lora_out_dir),
    ]
    if cfg:
        args += ["--cfg", str(cfg)]
    run_py(script, args, cwd=p.root)


def step_batch_infer(p: ProjectPaths,
                     k: int = 8,
                     max_new_tokens: int = 700,
                     temperature: float = 0.2,
                     base_model: Optional[Path] = None) -> None:
    """08: Batch RAG inference -> model/batch_outputs.jsonl"""
    script = p.bf_model_scripts / "08_batch_rag_infer.py"
    base_dir = base_model.resolve() if base_model else p.base_model_dir
    args = [
        "--base", str(base_dir),
        "--adapter", str(p.lora_out_dir),
        "--input", str(p.prompts_txt),
        "--out", str(p.batch_outputs),
        "--corpus-root", str(p.bf_corpus),
        "--k", str(k),
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
    ]
    run_py(script, args, cwd=p.root)


def step_validate_outputs(p: ProjectPaths) -> None:
    """10: Validate batch JSON outputs -> summary JSON on stdout"""
    script = p.bf_model_scripts / "10_validate_outputs.py"
    args = ["--file", str(p.batch_outputs)]
    run_py(script, args, cwd=p.root)


def step_merge_lora(p: ProjectPaths, base_model: Optional[Path] = None) -> None:
    """09: Merge LoRA into a full model folder -> model/merged/qwen2.5-7b-wbs"""
    script = p.bf_model_scripts / "09_merge_lora.py"
    base_dir = base_model.resolve() if base_model else p.base_model_dir
    args = [
        "--base", str(base_dir),
        "--adapter", str(p.lora_out_dir),
        "--out", str(p.merged_out_dir),
    ]
    run_py(script, args, cwd=p.root)


# ---------------------------
# CLI
# ---------------------------

ALL_STEPS: List[str] = [
    "download",
    "parse",
    "chunk",
    "index",
    "supervise",
    "prepare-sft",
    "train",
    "batch-infer",
    "validate",
    "merge",
]


def _paths_from_args(args: argparse.Namespace) -> ProjectPaths:
    base_model = Path(args.base_model).resolve() if args.base_model else None
    embed_model = Path(args.embed_model).resolve() if args.embed_model else None
    return ProjectPaths.from_root(
        root=Path(args.project_root),
        base_model_dir=base_model,
        embed_model_dir=embed_model,
    )


def _add_common_opts(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--project-root", default=".", help="Repository root (where base_files/ lives)")
    ap.add_argument("--base-model", default=None, help="Path to local base model dir (config.json present)")
    ap.add_argument("--embed-model", default=None, help="Path to local SentenceTransformer dir")


def cli() -> None:
    ap = argparse.ArgumentParser(description="WBS pipeline orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run-all
    p = sub.add_parser("run-all", help="Run the entire pipeline end-to-end")
    _add_common_opts(p)
    p.add_argument("--reactor-types", default="PWR,BWR,MSR,SFR,HTGR")
    p.add_argument("--disciplines", default="seismic,qa,emc,electrical,th,core,materials,ep,fire,iet")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--skip-merge", action="store_true", help="Skip LoRA merge step")

    # run (selected steps)
    p = sub.add_parser("run", help="Run selected steps in order")
    _add_common_opts(p)
    p.add_argument(
        "--steps",
        nargs="+",
        required=True,
        choices=ALL_STEPS,
        help=f"One or more of: {', '.join(ALL_STEPS)}",
    )

    # Individual step commands
    for name, help_text in [
        ("download", "01 Download"),
        ("parse", "02 Parse"),
        ("chunk", "03 Chunk+Embed"),
        ("index", "04 Build indices"),
        ("supervise", "05 Supervision pairs"),
        ("prepare-sft", "06 Prepare SFT splits"),
        ("train", "07 Train LoRA"),
        ("batch-infer", "08 Batch RAG inference"),
        ("validate", "10 Validate outputs"),
        ("merge", "09 Merge LoRA into base"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        _add_common_opts(sp)
        # Optional per-step flags
        if name == "batch-infer":
            sp.add_argument("--k", type=int, default=8)
            sp.add_argument("--max-new-tokens", type=int, default=700)
            sp.add_argument("--temperature", type=float, default=0.2)
        if name == "supervise":
            sp.add_argument("--reactor-types", default="PWR,BWR,MSR,SFR,HTGR")
            sp.add_argument("--disciplines", default="seismic,qa,emc,electrical,th,core,materials,ep,fire,iet")
        if name == "prepare-sft":
            sp.add_argument("--val-ratio", type=float, default=0.2)

    args = ap.parse_args()
    paths = _paths_from_args(args)

    # Ensure expected dirs exist
    paths.raw.mkdir(parents=True, exist_ok=True)
    paths.parsed.mkdir(parents=True, exist_ok=True)
    paths.chunks.mkdir(parents=True, exist_ok=True)
    paths.index.mkdir(parents=True, exist_ok=True)
    paths.sft_out_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.merged_out_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.cmd == "run-all":
        step_download(paths)
        step_parse(paths)
        step_chunk_and_embed(paths)
        step_build_indices(paths)
        step_supervision(paths, reactor_types=args.reactor_types, disciplines=args.disciplines)
        step_prepare_sft(paths, val_ratio=args.val_ratio)
        step_train_lora(paths)
        step_batch_infer(paths)
        step_validate_outputs(paths)
        if not args.skip_merge:
            step_merge_lora(paths)
        log.info("[DONE] run-all complete")
        return

    if args.cmd == "run":
        for s in args.steps:
            _dispatch_single_step(s, paths, args)
        log.info("[DONE] run steps complete")
        return

    # Individual
    _dispatch_single_step(args.cmd, paths, args)


def _dispatch_single_step(step_name: str, p: ProjectPaths, args: argparse.Namespace) -> None:
    if step_name == "download":
        step_download(p)
    elif step_name == "parse":
        step_parse(p)
    elif step_name == "chunk":
        step_chunk_and_embed(p)
    elif step_name == "index":
        step_build_indices(p)
    elif step_name == "supervise":
        step_supervision(p, reactor_types=getattr(args, "reactor_types", "PWR,BWR,MSR,SFR,HTGR"),
                         disciplines=getattr(args, "disciplines", "seismic,qa,emc,electrical,th,core,materials,ep,fire,iet"))
    elif step_name == "prepare-sft":
        step_prepare_sft(p, val_ratio=getattr(args, "val_ratio", 0.2))
    elif step_name == "train":
        step_train_lora(p)
    elif step_name == "batch-infer":
        step_batch_infer(
            p,
            k=getattr(args, "k", 8),
            max_new_tokens=getattr(args, "max_new_tokens", 700),
            temperature=getattr(args, "temperature", 0.2),
        )
    elif step_name == "validate":
        step_validate_outputs(p)
    elif step_name == "merge":
        step_merge_lora(p)
    else:
        raise SystemExit(f"Unknown step: {step_name}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
