# wbs_core/rag.py
# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) utilities.

This module provides:
  • RAGEngine – loads dense FAISS and TF-IDF indices and retrieves top-k chunks
  • prompt building helpers with explicit citation requirements
  • local inference with a base HF model folder and an optional LoRA adapter
    (no internet required, avoids bitsandbytes / torch.compile pitfalls)

Expected on-disk layout (built by your corpus/index scripts):
  corpus/
    chunks/                         # *.jsonl chunk files (optional for RAG display)
    index/
      dense_open_bge_small_ip.faiss # Inner-product FAISS index for BGE-small
      meta.jsonl                    # One JSON per vector row, with fields {id, text, url, ...}
      tfidf_matrix.npz              # CSR matrix for TF-IDF fallback
      tfidf_vectorizer.joblib       # Fitted vectorizer for fallback
    models/
      bge-small-en-v1.5/            # SentenceTransformer directory (local)

Usage (programmatic)
--------------------
from wbs_core.rag import RAGEngine, default_system_prompt, build_user_prompt, chat_generate

engine = RAGEngine(
    corpus_root=Path("base_files/corpus"),
    embed_model_dir=Path("base_files/corpus/models/bge-small-en-v1.5")
)
hits = engine.retrieve("Design seismic classification...", k=8)

messages = [
    {"role": "system", "content": default_system_prompt()},
    {"role": "user", "content": build_user_prompt("Design seismic classification...", hits)}
]

text = chat_generate(
    messages=messages,
    base_model_dir=Path("models/Qwen2.5-3B-Instruct"),
    adapter_dir=Path("base_files/model/checkpoints/wbs-lora"),
    max_new_tokens=700,
    temperature=0.2
)

Notes
-----
• We set env flags to run fully offline and bypass bitsandbytes runtime imports that
  are incompatible with Python 3.14's torch.compile. Inference is FP16/BF16 by default.
• FAISS is preferred; if missing, we fall back to TF-IDF. If neither is present, a
  clear error is raised.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional imports guarded to keep failure modes clear
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

# Sentence embeddings (LOCAL directory only; no network)
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("sentence_transformers is required for dense retrieval.") from e

# HF inference bits
# Avoid bitsandbytes auto-imports that trigger torch.compile on Python 3.14+
os.environ.setdefault("BNB_DISABLE_CUSTOM_IMPORTS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:true")

import torch  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore


# -----------------------------
# Small logging helper
# -----------------------------

def _log(msg: str) -> None:
    print(f"[RAG] {msg}", flush=True)


# -----------------------------
# Retrieval datastructures
# -----------------------------

@dataclass
class RetrievedChunk:
    idx: int
    score: float
    id: str
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------
# RAG Engine
# -----------------------------

class RAGEngine:
    """
    Loads indices and provides retrieval over the local corpus.
    Prefers dense FAISS (BGE-small, inner product); falls back to TF-IDF.
    """

    def __init__(self,
                 corpus_root: Path,
                 embed_model_dir: Optional[Path] = None):
        self.corpus_root = Path(corpus_root).resolve()
        self.index_dir = self.corpus_root / "index"
        self.chunks_dir = self.corpus_root / "chunks"
        self.embed_model_dir = Path(embed_model_dir).resolve() if embed_model_dir else None

        # Paths
        self.faiss_path = self.index_dir / "dense_open_bge_small_ip.faiss"
        self.meta_path = self.index_dir / "meta.jsonl"
        self.tfidf_matrix_path = self.index_dir / "tfidf_matrix.npz"
        self.tfidf_vectorizer_path = self.index_dir / "tfidf_vectorizer.joblib"

        # State
        self._faiss_index = None
        self._dense_ready = False
        self._tfidf_ready = False
        self._meta: List[Dict[str, Any]] = []
        self._embedder = None

        self._load_meta()
        self._try_load_dense()
        if not self._dense_ready:
            self._try_load_tfidf()
            if not self._tfidf_ready:
                raise RuntimeError("No usable index found (neither FAISS nor TF-IDF). "
                                   "Did you run '04_build_faiss.py'?")

    # ---------- load helpers ----------

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing meta.jsonl at {self.meta_path}")
        meta: List[Dict[str, Any]] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    meta.append(json.loads(line))
                except Exception:
                    # be tolerant but warn
                    _log("Warning: could not parse a meta.jsonl line; skipping.")
        if not meta:
            raise RuntimeError("meta.jsonl parsed but empty.")
        self._meta = meta
        _log(f"Loaded meta for {len(self._meta)} vectors.")

    def _try_load_dense(self) -> None:
        if faiss is None:
            _log("FAISS not available; skipping dense index.")
            self._dense_ready = False
            return
        if not self.faiss_path.exists():
            _log("FAISS index not found; skipping dense retrieval.")
            self._dense_ready = False
            return
        if self.embed_model_dir is None:
            _log("No embed_model_dir provided; skipping dense retrieval.")
            self._dense_ready = False
            return

        # Load FAISS
        self._faiss_index = faiss.read_index(str(self.faiss_path))
        # Load local embedder
        self._embedder = SentenceTransformer(str(self.embed_model_dir))
        # SentenceTransformer returns normalized embeddings for BGE by default when you
        # call encode(normalize_embeddings=True). We'll do that explicitly.
        self._dense_ready = True
        _log("Dense retrieval ready (FAISS + local BGE).")

    def _try_load_tfidf(self) -> None:
        if joblib is None:
            _log("joblib not available; cannot enable TF-IDF fallback.")
            self._tfidf_ready = False
            return
        if not self.tfidf_matrix_path.exists() or not self.tfidf_vectorizer_path.exists():
            _log("TF-IDF files not found; TF-IDF fallback disabled.")
            self._tfidf_ready = False
            return

        self._tfidf_X = self._load_sparse_npz(self.tfidf_matrix_path)
        self._tfidf_v = joblib.load(self.tfidf_vectorizer_path)
        self._tfidf_ready = True
        _log("TF-IDF fallback ready.")

    @staticmethod
    def _load_sparse_npz(path: Path):
        # Lazy import to keep module import cheap without NumPy/SciPy
        from scipy.sparse import load_npz  # type: ignore
        return load_npz(path)

    # ---------- retrieval ----------

    def retrieve(self, query: str, k: int = 8) -> List[RetrievedChunk]:
        if self._dense_ready:
            return self._retrieve_dense(query, k)
        if self._tfidf_ready:
            return self._retrieve_tfidf(query, k)
        raise RuntimeError("No retrieval method is ready.")

    def _retrieve_dense(self, query: str, k: int) -> List[RetrievedChunk]:
        assert self._embedder is not None and self._faiss_index is not None
        qvec = self._embedder.encode([query], normalize_embeddings=True)
        D, I = self._faiss_index.search(np.asarray(qvec, dtype=np.float32), k)
        # D: (1, k) similarity (inner product); I: (1, k) indices
        out: List[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist())):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            out.append(RetrievedChunk(
                idx=idx,
                score=float(score),
                id=str(m.get("id", idx)),
                text=str(m.get("text", "")),
                url=m.get("url"),
                title=m.get("title"),
                meta=m,
            ))
        return out

    def _retrieve_tfidf(self, query: str, k: int) -> List[RetrievedChunk]:
        v = self._tfidf_v.transform([query])  # 1 x N
        scores = (self._tfidf_X @ v.T).toarray().reshape(-1)  # cosine approx (no norm)
        top_idx = np.argsort(-scores)[:k]
        out: List[RetrievedChunk] = []
        for idx in top_idx.tolist():
            sc = float(scores[idx])
            m = self._meta[idx]
            out.append(RetrievedChunk(
                idx=idx,
                score=sc,
                id=str(m.get("id", idx)),
                text=str(m.get("text", "")),
                url=m.get("url"),
                title=m.get("title"),
                meta=m,
            ))
        return out


# -----------------------------
# Prompt helpers
# -----------------------------

def default_system_prompt() -> str:
    """
    System instruction for WBS synthesis with strict JSON and citations.
    You can override this in your app to tweak the output schema.
    """
    return (
        "You are a nuclear engineering assistant that generates a Work Breakdown Structure (WBS) "
        "for reactor design tasks. Always produce STRICT JSON (no markdown) matching this schema:\n"
        "{\n"
        '  "wbs_title": str,\n'
        '  "assumptions": [str, ...],\n'
        '  "tasks": [\n'
        "    {\n"
        '      "id": str,                      // short hierarchical ID, e.g., "1.2.3"\n'
        '      "title": str,                   // concise task name\n'
        '      "discipline": str,              // e.g., seismic, TH, core physics, QA, EP\n'
        '      "objective": str,               // what this task proves/delivers\n'
        '      "deliverables": [str, ...],     // reports, calc packages, models, drawings\n'
        '      "inputs": [str, ...],           // required data from user/site/plant\n'
        '      "methods": [str, ...],          // analyses, ML models, correlations used\n'
        '      "acceptance_criteria": [str, ...],\n'
        '      "citations": [\n'
        "        {\"ref\": str, \"quote\": str} // point to CFR, RG, SRP, IAEA etc\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "• Base *all* tasks and acceptance criteria on the provided sources; never invent regulations.\n"
        "• Each citation.ref must be a short tag like '10 CFR 50 App S §IV(a)' or 'RG 1.29 Rev.6 §C.1'.\n"
        "• Each citation.quote should be a short excerpt (≤40 words) from the provided passages.\n"
        "• If the prompt is ambiguous, list explicit assumptions in `assumptions`.\n"
        "• Keep outputs compact and actionable."
    )


def build_user_prompt(query: str, hits: Sequence[RetrievedChunk]) -> str:
    """
    Builds the user content block: a concise instruction plus an appendix of sources.

    We include only the top-k retrieved texts (with ids) and lightweight metadata so the
    model can cite by id or by a normalized 'ref' string.
    """
    src_lines = []
    for i, h in enumerate(hits, 1):
        # Prefer an explicit 'ref' in metadata if present; else synthesize from title/url/id
        ref = (
            h.meta.get("ref") if h.meta else None
        ) or (
            f"{h.title or h.id}".strip()
        )
        header = f"[{i}] id={h.id} ref={ref}"
        url_line = f"URL: {h.url}" if h.url else ""
        text = h.text.replace("\n", " ").strip()
        if len(text) > 1200:
            text = text[:1200] + " ..."
        src_lines.append(f"{header}\n{text}\n{url_line}".strip())

    sources = "\n\n".join(src_lines) if src_lines else "(no sources found)"
    prompt = (
        f"User request:\n{query}\n\n"
        "Use ONLY the following sources to build the WBS. Cite by id/ref and include short quotes.\n"
        "SOURCES BEGIN\n"
        f"{sources}\n"
        "SOURCES END\n\n"
        "Return STRICT JSON only, no prose."
    )
    return prompt


# -----------------------------
# Local inference (chat style)
# -----------------------------

def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # L4 supports bf16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_local_model(base_model_dir: Path,
                     adapter_dir: Optional[Path] = None) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """
    Loads a local CausalLM and attaches an optional LoRA adapter (no internet).
    Ensures we don't accidentally import bitsandbytes.
    """
    dtype = _select_dtype()
    tok = AutoTokenizer.from_pretrained(str(base_model_dir), use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    # Attach LoRA adapter in inference mode if provided
    if adapter_dir is not None:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but an adapter was provided.")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)

    model.eval()
    return tok, model


def chat_generate(messages: List[Dict[str, str]],
                  base_model_dir: Path,
                  adapter_dir: Optional[Path] = None,
                  max_new_tokens: int = 700,
                  temperature: float = 0.2,
                  top_p: float = 0.9,
                  top_k: int = 50,
                  do_sample: bool = True) -> str:
    """
    Simple chat-style generation using the model's chat template if available.

    Args:
      messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
      base_model_dir: local folder for the base model
      adapter_dir: optional LoRA folder
    """
    tok, model = load_local_model(Path(base_model_dir), Path(adapter_dir) if adapter_dir else None)

    if hasattr(tok, "apply_chat_template"):
        text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to a naive concatenation
        def fmt(m): return f"<<{m['role']}>>:\n{m['content'].strip()}"
        text_in = "\n\n".join(fmt(m) for m in messages) + "\n\n<<assistant>>:\n"

    device = model.device
    inputs = tok([text_in], return_tensors="pt").to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen_ids[0], skip_special_tokens=True)

    # If the tokenizer appended the full conversation, strip the prefix
    # to return only the assistant segment.
    # Heuristic: take the substring after last occurrence of user's question.
    user_last = messages[-1]["content"].strip()
    pos = out.rfind(user_last)
    if pos != -1:
        candidate = out[pos + len(user_last):].strip()
        if candidate:
            out = candidate

    return out.strip()


# -----------------------------
# High-level one-shot answer
# -----------------------------

def rag_answer(query: str,
               corpus_root: Path,
               base_model_dir: Path,
               adapter_dir: Optional[Path] = None,
               embed_model_dir: Optional[Path] = None,
               k: int = 8,
               max_new_tokens: int = 700,
               temperature: float = 0.2,
               system_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Full RAG pipeline for a single query:
      1) retrieve top-k chunks
      2) build messages (system + user)
      3) run local inference
      4) return text and the retrieved subset (for UI display)

    Returns:
      {
        "query": str,
        "retrieved": [RetrievedChunk as dict...],
        "output": str
      }
    """
    engine = RAGEngine(corpus_root=corpus_root, embed_model_dir=embed_model_dir)
    hits = engine.retrieve(query, k=k)

    sys_prompt = system_prompt or default_system_prompt()
    user_prompt = build_user_prompt(query, hits)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = chat_generate(
        messages=messages,
        base_model_dir=base_model_dir,
        adapter_dir=adapter_dir,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Convert RetrievedChunk list to dicts for easy JSON serialization
    retrieved_as_dicts = [
        {
            "idx": h.idx,
            "score": h.score,
            "id": h.id,
            "text": h.text,
            "url": h.url,
            "title": h.title,
            "meta": h.meta,
        }
        for h in hits
    ]

    return {
        "query": query,
        "retrieved": retrieved_as_dicts,
        "output": output,
    }
