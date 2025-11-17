# wbs_core/retrievy.py
# -*- coding: utf-8 -*-
"""
Lightweight retrieval utilities (dense FAISS, TF-IDF, and hybrid) with a simple CLI.

This module is intentionally generation-free: it only loads the local indices you built
under `corpus/index/` and returns the top-k matching chunks with their metadata.

It complements `wbs_core.rag.RAGEngine` by exposing retrieval primitives and a CLI.

Index layout expected (created by your corpus/index scripts):
  corpus/
    index/
      dense_open_bge_small_ip.faiss   # optional (dense retrieval)
      meta.jsonl                      # REQUIRED (row-aligned with FAISS/TF-IDF)
      tfidf_matrix.npz                # optional (TF-IDF fallback)
      tfidf_vectorizer.joblib         # optional (TF-IDF fallback)
    models/
      bge-small-en-v1.5/              # optional (local SentenceTransformer)

Examples
--------
# Programmatic
from pathlib import Path
from wbs_core.retrievy import HybridRetriever

ret = HybridRetriever(
    corpus_root=Path("base_files/corpus"),
    embed_model_dir=Path("base_files/corpus/models/bge-small-en-v1.5"),
    alpha=0.7,                      # weight for dense; (1-alpha) for TF-IDF
)
hits = ret.search("Design seismic classification for RCPs", k=8)
for h in hits:
    print(h.id, h.score, h.title, h.url)

# CLI (JSONL)
python -m wbs_core.retrievy \
  --corpus-root base_files/corpus \
  --embed-model-dir base_files/corpus/models/bge-small-en-v1.5 \
  --backend hybrid --alpha 0.7 --k 8 \
  --query "Design the seismic classification for reactor coolant pumps in a PWR" \
  --jsonl

# CLI (pretty)
python -m wbs_core.retrievy \
  --corpus-root base_files/corpus \
  --backend tfidf --k 5 \
  --query-file prompts.txt \
  --pretty --max-chars 500
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional deps with graceful fallbacks
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

# Sentence embeddings for dense retrieval
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# SciPy sparse for TF-IDF matrix load
def _load_sparse_npz(path: Path):
    from scipy.sparse import load_npz  # type: ignore
    return load_npz(path)


# -----------------------------
# Small helpers
# -----------------------------

def _log(msg: str) -> None:
    print(f"[retrievy] {msg}", flush=True)


def _zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score per array; robust to constant vectors."""
    if x.size == 0:
        return x
    m = x.mean()
    s = x.std()
    if s == 0:
        return np.zeros_like(x)
    return (x - m) / s


def _minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; robust to constant vectors."""
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


@dataclass
class Retrieved:
    idx: int
    score: float
    id: str
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------
# Base retriever
# -----------------------------

class BaseRetriever:
    def __init__(self, corpus_root: Path):
        self.corpus_root = Path(corpus_root).resolve()
        self.index_dir = self.corpus_root / "index"
        self.meta_path = self.index_dir / "meta.jsonl"
        self._meta: List[Dict[str, Any]] = []
        self._load_meta()

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
                    _log("Warning: could not parse one meta.jsonl line; skipping.")
        if not meta:
            raise RuntimeError("meta.jsonl parsed but empty.")
        self._meta = meta
        _log(f"Loaded meta for {len(self._meta)} vectors.")

    @property
    def meta(self) -> List[Dict[str, Any]]:
        return self._meta

    def _to_hits(self, indices: Sequence[int], scores: Sequence[float]) -> List[Retrieved]:
        hits: List[Retrieved] = []
        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[idx]
            hits.append(Retrieved(
                idx=idx,
                score=float(score),
                id=str(m.get("id", idx)),
                text=str(m.get("text", "")),
                url=m.get("url"),
                title=m.get("title"),
                meta=m,
            ))
        return hits

    # Interface
    def search(self, query: str, k: int = 8) -> List[Retrieved]:  # pragma: no cover
        raise NotImplementedError


# -----------------------------
# Dense (FAISS) retriever
# -----------------------------

class DenseRetriever(BaseRetriever):
    """
    FAISS inner-product retriever using a local SentenceTransformer (BGE-small v1.5).
    """

    def __init__(self, corpus_root: Path, embed_model_dir: Path):
        super().__init__(corpus_root=corpus_root)
        if faiss is None:
            raise RuntimeError("FAISS is not available but required for DenseRetriever.")
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is required for DenseRetriever.")

        self.faiss_path = self.index_dir / "dense_open_bge_small_ip.faiss"
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.faiss_path}")
        self._index = faiss.read_index(str(self.faiss_path))

        self.embed_model_dir = Path(embed_model_dir).resolve()
        if not self.embed_model_dir.exists():
            raise FileNotFoundError(f"Embedding model dir not found: {self.embed_model_dir}")

        # Prefer GPU if available; SentenceTransformer handles device placement internally.
        device = "cuda" if _cuda_available() else "cpu"
        self._embedder = SentenceTransformer(str(self.embed_model_dir), device=device)
        _log(f"Dense retriever ready (device={device}).")

    def search(self, query: str, k: int = 8) -> List[Retrieved]:
        q = self._embedder.encode([query], normalize_embeddings=True)
        D, I = self._index.search(np.asarray(q, dtype=np.float32), k)
        return self._to_hits(I[0].tolist(), D[0].tolist())


# -----------------------------
# TF-IDF retriever
# -----------------------------

class TfIdfRetriever(BaseRetriever):
    """
    TF-IDF cosine-like retriever using precomputed sparse matrix and vectorizer.
    """

    def __init__(self, corpus_root: Path):
        super().__init__(corpus_root=corpus_root)
        if joblib is None:
            raise RuntimeError("joblib is required for TF-IDF retrieval.")

        self.tfidf_matrix_path = self.index_dir / "tfidf_matrix.npz"
        self.tfidf_vectorizer_path = self.index_dir / "tfidf_vectorizer.joblib"
        if not self.tfidf_matrix_path.exists() or not self.tfidf_vectorizer_path.exists():
            raise FileNotFoundError(
                f"Missing TF-IDF files: {self.tfidf_matrix_path} / {self.tfidf_vectorizer_path}"
            )

        self._X = _load_sparse_npz(self.tfidf_matrix_path)
        self._v = joblib.load(self.tfidf_vectorizer_path)
        _log("TF-IDF retriever ready.")

    def search(self, query: str, k: int = 8) -> List[Retrieved]:
        q = self._v.transform([query])          # (1, V)
        # Score = raw dot product (approximate); you can L2-normalize rows for cosine if desired.
        scores = (self._X @ q.T).toarray().reshape(-1)
        top = np.argsort(-scores)[:k]
        return self._to_hits(top.tolist(), scores[top].tolist())


# -----------------------------
# Hybrid retriever
# -----------------------------

class HybridRetriever(BaseRetriever):
    """
    Hybrid of dense and TF-IDF. If one backend is missing, it degrades gracefully to the other.

    score = alpha * dense_score_norm + (1-alpha) * tfidf_score_norm
    """

    def __init__(self,
                 corpus_root: Path,
                 embed_model_dir: Optional[Path] = None,
                 alpha: float = 0.7):
        super().__init__(corpus_root=corpus_root)
        self.alpha = float(alpha)
        self._dense: Optional[DenseRetriever] = None
        self._tfidf: Optional[TfIdfRetriever] = None

        # Try dense first if embed model is given and FAISS exists
        faiss_path = self.corpus_root / "index" / "dense_open_bge_small_ip.faiss"
        if embed_model_dir and faiss is not None and faiss_path.exists() and SentenceTransformer is not None:
            try:
                self._dense = DenseRetriever(corpus_root, embed_model_dir)
            except Exception as e:
                _log(f"Dense backend unavailable: {e}")

        # Try TF-IDF
        try:
            self._tfidf = TfIdfRetriever(corpus_root)
        except Exception as e:
            _log(f"TF-IDF backend unavailable: {e}")

        if self._dense is None and self._tfidf is None:
            raise RuntimeError("No retrieval backend available (neither dense nor TF-IDF).")

    def search(self, query: str, k: int = 8) -> List[Retrieved]:
        # If only one backend available, use it directly
        if self._dense is None and self._tfidf is not None:
            return self._tfidf.search(query, k)
        if self._tfidf is None and self._dense is not None:
            return self._dense.search(query, k)
        assert self._dense is not None and self._tfidf is not None

        # Get a wider pool from each side to reduce chance of missing overlaps
        pool_k = max(k * 3, 24)
        d_hits = self._dense.search(query, pool_k)
        t_hits = self._tfidf.search(query, pool_k)

        # Build score vectors aligned on global row indices
        # Collect union of indices
        idxs = sorted({h.idx for h in d_hits}.union({h.idx for h in t_hits}))
        if not idxs:
            return []

        dense_scores = np.zeros(len(idxs), dtype=np.float32)
        tfidf_scores = np.zeros(len(idxs), dtype=np.float32)

        pos = {idx: i for i, idx in enumerate(idxs)}
        for h in d_hits:
            dense_scores[pos[h.idx]] = h.score
        for h in t_hits:
            tfidf_scores[pos[h.idx]] = h.score

        # Normalize scores per query then fuse
        d_norm = _minmax_normalize(dense_scores)
        t_norm = _minmax_normalize(tfidf_scores)
        fused = self.alpha * d_norm + (1.0 - self.alpha) * t_norm

        # Rank, slice top-k
        order = np.argsort(-fused)[:k]
        top_idxs = [idxs[i] for i in order.tolist()]
        top_scores = [float(fused[i]) for i in order.tolist()]
        return self._to_hits(top_idxs, top_scores)


# -----------------------------
# CLI
# -----------------------------

def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def _iter_queries(args: argparse.Namespace) -> Iterable[str]:
    if args.query:
        yield args.query
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    yield q


def _print_pretty(q: str, hits: List[Retrieved], max_chars: int = 600) -> None:
    print("=" * 80)
    print(f"QUERY: {q}")
    print("-" * 80)
    for i, h in enumerate(hits, 1):
        txt = h.text.replace("\n", " ").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars] + " ..."
        title = f"{h.title} " if h.title else ""
        url = f"\nURL: {h.url}" if h.url else ""
        print(f"[{i}] score={h.score:.4f} id={h.id} {title}{url}")
        print(txt)
        print("-" * 80)


def _emit_jsonl(q: str, hits: List[Retrieved]) -> None:
    rec = {
        "query": q,
        "results": [
            {
                "rank": i + 1,
                "score": h.score,
                "id": h.id,
                "text": h.text,
                "url": h.url,
                "title": h.title,
                "meta": h.meta,
                "row_index": h.idx,
            }
            for i, h in enumerate(hits)
        ],
    }
    print(json.dumps(rec, ensure_ascii=False))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Local retrieval (dense / tfidf / hybrid)")
    parser.add_argument("--corpus-root", type=str, required=True, help="Path to corpus root folder")
    parser.add_argument("--embed-model-dir", type=str, default=None, help="Local SentenceTransformer dir (for dense or hybrid)")
    parser.add_argument("--backend", type=str, choices=["dense", "tfidf", "hybrid"], default="hybrid", help="Retriever backend")
    parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid weight for dense [0..1]")
    parser.add_argument("--k", type=int, default=8, help="Top-k results")
    parser.add_argument("--query", type=str, help="Single query string")
    parser.add_argument("--query-file", type=str, help="File with one query per line")
    parser.add_argument("--jsonl", action="store_true", help="Emit JSONL records")
    parser.add_argument("--pretty", action="store_true", help="Pretty print results")
    parser.add_argument("--max-chars", type=int, default=600, help="Max characters per hit when pretty printing")

    args = parser.parse_args(argv)

    corpus_root = Path(args.corpus_root).resolve()
    embed_model_dir = Path(args.embed_model_dir).resolve() if args.embed_model_dir else None

    # Instantiate backend
    if args.backend == "dense":
        if embed_model_dir is None:
            parser.error("--embed-model-dir is required for dense backend")
        retriever = DenseRetriever(corpus_root=corpus_root, embed_model_dir=embed_model_dir)
    elif args.backend == "tfidf":
        retriever = TfIdfRetriever(corpus_root=corpus_root)
    else:  # hybrid
        retriever = HybridRetriever(corpus_root=corpus_root, embed_model_dir=embed_model_dir, alpha=args.alpha)

    any_output = False
    for q in _iter_queries(args):
        hits = retriever.search(q, k=args.k)
        if args.jsonl:
            _emit_jsonl(q, hits)
        else:
            # default to pretty if no explicit output mode
            _print_pretty(q, hits, max_chars=args.max_chars)
        any_output = True

    if not any_output:
        parser.error("Provide --query or --query-file")

if __name__ == "__main__":
    main()
