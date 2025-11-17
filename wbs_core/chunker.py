
"""
wbs_core.chunker
=================

A self-contained chunking & embedding utility that works directly with your
new repository layout where all working assets live under ``base_files/``.

It can:
  * iterate parsed documents in ``base_files/corpus/parsed``,
  * create JSONL chunks in ``base_files/corpus/chunks``,
  * build a TF‑IDF index (always) and a FAISS dense index (if faiss is installed),
  * provide simple retrieval helpers for later RAG steps, and
  * optionally delegate to your original scripts if you prefer.

This module is designed to be called from your Streamlit app or any Python
workflow. It is offline/HPC friendly (never phones home).

Typical usage
-------------
>>> from wbs_core.chunker import build_all, retrieve
>>> build_all()  # chunk + embed + build indexes using defaults
>>> hits = retrieve("seismic qualification of pumps", top_k=8)
>>> hits[0]["text"][:120]

If you prefer to call the *original* scripts you parked in ``base_files/``,
use the delegating wrapper in :mod:`wbs_core.__init__` (``run_chunk_embed``,
``run_build_faiss``) or call ``delegate_to_scripts()`` defined below.

Notes
-----
* Defaults expect a local embedding model at
  ``base_files/corpus/models/bge-small-en-v1.5`` created earlier.
* We normalize embeddings and use an Inner-Product FAISS index to emulate
  cosine similarity (cosine == dot-product when vectors are L2-normalized).
"""
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Optional deps
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from scipy import sparse as sp  # type: ignore
    import joblib  # type: ignore
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# Local helpers
from . import paths as _discover_paths, offline_env, run_script, WbsPaths


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """One retrieval unit."""
    id: str
    doc_id: str
    title: str
    cite: Optional[str]
    source: Optional[str]
    text: str


# ---------------------------------------------------------------------------
# Loading parsed docs
# ---------------------------------------------------------------------------

def _iter_parsed_sections(parsed_dir: Path) -> Iterator[Tuple[Dict, str]]:
    """
    Yield (meta, text) from parsed JSON files.

    We try to be liberal about schema:
      - If file has {"sections":[{"text":..., "title":..., "cite":...}, ...]}
        we yield each section.
      - Else if {"text": ...} at top, we yield one record.
      - Else if {"content": ...} we yield one record.
    """
    for f in sorted(parsed_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        doc_id = data.get("id") or f.stem
        doc_title = data.get("title") or doc_id
        source = data.get("source") or data.get("url") or None

        if isinstance(data.get("sections"), list) and data["sections"]:
            for i, s in enumerate(data["sections"]):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                title = s.get("title") or doc_title
                cite = s.get("cite") or s.get("id") or None
                meta = {"doc_id": doc_id, "title": title, "cite": cite, "source": source, "sec_index": i}
                yield meta, text
        else:
            # Single blob fallback
            text = (data.get("text") or data.get("content") or "").strip()
            if text:
                meta = {"doc_id": doc_id, "title": doc_title, "cite": None, "source": source, "sec_index": 0}
                yield meta, text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def make_chunks(
    parsed_dir: Path,
    out_dir: Path,
    max_chars: int = 1200,
    overlap: int = 200,
    min_chars: int = 200,
) -> List[Path]:
    """
    Creates JSONL chunk files (one per source document) in ``out_dir``.
    Returns the list of created JSONL paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    def split_text(txt: str) -> List[str]:
        txt = _normalize_ws(txt)
        if len(txt) <= max_chars:
            return [txt] if len(txt) >= min_chars else []
        chunks = []
        stride = max_chars - overlap
        i = 0
        while i < len(txt):
            piece = txt[i:i+max_chars]
            if len(piece) >= min_chars:
                chunks.append(piece)
            i += stride
        return chunks

    # Group chunks by source file to write one JSONL per doc
    buffers: Dict[str, List[ChunkRecord]] = {}
    for meta, text in _iter_parsed_sections(parsed_dir):
        chunks = split_text(text)
        if not chunks:
            continue
        key = meta["doc_id"]
        if key not in buffers:
            buffers[key] = []
        for j, ch in enumerate(chunks):
            cid = f"{meta['doc_id']}::{meta['sec_index']}.{j}"
            buffers[key].append(
                ChunkRecord(
                    id=cid,
                    doc_id=meta["doc_id"],
                    title=meta["title"],
                    cite=meta.get("cite"),
                    source=meta.get("source"),
                    text=ch,
                )
            )

    # Write files
    for doc_id, items in buffers.items():
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", doc_id).strip("-").lower()
        outp = out_dir / f"{safe}.jsonl"
        with outp.open("w", encoding="utf-8") as w:
            for rec in items:
                w.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        created.append(outp)

    return created


# ---------------------------------------------------------------------------
# Embedding + Indexes
# ---------------------------------------------------------------------------

def _load_embedder(local_path: Path):
    if not HAS_ST:
        raise RuntimeError("sentence-transformers is not installed in this environment.")
    if not local_path.exists():
        raise FileNotFoundError(f"Local embedding model not found at: {local_path}")
    # Pure local load (no internet)
    return SentenceTransformer(str(local_path))


def _iter_all_chunks(chunks_dir: Path) -> Iterator[Tuple[Dict, str]]:
    for f in sorted(chunks_dir.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as r:
            for line in r:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                yield obj, obj.get("text", "")


def _embed_texts(embedder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    # sentence-transformers returns numpy array already
    embs = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # ensures cosine == dot
    )
    return embs.astype("float32", copy=False)


def build_tfidf_index(
    chunks_dir: Path,
    index_dir: Path,
    vectorizer_name: str = "tfidf_vectorizer.joblib",
    matrix_name: str = "tfidf_matrix.npz",
    meta_name: str = "meta.jsonl",
) -> Path:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn / scipy / joblib are required to build TF-IDF index.")
    index_dir.mkdir(parents=True, exist_ok=True)

    texts: List[str] = []
    metas: List[Dict] = []
    for meta, text in _iter_all_chunks(chunks_dir):
        texts.append(text)
        metas.append(meta)

    if not texts:
        raise RuntimeError(f"No chunks found in {chunks_dir}")

    vec = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)
    mat = vec.fit_transform(texts)  # sparse CSR

    # Save
    joblib.dump(vec, index_dir / vectorizer_name)
    sp.save_npz(index_dir / matrix_name, mat)

    # Save meta aligned to row order
    with (index_dir / meta_name).open("w", encoding="utf-8") as w:
        for m in metas:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")

    return index_dir / matrix_name


def build_faiss_index(
    chunks_dir: Path,
    index_dir: Path,
    embedder_local_path: Path,
    faiss_name: str = "dense_open_bge_small_ip.faiss",
    meta_name: str = "meta.jsonl",
    batch_size: int = 64,
) -> Optional[Path]:
    if not HAS_FAISS:
        # Caller can still use TF-IDF
        return None

    index_dir.mkdir(parents=True, exist_ok=True)

    # Collect texts + meta
    texts: List[str] = []
    metas: List[Dict] = []
    for meta, text in _iter_all_chunks(chunks_dir):
        texts.append(text)
        metas.append(meta)
    if not texts:
        raise RuntimeError(f"No chunks found in {chunks_dir}")

    embedder = _load_embedder(embedder_local_path)
    embs = _embed_texts(embedder, texts, batch_size=batch_size)  # [N, d], L2-normalized

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via normalized vectors
    index.add(embs)

    faiss_path = index_dir / faiss_name
    faiss.write_index(index, str(faiss_path))

    # Save meta (aligned to index ids)
    with (index_dir / meta_name).open("w", encoding="utf-8") as w:
        for m in metas:
            w.write(json.dumps(m, ensure_ascii=False) + "\n")

    return faiss_path


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _load_meta(meta_path: Path) -> List[Dict]:
    metas: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as r:
        for line in r:
            try:
                metas.append(json.loads(line))
            except Exception:
                continue
    return metas


def _tfidf_search(
    query: str,
    index_dir: Path,
    vectorizer_name: str = "tfidf_vectorizer.joblib",
    matrix_name: str = "tfidf_matrix.npz",
    meta_name: str = "meta.jsonl",
    top_k: int = 8,
) -> List[Dict]:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn / scipy / joblib are required for TF-IDF retrieval.")

    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    from scipy import sparse as sp

    vec: TfidfVectorizer = joblib.load(index_dir / vectorizer_name)
    mat = sp.load_npz(index_dir / matrix_name)
    q = vec.transform([query])
    scores = (mat @ q.T).toarray().ravel()  # cosine-like for tf-idf
    top = np.argsort(-scores)[:top_k]

    metas = _load_meta(index_dir / meta_name)
    hits = []
    for i in top:
        m = metas[int(i)]
        m = dict(m)
        m["_score"] = float(scores[int(i)])
        hits.append(m)
    return hits


def _faiss_search(
    query: str,
    index_dir: Path,
    embedder_local_path: Path,
    faiss_name: str = "dense_open_bge_small_ip.faiss",
    meta_name: str = "meta.jsonl",
    top_k: int = 8,
) -> List[Dict]:
    if not HAS_FAISS:
        raise RuntimeError("FAISS not available in this environment.")

    index_path = index_dir / faiss_name
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    embedder = _load_embedder(embedder_local_path)
    q_emb = _embed_texts(embedder, [query], batch_size=1)  # [1, d], already normalized

    D, I = index.search(q_emb, top_k)  # inner product scores
    metas = _load_meta(index_dir / meta_name)

    hits = []
    for rank, (idx, score) in enumerate(zip(I[0].tolist(), D[0].tolist())):
        if idx < 0 or idx >= len(metas):
            continue
        m = dict(metas[idx])
        m["_score"] = float(score)
        m["_rank"] = int(rank + 1)
        hits.append(m)
    return hits


def retrieve(
    query: str,
    p: Optional[WbsPaths] = None,
    prefer_faiss: bool = True,
    top_k: int = 8,
) -> List[Dict]:
    """
    Convenience retrieval that prefers FAISS if present, else TF-IDF.
    Returns meta dicts (including 'text' and optional '_score', '_rank').
    """
    p = p or _discover_paths()
    index_dir = p.corpus_root / "index"

    if prefer_faiss and HAS_FAISS and (index_dir / "dense_open_bge_small_ip.faiss").exists():
        try:
            return _faiss_search(query, index_dir, p.embed_local_path, top_k=top_k)
        except Exception:
            # fall through
            pass

    # TF-IDF fallback
    if (index_dir / "tfidf_matrix.npz").exists():
        return _tfidf_search(query, index_dir, top_k=top_k)
    raise FileNotFoundError("No index found. Run build_all() first.")


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def build_all(
    p: Optional[WbsPaths] = None,
    max_chars: int = 1200,
    overlap: int = 200,
    min_chars: int = 200,
    build_dense: bool = True,
    batch_size: int = 64,
) -> Dict[str, Optional[str]]:
    """
    End-to-end:
      1) chunk parsed docs -> corpus/chunks/*.jsonl
      2) build TF-IDF index  -> corpus/index/tfidf_*.{joblib,npz}
      3) build FAISS index   -> corpus/index/dense_*.faiss (if requested & faiss available)
    Returns a small dict with paths for convenience.
    """
    p = p or _discover_paths()
    p.ensure_dirs()

    parsed_dir = p.corpus_root / "parsed"
    chunks_dir = p.corpus_root / "chunks"
    index_dir = p.corpus_root / "index"

    created = make_chunks(parsed_dir, chunks_dir, max_chars=max_chars, overlap=overlap, min_chars=min_chars)

    # Always build TF-IDF
    tfidf_path = build_tfidf_index(chunks_dir, index_dir)
    faiss_path = None
    if build_dense:
        faiss_path = build_faiss_index(chunks_dir, index_dir, embedder_local_path=p.embed_local_path, batch_size=batch_size)

    return {
        "chunks_dir": str(chunks_dir),
        "tfidf_matrix": str(tfidf_path),
        "faiss_index": str(faiss_path) if faiss_path else None,
    }


# ---------------------------------------------------------------------------
# Delegation (optional): call original scripts
# ---------------------------------------------------------------------------

def delegate_to_scripts(
    p: Optional[WbsPaths] = None,
    stream: Optional[callable] = None,
) -> Tuple[int, str, int, str]:
    """
    If you prefer the canonical pipeline you've already validated, this runs:
      - base_files/corpus/scripts/03_chunk_and_embed.py
      - base_files/corpus/scripts/04_build_faiss.py
    Returns a tuple: (ret1, out1, ret2, out2).
    """
    p = p or _discover_paths()
    # 03
    r1, o1 = run_script(
        ["python", str(p.corpus_scripts / "03_chunk_and_embed.py")],
        cwd=p.repo_root,
        env=offline_env(p, extra={"EMBED_LOCAL_PATH": str(p.embed_local_path)}),
        stream=stream,
    )
    # 04
    r2, o2 = run_script(
        ["python", str(p.corpus_scripts / "04_build_faiss.py")],
        cwd=p.repo_root,
        env=offline_env(p),
        stream=stream,
    )
    return r1, o1, r2, o2
