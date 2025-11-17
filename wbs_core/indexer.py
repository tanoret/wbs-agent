"""
wbs_core.indexer
================

Builds retrieval indexes from chunked corpus files:

1) Dense FAISS (inner-product, cosine-equivalent with L2-normalized embeddings)
   - Uses a *local* Sentence-Transformers model (no internet required)
   - Saves:
       base_files/corpus/index/dense_*.faiss
       base_files/corpus/index/meta.jsonl          (1:1 with vectors)
       base_files/corpus/index/dense_config.json   (metadata: dim, model, etc.)

2) TF-IDF fallback (scikit-learn)
   - Saves:
       base_files/corpus/index/tfidf_vectorizer.joblib
       base_files/corpus/index/tfidf_matrix.npz

By default, this module expects:
   chunks in: base_files/corpus/chunks/*.jsonl
   index out: base_files/corpus/index/

It aligns with your earlier scripts (03_chunk_and_embed / 04_build_faiss),
but is more robust and offline-friendly.

Typical usage
-------------
>>> from wbs_core.indexer import build_dense_faiss, build_tfidf, build_all
>>> build_all()  # builds FAISS + TF-IDF with sane defaults

CLI
---
python -m wbs_core.indexer --help

Notes
-----
• Ensure you have a local embed model at:
    base_files/corpus/models/bge-small-en-v1.5
  or override via --model-dir.
• The chunk JSONL must contain at least a "text" field. "meta" is optional.

"""
from __future__ import annotations

import os
import re
import io
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Optional heavy deps are imported lazily with clear error messages.
_FAISS = None
_SENTENCE_TRANSFORMERS = None

from . import paths as _discover_paths, WbsPaths

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIRNAME = "bge-small-en-v1.5"
DEFAULT_DENSE_NAME = "dense_open_bge_small_ip.faiss"

def _lazy_import_faiss():
    global _FAISS
    if _FAISS is None:
        try:
            import faiss  # type: ignore
            _FAISS = faiss
        except Exception as e:
            raise RuntimeError(
                "FAISS is required. On CPU-only systems: `pip install faiss-cpu`. "
                "On GPU systems: `pip install faiss-gpu`."
            ) from e
    return _FAISS

def _lazy_import_st():
    global _SENTENCE_TRANSFORMERS
    if _SENTENCE_TRANSFORMERS is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _SENTENCE_TRANSFORMERS = SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers is required. Install with:\n"
                "  pip install sentence-transformers"
            ) from e
    return _SENTENCE_TRANSFORMERS

def _iter_chunk_files(chunks_dir: Path) -> List[Path]:
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl chunk files found in {chunks_dir}")
    return files

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip malformed rows but keep going
                continue

def _clean_text(s: str) -> str:
    # Lightweight clean to normalize whitespace
    return re.sub(r"\s+", " ", s).strip()

def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# ---------------------------------------------------------------------------
# FAISS (dense) index building
# ---------------------------------------------------------------------------

def build_dense_faiss(
    p: Optional[WbsPaths] = None,
    chunks_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    batch_size: int = 64,
    normalize: bool = True,
    index_name: str = DEFAULT_DENSE_NAME,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """
    Build a dense FAISS (IP) index from JSONL chunks using a local ST model.

    Parameters
    ----------
    p : WbsPaths
        Optional project paths. If None, will auto-detect.
    chunks_dir : Path
        Directory with *.jsonl chunk files (default: base_files/corpus/chunks)
    out_dir : Path
        Output directory for index (default: base_files/corpus/index)
    model_dir : Path
        Local Sentence-Transformers model directory
        (default: base_files/corpus/models/bge-small-en-v1.5)
    batch_size : int
        Encode batch size
    normalize : bool
        L2 normalize embeddings before adding to IP index (cosine sim)
    index_name : str
        Output FAISS filename
    limit : int or None
        For debugging: maximum number of chunks to index

    Returns
    -------
    dict with keys: total, dim, index_path, meta_path, config_path
    """
    paths = p or _discover_paths()
    chunks_dir = chunks_dir or (paths.corpus_root / "chunks")
    out_dir = out_dir or (paths.corpus_root / "index")
    _ensure_dirs(out_dir)

    model_dir = model_dir or (paths.corpus_models / DEFAULT_MODEL_DIRNAME)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local embed model not found at {model_dir}.\n"
            "Download a Sentence-Transformers model to this folder "
            "(e.g., bge-small-en-v1.5) or pass --model-dir."
        )

    SentenceTransformer = _lazy_import_st()
    faiss = _lazy_import_faiss()

    model = SentenceTransformer(str(model_dir))
    files = _iter_chunk_files(chunks_dir)

    # Prepare FAISS after getting dim from a test encode
    # We stream to avoid loading everything into RAM at once
    meta_path = out_dir / "meta.jsonl"
    index_path = out_dir / index_name
    config_path = out_dir / "dense_config.json"

    # truncate meta if exists (we rebuild cleanly)
    with meta_path.open("w", encoding="utf-8") as _:
        pass

    # First batch to determine dim
    texts_probe: List[str] = []
    probe_meta: List[dict] = []
    count = 0
    for f in files:
        for j in _iter_jsonl(f):
            t = _clean_text(j.get("text", ""))
            if not t:
                continue
            texts_probe.append(t)
            probe_meta.append(j.get("meta", {}))
            count += 1
            if len(texts_probe) >= 8:  # small probe
                break
        if len(texts_probe) >= 8:
            break

    if not texts_probe:
        raise RuntimeError(f"No text rows found under {chunks_dir}")

    # get test dim
    emb_probe = model.encode(texts_probe, batch_size=len(texts_probe), convert_to_numpy=True, normalize_embeddings=normalize)
    dim = int(emb_probe.shape[1])

    # Create FAISS IP index (cosine when normalized)
    index = faiss.IndexFlatIP(dim)

    # Now stream through everything again, batching
    total = 0
    batch_texts: List[str] = []
    batch_metas: List[dict] = []
    with meta_path.open("a", encoding="utf-8") as meta_out:
        for f in files:
            for ln, j in enumerate(_iter_jsonl(f), 1):
                txt = _clean_text(j.get("text", ""))
                if not txt:
                    continue
                batch_texts.append(txt)
                # Build a consistent meta line
                m = j.get("meta", {})
                m_out = {
                    "chunk_file": f.name,
                    "line": ln,
                    "source_id": m.get("source_id") or m.get("doc_id") or "",
                    "section_id": m.get("section_id") or m.get("sec_id") or "",
                    "cite": m.get("cite") or "",
                    "text_len": len(txt),
                }
                batch_metas.append(m_out)

                if len(batch_texts) >= batch_size:
                    emb = model.encode(
                        batch_texts,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                        show_progress_bar=False,
                    )
                    index.add(emb.astype(np.float32))

                    for mm in batch_metas:
                        meta_out.write(json.dumps(mm, ensure_ascii=False) + "\n")

                    total += len(batch_texts)
                    batch_texts.clear()
                    batch_metas.clear()

                if limit is not None and total >= limit:
                    break
            if limit is not None and total >= limit:
                break

        # flush tail
        if batch_texts:
            emb = model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            index.add(emb.astype(np.float32))
            for mm in batch_metas:
                meta_out.write(json.dumps(mm, ensure_ascii=False) + "\n")
            total += len(batch_texts)
            batch_texts.clear()
            batch_metas.clear()

    # Save FAISS index
    faiss.write_index(index, str(index_path))

    # Config for traceability
    cfg = {
        "created_utc": _now(),
        "dim": dim,
        "total": total,
        "index_type": "IndexFlatIP",
        "normalize": normalize,
        "model_dir": str(model_dir),
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "note": "Cosine similarity via L2-normalized embeddings in IP index",
    }
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return {
        "total": total,
        "dim": dim,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
        "config_path": str(config_path),
    }

# ---------------------------------------------------------------------------
# TF-IDF building
# ---------------------------------------------------------------------------

def build_tfidf(
    p: Optional[WbsPaths] = None,
    chunks_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    max_features: int = 200_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, object]:
    """
    Build TF-IDF vectorizer + matrix aligned to the same row order as meta.jsonl
    produced by build_dense_faiss (or simply in chunk traversal order if dense
    has not been built yet).

    Saves:
      tfidf_vectorizer.joblib
      tfidf_matrix.npz

    Parameters
    ----------
    p : WbsPaths
    chunks_dir : Path
    out_dir : Path
    limit : int or None
        For debugging: maximum number of chunks to include
    max_features : int
    ngram_range : (min_n, max_n)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from scipy.sparse import csr_matrix, save_npz  # type: ignore
    import joblib  # type: ignore

    paths = p or _discover_paths()
    chunks_dir = chunks_dir or (paths.corpus_root / "chunks")
    out_dir = out_dir or (paths.corpus_root / "index")
    _ensure_dirs(out_dir)

    files = _iter_chunk_files(chunks_dir)

    texts: List[str] = []
    count = 0
    for f in files:
        for j in _iter_jsonl(f):
            txt = _clean_text(j.get("text", ""))
            if not txt:
                continue
            texts.append(txt)
            count += 1
            if limit is not None and count >= limit:
                break
        if limit is not None and count >= limit:
            break

    if not texts:
        raise RuntimeError(f"No text rows found under {chunks_dir}")

    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
    )
    mat: csr_matrix = vec.fit_transform(texts)

    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")
    save_npz(out_dir / "tfidf_matrix.npz", mat)

    return {
        "total": mat.shape[0],
        "vocab_size": len(vec.vocabulary_),
        "vectorizer_path": str(out_dir / "tfidf_vectorizer.joblib"),
        "matrix_path": str(out_dir / "tfidf_matrix.npz"),
    }

# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def build_all(
    p: Optional[WbsPaths] = None,
    chunks_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    batch_size: int = 64,
    normalize: bool = True,
    index_name: str = DEFAULT_DENSE_NAME,
    limit: Optional[int] = None,
    tfidf_max_features: int = 200_000,
    tfidf_ngram_range: Tuple[int, int] = (1, 2),
) -> Dict[str, object]:
    """
    Build dense FAISS + TF-IDF in one call.
    """
    rep_dense = build_dense_faiss(
        p=p,
        chunks_dir=chunks_dir,
        out_dir=out_dir,
        model_dir=model_dir,
        batch_size=batch_size,
        normalize=normalize,
        index_name=index_name,
        limit=limit,
    )
    rep_tfidf = build_tfidf(
        p=p,
        chunks_dir=chunks_dir,
        out_dir=out_dir,
        limit=limit,
        max_features=tfidf_max_features,
        ngram_range=tfidf_ngram_range,
    )
    return {"dense": rep_dense, "tfidf": rep_tfidf}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build dense FAISS and/or TF-IDF indexes from chunked corpus.")
    ap.add_argument("--kind", choices=["dense", "tfidf", "both"], default="both",
                    help="Which index to build (default: both)")
    ap.add_argument("--chunks-dir", type=str, default="",
                    help="Path to chunks dir (default: base_files/corpus/chunks)")
    ap.add_argument("--out-dir", type=str, default="",
                    help="Path to index out dir (default: base_files/corpus/index)")
    ap.add_argument("--model-dir", type=str, default="",
                    help="Local Sentence-Transformers model dir (default: base_files/corpus/models/bge-small-en-v1.5)")
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (dense)")
    ap.add_argument("--no-normalize", action="store_true", help="Do NOT L2 normalize embeddings before IP index")
    ap.add_argument("--index-name", type=str, default=DEFAULT_DENSE_NAME, help="Output FAISS filename")
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit number of chunks")
    ap.add_argument("--tfidf-max-features", type=int, default=200_000, help="Max features for TF-IDF")
    ap.add_argument("--tfidf-ngram-max", type=int, default=2, help="Max n-gram size (min is 1)")
    return ap.parse_args()

def main():
    args = _parse_args()
    p = _discover_paths()

    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else (p.corpus_root / "chunks")
    out_dir = Path(args.out_dir) if args.out_dir else (p.corpus_root / "index")
    model_dir = Path(args.model_dir) if args.model_dir else (p.corpus_models / DEFAULT_MODEL_DIRNAME)
    limit = args.limit if args.limit and args.limit > 0 else None
    normalize = not args.no_normalize
    ngram_range = (1, max(1, int(args.tfidf_ngram_max)))

    if args.kind in ("dense", "both"):
        rep_d = build_dense_faiss(
            p=p,
            chunks_dir=chunks_dir,
            out_dir=out_dir,
            model_dir=model_dir,
            batch_size=args.batch_size,
            normalize=normalize,
            index_name=args.index_name,
            limit=limit,
        )
        print(json.dumps({"dense": rep_d}, indent=2))

    if args.kind in ("tfidf", "both"):
        rep_t = build_tfidf(
            p=p,
            chunks_dir=chunks_dir,
            out_dir=out_dir,
            limit=limit,
            max_features=args.tfidf_max_features,
            ngram_range=ngram_range,
        )
        print(json.dumps({"tfidf": rep_t}, indent=2))

if __name__ == "__main__":
    main()
