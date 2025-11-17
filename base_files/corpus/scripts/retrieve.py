# scripts/retrieve.py
# Example: python scripts/retrieve.py --q "technical specifications will include items in the following categories"

import json, numpy as np
from pathlib import Path
import argparse

ROOT = Path(__file__).parents[1]
INDEX = ROOT/"index"

def load_dense():
    import faiss, json
    idx = faiss.read_index(str(INDEX/"dense_open_bge_small_ip.faiss"))
    meta = [json.loads(l) for l in (INDEX/"meta.jsonl").read_text().splitlines()]
    return idx, meta

def load_tfidf():
    import joblib, scipy.sparse as sp, json
    vect = joblib.load(INDEX/"tfidf_vectorizer.joblib")
    mat = sp.load_npz(INDEX/"tfidf_matrix.npz")
    meta = [json.loads(l) for l in (INDEX/"meta.jsonl").read_text().splitlines()]
    return vect, mat, meta

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--q", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    dense_path = INDEX/"dense_open_bge_small_ip.faiss"
    if dense_path.exists():
        from sentence_transformers import SentenceTransformer
        import os
        LOCAL_MODEL = Path(os.getenv("EMBED_LOCAL_PATH","models/bge-small-en-v1.5"))
        model = SentenceTransformer(str(LOCAL_MODEL))
        idx, meta = load_dense()
        qv = model.encode([args.q], normalize_embeddings=True).astype("float32")
        D, I = idx.search(qv, args.k)
        for d, i in zip(D[0], I[0]):
            m = meta[i]
            print(f"{d:.3f} | {m['doc_id']}#{m['chunk_no']} | {m['title']} | {m['source_url']}\n  {m['text'][:200]}…\n")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        vect, mat, meta = load_tfidf()
        qv = vect.transform([args.q])
        sims = linear_kernel(qv, mat).ravel()
        top = sims.argsort()[::-1][:args.k]
        for i in top:
            m = meta[i]
            print(f"{sims[i]:.3f} | {m['doc_id']}#{m['chunk_no']} | {m['title']} | {m['source_url']}\n  {m['text'][:200]}…\n")
