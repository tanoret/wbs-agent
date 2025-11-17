# scripts/04_build_index.py
import os, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
CHUNKS, INDEX = ROOT/"chunks", ROOT/"index"
INDEX.mkdir(exist_ok=True)

# Gather all chunks
meta, texts = [], []
for f in CHUNKS.glob("*.jsonl"):
    for line in f.read_text().splitlines():
        obj = json.loads(line)
        meta.append(obj)
        texts.append(obj["text"])

# Try local sentence-transformers model first
LOCAL_MODEL = Path(os.getenv("EMBED_LOCAL_PATH","models/bge-small-en-v1.5"))
USE_TFIDF = True
emb = None

if LOCAL_MODEL.exists():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(LOCAL_MODEL))
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        USE_TFIDF = False
        print(f"Using local dense model at {LOCAL_MODEL}")
    except Exception as e:
        print("Dense model load failed, falling back to TF-IDF:", e)

if USE_TFIDF:
    print("Building offline TF-IDF index…")
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib, scipy.sparse as sp
    vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=2)
    mat = vect.fit_transform(texts)   # sparse CSR
    joblib.dump(vect, INDEX/"tfidf_vectorizer.joblib")
    sp.save_npz(INDEX/"tfidf_matrix.npz", mat)
    (INDEX/"meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    print("TF-IDF index built:", mat.shape)
else:
    import faiss
    vecs = np.array(emb, dtype="float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX/"dense_open_bge_small_ip.faiss"))
    (INDEX/"meta.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
    print("FAISS index built:", len(meta))
