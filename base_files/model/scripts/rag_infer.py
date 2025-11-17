#!/usr/bin/env python3
import argparse, json, os, sys, re, math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Avoid bitsandbytes import on Python 3.14 by forcing PEFT to think bnb is unavailable
import peft
try:
    import peft.import_utils as _peft_import_utils
    _peft_import_utils.is_bnb_available = lambda: False
    import peft.tuners.lora.model as _lora_model
    _lora_model.is_bnb_available = lambda: False
except Exception:
    pass
from peft import PeftModel

# ---- Retrieval helpers -------------------------------------------------------

def _load_meta(meta_path: Path):
    items = []
    if meta_path.suffix.lower() == ".jsonl":
        for line in open(meta_path, "r", encoding="utf-8"):
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    else:
        try:
            items = json.loads(open(meta_path, "r", encoding="utf-8").read())
        except Exception:
            items = []
    return items

def _load_faiss_retriever(corpus_root: Path):
    try:
        import faiss  # type: ignore
        import numpy as np  # noqa: F401
    except Exception as e:
        print(f"[WARN] FAISS unavailable: {e}", file=sys.stderr)
        return None

    index_dir = corpus_root / "index"
    idx_candidates = ["faiss.index", "faiss.bin", "index.faiss", "index.bin"]
    idx_path = next((index_dir / n for n in idx_candidates if (index_dir / n).exists()), None)
    meta_candidates = ["meta.jsonl", "meta.json", "chunks.jsonl"]
    meta_path = next((index_dir / n for n in meta_candidates if (index_dir / n).exists()), None)

    if not idx_path or not meta_path:
        print("[INFO] No FAISS index found; will fallback to TF-IDF.", file=sys.stderr)
        return None

    print(f"[INFO] Using FAISS index: {idx_path.name} with meta: {meta_path.name}")
    index = faiss.read_index(str(idx_path))
    meta  = _load_meta(meta_path)
    if not meta:
        print("[WARN] Meta is empty; fallback to TF-IDF.", file=sys.stderr)
        return None

    st_model = None
    embed_local = os.environ.get("EMBED_LOCAL_PATH")
    if embed_local:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            st_model = SentenceTransformer(embed_local)
            print(f"[INFO] Loaded local embedder: {embed_local}")
        except Exception as e:
            print(f"[WARN] Could not load local embedder ({embed_local}): {e}", file=sys.stderr)

    def _encode(q: str):
        if st_model is None:
            raise RuntimeError("No local embedding model available; set EMBED_LOCAL_PATH for FAISS.")
        v = st_model.encode([q], normalize_embeddings=True)
        return v

    def search(query: str, k: int = 6):
        v = _encode(query)
        D, I = index.search(v, k)
        out = []
        for d, i in zip(D[0], I[0]):
            if i < 0 or i >= len(meta): 
                continue
            m = meta[i]
            out.append({
                "doc_id": m.get("doc_id") or m.get("id") or m.get("source_id") or f"meta{i}",
                "chunk_no": int(m.get("chunk_no", i)),
                "text": m.get("text") or m.get("chunk_text") or "",
                "score": float(d),
                "title": m.get("title") or m.get("doc_title"),
                "source": m.get("source") or m.get("path") or m.get("url"),
            })
        return out

    return search

def _extract_chunks_from_obj(obj, fallback_doc_id, fallback_title, fallback_source):
    """Return list[dict(doc_id, chunk_no, text, title, source)] from many possible shapes."""
    chunks = []

    def _add(txt, j, doc_id=None, title=None, source=None):
        t = (txt or "").strip()
        if not t:
            return
        chunks.append({
            "doc_id": doc_id or fallback_doc_id,
            "chunk_no": int(j),
            "text": t,
            "title": title or fallback_title,
            "source": source or fallback_source,
        })

    if isinstance(obj, dict):
        doc_id = obj.get("id") or obj.get("doc_id") or fallback_doc_id
        title  = obj.get("title") or fallback_title
        source = obj.get("source") or obj.get("url") or fallback_source

        # Case A: {"chunks":[...]}
        chs = obj.get("chunks")
        if isinstance(chs, list):
            for j, ch in enumerate(chs):
                if isinstance(ch, dict):
                    txt = ch.get("text") or ch.get("chunk_text") or ch.get("content")
                    if not txt and isinstance(ch.get("lines"), list):
                        txt = "\n".join(ch["lines"])
                else:
                    txt = str(ch)
                _add(txt, j, doc_id, title, source)
            return chunks

        # Case B: {"content": ...} in list or str
        content = obj.get("content") or obj.get("body") or obj.get("text")
        if isinstance(content, list):
            for j, item in enumerate(content):
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or ""
                else:
                    txt = str(item)
                _add(txt, j, doc_id, title, source)
            return chunks
        if isinstance(content, str):
            # naive split if it's a long string
            step = 1000
            for j in range(0, len(content), step):
                _add(content[j:j+step], j//step, doc_id, title, source)
            return chunks

        # Case C: {"sections":[...]} or {"paragraphs":[...]}
        sections = obj.get("sections") or obj.get("paragraphs")
        if isinstance(sections, list):
            j = 0
            for sec in sections:
                if isinstance(sec, dict):
                    txt = sec.get("text") or sec.get("content") or ""
                else:
                    txt = str(sec)
                if (txt or "").strip():
                    _add(txt, j, doc_id, title, source)
                    j += 1
            return chunks

    # Case D: List of chunks directly
    if isinstance(obj, list):
        for j, sec in enumerate(obj):
            if isinstance(sec, dict):
                txt = sec.get("text") or sec.get("content") or ""
            else:
                txt = str(sec)
            _add(txt, j, fallback_doc_id, fallback_title, fallback_source)
        return chunks

    return chunks

def _load_tfidf_retriever(corpus_root: Path):
    """Build a TF-IDF retriever over parsed/*.json and parsed/*.jsonl."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    parsed_dir = corpus_root / "parsed"

    files = sorted(list(parsed_dir.glob("*.json")) + list(parsed_dir.glob("*.jsonl")))
    if not files:
        raise SystemExit("[FATAL] No parsed JSON/JSONL files found under corpus/parsed")

    docs = []
    for p in files:
        fallback_doc_id = p.stem
        fallback_title  = p.stem
        fallback_source = p.name
        try:
            if p.suffix.lower() == ".jsonl":
                for i, line in enumerate(open(p, "r", encoding="utf-8")):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    chunks = _extract_chunks_from_obj(obj, f"{fallback_doc_id}", fallback_title, fallback_source)
                    # ensure unique chunk_no within file by offset
                    for j, ch in enumerate(chunks):
                        ch["chunk_no"] = j  # reindex within this file
                        docs.append(ch)
            else:
                obj = json.loads(open(p, "r", encoding="utf-8").read())
                chunks = _extract_chunks_from_obj(obj, fallback_doc_id, fallback_title, fallback_source)
                for j, ch in enumerate(chunks):
                    ch["chunk_no"] = j
                    docs.append(ch)
        except Exception as e:
            print(f"[WARN] Could not parse {p.name}: {e}", file=sys.stderr)

    # Filter any empties
    docs = [d for d in docs if (d.get("text") or "").strip()]
    if not docs:
        raise SystemExit("[FATAL] Parsed directory was found but no text chunks could be extracted. "
                         "Check the JSON structure or rebuild with 02_parse_normalize.py")

    corpus = [d["text"] for d in docs]
    tfidf = TfidfVectorizer(max_features=200_000)
    X = tfidf.fit_transform(corpus)
    print(f"[INFO] TF-IDF built over {len(files)} files / {len(docs)} chunks.")

    def search(query: str, k: int = 6):
        qv = tfidf.transform([query])
        sims = (qv @ X.T).toarray()[0]
        idx = sims.argsort()[::-1][:k]
        out = []
        for i in idx:
            out.append({**docs[i], "score": float(sims[i])})
        return out

    return search

def load_retriever(corpus_root: Path):
    s = _load_faiss_retriever(corpus_root)
    if s is None:
        s = _load_tfidf_retriever(corpus_root)
    return s

def build_prompt(query: str, retrieved):
    header = (
        "You are a nuclear licensing & engineering assistant.\n"
        "Use ONLY the context chunks below. Answer with a structured WBS JSON that cites the chunk IDs you used.\n\n"
    )
    ctx_lines = []
    for r in retrieved:
        tag = f"{r['doc_id']}#{r['chunk_no']}"
        ctx_lines.append(f"[{tag}] {r['text']}".strip())
    ctx = "\n\n".join(ctx_lines)

    instr = (
        "\n\nTask:\n"
        f"{query}\n\n"
        "Output JSON schema:\n"
        "{\n"
        '  "wbs_title": str,\n'
        '  "tasks": [\n'
        "    {\n"
        '      "name": str,\n'
        '      "description": str,\n'
        '      "applicable_codes": [str],\n'
        '      "guidance": str,\n'
        '      "citations": [{"id": "doc#chunk"}]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Ensure every task has at least one citation id from the provided context.\n"
    )
    return header + ctx + instr

# ---- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="LOCAL base model path (merged or original)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (omit if you merged)")
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--corpus-root", default="corpus")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load retriever
    search = load_retriever(Path(args.corpus_root))

    # Load tokenizer & model locally
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)

    model.eval().to(device)

    # Build prompt with retrieved context
    retrieved = search(args.q, args.k)
    prompt = build_prompt(args.q, retrieved)

    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
    text = tok.decode(out[0], skip_special_tokens=True)

    m = re.search(r"\{.*\}\s*$", text, re.S)
    payload = text[m.start():] if m else text

    result = {
        "query": args.q,
        "retrieved": [
            {"id": f"{r['doc_id']}#{r['chunk_no']}", "score": r.get("score"), "title": r.get("title"), "source": r.get("source")}
            for r in retrieved
        ],
        "output": payload,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    Path("model").mkdir(exist_ok=True)
    Path("model/last_response.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] Wrote model/last_response.json")

if __name__ == "__main__":
    main()
