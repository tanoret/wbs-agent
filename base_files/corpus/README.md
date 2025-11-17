# Corpus Build — README

This README covers how to build and use a reproducible, offline-friendly corpus and search index of NRC/IAEA rules and guidance to ground the WBS generator.

---

## What you get

- Deterministic fetch of **CFR**, **NRC Regulatory Guides (RGs)**, and **NUREG-0800 (SRP)** using mirrors that avoid eCFR/FederalRegister CAPTCHA walls.
- Robust parsing (PDF/HTML → normalized text JSON).
- Legally meaningful **chunking** (good for citations).
- A search index:
  - **Dense (FAISS)** if a local embedding model is available, or
  - **Pure offline TF‑IDF** fallback (no internet required).

Outputs live under `parsed/`, `chunks/`, and `index/` for easy integration with the WBS generator (Task 1).

---

## Repository layout

```
corpus/
  manifests/
    corpus_manifest.yaml   # or manifest.yml (both supported)
  raw/                     # downloaded source files (read-only)
  parsed/                  # 1 JSON per document (clean text + metadata)
  chunks/                  # JSONL per document (chunked passages)
  index/                   # FAISS or TF-IDF artifacts (+ meta)
  scripts/
    01_download.py
    02_parse_normalize.py
    03_chunk.py
    04_build_faiss.py
    retrieve.py            # (optional) quick local search
```

---

## Requirements

Install once (CPU-only is fine):

```bash
pip install pdfminer.six trafilatura requests pyyaml
pip install sentence-transformers faiss-cpu      # optional (dense mode)
pip install scikit-learn joblib scipy            # used by TF-IDF fallback
```

> **Corporate SSL**: if your network uses a TLS‑intercepting proxy, set the CA bundle (ask IT for the PEM):
> ```bash
> export REQUESTS_CA_BUNDLE=/path/to/corporate_root_ca.pem
> export SSL_CERT_FILE=/path/to/corporate_root_ca.pem
> ```
> **Proxies (if required)**: set `HTTP_PROXY`, `HTTPS_PROXY`.

---

## Manifests

Both are supported:
- `manifests/corpus_manifest.yaml` with top-level `documents`, or
- `manifests/manifest.yml` with top-level `sources`.

Each entry has a stable `id`, a primary URL and mirrors, and (optionally) a deterministic `filename`. Regulatory Guides point to **direct ADAMS PDF links** to avoid blocked package pages.

**Limit to a subset during testing**:
```bash
ONLY_IDS="cfr-10-50-36,rg-1-28-rev6" python scripts/01_download.py
```

---

## Step-by-step

### 1) Download (mirror-aware, deterministic filenames)

```bash
python scripts/01_download.py
```

- Auto-detects which manifest file you’re using.
- Detects eCFR/FR “Request Access” HTML and falls back to mirrors (NRC Reading Room, LII, GovInfo, ww2.nrc).
- Saves to a deterministic filename from the manifest (`filename`) or `{id}.{format}`.
- Honors corporate CA/proxy env vars.

Expected log snippet:
```
[INFO] Using manifest: .../manifests/manifest.yml
downloaded: cfr-10-50-36 <- https://... -> raw/cfr_10_50_36.html
...
```

### 2) Parse & normalize (PDF/HTML → JSON)

```bash
python scripts/02_parse_normalize.py
```

Emits `parsed/<id>.json` with:
```json
{
  "doc_id": "...",
  "source_url": "...",
  "source_type": "cfr|regulatory_guide|nureg|iaea",
  "title": "...",
  "rev_date": "YYYY-MM-DD",
  "publisher": "NRC|IAEA|Other",
  "license": "public-domain",
  "text": "normalized plain text..."
}
```

The parser fails **loudly** if any raw files are missing and prints the IDs plus the filename attempts it made.

### 3) Chunk (legally meaningful slices)

```bash
python scripts/03_chunk.py
```

- Splits by section headings / “§” markers / numbered headings.
- Targets ~700–1200 words; merges tiny tails to avoid splitting acceptance criteria mid-list.
- Writes `chunks/<doc_id>.jsonl` (one JSON per chunk).

Example record:
```json
{
  "doc_id": "cfr-10-50-36",
  "source_url": "https://...",
  "source_type": "cfr",
  "title": "10 CFR §50.36 — Technical specifications",
  "rev_date": "2025-11-14",
  "text": "…chunk text…",
  "chunk_no": 12
}
```

### 4) Build the index (dense FAISS or offline TF‑IDF)

```bash
python scripts/04_build_faiss.py
```

- If a **local** Sentence‑Transformers model exists (e.g., `models/bge-small-en-v1.5`) and the library is installed:
  - Encodes all chunks and writes `index/dense_open_bge_small_ip.faiss`.
- Otherwise (no model or no internet):
  - Builds TF‑IDF artifacts: `index/tfidf_vectorizer.joblib` and `index/tfidf_matrix.npz`.
- Always writes `index/meta.jsonl` aligned with vectors/rows.

**Point to a local model path (no internet needed):**
```bash
export EMBED_LOCAL_PATH=models/bge-small-en-v1.5
```

---

## Quick smoke tests

Use the helper to query either index type:

```bash
python scripts/retrieve.py --q "technical specifications will include items in the following categories" --k 5
```

Expected: top hits should be from **10 CFR §50.36(c)** with a relevant snippet.

---

## How to use with the WBS generator

At inference time, perform retrieval (dense or TF‑IDF) with the user’s query plus topic expansions (e.g., “GDC 3”, “Appendix B”, “RG 1.28”). For each hit, pass `{citation, source_url, snippet, chunk_id}` to the LLM and **require it to cite** the chunks used. Validate outputs against your JSON schema.

---

## Configuration knobs

- `ONLY_IDS` — limit downloads to a subset of manifest IDs.
- `REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` — trust your corporate root CA for SSL interception.
- `HTTP_PROXY` / `HTTPS_PROXY` — proxy support.
- `EMBED_LOCAL_PATH` — local directory containing a Sentence‑Transformers model (e.g., `models/bge-small-en-v1.5`).

---

## Troubleshooting

**Downloader prints “exhausted URLs”**  
• The ADAMS or mirror link may be stale. Swap in another mirror (GovInfo/ww2.nrc/LII) or update the manifest entry.

**Parser says “Parsed 0 documents.”**  
• Manifest/top-level key mismatch. Use either `documents` (corpus_manifest.yaml) or `sources` (manifest.yml). The provided parser handles both.

**Hugging Face SSL errors (self‑signed in chain)**  
• Use offline TF‑IDF mode (fallback), or download the embedding model externally and copy it into `models/`, then set `EMBED_LOCAL_PATH`.

**eCFR / Federal Register “Request Access” wall**  
• The downloader detects this and falls back to mirrors. If a specific CFR section still fails, add another mirror to the manifest.

---

## Data governance & licensing

- **U.S. NRC / CFR** texts are public domain.  
- **IAEA PDFs** (if added later) are fine for internal retrieval; avoid redistribution.  
- Keep `raw/` read‑only and preserve provenance in `parsed/` to ease audits.

---

## Next steps (optional)

- Stand up a tiny FastAPI retrieval service that exposes a `/search?q=&k=` endpoint returning `{citation, url, snippet}` for top‑k.  
- Add IAEA SSR/SSG documents (Design/DSA/PSA) for broader coverage of non‑LWR designs.  
- Add JSON Schema & guardrails to force complete WBS tasks with mapped citations.

---

**Changelog**  
- 2025‑11‑14: Initial version with mirror-aware downloader, dual-schema manifest support, offline TF‑IDF fallback, and deterministic filenames.
