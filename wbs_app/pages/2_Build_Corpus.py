# wbs_app/pages/2_Build_Corpus.py
# Streamlit page: step-by-step corpus builder
# - Select manifest
# - Run individual stages (download / parse / chunk+embed / indexing)
# - Works with wbs_core.* if available; falls back to base_files/*/scripts
# - Shows progress, logs, and artifact previews

import os
import sys
import json
import time
import yaml
import shutil
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st

# ------------------------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------------------------
APP_ROOT = Path.cwd()
BASE_FILES = APP_ROOT / "base_files"

CORPUS_ROOT = BASE_FILES / "corpus"
SUP_ROOT = BASE_FILES / "supervision"
MODEL_ROOT = BASE_FILES / "model"
MODELS_DIR = BASE_FILES / "models"

RAW_DIR = CORPUS_ROOT / "raw"
PARSED_DIR = CORPUS_ROOT / "parsed"
CHUNKS_DIR = CORPUS_ROOT / "chunks"
INDEX_DIR = CORPUS_ROOT / "index"

MANIFESTS_DIR = CORPUS_ROOT / "manifests"
DEFAULT_MANIFEST = MANIFESTS_DIR / "corpus_manifest.yaml"

EMBED_DEFAULT = CORPUS_ROOT / "models" / "bge-small-en-v1.5"  # local Sentence-Transformer

SCRIPTS_CORPUS = CORPUS_ROOT / "scripts"
SCRIPT_DL = SCRIPTS_CORPUS / "01_download.py"
SCRIPT_PARSE = SCRIPTS_CORPUS / "02_parse_normalize.py"
SCRIPT_CHUNK = SCRIPTS_CORPUS / "03_chunk_and_embed.py"
SCRIPT_INDEX = SCRIPTS_CORPUS / "04_build_faiss.py"

# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def run_subprocess(py_script: Path, args: List[str], env: Optional[dict] = None) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(py_script)] + list(args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr

def artifact_counts() -> Dict[str, int]:
    def count_files(p: Path, suffixes=None):
        if not p.exists():
            return 0
        if suffixes is None:
            return sum(1 for _ in p.rglob("*") if _.is_file())
        else:
            return sum(1 for _ in p.rglob("*") if _.is_file() and _.suffix.lower() in suffixes)
    return {
        "raw": count_files(RAW_DIR),
        "parsed_json": count_files(PARSED_DIR, {".json"}),
        "chunks_jsonl": count_files(CHUNKS_DIR, {".jsonl"}),
        "index_files": count_files(INDEX_DIR),
        "faiss_index_present": 1 if (INDEX_DIR / "dense_open_bge_small_ip.faiss").exists() else 0,
        "tfidf_present": 1 if (INDEX_DIR / "tfidf_vectorizer.joblib").exists() else 0,
    }

def preview_parsed(limit: int = 5) -> List[Path]:
    files = sorted(PARSED_DIR.glob("*.json"))
    return files[:limit]

def preview_chunks(limit: int = 5) -> List[Path]:
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    return files[:limit]

def read_text(p: Path, max_bytes: int = 40_000) -> str:
    try:
        data = p.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def try_import_wbs_core():
    try:
        from wbs_core.downloader import Downloader  # type: ignore
        from wbs_core.parser import Parser  # type: ignore
        from wbs_core.chunker import Chunker  # type: ignore
        from wbs_core.indexer import Indexer  # type: ignore
        return Downloader, Parser, Chunker, Indexer
    except Exception as e:
        return None

# ------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="2 · Build Corpus (Step-by-step)", layout="wide")
st.title("② Build Corpus — Step-by-step")

# Sidebar — global settings
st.sidebar.header("Paths & Environment")
st.sidebar.caption("All paths are relative to the project root (this app's working directory).")
embed_dir_in = st.sidebar.text_input("Embedding model dir", value=str(EMBED_DEFAULT))
st.sidebar.caption("Local Sentence-Transformer (e.g., bge-small-en-v1.5).")
st.sidebar.write("**Artifacts summary**")
st.sidebar.json(artifact_counts())

st.divider()

# Manifest chooser
st.subheader("Select Manifest")
ensure_dir(MANIFESTS_DIR)
manifests = sorted(MANIFESTS_DIR.glob("*.yaml"))
if not manifests and DEFAULT_MANIFEST.exists():
    manifests = [DEFAULT_MANIFEST]
manifest_path_str = st.selectbox(
    "Manifest file",
    options=[str(p) for p in manifests] or [str(DEFAULT_MANIFEST)],
)
manifest_path = Path(manifest_path_str)

if not manifest_path.exists():
    st.warning("The selected manifest does not exist yet. Create/save one in page ① or here manually.")
with st.expander("View selected manifest", expanded=False):
    st.code(read_text(manifest_path) if manifest_path.exists() else "(missing)")

# Options per stage
st.subheader("Stage Options")

c1, c2, c3, c4 = st.columns(4)
with c1:
    do_download = st.checkbox("Download", value=True)
with c2:
    do_parse = st.checkbox("Parse/Normalize", value=True)
with c3:
    do_chunk = st.checkbox("Chunk + Embed", value=True)
with c4:
    do_index = st.checkbox("Build Index (TF-IDF + FAISS)", value=True)

st.caption("Uncheck any stage you want to skip.")

# Chunk/Embed options
with st.expander("Chunking & Embedding Options", expanded=False):
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=5000, value=1200, step=50)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=1000, value=200, step=10)
    embed_batch = st.number_input("Embedding batch size (may be ignored by script)", min_value=1, max_value=256, value=64, step=1)
    normalize_emb = st.checkbox("L2 normalize embeddings (recommended for inner-product FAISS)", value=True)

# Index options
with st.expander("Index Options", expanded=False):
    faiss_metric = st.selectbox("FAISS metric", options=["inner_product", "l2"], index=0)
    rebuild_tfidf = st.checkbox("Rebuild TF-IDF", value=True)
    rebuild_faiss = st.checkbox("Rebuild FAISS", value=True)

# Cleaning options
with st.expander("Cleanup / Reset", expanded=False):
    clean_raw = st.checkbox("Clean raw/ before Download", value=False)
    clean_parsed = st.checkbox("Clean parsed/ before Parse", value=False)
    clean_chunks = st.checkbox("Clean chunks/ before Chunk", value=False)
    clean_index = st.checkbox("Clean index/ before Index", value=False)

# Log container
log = st.container(border=True)
st.divider()

DownloaderParserChunkerIndexer = try_import_wbs_core()
using_wbs_core = DownloaderParserChunkerIndexer is not None
st.info(f"Execution mode: {'wbs_core modules' if using_wbs_core else 'base_files scripts fallback'}")

# ------------------------------------------------------------------------------------
# Runners for each stage
# ------------------------------------------------------------------------------------
def stage_download():
    if clean_raw and RAW_DIR.exists():
        shutil.rmtree(RAW_DIR, ignore_errors=True)
    ensure_dir(RAW_DIR)

    if using_wbs_core:
        Downloader, _, _, _ = DownloaderParserChunkerIndexer
        try:
            d = Downloader(corpus_root=CORPUS_ROOT, manifest_path=manifest_path)
            with st.spinner("Downloading documents..."):
                n_ok, n_fail = d.run()
            log.success(f"Download complete. OK={n_ok}, Failed={n_fail}")
        except Exception as e:
            log.error(f"Downloader error: {e}")
            raise
    else:
        if not SCRIPT_DL.exists():
            raise FileNotFoundError(f"Missing script: {SCRIPT_DL}")
        env = os.environ.copy()
        with st.spinner("Running 01_download.py ..."):
            rc, out, err = run_subprocess(SCRIPT_DL, [], env=env)
        if out.strip():
            log.code(out)
        if err.strip():
            log.warning(err)
        if rc != 0:
            raise RuntimeError("01_download.py failed")
        log.success("01_download.py completed.")

def stage_parse():
    if clean_parsed and PARSED_DIR.exists():
        shutil.rmtree(PARSED_DIR, ignore_errors=True)
    ensure_dir(PARSED_DIR)

    if using_wbs_core:
        _, Parser, _, _ = DownloaderParserChunkerIndexer
        try:
            p = Parser(corpus_root=CORPUS_ROOT)
            with st.spinner("Parsing & normalizing..."):
                n = p.run()
            log.success(f"Parsed {n} documents.")
        except Exception as e:
            log.error(f"Parser error: {e}")
            raise
    else:
        if not SCRIPT_PARSE.exists():
            raise FileNotFoundError(f"Missing script: {SCRIPT_PARSE}")
        with st.spinner("Running 02_parse_normalize.py ..."):
            rc, out, err = run_subprocess(SCRIPT_PARSE, [])
        if out.strip():
            log.code(out)
        if err.strip():
            log.warning(err)
        if rc != 0:
            raise RuntimeError("02_parse_normalize.py failed")
        log.success("02_parse_normalize.py completed.")

def stage_chunk_embed(embed_dir: Path):
    if clean_chunks and CHUNKS_DIR.exists():
        shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    ensure_dir(CHUNKS_DIR)

    if using_wbs_core:
        _, _, Chunker, _ = DownloaderParserChunkerIndexer
        try:
            c = Chunker(
                corpus_root=CORPUS_ROOT,
                embed_model_dir=embed_dir,
                chunk_size=int(chunk_size),
                chunk_overlap=int(chunk_overlap),
                normalize_embeddings=bool(normalize_emb),
                batch_size=int(embed_batch),
            )
            with st.spinner("Chunking & embedding..."):
                stats = c.run()
            log.success(f"Chunk/Embed complete. Stats: {json.dumps(stats, indent=2)}")
        except Exception as e:
            log.error(f"Chunker error: {e}")
            raise
    else:
        if not SCRIPT_CHUNK.exists():
            raise FileNotFoundError(f"Missing script: {SCRIPT_CHUNK}")
        env = os.environ.copy()
        env["EMBED_LOCAL_PATH"] = str(embed_dir)               # used by your legacy script
        env["CHUNK_SIZE"] = str(int(chunk_size))               # your script may ignore; harmless
        env["CHUNK_OVERLAP"] = str(int(chunk_overlap))
        env["NORMALIZE_EMB"] = "1" if normalize_emb else "0"
        with st.spinner("Running 03_chunk_and_embed.py ..."):
            rc, out, err = run_subprocess(SCRIPT_CHUNK, [], env=env)
        if out.strip():
            log.code(out)
        if err.strip():
            log.warning(err)
        if rc != 0:
            raise RuntimeError("03_chunk_and_embed.py failed")
        log.success("03_chunk_and_embed.py completed.")

def stage_index(embed_dir: Path):
    if clean_index and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    ensure_dir(INDEX_DIR)

    if using_wbs_core:
        _, _, _, Indexer = DownloaderParserChunkerIndexer
        try:
            idx = Indexer(
                corpus_root=CORPUS_ROOT,
                embed_model_dir=embed_dir,
                build_tfidf=rebuild_tfidf,
                build_faiss=rebuild_faiss,
                faiss_metric=faiss_metric,
                normalize_embeddings=normalize_emb,
            )
            with st.spinner("Building indices (TF-IDF & FAISS)..."):
                stats = idx.run()
            log.success(f"Index build complete. Stats: {json.dumps(stats, indent=2)}")
        except Exception as e:
            log.error(f"Indexer error: {e}")
            raise
    else:
        if not SCRIPT_INDEX.exists():
            raise FileNotFoundError(f"Missing script: {SCRIPT_INDEX}")
        env = os.environ.copy()
        env["EMBED_LOCAL_PATH"] = str(embed_dir)  # used by your legacy script
        # Optional hints; legacy script may ignore
        env["FAISS_METRIC"] = faiss_metric
        env["NORMALIZE_EMB"] = "1" if normalize_emb else "0"
        with st.spinner("Running 04_build_faiss.py ..."):
            rc, out, err = run_subprocess(SCRIPT_INDEX, [], env=env)
        if out.strip():
            log.code(out)
        if err.strip():
            log.warning(err)
        if rc != 0:
            raise RuntimeError("04_build_faiss.py failed")
        log.success("04_build_faiss.py completed.")

# ------------------------------------------------------------------------------------
# Buttons
# ------------------------------------------------------------------------------------
colA, colB, colC, colD, colE = st.columns([1,1,1,1,2])
run_any = False
with colA:
    if st.button("Run Download", disabled=not do_download):
        run_any = True
        stage_download()
with colB:
    if st.button("Run Parse", disabled=not do_parse):
        run_any = True
        stage_parse()
with colC:
    if st.button("Run Chunk+Embed", disabled=not do_chunk):
        run_any = True
        stage_chunk_embed(Path(embed_dir_in))
with colD:
    if st.button("Build Index", disabled=not do_index):
        run_any = True
        stage_index(Path(embed_dir_in))
with colE:
    if st.button("Run Selected →", type="primary"):
        run_any = True
        if do_download:
            stage_download()
        if do_parse:
            stage_parse()
        if do_chunk:
            stage_chunk_embed(Path(embed_dir_in))
        if do_index:
            stage_index(Path(embed_dir_in))

if run_any:
    st.success("Completed selected stage(s).")
    st.sidebar.json(artifact_counts())

st.divider()

# ------------------------------------------------------------------------------------
# Artifact previews
# ------------------------------------------------------------------------------------
st.subheader("Artifact Previews")

cP, cC = st.columns(2)
with cP:
    st.write("**Parsed JSON (sample)**")
    files = preview_parsed(limit=5)
    if not files:
        st.caption("No parsed files found.")
    for p in files:
        st.markdown(f"* {p.name}")
        with st.expander(f"Preview {p.name}", expanded=False):
            st.code(read_text(p))

with cC:
    st.write("**Chunked JSONL (sample)**")
    files = preview_chunks(limit=5)
    if not files:
        st.caption("No chunks found.")
    for p in files:
        st.markdown(f"* {p.name}")
        with st.expander(f"Preview head(20) of {p.name}", expanded=False):
            # Show only first 20 lines
            try:
                lines = []
                with p.open("r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh):
                        lines.append(line.rstrip("\n"))
                        if i >= 19:
                            break
                st.code("\n".join(lines))
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")

st.divider()
st.caption("Tip: If fully offline, ensure local models & set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. Also set EMBED_LOCAL_PATH to your local Sentence-Transformer.")
