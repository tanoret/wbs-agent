# wbs_app/pages/1_Manifest_and_Data.py
# Streamlit page: Manifest & Data (build or import corpus artifacts)
# - Edit/validate/save manifest.yaml
# - Upload prebuilt database (zip) with raw/parsed/chunks/index
# - Run pipeline: 01_download -> 02_parse_normalize -> 03_chunk_and_embed -> 04_build_faiss
# - Optional: auto-run supervision + SFT preparation
# - Robust to offline/HPC environments

import io
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
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st

# ---------------------------------------------------------------------
# Defaults & Paths
# ---------------------------------------------------------------------
APP_ROOT = Path.cwd()
BASE_FILES = APP_ROOT / "base_files"
CORPUS_ROOT = BASE_FILES / "corpus"
SUP_ROOT = BASE_FILES / "supervision"
MODEL_ROOT = BASE_FILES / "model"
MODELS_DIR = BASE_FILES / "models"

MANIFESTS_DIR = CORPUS_ROOT / "manifests"
DEFAULT_MANIFEST = MANIFESTS_DIR / "corpus_manifest.yaml"

RAW_DIR = CORPUS_ROOT / "raw"
PARSED_DIR = CORPUS_ROOT / "parsed"
CHUNKS_DIR = CORPUS_ROOT / "chunks"
INDEX_DIR = CORPUS_ROOT / "index"
EMBED_DIR = CORPUS_ROOT / "models" / "bge-small-en-v1.5"

SCRIPTS_CORPUS = CORPUS_ROOT / "scripts"
SCRIPTS_SUP = SUP_ROOT / "scripts"
SCRIPTS_MODEL = MODEL_ROOT / "scripts"

WEAK_MAP = SUP_ROOT / "configs" / "weak_label_map.yml"
SUP_JSONL = SUP_ROOT / "data" / "instructions.jsonl"
SFT_OUT_DIR = MODEL_ROOT / "data"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

def write_text(p: Path, content: str) -> None:
    ensure_dir(p.parent)
    p.write_text(content, encoding="utf-8")

def validate_manifest_yaml(raw_yaml: str) -> Tuple[bool, List[str]]:
    errs = []
    try:
        data = yaml.safe_load(raw_yaml) or {}
    except Exception as e:
        return False, [f"YAML parse error: {e}"]

    if not isinstance(data, dict):
        errs.append("Root of YAML must be a mapping (dict).")
        return False, errs

    docs = data.get("documents")
    if not isinstance(docs, list) or not docs:
        errs.append("Missing or empty 'documents' list.")
        return False, errs

    required = {"id", "title", "url"}
    for i, d in enumerate(docs, 1):
        if not isinstance(d, dict):
            errs.append(f"documents[{i}] must be a mapping (dict).")
            continue
        missing = required - set(d.keys())
        if missing:
            errs.append(f"documents[{i}] missing required keys: {sorted(missing)}")
        # gentle sanity: optional 'mirrors' as list
        if "mirrors" in d and not isinstance(d["mirrors"], list):
            errs.append(f"documents[{i}].mirrors must be a list if provided.")
    return (len(errs) == 0), errs

def extract_database(archive_file: Path, dest_root: Path) -> None:
    """Extracts a ZIP/TAR file that contains raw/parsed/chunks/index subfolders."""
    ensure_dir(dest_root)
    tmp = dest_root.parent / "_import_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    ensure_dir(tmp)

    try:
        if zipfile.is_zipfile(archive_file):
            with zipfile.ZipFile(archive_file, "r") as z:
                z.extractall(tmp)
        elif tarfile.is_tarfile(archive_file):
            with tarfile.open(archive_file, "r:*") as t:
                t.extractall(tmp)
        else:
            raise RuntimeError("Unsupported archive format. Please upload .zip or .tar(.gz/.bz2).")

        # Find the first subdir that has expected layout, or use tmp root
        candidate = tmp
        for sub in tmp.glob("*"):
            if sub.is_dir() and (sub / "chunks").exists():
                candidate = sub
                break

        # Move relevant directories over
        for name in ["raw", "parsed", "chunks", "index"]:
            src = candidate / name
            if src.exists():
                dst = dest_root / name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

def artifact_counts() -> Dict[str, int]:
    def count_files(p: Path, suffixes=None):
        if not p.exists():
            return 0
        if suffixes is None:
            return sum(1 for _ in p.rglob("*") if _.is_file())
        else:
            return sum(1 for _ in p.rglob("*") if _.is_file() and _.suffix.lower() in suffixes)

    return {
        "raw_files": count_files(RAW_DIR),
        "parsed_json": count_files(PARSED_DIR, {".json"}),
        "chunks_jsonl": count_files(CHUNKS_DIR, {".jsonl"}),
        "index_bits": count_files(INDEX_DIR),
        "faiss_index": 1 if (INDEX_DIR / "dense_open_bge_small_ip.faiss").exists() else 0,
        "tfidf": 1 if (INDEX_DIR / "tfidf_vectorizer.joblib").exists() else 0,
    }

def run_subprocess(script: Path, args: List[str]) -> Tuple[int, str, str]:
    """Run a python script as a subprocess, capture output."""
    cmd = [sys.executable, str(script)] + args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def try_wbs_core_pipeline(manifest_path: Path, embed_dir: Path) -> Tuple[bool, str]:
    """Try using wbs_core Pipeline if available. Returns (ok, msg)."""
    try:
        from wbs_core.pipeline import Pipeline  # type: ignore
    except Exception as e:
        return False, f"wbs_core Pipeline not available: {e}"

    try:
        pipe = Pipeline(
            corpus_root=CORPUS_ROOT,
            manifest_path=manifest_path,
            embed_model_dir=embed_dir,
        )
        pipe.run_all()  # expected to orchestrate download->parse->chunk->index
        return True, "wbs_core pipeline completed."
    except Exception as e:
        return False, f"wbs_core pipeline failed: {e}"

def run_corpus_scripts(manifest_path: Path, embed_dir: Path, log_box) -> None:
    """Fallback: call the known base_files scripts in order."""
    steps = [
        ("01_download.py", []),
        ("02_parse_normalize.py", []),
        ("03_chunk_and_embed.py", []),
        ("04_build_faiss.py", []),
    ]
    # Export env var for embed local path (used by some scripts)
    env = os.environ.copy()
    env["EMBED_LOCAL_PATH"] = str(embed_dir)

    for script_name, extra_args in steps:
        script = SCRIPTS_CORPUS / script_name
        if not script.exists():
            log_box.error(f"Script missing: {script}")
            raise FileNotFoundError(f"Missing script: {script}")

        log_box.info(f"Running {script_name} ...")
        rc, out, err = run_subprocess(script, extra_args)
        if out.strip():
            log_box.code(out)
        if err.strip():
            log_box.warning(err)
        if rc != 0:
            raise RuntimeError(f"{script_name} failed (rc={rc}).")

def run_supervision_and_sft(auto_num: int, k: int, log_box) -> None:
    """Generate weakly-labeled instructions and prep SFT splits."""
    ensure_dir(SUP_JSONL.parent)
    ensure_dir(SFT_OUT_DIR)

    # 05_make_supervision.py
    sup_script = SCRIPTS_SUP / "05_make_supervision.py"
    if not sup_script.exists():
        log_box.error(f"Missing: {sup_script}")
        raise FileNotFoundError(str(sup_script))

    sup_args = [
        "--chunks-dir", str(CHUNKS_DIR),
        "--index-dir", str(INDEX_DIR),
        "--weak-map", str(WEAK_MAP),
        "--num", str(auto_num),
        "--k", str(k),
        "--out", str(SUP_JSONL),
    ]
    log_box.info("Running 05_make_supervision.py ...")
    rc, out, err = run_subprocess(sup_script, sup_args)
    if out.strip():
        log_box.code(out)
    if err.strip():
        log_box.warning(err)
    if rc != 0:
        raise RuntimeError("05_make_supervision.py failed")

    # 06_prepare_sft.py
    sft_script = SCRIPTS_MODEL / "06_prepare_sft.py"
    if not sft_script.exists():
        log_box.error(f"Missing: {sft_script}")
        raise FileNotFoundError(str(sft_script))

    sft_args = [
        "--in", str(SUP_JSONL),
        "--out-dir", str(SFT_OUT_DIR),
        "--val-ratio", "0.2",
    ]
    log_box.info("Running 06_prepare_sft.py ...")
    rc, out, err = run_subprocess(sft_script, sft_args)
    if out.strip():
        log_box.code(out)
    if err.strip():
        log_box.warning(err)
    if rc != 0:
        raise RuntimeError("06_prepare_sft.py failed")

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="1 · Manifest & Data", layout="wide")
st.title("① Manifest & Data")

st.sidebar.header("Paths & Settings")
root_disp = st.sidebar.text_input("Project root", value=str(APP_ROOT))
st.sidebar.caption("Tip: All relative paths below are resolved from this project root.")
embed_guess = st.sidebar.text_input("Embedding model dir", value=str(EMBED_DIR))
st.sidebar.caption("Should point to local Sentence-Transformer (e.g., bge-small-en-v1.5).")

st.sidebar.divider()
st.sidebar.write("**Artifacts summary**")
counts = artifact_counts()
st.sidebar.json(counts)

st.divider()

# Tabs for workflow
tab_manifest, tab_upload, tab_build = st.tabs(["📝 Manifest", "📦 Upload Database", "⚙️ Build Pipeline"])

# ---------------- Manifest Tab ----------------
with tab_manifest:
    st.subheader("Edit or Create Manifest (YAML)")

    ensure_dir(MANIFESTS_DIR)
    manifest_files = sorted(MANIFESTS_DIR.glob("*.yaml"))
    sel = st.selectbox("Choose a manifest file", options=[str(p.name) for p in manifest_files] + ["(new)"], index=0 if manifest_files else 0)

    if sel == "(new)":
        default_yaml = """documents:
  - id: cfr-10-50-app-s
    title: 10 CFR Part 50, Appendix S
    url: https://example.gov/cfr_10_part50_appendixS.html
    mirrors:
      - https://mirror1.example/cfr_10_part50_appendixS.html
"""
        manifest_text = st.text_area("Manifest YAML", value=default_yaml, height=260)
        target_name = st.text_input("Save as", value="corpus_manifest.yaml")
    else:
        target_name = sel
        manifest_text = st.text_area("Manifest YAML", value=read_text(MANIFESTS_DIR / sel), height=260)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Validate"):
            ok, errs = validate_manifest_yaml(manifest_text)
            if ok:
                st.success("Manifest looks good ✅")
            else:
                st.error("Manifest validation failed.")
                st.write(errs)
    with c2:
        if st.button("Save"):
            ok, errs = validate_manifest_yaml(manifest_text)
            if not ok:
                st.error("Refusing to save invalid YAML. Please fix errors first.")
                st.write(errs)
            else:
                out_path = MANIFESTS_DIR / target_name
                write_text(out_path, manifest_text)
                st.success(f"Saved: {out_path}")
    with c3:
        downloaded = st.download_button(
            "Download manifest.yaml",
            data=manifest_text.encode("utf-8"),
            file_name="corpus_manifest.yaml",
            mime="text/yaml"
        )

# ---------------- Upload Database Tab ----------------
with tab_upload:
    st.subheader("Upload Prebuilt Database (ZIP/TAR)")
    st.caption("Upload an archive containing any of: raw/, parsed/, chunks/, index/ to replace current artifacts.")

    uploaded = st.file_uploader("Archive (.zip / .tar / .tar.gz)", type=["zip", "tar", "gz", "bz2"])
    if uploaded:
        tmp_file = APP_ROOT / "_upload.tmp"
        with open(tmp_file, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            with st.spinner("Extracting..."):
                extract_database(tmp_file, CORPUS_ROOT)
            st.success("Imported successfully.")
            st.json(artifact_counts())
        except Exception as e:
            st.error(f"Import failed: {e}")
        finally:
            tmp_file.unlink(missing_ok=True)

# ---------------- Build Pipeline Tab ----------------
with tab_build:
    st.subheader("Build / Refresh Corpus")
    st.caption("Runs: download → parse/normalize → chunk/embed → TF-IDF + FAISS index")

    st.write("**Manifest to use**")
    manifest_options = sorted(MANIFESTS_DIR.glob("*.yaml"))
    manifest_choice = st.selectbox("Manifest file", options=[str(p) for p in manifest_options] or [str(DEFAULT_MANIFEST)])
    if not Path(manifest_choice).exists():
        st.warning("Selected manifest does not exist yet. Please create/save one in the 'Manifest' tab.")

    auto_sup = st.checkbox("Also generate supervision pairs and prepare SFT", value=False)
    num_sup = st.number_input("Supervision: number of pairs", min_value=10, max_value=10000, value=120, step=10, disabled=not auto_sup)
    k_sup = st.number_input("Supervision: retrieval top-k", min_value=2, max_value=32, value=6, step=1, disabled=not auto_sup)

    # Log box
    log_box = st.container(border=True)
    run_btn = st.button("Run Pipeline", type="primary")

    if run_btn:
        # Resolve embed path
        embed_dir = Path(embed_guess).resolve()
        if not embed_dir.exists():
            st.error(f"Embedding model dir not found: {embed_dir}")
            st.stop()

        # Ensure subfolders
        for d in [RAW_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR]:
            ensure_dir(d)

        # Try wbs_core pipeline first
        with st.spinner("Running corpus pipeline..."):
            ok, msg = try_wbs_core_pipeline(Path(manifest_choice), embed_dir)
            if ok:
                log_box.success(msg)
            else:
                log_box.warning(f"{msg}\nFalling back to base_files/*/scripts ...")
                try:
                    run_corpus_scripts(Path(manifest_choice), embed_dir, log_box)
                    log_box.success("Corpus scripts completed.")
                except Exception as e:
                    log_box.error(f"Corpus pipeline failed: {e}")
                    st.stop()

        st.success("Corpus build complete.")
        st.json(artifact_counts())

        if auto_sup:
            with st.spinner("Generating supervision + preparing SFT ..."):
                try:
                    run_supervision_and_sft(int(num_sup), int(k_sup), log_box)
                    st.success("Supervision + SFT complete.")
                    st.write("SFT files:")
                    st.code(f"{SFT_OUT_DIR}/train.jsonl\n{SFT_OUT_DIR}/val.jsonl")
                except Exception as e:
                    st.error(f"Supervision/SFT failed: {e}")

# Footer
st.divider()
st.caption("Tip: If you’re fully offline, make sure base models and the sentence-transformer are cloned locally, and set HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.")