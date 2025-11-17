# wbs_app/pages/3_Supervision_and_SFT.py
# Streamlit page: Supervision (instruction pairs) + SFT dataset split
# - Generate instruction pairs from chunked corpus + weak label map
# - Or upload your own instructions.jsonl
# - Split into Train/Val for SFT
# - Uses wbs_core if present; falls back to base_files/.../scripts CLIs

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import streamlit as st

# ------------------------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------------------------
APP_ROOT = Path.cwd()
BASE_FILES = APP_ROOT / "base_files"

# Corpus
CORPUS_ROOT = BASE_FILES / "corpus"
CHUNKS_DIR = CORPUS_ROOT / "chunks"

# Supervision
SUP_ROOT = BASE_FILES / "supervision"
SUP_CONFIGS = SUP_ROOT / "configs"
SUP_DATA = SUP_ROOT / "data"
SUP_LOGS = SUP_ROOT / "logs"
SUP_SCRIPT = SUP_ROOT / "scripts" / "05_make_supervision.py"

# Model data
MODEL_ROOT = BASE_FILES / "model"
MODEL_DATA = MODEL_ROOT / "data"
SFT_SCRIPT = MODEL_ROOT / "scripts" / "06_prepare_sft.py"

# Defaults
DEFAULT_WEAK_MAP = SUP_CONFIGS / "weak_label_map.yml"
DEFAULT_INSTRUCTIONS = SUP_DATA / "instructions.jsonl"
DEFAULT_TRAIN = MODEL_DATA / "train.jsonl"
DEFAULT_VAL = MODEL_DATA / "val.jsonl"

# Make sure directories exist
for d in [CHUNKS_DIR, SUP_CONFIGS, SUP_DATA, SUP_LOGS, MODEL_DATA]:
    d.mkdir(parents=True, exist_ok=True)

# Make project import available for wbs_core if user installed it as a local package tree
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def try_import_wbs_core():
    """
    Try importing wbs_core implementations.
    Returns (SupervisionBuilder, SFTPreparer) or None on failure.
    """
    try:
        from wbs_core.supervision import SupervisionBuilder  # type: ignore
        from wbs_core.stf import SFTPreparer  # (module named 'stf' per your repo) # type: ignore
        return SupervisionBuilder, SFTPreparer
    except Exception:
        return None

def run_subprocess(py_script: Path, args: List[str], env: Optional[dict] = None) -> Tuple[int, str, str]:
    cmd = [sys.executable, str(py_script)] + list(args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr

def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    i = 0
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for _ in fh:
            i += 1
    return i

def head_jsonl(p: Path, n: int = 3) -> List[Dict]:
    out = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            try:
                out.append(json.loads(line))
            except Exception:
                out.append({"_raw": line.strip(), "_error": "JSON decode failed"})
            if i + 1 >= n:
                break
    return out

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------
# Streamlit page
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="3 · Supervision & SFT", layout="wide")
st.title("③ Supervision & SFT")

st.sidebar.header("Locations")
st.sidebar.caption("All paths are relative to this app's working directory.")
st.sidebar.write("**Corpus chunks dir:**")
st.sidebar.code(str(CHUNKS_DIR))
st.sidebar.write("**Supervision data dir:**")
st.sidebar.code(str(SUP_DATA))
st.sidebar.write("**Model data dir:**")
st.sidebar.code(str(MODEL_DATA))

using_wbs_core = try_import_wbs_core()
st.info(f"Execution mode: {'wbs_core modules' if using_wbs_core else 'base_files scripts fallback'}")

st.divider()
st.subheader("A) Build Supervision (Instruction Pairs)")

tab_gen, tab_upload = st.tabs(["Generate from corpus", "Upload existing instructions.jsonl"])

with tab_gen:
    st.markdown("Use the chunked corpus + weak label map to synthesize instruction/target pairs.")

    # Controls
    weak_map_path = st.text_input(
        "Weak label map (YAML)",
        value=str(DEFAULT_WEAK_MAP),
        help="Mapping of keywords → regulatory citations & SRP/RG anchors"
    )
    reactor_types = st.multiselect(
        "Reactor types",
        options=["PWR", "BWR", "SFR", "HTGR", "MSR", "VHTR"],
        default=["PWR", "BWR", "MSR"]
    )
    disciplines = st.multiselect(
        "Disciplines / Topics",
        options=[
            "seismic", "QA", "core physics", "thermal-hydraulics",
            "I&C", "containment", "radiation protection", "licensing",
            "probabilistic risk", "EP"
        ],
        default=["seismic", "QA", "core physics", "thermal-hydraulics"]
    )
    num_pairs = st.number_input("Number of instruction pairs", min_value=10, max_value=10000, value=200, step=10)
    output_instructions = st.text_input("Output instructions.jsonl", value=str(DEFAULT_INSTRUCTIONS))

    colA, colB = st.columns([1, 3])
    with colA:
        btn_generate = st.button("Generate Supervision", type="primary", use_container_width=True)
    with colB:
        if st.button("Clean previous instructions", use_container_width=True):
            try:
                if Path(output_instructions).exists():
                    Path(output_instructions).unlink()
                st.success("Removed previous instructions file.")
            except Exception as e:
                st.error(f"Could not remove file: {e}")

    if btn_generate:
        # Sanity checks
        if not CHUNKS_DIR.exists():
            st.error("Chunks directory not found. Please run page ② Build Corpus first.")
        elif not Path(weak_map_path).exists():
            st.error("Weak label map not found.")
        else:
            ensure_parent(Path(output_instructions))
            if using_wbs_core:
                SupervisionBuilder, _SFTPreparer = using_wbs_core
                try:
                    with st.spinner("Generating instruction pairs with wbs_core.supervision..."):
                        builder = SupervisionBuilder(
                            chunks_dir=CHUNKS_DIR,
                            weak_label_map=Path(weak_map_path),
                            reactor_types=reactor_types,
                            disciplines=disciplines,
                            num=int(num_pairs),
                            out_path=Path(output_instructions),
                        )
                        stats = builder.run()
                    st.success(f"Supervision generated. Stats: {json.dumps(stats, indent=2)}")
                except Exception as e:
                    st.error(f"SupervisionBuilder error: {e}")
            else:
                # Fallback CLI: 05_make_supervision.py --chunks-dir ... --map ... --reactor-types ... --disciplines ... --num ... --out ...
                args = [
                    "--chunks-dir", str(CHUNKS_DIR),
                    "--map", str(weak_map_path),
                    "--num", str(int(num_pairs)),
                    "--out", str(output_instructions),
                ]
                if reactor_types:
                    args += ["--reactor-types", ",".join(reactor_types)]
                if disciplines:
                    args += ["--disciplines", ",".join(disciplines)]

                with st.spinner("Running base_files/supervision/scripts/05_make_supervision.py ..."):
                    rc, out, err = run_subprocess(SUP_SCRIPT, args)
                if out.strip():
                    st.code(out, language="text")
                if err.strip():
                    st.warning(err)
                if rc != 0:
                    st.error("Supervision script failed.")
                else:
                    st.success(f"Supervision generated at {output_instructions}")

    # Preview generated file
    st.markdown("**Preview (first 3 records)**")
    if Path(output_instructions).exists():
        preview = head_jsonl(Path(output_instructions), n=3)
        st.json(preview)
    else:
        st.caption("No instructions.jsonl found yet.")

with tab_upload:
    st.markdown("If you already have an `instructions.jsonl`, you can upload it here.")
    uploaded = st.file_uploader("Upload instructions.jsonl", type=["jsonl", "txt", "json"])
    uploaded_target = st.text_input("Save to path", value=str(DEFAULT_INSTRUCTIONS))

    if uploaded and st.button("Save uploaded file"):
        try:
            data = uploaded.read()
            ensure_parent(Path(uploaded_target))
            Path(uploaded_target).write_bytes(data)
            st.success(f"Saved to {uploaded_target} ({len(data)} bytes).")
        except Exception as e:
            st.error(f"Failed to save: {e}")

    if Path(uploaded_target).exists():
        st.caption("Preview of saved file (first 3 lines):")
        st.json(head_jsonl(Path(uploaded_target), n=3))

st.divider()
st.subheader("B) Prepare SFT Train/Val")

input_instructions = st.text_input("Input instructions.jsonl", value=str(DEFAULT_INSTRUCTIONS))
out_train = st.text_input("Train output", value=str(DEFAULT_TRAIN))
out_val = st.text_input("Val output", value=str(DEFAULT_VAL))
val_ratio = st.slider("Validation ratio", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    btn_split = st.button("Split Train/Val", type="primary", use_container_width=True)
with c2:
    if st.button("Clean previous splits", use_container_width=True):
        try:
            if Path(out_train).exists():
                Path(out_train).unlink()
            if Path(out_val).exists():
                Path(out_val).unlink()
            st.success("Removed previous train/val files.")
        except Exception as e:
            st.error(f"Could not remove files: {e}")

if btn_split:
    if not Path(input_instructions).exists():
        st.error("Input instructions.jsonl not found.")
    else:
        ensure_parent(Path(out_train))
        ensure_parent(Path(out_val))
        if using_wbs_core:
            _SupervisionBuilder, SFTPreparer = using_wbs_core
            try:
                with st.spinner("Preparing SFT splits with wbs_core.stf..."):
                    prep = SFTPreparer(
                        inp=Path(input_instructions),
                        out_dir=Path(out_train).parent,  # SFTPreparer writes to out_dir/train.jsonl + out_dir/val.jsonl
                        val_ratio=float(val_ratio),
                        train_path=Path(out_train),
                        val_path=Path(out_val),
                    )
                    info = prep.run()
                st.success(f"SFT split complete. {json.dumps(info, indent=2)}")
            except Exception as e:
                st.error(f"SFTPreparer error: {e}")
        else:
            # Fallback CLI: 06_prepare_sft.py --in ... --out-dir <dir> --val-ratio 0.2
            args = ["--in", str(input_instructions), "--out-dir", str(Path(out_train).parent), "--val-ratio", str(float(val_ratio))]
            with st.spinner("Running base_files/model/scripts/06_prepare_sft.py ..."):
                rc, out, err = run_subprocess(SFT_SCRIPT, args)
            if out.strip():
                st.code(out, language="text")
            if err.strip():
                st.warning(err)
            if rc != 0:
                st.error("SFT split script failed.")
            else:
                # The CLI writes to out_dir/train.jsonl and out_dir/val.jsonl, but user may have different filenames requested.
                # If the requested filenames differ, copy them.
                default_train = Path(out_train).parent / "train.jsonl"
                default_val = Path(out_val).parent / "val.jsonl"
                try:
                    if str(default_train) != str(out_train) and default_train.exists():
                        shutil.copy2(default_train, out_train)
                    if str(default_val) != str(out_val) and default_val.exists():
                        shutil.copy2(default_val, out_val)
                    st.success(f"SFT split complete. Train={out_train}  Val={out_val}")
                except Exception as e:
                    st.warning(f"Split OK but could not copy to requested names: {e}")

# Preview counts and head
st.markdown("**Artifacts**")
row1, row2, row3 = st.columns(3)
with row1:
    st.metric("instructions.jsonl lines", count_lines(Path(input_instructions)))
with row2:
    st.metric("train.jsonl lines", count_lines(Path(out_train)))
with row3:
    st.metric("val.jsonl lines", count_lines(Path(out_val)))

st.markdown("**Preview train.jsonl (first 3 records)**")
if Path(out_train).exists():
    st.json(head_jsonl(Path(out_train), n=3))
else:
    st.caption("No train.jsonl yet.")

st.markdown("**Preview val.jsonl (first 3 records)**")
if Path(out_val).exists():
    st.json(head_jsonl(Path(out_val), n=3))
else:
    st.caption("No val.jsonl yet.")

st.divider()
st.caption("Tip: This page is offline-friendly. It only uses your local chunked corpus and local scripts/modules.")
