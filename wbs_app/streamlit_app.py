# wbs_app/streamlit_app.py
# Landing page for the WBS Builder app.
# - Shows overall status of corpus & models
# - Provides quick links to the multipage workflow
# - Optional: "one-click" pipeline runner for corpus build & evaluation
# - Reads local paths relative to your current working directory (expects the repo layout you shared)

from __future__ import annotations

import os
import sys
import re
import json
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
# Ensure app root on sys.path for relative imports
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))


# Local utils shared by all pages
from utils import (
    AppPaths,
    ensure_dirs,
    corpus_status,
    find_base_models,
    find_merged_models,
    find_adapters,
    build_offline_env,
    run_cli,
    show_logs,
    load_last_response,
    ok_bad,
    human_count,
    timestamp,
    cli_corpus_download,
    cli_corpus_parse,
    cli_corpus_chunk_embed,
    cli_corpus_build_index,
    cli_make_supervision,
    cli_prepare_sft,
    cli_train_lora,
    cli_batch_rag,
    cli_merge_lora,
    cli_validate_outputs,
)

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="WBS Builder · Landing",
    page_icon="🧰",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# Helpers (UI-only)
# --------------------------------------------------------------------------------------

def _short(p: Path, root: Optional[Path] = None) -> str:
    try:
        if root:
            return str(p.relative_to(root))
        return str(p)
    except Exception:
        return str(p)

def _optional_torch_info() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "PyTorch: not installed"
    cuda = torch.cuda.is_available()
    if not cuda:
        return f"PyTorch: {torch.__version__} (CPU)"
    n = torch.cuda.device_count()
    names = []
    for i in range(n):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append(f"GPU{i}")
    return f"PyTorch: {torch.__version__} (CUDA, {n} x {', '.join(names)})"

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try hard to extract a single JSON object from free-form text."""
    if not text:
        return None
    # ```json ... ```
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    for blk in blocks:
        try:
            return json.loads(blk)
        except Exception:
            pass
    # first balanced-ish {...}
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = text.find("{", start + 1)
    # try whole
    try:
        return json.loads(text)
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Session state & paths
# --------------------------------------------------------------------------------------

if "paths" not in st.session_state:
    st.session_state["paths"] = AppPaths.detect()

PATHS: AppPaths = st.session_state["paths"]

# Make sure export dirs exist
ensure_dirs([PATHS.index_dir, PATHS.model_merged_dir, PATHS.model_checkpoints_dir, PATHS.exports_dir])

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------

with st.sidebar:
    st.title("WBS Builder")
    st.caption("End-to-end pipeline for reactor WBS generation")

    st.markdown("### Directories")
    st.write(f"• Base files: `{_short(PATHS.base_files, PATHS.app_root)}`")
    st.write(f"• Corpus: `{_short(PATHS.corpus_root, PATHS.app_root)}`")
    st.write(f"• Models: `{_short(PATHS.models_dir, PATHS.app_root)}`")
    st.write(f"• Checkpoints: `{_short(PATHS.model_checkpoints_dir, PATHS.app_root)}`")
    st.write(f"• Merged models: `{_short(PATHS.model_merged_dir, PATHS.app_root)}`")

    st.markdown("---")
    st.markdown("### Navigation")
    # Streamlit auto-generates a "Pages" section; these links help too on newer versions.
    try:
        st.page_link("pages/1_Manifest_and_Data.py", label="1) Manifest & Data", icon="📄")
        st.page_link("pages/2_Build_Corpus.py", label="2) Build Corpus", icon="🧱")
        st.page_link("pages/3_Supervision_and_SFT.py", label="3) Supervision & SFT", icon="🧭")
        st.page_link("pages/4_Train_and_Evaluate.py", label="4) Train & Evaluate", icon="🎯")
        st.page_link("pages/5_WBS_Generator.py", label="5) WBS Generator", icon="🗂️")
    except Exception:
        st.info("Use the Pages menu (top-left) to navigate steps 1→5.")

    st.markdown("---")
    st.markdown("### Runtime")
    st.write(_optional_torch_info())
    st.write(f"Python: {platform.python_version()} on {platform.system()} {platform.release()}")

# --------------------------------------------------------------------------------------
# Main: Overview & Quick Actions
# --------------------------------------------------------------------------------------

st.title("🔧 WBS Builder — Landing")

st.markdown(
    """
Welcome! This app guides you through:
1) curating/ingesting regulatory corpus,  
2) building dense/TF-IDF indices,  
3) generating supervision & SFT splits,  
4) LoRA training, evaluation & merging, and  
5) **generating a WBS** for a design prompt grounded in the corpus (RAG + citations).

Use the sidebar to jump to each step. This landing page provides **status** and **one-click helpers**.
"""
)

# --- Status cards ---
status = corpus_status(PATHS)
cols = st.columns(6)
cols[0].metric("Raw docs", human_count(status["raw_count"]))
cols[1].metric("Parsed JSON", human_count(status["parsed_count"]))
cols[2].metric("Chunks", human_count(status["chunks_count"]))
cols[3].metric("FAISS", "ready" if status["has_faiss"] else "missing")
cols[4].metric("TF-IDF", "ready" if status["has_tfidf"] else "missing")
cols[5].metric("Embed model", "found" if status["has_embed_model"] else "missing")

st.markdown("---")

# --------------------------------------------------------------------------------------
# Quick Corpus Pipeline (optional one-click)
# --------------------------------------------------------------------------------------

with st.expander("⚡ Quick Corpus Pipeline (optional)", expanded=False):
    st.markdown(
        "Run the canonical corpus pipeline in order. "
        "These are exactly the same scripts available on page ② Build Corpus."
    )
    c1, c2, c3, c4 = st.columns(4)
    do_dl = c1.checkbox("01 Download", value=False)
    do_parse = c2.checkbox("02 Parse/Normalize", value=False)
    do_chunk = c3.checkbox("03 Chunk/Embed", value=False)
    do_index = c4.checkbox("04 Build Index", value=False)

    run_now = st.button("Run selected steps")
    if run_now:
        env = build_offline_env(PATHS.embed_model_dir)

        if do_dl:
            with st.spinner("01 Downloading corpus…"):
                rc, out, err = cli_corpus_download(PATHS)
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        if do_parse:
            with st.spinner("02 Parsing & normalizing…"):
                rc, out, err = cli_corpus_parse(PATHS)
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        if do_chunk:
            with st.spinner("03 Chunking & embedding…"):
                rc, out, err = cli_corpus_chunk_embed(PATHS)
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        if do_index:
            with st.spinner("04 Building FAISS / TF-IDF…"):
                rc, out, err = cli_corpus_build_index(PATHS)
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        st.success("Done. Refreshing status…")
        status = corpus_status(PATHS)

# --------------------------------------------------------------------------------------
# Quick Supervision + SFT (optional shortcut)
# --------------------------------------------------------------------------------------

with st.expander("🧭 Quick Supervision + SFT (optional)", expanded=False):
    st.markdown("These are the same operations available on page ③ Supervision & SFT.")

    default_instr = PATHS.supervision_root / "data" / "instructions.jsonl"
    default_sup_schema = PATHS.supervision_root / "configs" / "schema.json"
    default_sup_map = PATHS.supervision_root / "configs" / "weak_label_map.yml"
    sft_out_dir = PATHS.model_root / "data"

    c1, c2 = st.columns(2)
    with c1:
        gen_sup = st.checkbox("Generate supervision", value=False)
        prep_sft = st.checkbox("Prepare SFT splits", value=False)
    with c2:
        val_ratio = st.slider("Validation ratio", 0.05, 0.5, 0.2, 0.05)

    if st.button("Run selected supervision/SFT"):
        if gen_sup:
            with st.spinner("Generating supervision pairs…"):
                rc, out, err = cli_make_supervision(
                    PATHS,
                    corpus_root=PATHS.corpus_root,
                    schema_path=default_sup_schema,
                    weak_map=default_sup_map,
                    out_path=default_instr,
                )
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        if prep_sft:
            with st.spinner("Preparing SFT splits…"):
                rc, out, err = cli_prepare_sft(
                    PATHS,
                    instructions=default_instr,
                    out_dir=sft_out_dir,
                    val_ratio=val_ratio,
                )
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        st.success("Done.")

# --------------------------------------------------------------------------------------
# Quick Train/Eval/Merge (optional shortcut)
# --------------------------------------------------------------------------------------

with st.expander("🎯 Quick Train / Evaluate / Merge (optional)", expanded=False):
    st.markdown("These controls mirror page ④ Train & Evaluate.")

    base_models = find_base_models(PATHS)
    adapters = find_adapters(PATHS)
    merged_models = find_merged_models(PATHS)

    if not base_models:
        st.warning("No base models found under base_files/models. Add models before training.")
    base_choice = st.selectbox(
        "Base model",
        base_models,
        format_func=lambda p: p.name,
        disabled=not base_models,
    )

    c1, c2 = st.columns(2)
    with c1:
        train_now = st.checkbox("Train LoRA", value=False)
        merge_now = st.checkbox("Merge LoRA", value=False)
    with c2:
        batch_now = st.checkbox("Batch RAG infer", value=False)
        validate_now = st.checkbox("Validate outputs", value=False)

    # Training config lives in YAML (as per your working scripts)
    cfg_path = PATHS.model_root / "configs" / "lora.yaml"
    train_out = PATHS.model_checkpoints_dir / "wbs-lora"
    data_train = PATHS.model_root / "data" / "train.jsonl"
    data_val = PATHS.model_root / "data" / "val.jsonl"

    if st.button("Run selected model steps"):
        if train_now:
            with st.spinner("Training LoRA…"):
                rc, out, err = cli_train_lora(
                    PATHS,
                    base_model_dir=base_choice,
                    data_train=data_train,
                    data_val=data_val,
                    out_dir=train_out,
                    cfg=cfg_path if cfg_path.exists() else None,
                )
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        # refresh adapters
        adapters = find_adapters(PATHS)

        if merge_now:
            if not adapters:
                st.error("No LoRA adapter found. Train first.")
            else:
                adapter_choice = st.selectbox(
                    "Adapter to merge",
                    adapters,
                    index=len(adapters) - 1,
                    format_func=lambda p: p.name,
                )
                merged_out = PATHS.model_merged_dir / f"{base_choice.name}-merged-{timestamp()}"
                with st.spinner("Merging adapter into base…"):
                    rc, out, err = cli_merge_lora(
                        PATHS, base_model_dir=base_choice, adapter_dir=adapter_choice, out_dir=merged_out
                    )
                    st.write(f"Exit code: {rc}")
                    show_logs(out, err)

        if batch_now:
            prompts_file = PATHS.model_root / "prompts" / "example_prompt.txt"
            batch_out = PATHS.model_root / "batch_outputs.jsonl"
            # Prefer merged model if available
            merged_models = find_merged_models(PATHS)
            base_for_infer = merged_models[-1] if merged_models else base_choice
            adapter_for_infer = None if merged_models else (train_out if train_out.exists() else None)

            with st.spinner("Batch RAG inference…"):
                rc, out, err = cli_batch_rag(
                    PATHS,
                    base_or_merged=base_for_infer,
                    prompts_file=prompts_file,
                    out_file=batch_out,
                    k=8,
                    max_new_tokens=700,
                    temperature=0.2,
                    adapter_dir=adapter_for_infer,
                )
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        if validate_now:
            jsonl_path = PATHS.model_root / "batch_outputs.jsonl"
            with st.spinner("Validating outputs…"):
                rc, out, err = cli_validate_outputs(PATHS, jsonl_file=jsonl_path)
                st.write(f"Exit code: {rc}")
                show_logs(out, err)

        st.success("Done.")

# --------------------------------------------------------------------------------------
# Last Response (single-query RAG page uses this too)
# --------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("📦 Last Response Snapshot")

lr = load_last_response(PATHS)
if not lr:
    st.info("No last_response.json found yet. After you run single-query RAG or batch inference, results will appear here.")
else:
    # Robustly pick the model text (your CLI writes 'output')
    model_text = lr.get("output_text") or lr.get("output") or lr.get("text") or ""
    # Try to show parsed JSON whether writer called it 'output_json' or only embedded it in text.
    parsed_json = lr.get("output_json") or _extract_json(model_text) or {}

    tabs = st.tabs(["Summary", "Output JSON", "Citations / Retrieved", "Raw"])
    with tabs[0]:
        st.write("**Query**")
        st.code(lr.get("query", ""), language="markdown")
        st.write("**Model Text**")
        st.code(model_text, language="markdown")
        if "timing" in lr:
            st.write("**Timing**")
            st.json(lr["timing"])
    with tabs[1]:
        st.write("**Parsed JSON (if any)**")
        st.json(parsed_json)
    with tabs[2]:
        st.write("**Retrieved (top-k)**")
        retrieved = lr.get("retrieved", [])
        if not retrieved:
            st.info("No retrieved entries in snapshot.")
        else:
            # Your snapshot: {"id": "doc#chunk", "title": "...", "source": "...", "score": ...}
            # Display as a compact table.
            rows = []
            for r in retrieved:
                rid = str(r.get("id", ""))
                title = str(r.get("title", ""))
                source = str(r.get("source", ""))
                score = r.get("score", "")
                rows.append(
                    {
                        "id": rid,
                        "title": title,
                        "source": source,
                        "score": f"{score:.3f}" if isinstance(score, (int, float)) else str(score),
                    }
                )
            st.dataframe(rows, use_container_width=True)
    with tabs[3]:
        st.write("**Raw last_response.json**")
        st.code(json.dumps(lr, indent=2, ensure_ascii=False), language="json")

st.markdown("---")
st.caption(
    "Tip: Use the left sidebar to navigate steps 1→5. "
    "The landing page gives you quick status and optional one-click actions."
)

# --- BEGIN: WBS Analytics Section (Altair-free) ---

import pandas as _pd
import re as _re
from typing import Tuple as _Tuple, List as _List, Dict as _Dict, Any as _Any, Optional as _Optional

def _extract_json_all(text: str) -> _List[_Dict[str, _Any]]:
    if not text: return []
    objs: _List[_Dict[str,_Any]] = []
    for blk in _re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=_re.DOTALL):
        try: objs.append(json.loads(blk))
        except Exception: pass
    start, N = 0, len(text)
    while True:
        s = text.find("{", start)
        if s == -1: break
        depth, i = 0, s
        while i < N:
            ch = text[i]
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[s:i+1]
                    try: objs.append(json.loads(cand))
                    except Exception: pass
                    start = i+1; break
            i += 1
        else: break
    uniq, seen = [], set()
    for o in objs:
        k = json.dumps(o, sort_keys=True, ensure_ascii=False)
        if k not in seen: seen.add(k); uniq.append(o)
    return uniq

def _coerce_wbs(obj: _Dict[str,_Any]) -> _Optional[_Dict[str,_Any]]:
    if not isinstance(obj, dict): return None
    title = obj.get("wbs_title") or obj.get("project") or obj.get("title") or "WBS"
    tasks = obj.get("tasks") or obj.get("items") or obj.get("work_breakdown") or (obj.get("wbs") or {}).get("tasks") or []
    if isinstance(tasks, dict): tasks = list(tasks.values())
    if not isinstance(tasks, list): return None
    norm: _List[_Dict[str,_Any]] = []
    for idx, t in enumerate(tasks, 1):
        if not isinstance(t, dict):
            norm.append({"id": f"T{idx}", "name": str(t), "description": "", "children": []}); continue
        citations_raw = t.get("citations") or t.get("sources") or []
        citations = []
        if isinstance(citations_raw, list):
            for c in citations_raw:
                if isinstance(c, dict) and "id" in c: citations.append({"id": str(c["id"])})
                else: citations.append({"id": str(c)})
        norm.append({
            "id": t.get("id") or t.get("task_id") or f"T{idx}",
            "name": t.get("name") or t.get("title") or f"Task {idx}",
            "description": t.get("description") or t.get("objective") or "",
            "standards": t.get("standards") or t.get("applicable_codes") or [],
            "citations": citations,
            "children": t.get("children") or t.get("subtasks") or [],
        })
    return {"project": title, "tasks": norm}

def _split_lines(desc: str) -> _List[str]:
    lines = [ln.strip() for ln in desc.splitlines() if ln.strip()]
    picked = []
    for ln in lines:
        m = _re.match(r"^\s*[-*•]\s+(.+)$", ln) or _re.match(r"^\s*\d+[.)]\s+(.+)$", ln)
        if m: picked.append(m.group(1).strip())
    if len(picked) >= 2: return picked
    text = desc.replace("—",";").replace("–",";")
    parts = _re.split(r"[;•]+", text)
    parts = [p.strip(" .") for p in parts if len(p.strip())>2]
    if len(parts) >= 2: return parts[:12]
    sents = _re.split(r"(?<=[.!?])\s+", desc)
    sents = [s.strip() for s in sents if 3 <= len(s.strip()) <= 180]
    return sents[:10] if len(sents) >= 2 else []

def _expand_heuristic(wbs: _Dict[str,_Any]) -> _Dict[str,_Any]:
    def expand(t: _Dict[str,_Any]) -> _Dict[str,_Any]:
        kids = t.get("children") or []
        if kids:
            t["children"] = [expand(ch) for ch in kids]; return t
        desc = t.get("description") or ""
        if len(desc) < 60: return t
        frags = _split_lines(desc)
        if len(frags) < 2: return t
        new_kids = []
        for i, frag in enumerate(frags, 1):
            nm = (frag[:57] + "…") if len(frag) > 60 else frag
            new_kids.append({"id": f"{t.get('id','T')}.{i}", "name": nm, "description": frag, "children": [], "standards": t.get("standards", []), "citations": t.get("citations", [])})
        t["children"] = new_kids; return t
    wbs["tasks"] = [expand(t) for t in (wbs.get("tasks") or [])]
    return wbs

def _build_wbs_from_text(txt: str) -> _Optional[_Dict[str,_Any]]:
    cands = _extract_json_all(txt)
    scored = []
    for o in cands:
        w = _coerce_wbs(o)
        if w: scored.append((len(w.get("tasks") or []), w))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    tasks = []
    for line in txt.splitlines():
        m = _re.match(r'^\s*[-*•]\s*(.+)', line) or _re.match(r'^\s*\d+\.\s*(.+)', line)
        if m:
            label = m.group(1).strip()
            if label: tasks.append({"id": f"T{len(tasks)+1}", "name": label, "description": "", "children": []})
    return {"project": "WBS", "tasks": tasks} if tasks else None

def _flatten_with_depth(wbs: _Dict[str,_Any]) -> _List[_Dict[str,_Any]]:
    out = []
    def walk(t: _Dict[str,_Any], d: int, parent: _Optional[str]):
        out.append({"id": t.get("id"), "name": t.get("name"), "depth": d, "is_leaf": not bool(t.get("children")), "citations": t.get("citations") or [], "standards": t.get("standards") or [], "parent": parent})
        for ch in (t.get("children") or []):
            walk(ch, d+1, t.get("id"))
    for t in (wbs.get("tasks") or []): walk(t, 0, None)
    return out

def _bar_plot_or_table(series, title: str, xlabel: str):
    """Try to plot with matplotlib; if unavailable, show a table."""
    try:
        import matplotlib.pyplot as plt  # no seaborn, no explicit colors
        fig, ax = plt.subplots()
        if hasattr(series, "index") and hasattr(series, "values"):
            x = [str(x) for x in series.index.tolist()]
            y = [float(v) for v in series.values.tolist()]
        else:
            x = [str(k) for k in series.keys()]
            y = [float(v) for v in series.values()]
        ax.bar(range(len(y)), y)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=0, ha="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        st.pyplot(fig, clear_figure=True)
    except Exception:
        # fallback: simple table
        if hasattr(series, "reset_index"):
            st.dataframe(series.reset_index().rename(columns={0:"count"}), use_container_width=True)
        else:
            st.json(series)

st.markdown("### 📊 WBS Analytics (last response)")
_lr = load_last_response(PATHS)
if not _lr:
    st.info("No last_response.json to analyze yet.")
else:
    _raw_text = (_lr.get("output_text") or "") or (_lr.get("output") or "")
    _wbs = _lr.get("output_json")
    _wbs = _coerce_wbs(_wbs) if isinstance(_wbs, dict) and _wbs.get("tasks") else _build_wbs_from_text(_raw_text)
    if _wbs:
        _wbs = _expand_heuristic(_wbs)
        _rows = _flatten_with_depth(_wbs)
        df = _pd.DataFrame(_rows)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total tasks", int(df.shape[0]))
        with c2: st.metric("Max depth", int(df["depth"].max()) if not df.empty else 0)
        with c3:
            coverage = 100.0 * (df["citations"].apply(lambda xs: len(xs) > 0).sum() / max(1, df.shape[0]))
            st.metric("Tasks with citations", f"{coverage:.1f}%")

        st.markdown("**Tasks per level**")
        lvl = df.groupby("depth").size().reset_index(name="count").sort_values("depth").set_index("depth")["count"]
        _bar_plot_or_table(lvl, "Tasks per level", "level")

        st.markdown("**Leaf vs Parent**")
        leaf = df["is_leaf"].map({True:"Leaf", False:"Parent"}).value_counts()
        # ensure stable order
        leaf = leaf.reindex(["Leaf","Parent"]).dropna()
        _bar_plot_or_table(leaf, "Leaf vs parent", "type")

        st.markdown("**Top Standards (if any)**")
        stds = []
        for xs in df["standards"].tolist():
            for s in (xs or []): stds.append(str(s))
        if stds:
            s = _pd.Series(stds).value_counts().head(12).sort_values(ascending=True)
            _bar_plot_or_table(s, "Top standards", "standard")
        else:
            st.info("No standards listed in WBS.")

        st.markdown("**Flat Task Table (id, name, depth)**")
        st.dataframe(df[["id","name","depth","is_leaf"]], use_container_width=True)
    else:
        st.warning("Could not coerce a WBS from the last response.")
# --- END: WBS Analytics Section ---
