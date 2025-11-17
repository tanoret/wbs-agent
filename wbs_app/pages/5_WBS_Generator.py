# wbs_app/pages/5_WBS_Generator.py
from __future__ import annotations
import os, sys, re, json, textwrap, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st

APP_ROOT = Path.cwd()
BASE_FILES = APP_ROOT / "base_files"
MODEL_SCRIPTS = BASE_FILES / "model" / "scripts"
RAG_INFER = MODEL_SCRIPTS / "rag_infer.py"
CORPUS_ROOT = BASE_FILES / "corpus"
INDEX_DIR = CORPUS_ROOT / "index"
EMBED_MODEL_DIR = CORPUS_ROOT / "models" / "bge-small-en-v1.5"
MODELS_DIR = BASE_FILES / "models"
DEFAULT_MERGED = BASE_FILES / "model" / "merged" / "qwen2.5-7b-wbs"
DEFAULT_BASE_3B = MODELS_DIR / "Qwen2.5-3B-Instruct"
DEFAULT_BASE_7B = MODELS_DIR / "Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = BASE_FILES / "model" / "checkpoints" / "wbs-lora"
LAST_RESPONSE_CANDIDATES = [BASE_FILES / "model" / "last_response.json", APP_ROOT / "model" / "last_response.json"]
EXPORTS_DIR = BASE_FILES / "model" / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_PYTHON = os.environ.get("WBS_PYTHON", sys.executable)

def newest_existing(paths: List[Path]) -> Optional[Path]:
    xs = [p for p in paths if p.exists()]
    if not xs: return None
    xs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return xs[0]

def run_cli(py_script: Path, args: List[str], env: Optional[Dict[str, str]] = None, python_bin: Optional[str] = None) -> Tuple[int, str, str]:
    py = python_bin or DEFAULT_PYTHON
    cmd = [py, str(py_script)] + list(map(str, args))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr

def show_logs(stdout: str, stderr: str):
    if stdout.strip():
        with st.expander("stdout", expanded=False): st.code(stdout, language="text")
    if stderr.strip():
        with st.expander("stderr", expanded=False): st.code(stderr, language="text")

def load_last_response() -> Optional[Dict[str, Any]]:
    p = newest_existing(LAST_RESPONSE_CANDIDATES)
    if not p: return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    for blk in re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        try: return json.loads(blk)
        except Exception: pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start:i+1]
                    try: return json.loads(cand)
                    except Exception: break
        start = text.find("{", start+1)
    try: return json.loads(text)
    except Exception: return None

def extract_all_json(text: str) -> List[Dict[str, Any]]:
    if not text: return []
    objs: List[Dict[str, Any]] = []
    for blk in re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
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
                    start = i + 1
                    break
            i += 1
        else: break
    uniq, seen = [], set()
    for o in objs:
        k = json.dumps(o, sort_keys=True, ensure_ascii=False)
        if k not in seen:
            seen.add(k); uniq.append(o)
    return uniq

def coerce_to_wbs(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict): return None
    title = obj.get("wbs_title") or obj.get("project") or obj.get("title") or "WBS"
    tasks = obj.get("tasks") or obj.get("items") or obj.get("work_breakdown") or (obj.get("wbs") or {}).get("tasks") or []
    if isinstance(tasks, dict): tasks = list(tasks.values())
    if not isinstance(tasks, list): return None
    norm: List[Dict[str, Any]] = []
    for idx, t in enumerate(tasks, 1):
        if not isinstance(t, dict):
            norm.append({"id": f"T{idx}", "name": str(t), "description": "", "children": []}); continue
        name = t.get("name") or t.get("title") or f"Task {idx}"
        desc = t.get("description") or t.get("objective") or ""
        kids = t.get("children") or t.get("subtasks") or []
        standards = t.get("standards") or t.get("applicable_codes") or []
        citations_raw = t.get("citations") or t.get("sources") or []
        citations: List[Dict[str, str]] = []
        if isinstance(citations_raw, list):
            for c in citations_raw:
                if isinstance(c, dict) and "id" in c: citations.append({"id": str(c["id"])})
                else: citations.append({"id": str(c)})
        norm.append({
            "id": t.get("id") or t.get("task_id") or f"T{idx}",
            "name": name,
            "description": desc,
            "inputs": t.get("inputs") or [],
            "outputs": t.get("outputs") or [],
            "standards": standards,
            "citations": citations,
            "children": kids if isinstance(kids, list) else [],
        })
    return {"project": title, "tasks": norm}

# ---------- NEW: heuristic expander to split long descriptions ----------
_SPLIT_PATTERNS = [
    r"^\s*[-*•]\s+(.+)$",     # bullets
    r"^\s*\d+[.)]\s+(.+)$",   # numbered
]

def _split_lines(desc: str) -> List[str]:
    lines = [ln.strip() for ln in desc.splitlines() if ln.strip()]
    picked: List[str] = []
    # bullets / numbered
    for ln in lines:
        m = None
        for pat in _SPLIT_PATTERNS:
            m = re.match(pat, ln)
            if m: break
        if m:
            picked.append(m.group(1).strip())
    if len(picked) >= 2:  # found a decent list
        return picked
    # fallbacks: semicolons / em-dashes / periods (short fragments)
    text = desc.replace("—", ";").replace("–", ";")
    parts = re.split(r"[;•]+", text)
    parts = [p.strip(" .") for p in parts if len(p.strip()) > 2]
    if len(parts) >= 2:
        return parts[:12]
    # last resort: split by sentences and keep short ones
    sents = re.split(r"(?<=[.!?])\s+", desc)
    sents = [s.strip() for s in sents if 3 <= len(s.strip()) <= 180]
    return sents[:10] if len(sents) >= 2 else []

def heuristically_expand_tasks(wbs: Dict[str, Any]) -> Dict[str, Any]:
    """If a task has a long description and no children, split into subtasks."""
    def expand_task(t: Dict[str, Any]) -> Dict[str, Any]:
        kids = t.get("children") or []
        if kids:  # already structured
            t["children"] = [expand_task(ch) for ch in kids]
            return t
        desc = t.get("description") or ""
        if len(desc) < 60:   # not long enough to try
            return t
        frags = _split_lines(desc)
        if len(frags) < 2:   # no useful split
            return t
        # make subtasks
        new_kids = []
        for i, frag in enumerate(frags, 1):
            name = frag
            # a short "title-ish" name
            if len(name) > 60:
                name = name[:57].rstrip() + "…"
            new_kids.append({
                "id": f"{t.get('id','T')}.{i}",
                "name": name,
                "description": frag,
                "inputs": [],
                "outputs": [],
                "standards": t.get("standards", []),
                "citations": t.get("citations", []),
                "children": [],
            })
        t["children"] = new_kids
        return t

    wbs["tasks"] = [expand_task(t) for t in (wbs.get("tasks") or [])]
    return wbs
# ------------------------------------------------------------------------

def build_wbs_from_text(model_text: str) -> Optional[Dict[str, Any]]:
    cands = extract_all_json(model_text)
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for o in cands:
        w = coerce_to_wbs(o)
        if w: scored.append((len(w.get("tasks") or []), w))
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
    tasks: List[Dict[str, Any]] = []
    for line in model_text.splitlines():
        m = re.match(r'^\s*[-*•]\s*(.+)', line) or re.match(r'^\s*\d+\.\s*(.+)', line)
        if m:
            label = m.group(1).strip()
            if label:
                tasks.append({"id": f"T{len(tasks)+1}", "name": label, "description": "", "children": []})
    if tasks: return {"project": "WBS", "tasks": tasks}
    return None

def wrap_label(s: str, width: int = 28, max_lines: int = 6) -> str:
    if not s: return ""
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines: lines = lines[:max_lines-1] + ["…"]
    return "\\n".join(lines)

def wbs_to_nodes_edges(wbs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    nodes: List[Dict[str, Any]] = []; edges: List[Tuple[str, str]] = []
    def add_task(task: Dict[str, Any], parent_id: Optional[str] = None):
        tid = str(task.get("id") or task.get("task_id") or task.get("name") or f"node_{len(nodes)+1}")
        name = task.get("name") or task.get("title") or tid
        desc = task.get("description") or task.get("objective") or ""
        node_label = f"{name}"
        if desc: node_label += f"\\n{wrap_label(desc, 28, 4)}"
        nodes.append({"id": tid, "label": node_label, "raw": task})
        if parent_id: edges.append((parent_id, tid))
        for ch in (task.get("children") or task.get("subtasks") or []):
            add_task(ch, tid)
    for t in (wbs.get("tasks") or []): add_task(t, None)
    return nodes, edges

def wbs_to_dot(wbs: Dict[str, Any]) -> str:
    title = wbs.get("project") or wbs.get("wbs_title") or wbs.get("title") or "WBS"
    nodes, edges = wbs_to_nodes_edges(wbs)
    lines = [
        "digraph WBS {",
        "  rankdir=TB;",  # top->bottom reads more like a WBS tree
        '  node [shape=box, style="rounded,filled", fillcolor="#f6f8fa", color="#c9d1d9", fontname="Helvetica"];',
        '  edge [color="#95a5a6"];',
        f'  labelloc="t"; label="{title}"; fontsize=18;'
    ]
    for n in nodes:
        nid = n["id"].replace('"', '\\"'); lab = n["label"].replace('"', '\\"')
        lines.append(f'  "{nid}" [label="{lab}"];')
    for a, b in edges:
        a = a.replace('"', '\\"'); b = b.replace('"', '\\"')
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    return "\n".join(lines)

def show_citations(retrieved: List[Dict[str, Any]]):
    st.markdown("#### Retrieved Chunks (for grounding)")
    if not retrieved:
        st.info("No retrieved chunks found."); return
    rows = []
    for r in retrieved:
        if "meta" in r:
            m = r.get("meta", {}) or {}
            src = m.get("source") or m.get("doc_id") or m.get("id") or ""
            title = m.get("title") or m.get("section") or ""; score = r.get("score", "")
        else:
            src = r.get("id", "") or (r.get("source", "") or "")
            title = r.get("title", ""); score = r.get("score", "")
        rows.append({"source/id": str(src), "title/section": str(title), "score": f"{score:.3f}" if isinstance(score, (int, float)) else ""})
    st.dataframe(rows, use_container_width=True)

def save_export(wbs: Dict[str, Any], dot: str) -> Tuple[Path, Path]:
    base = EXPORTS_DIR / "wbs_export"; idx = 1
    json_path = base.with_suffix(".json"); dot_path = base.with_suffix(".dot")
    while json_path.exists() or dot_path.exists():
        idx += 1
        json_path = EXPORTS_DIR / f"wbs_export_{idx}.json"
        dot_path = EXPORTS_DIR / f"wbs_export_{idx}.dot"
    json_path.write_text(json.dumps(wbs, indent=2), encoding="utf-8")
    dot_path.write_text(dot, encoding="utf-8")
    return json_path, dot_path

# ----------------------- UI --------------------------------------------------
st.set_page_config(page_title="5 · WBS Generator", layout="wide")
st.title("⑤ WBS Generator (RAG-grounded)")

with st.sidebar:
    st.header("Environment")
    st.code(str(CORPUS_ROOT)); st.code(str(INDEX_DIR)); st.code(str(EMBED_MODEL_DIR))
    st.markdown("---")
    python_bin = st.text_input("Python for CLI (optional)", value=DEFAULT_PYTHON)

st.subheader("Model Selection")
mode = st.radio("Choose inference mode", ["Use merged model", "Use base + LoRA adapter"], index=0, horizontal=True)
c1, c2 = st.columns(2)
if mode == "Use merged model":
    merged_dir = c1.text_input("Merged model path", value=str(DEFAULT_MERGED))
    adapter_dir = None; base_dir = merged_dir
else:
    base_dir = c1.text_input("Base model path", value=str(DEFAULT_BASE_3B if DEFAULT_BASE_3B.exists() else DEFAULT_BASE_7B))
    adapter_dir = c2.text_input("LoRA adapter path", value=str(DEFAULT_ADAPTER))

st.subheader("RAG & Decoding")
d1, d2, d3, d4 = st.columns(4)
with d1: top_k = st.number_input("Top-k retrieved", 2, 32, 8, 1)
with d2: max_new_tokens = st.number_input("Max new tokens", 64, 4096, 700, 16)
with d3: temperature = st.number_input("Temperature", 0.0, 1.0, 0.2, 0.05)
with d4: force_json = st.checkbox("Force JSON in prompt", True)

st.subheader("Prompt")
default_prompt = "Design the seismic classification and qualification WBS for the reactor coolant pumps in a PWR."
user_prompt = st.text_area("Enter your design/engineering request", value=default_prompt, height=120)
run_btn = st.button("Generate WBS", type="primary", use_container_width=True)

if run_btn:
    if not Path(base_dir).exists(): st.error("Selected base/merged model path does not exist."); st.stop()
    if mode == "Use base + LoRA adapter" and (not adapter_dir or not Path(adapter_dir).exists()):
        st.error("Adapter path not found."); st.stop()
    if not CORPUS_ROOT.exists(): st.error("Corpus root not found. Build it on Page ②."); st.stop()

    query = user_prompt.strip()
    if force_json:
        query += (
            "\n\nRespond ONLY with a JSON object following this schema keys: "
            "{'wbs_title': str, 'tasks': [{'name': str, 'description': str, "
            "'applicable_codes': [str], 'guidance': str, 'citations': [{'id': 'doc#chunk'}]}]}."
            "Do not include any prose before or after the JSON."
        )

    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
    if EMBED_MODEL_DIR.exists(): env["EMBED_LOCAL_PATH"] = str(EMBED_MODEL_DIR)

    args = ["--base", str(base_dir), "--q", query, "--corpus-root", str(CORPUS_ROOT), "--k", str(int(top_k)), "--max-new-tokens", str(int(max_new_tokens)), "--temperature", str(float(temperature))]
    if mode == "Use base + LoRA adapter": args += ["--adapter", str(adapter_dir)]

    with st.spinner("Running single-prompt RAG inference..."):
        rc, out, err = run_cli(RAG_INFER, args, env=env, python_bin=python_bin)
    show_logs(out, err)
    if rc != 0: st.error("RAG inference failed."); st.stop()

    payload = load_last_response() or {}
    if not payload:
        try: payload = json.loads(out)
        except Exception: payload = {}

    raw_text = (payload.get("output_text") or "") or (payload.get("output") or "") or (payload.get("text") or payload.get("response") or "")
    retrieved = payload.get("retrieved") or []

    st.subheader("Raw Model Output (truncated)")
    with st.expander("show text", expanded=False):
        st.code(raw_text[:5000] + ("..." if len(raw_text) > 5000 else ""), language="json")

    parsed_wbs = payload.get("output_json") or extract_json(raw_text) or {}
    wbs: Optional[Dict[str, Any]] = coerce_to_wbs(parsed_wbs) if isinstance(parsed_wbs, dict) else None
    if not wbs: wbs = build_wbs_from_text(raw_text)

    if not wbs or not (wbs.get("tasks") or []):
        st.error("Could not derive a WBS with tasks from the output text. Try re-running with 'Force JSON in prompt'.")
    else:
        # NEW: expand long descriptions into child tasks
        wbs = heuristically_expand_tasks(wbs)

        st.subheader("Parsed WBS (JSON)")
        st.json(wbs)

        st.subheader("WBS Graph")
        dot = wbs_to_dot(wbs)
        st.graphviz_chart(dot, use_container_width=True)

        show_citations(retrieved)

        st.subheader("Export")
        json_path, dot_path = save_export(wbs, dot)
        st.success("Artifacts saved.")
        st.download_button("Download WBS JSON", data=json.dumps(wbs, indent=2), file_name=json_path.name, mime="application/json", use_container_width=True)
        st.download_button("Download Graph (.dot)", data=dot, file_name=dot_path.name, mime="text/vnd.graphviz", use_container_width=True)
