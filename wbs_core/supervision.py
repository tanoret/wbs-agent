# wbs_core/supervision.py
# -*- coding: utf-8 -*-
"""
Create weakly-supervised instruction pairs for WBS training.

This module synthesizes (prompt, target) pairs and attaches a small set of
retrieved/citable chunks from your local corpus. It does NOT call any external
APIs and only uses what is in your local `corpus/` folder.

Outputs a JSONL where each line is:
{
  "prompt": "Design X for Y",
  "target": { ... WBS JSON object ... },
  "retrieved": [ { "id": "...", "title": "...", "url": "...", "text": "..." }, ... ],
  "meta": {
    "discipline": "seismic",
    "reactor_type": "PWR",
    "subject": "reactor coolant pumps",
    "citations": ["cfr-10-50-app-s","cfr-10-part100","rg-1-199-rev1"]
  }
}

CLI examples
------------
# Typical run with defaults; writes 200 samples
python -m wbs_core.supervision \
  --chunks-dir corpus/chunks \
  --index-dir corpus/index \
  --weak-map supervision/configs/weak_label_map.yml \
  --out supervision/data/instructions.jsonl \
  --num 200

# Restrict disciplines and reactor types; use k=8 retrieved chunks
python -m wbs_core.supervision \
  --disciplines seismic,ep,qa \
  --reactor-types PWR,BWR \
  --k 8 \
  --out supervision/data/instructions_small.jsonl

Programmatic
------------
from pathlib import Path
from wbs_core.supervision import SupervisionBuilder

bld = SupervisionBuilder()
recs = bld.build(num=50)
bld.write_jsonl(Path("supervision/data/instructions.jsonl"), recs)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    import joblib  # type: ignore
    import numpy as np  # type: ignore
    HAVE_SK = True
except Exception:
    HAVE_SK = False
    joblib = None
    np = None
    cosine_similarity = None  # type: ignore


# --------------------------
# Defaults and small vocab
# --------------------------

DEFAULT_REACTOR_TYPES = ["PWR", "BWR", "SFR", "MSR", "HTGR"]

# Subjects to sample per discipline (kept generic and non-proprietary)
SUBJECTS = {
    "seismic": [
        "reactor coolant pumps",
        "steam generators",
        "control room equipment",
        "containment structure",
        "spent fuel pool equipment",
    ],
    "qa": [
        "design control activities",
        "procurement documents",
        "test programs",
        "nonconformance reporting",
    ],
    "ep": [
        "emergency classification scheme",
        "offsite dose assessment",
        "emergency communications",
        "protective action recommendations",
    ],
    "cyber": [
        "reactor protection system network",
        "plant process computer network",
        "engineering workstations",
        "emergency preparedness systems",
    ],
    "licensing": [
        "combined license application",
        "environmental report",
        "ITAAC development",
        "inspections, tests, analyses, and acceptance criteria matrix",
    ],
    "radprot": [
        "ALARA program",
        "effluent monitoring",
        "access control and dose tracking",
        "radiation survey program",
    ],
    # Add more as needed; keep aligned with your available corpus
}

# Weak map fallback (used if YAML not provided). Keep IDs aligned with chunk doc_ids present in your corpus.
# Available (per user's tree): cfr-10-part50-full, cfr-10-50-36, cfr-10-50-47, cfr-10-50-55a,
# cfr-10-50-app-a, cfr-10-50-app-b, cfr-10-50-app-s, cfr-10-73-54, cfr-10-part20, cfr-10-part51,
# cfr-10-part52, cfr-10-part100, nureg-0800-index, rg-1-199-rev1
DEFAULT_WEAK_MAP = {
    "seismic": {
        "docs": ["cfr-10-50-app-s", "cfr-10-part100", "rg-1-199-rev1"],
        "srp_hints": ["3.7", "3.8"],  # hints only; we have the SRP index, not full chapters
    },
    "qa": {
        "docs": ["cfr-10-50-app-b"],
        "srp_hints": ["17"],  # QA/Trustworthiness topics
    },
    "ep": {
        "docs": ["cfr-10-50-47", "cfr-10-50-55a"],  # 50.47 + App E (file shows 50.55a is in corpus; App E not; keep 50.55a for eval)
        "srp_hints": ["13.3"],  # EP section number in SRP
    },
    "cyber": {
        "docs": ["cfr-10-73-54"],  # 10 CFR 73.54
        "srp_hints": ["13.6"],  # physical/cyber security interface in SRP
    },
    "licensing": {
        "docs": ["cfr-10-part52", "cfr-10-part51"],
        "srp_hints": ["1"],  # general considerations
    },
    "radprot": {
        "docs": ["cfr-10-part20"],
        "srp_hints": ["12"],  # radiation protection in SRP
    },
}

# Minimal WBS step templates per discipline
STEP_TEMPLATES = {
    "seismic": [
        "Identify safety-related SSCs subject to Seismic Category I classification.",
        "Derive site-specific design input (SSE/OBE) consistent with regulatory requirements.",
        "Perform seismic qualification strategy (analysis/test) selection for components.",
        "Define acceptance criteria and documentation for seismic qualification.",
    ],
    "qa": [
        "Establish organizational responsibilities and authority for QA.",
        "Define design control measures including verification and validation.",
        "Set procurement document controls and supplier qualification approach.",
        "Plan test, inspection, and nonconformance control processes.",
    ],
    "ep": [
        "Define emergency action levels and classification scheme.",
        "Establish notification, communication, and staffing arrangements.",
        "Develop dose assessment and protective action recommendation methods.",
        "Plan drills, training, and program maintenance for EP.",
    ],
    "cyber": [
        "Identify critical digital assets and critical systems.",
        "Perform cyber risk assessment and implement defensive architecture.",
        "Define access controls, monitoring, and incident response plans.",
        "Establish configuration control and periodic assessment program.",
    ],
    "licensing": [
        "Assemble application content and conformance statements.",
        "Map applicable regulations, exemptions, and licensing basis.",
        "Develop ITAAC and verification methods.",
        "Plan environmental report and associated evaluations.",
    ],
    "radprot": [
        "Define ALARA objectives and administrative control levels.",
        "Plan radiation monitoring and survey program.",
        "Establish effluent and environmental monitoring processes.",
        "Set personnel exposure tracking and access controls.",
    ],
}

# Inputs & deliverables skeletons per discipline
INPUTS = {
    "seismic": ["site seismic hazard", "equipment specs", "layout drawings"],
    "qa": ["org chart", "design procedures", "procurement specs"],
    "ep": ["site characteristics", "population data", "EAL scheme"],
    "cyber": ["network diagrams", "asset inventory", "threat model"],
    "licensing": ["COLA template", "applicant commitments", "ITAAC catalog"],
    "radprot": ["source term", "monitoring equipment list", "plant procedures"],
}
DELIVERABLES = {
    "seismic": ["seismic classification matrix", "qualification plan", "calculation package"],
    "qa": ["QA program description", "design control procedure", "supplier evaluation report"],
    "ep": ["emergency plan document", "dose assessment methodology", "training and drill plan"],
    "cyber": ["cyber security plan", "access control policy", "incident response plan"],
    "licensing": ["application package", "ITAAC matrix", "environmental report outline"],
    "radprot": ["ALARA plan", "radiation survey plan", "effluent monitoring procedure"],
}


# --------------------------
# Helpers
# --------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception as e:
                print(f"[supervision] WARN: skip line {i}: {e}", file=sys.stderr)
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _load_yaml(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        import yaml  # type: ignore
    except Exception:
        print("[supervision] WARN: PyYAML not installed; ignoring --weak-map file.", file=sys.stderr)
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------
# Retrieval backend (local)
# --------------------------

@dataclass
class LocalRetrieval:
    chunks_dir: Path
    index_dir: Optional[Path] = None
    use_tfidf: bool = True
    meta_rows: List[Dict[str, Any]] = field(default_factory=list)
    tfidf = None
    tfidf_mat = None

    def __post_init__(self):
        # Prefer prebuilt TF-IDF artifacts if present
        meta_p = None
        if self.index_dir:
            meta_p = self.index_dir / "meta.jsonl"
        if meta_p and meta_p.exists():
            self.meta_rows = _read_jsonl(meta_p)
        else:
            # Build meta by scanning chunks
            self.meta_rows = []
            for p in sorted(self.chunks_dir.glob("*.jsonl")):
                for row in _read_jsonl(p):
                    self.meta_rows.append(row)

        if self.use_tfidf and HAVE_SK and self.index_dir:
            vec_p = self.index_dir / "tfidf_vectorizer.joblib"
            mat_p = self.index_dir / "tfidf_matrix.npz"
            if vec_p.exists() and mat_p.exists():
                try:
                    self.tfidf = joblib.load(vec_p)  # type: ignore
                    self.tfidf_mat = np.load(mat_p)["arr_0"]  # type: ignore
                except Exception as e:
                    print(f"[supervision] WARN: failed loading TF-IDF ({e}); fallback to keyword scoring.", file=sys.stderr)
                    self.tfidf = None
                    self.tfidf_mat = None
            else:
                print("[supervision] INFO: TF-IDF artifacts not found; using keyword scoring.", file=sys.stderr)
                self.tfidf = None
                self.tfidf_mat = None
        else:
            if self.use_tfidf and not HAVE_SK:
                print("[supervision] INFO: scikit-learn not available; using keyword scoring.", file=sys.stderr)
            self.tfidf = None
            self.tfidf_mat = None

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        if self.tfidf is not None and self.tfidf_mat is not None:
            try:
                qv = self.tfidf.transform([query])  # type: ignore
                sims = cosine_similarity(qv, self.tfidf_mat).ravel()  # type: ignore
                top_idx = sims.argsort()[-k:][::-1]
                rows = [self.meta_rows[i] for i in top_idx]
                return self._compact(rows)
            except Exception as e:
                print(f"[supervision] WARN: TF-IDF search failed ({e}); using keyword scoring.", file=sys.stderr)
        # keyword fallback
        scores = []
        q_terms = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
        for i, r in enumerate(self.meta_rows):
            txt = (_safe_get(r, "text", default="") or "").lower()
            score = sum(txt.count(t) for t in q_terms)
            if score:
                scores.append((score, i))
        scores.sort(reverse=True)
        take = [self.meta_rows[i] for _, i in scores[:k]]
        return self._compact(take)

    def _compact(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            out.append(
                {
                    "id": _safe_get(r, "id", default=_safe_get(r, "row_index", default=None)),
                    "doc_id": r.get("doc_id"),
                    "title": r.get("title"),
                    "section": r.get("section"),
                    "url": r.get("url"),
                    "text": r.get("text") or r.get("chunk") or "",
                }
            )
        return out

    def doc_lookup(self) -> Dict[str, Dict[str, Any]]:
        """Map doc_id -> representative (title, url) using first seen row."""
        m: Dict[str, Dict[str, Any]] = {}
        for r in self.meta_rows:
            doc = r.get("doc_id")
            if not doc or doc in m:
                continue
            m[doc] = {"title": r.get("title"), "url": r.get("url")}
        return m


# --------------------------
# Builder
# --------------------------

@dataclass
class SupervisionBuilder:
    chunks_dir: Path = Path("corpus/chunks")
    index_dir: Path = Path("corpus/index")
    weak_map_path: Optional[Path] = None
    seed: int = 42
    k: int = 6

    def __post_init__(self):
        random.seed(self.seed)
        # Weak map
        wm = _load_yaml(self.weak_map_path)
        if isinstance(wm, dict) and wm:
            self.weak_map = wm
        else:
            self.weak_map = DEFAULT_WEAK_MAP
            if self.weak_map_path:
                print(f"[supervision] WARN: using built-in weak map (failed to load {self.weak_map_path})", file=sys.stderr)
        # Retrieval
        self.retr = LocalRetrieval(self.chunks_dir, self.index_dir)

        # doc meta lookup
        self.doc_meta = self.retr.doc_lookup()

    # ---------- Public API ----------

    def build(
        self,
        num: int = 200,
        disciplines: Optional[List[str]] = None,
        reactor_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not disciplines:
            disciplines = [d for d in self.weak_map.keys() if d in STEP_TEMPLATES]
        if not reactor_types:
            reactor_types = DEFAULT_REACTOR_TYPES

        pool: List[Tuple[str, str, str]] = []  # (discipline, reactor_type, subject)
        for d in disciplines:
            subs = SUBJECTS.get(d, ["system"])
            for rt in reactor_types:
                for subj in subs:
                    pool.append((d, rt, subj))

        if not pool:
            raise RuntimeError("Empty combination of disciplines/reactor_types/subjects.")

        # sample without replacement; if num > pool, cycle
        out: List[Dict[str, Any]] = []
        i = 0
        while len(out) < num:
            d, rt, subj = pool[i % len(pool)]
            prompt = self._make_prompt(d, rt, subj)
            citations_ids = self._citations_for_discipline(d)
            retrieved = self.retr.search(prompt, k=self.k)
            target = self._make_wbs_target(d, prompt, citations_ids, retrieved)
            rec = {
                "prompt": prompt,
                "target": target,
                "retrieved": retrieved,
                "meta": {
                    "discipline": d,
                    "reactor_type": rt,
                    "subject": subj,
                    "citations": citations_ids,
                },
            }
            out.append(rec)
            i += 1
        return out

    def write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> int:
        n = _write_jsonl(path, rows)
        print(f"[supervision] Wrote {n} instruction pairs -> {path}")
        return n

    # ---------- Internals ----------

    def _make_prompt(self, discipline: str, reactor_type: str, subject: str) -> str:
        if discipline == "seismic":
            return f"Design the seismic classification and qualification for the {subject} in a {reactor_type}"
        if discipline == "qa":
            return f"Develop the QA program scope and controls for {subject} in a {reactor_type}"
        if discipline == "ep":
            return f"Develop the emergency preparedness program for {subject} in a {reactor_type}"
        if discipline == "cyber":
            return f"Develop the cyber security program for the {subject} in a {reactor_type}"
        if discipline == "licensing":
            return f"Prepare the licensing submittal content for {subject} for a {reactor_type}"
        if discipline == "radprot":
            return f"Design the radiation protection program for {subject} in a {reactor_type}"
        return f"Design the {discipline} plan for {subject} in a {reactor_type}"

    def _citations_for_discipline(self, discipline: str) -> List[str]:
        docs = self.weak_map.get(discipline, {}).get("docs", [])
        return [str(d) for d in docs]

    def _code_entries(self, doc_ids: List[str]) -> List[Dict[str, str]]:
        out = []
        for did in doc_ids:
            meta = self.doc_meta.get(did, {})
            title = meta.get("title") or self._title_from_id(did)
            url = meta.get("url") or ""
            out.append({"id": did, "title": title or did, "section": "", "url": url})
        return out

    @staticmethod
    def _title_from_id(doc_id: str) -> str:
        # Friendly title fallback
        if doc_id.startswith("cfr-10-part"):
            part = doc_id.replace("cfr-10-part", "")
            return f"10 CFR Part {part}"
        if "app-a" in doc_id:
            return "10 CFR Part 50, Appendix A"
        if "app-b" in doc_id:
            return "10 CFR Part 50, Appendix B"
        if "app-s" in doc_id:
            return "10 CFR Part 50, Appendix S"
        if doc_id == "cfr-10-50-47":
            return "10 CFR 50.47"
        if doc_id == "cfr-10-50-36":
            return "10 CFR 50.36"
        if doc_id == "cfr-10-50-55a":
            return "10 CFR 50.55a"
        if doc_id == "cfr-10-73-54":
            return "10 CFR 73.54"
        if doc_id == "rg-1-199-rev1":
            return "Regulatory Guide 1.199 Rev.1"
        if doc_id == "nureg-0800-index":
            return "SRP NUREG-0800 (Index)"
        return doc_id

    def _make_wbs_target(
        self,
        discipline: str,
        prompt: str,
        citation_ids: List[str],
        retrieved: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        steps = STEP_TEMPLATES.get(discipline, ["Plan task execution.", "Document results."])
        inputs = INPUTS.get(discipline, ["design inputs"])
        deliverables = DELIVERABLES.get(discipline, ["deliverable(s)"])
        codes = self._code_entries(citation_ids)

        # Single top-level task (you can expand into multi-task WBS if desired)
        task_id = f"{discipline}-001"
        wbs = [
            {
                "id": task_id,
                "title": self._title_from_prompt(prompt),
                "discipline": discipline,
                "description": f"Execute the {discipline} engineering scope per cited regulations and guidance.",
                "inputs": inputs,
                "deliverables": deliverables,
                "codes": codes,
                "steps": steps,
            }
        ]

        # Create assumptions/notes with traceability to retrieved chunk IDs
        cite_ids = [str(r.get("id") or r.get("doc_id") or i) for i, r in enumerate(retrieved, 1)]
        assumptions = [
            "All design activities must conform to the cited regulations and guidance.",
            "Use plant-specific inputs where required (e.g., site hazard, layout, system configuration).",
        ]
        notes = [f"Retrieved context chunk refs: {', '.join(cite_ids)}"] if cite_ids else []

        return {"wbs": wbs, "assumptions": assumptions, "notes": notes}

    @staticmethod
    def _title_from_prompt(prompt: str) -> str:
        t = prompt.strip().rstrip(".")
        # Keep concise title
        if len(t) > 120:
            return t[:117] + "..."
        return t


# --------------------------
# CLI
# --------------------------

def _parse_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build weakly-supervised instruction pairs from local corpus")
    p.add_argument("--chunks-dir", type=str, default="corpus/chunks", help="Path to chunks JSONL directory")
    p.add_argument("--index-dir", type=str, default="corpus/index", help="Path to index directory (TF-IDF artifacts)")
    p.add_argument("--weak-map", type=str, default=None, help="YAML mapping of discipline -> doc ids")
    p.add_argument("--disciplines", type=str, default=None, help="Comma list (e.g., seismic,qa,ep,cyber,licensing,radprot)")
    p.add_argument("--reactor-types", type=str, default=None, help=f"Comma list of reactor types (default: {','.join(DEFAULT_REACTOR_TYPES)})")
    p.add_argument("--k", type=int, default=6, help="Top-k retrieved chunks to attach for each instruction")
    p.add_argument("--num", type=int, default=200, help="Number of instruction pairs to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=str, required=True, help="Output JSONL file")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    chunks_dir = Path(args.chunks_dir).resolve()
    index_dir = Path(args.index_dir).resolve()
    weak_map = Path(args.weak_map).resolve() if args.weak_map else None
    out = Path(args.out).resolve()

    if not chunks_dir.exists():
        raise FileNotFoundError(f"chunks-dir not found: {chunks_dir}")
    if not index_dir.exists():
        print(f"[supervision] INFO: index-dir missing ({index_dir}); retrieval will scan chunks only.", file=sys.stderr)

    b = SupervisionBuilder(
        chunks_dir=chunks_dir,
        index_dir=index_dir,
        weak_map_path=weak_map,
        seed=int(args.seed),
        k=int(args.k),
    )

    disciplines = _parse_list(args.disciplines)
    reactor_types = _parse_list(args.reactor_types)

    recs = b.build(num=int(args.num), disciplines=disciplines, reactor_types=reactor_types)
    b.write_jsonl(out, recs)


if __name__ == "__main__":
    main()
