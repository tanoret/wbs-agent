#!/usr/bin/env python3
"""
Create instruction pairs from your local corpus.

- Input: corpus/chunks/*.jsonl (created in Step A)
- Config: manifests/weak_label_map.yml
- Output: supervision/instructions.jsonl  (each line: {"prompt": "...", "target": {WBS...}, "weak_labels": {...}})

Usage:
  python scripts/05_make_supervision.py \
      --chunks-dir corpus/chunks \
      --out supervision/instructions.jsonl \
      --num 50 \
      --reactor-types "PWR,BWR,MSR,SFR,HTGR,SMR"

Notes:
- Pure offline (no internet).
- Citations come from your chunk files; if a mapped doc_id is missing, it’s skipped with a warning.
- You can safely re-run; it overwrites the output file.
"""

import argparse, json, random, re, sys
from pathlib import Path
import yaml

DEFAULT_DISCIPLINES = ["seismic","qa","core_physics","ep","cybersecurity"]

def load_chunks(chunks_dir: Path):
    """Return {doc_id: [chunk_records...]}."""
    index = {}
    for f in sorted(chunks_dir.glob("*.jsonl")):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                rec = json.loads(line)
                doc_id = rec.get("doc_id")
                if not doc_id: continue
                index.setdefault(doc_id, []).append(rec)
    # sort chunks per doc by chunk_no
    for did in index:
        index[did].sort(key=lambda r: r.get("chunk_no", 0))
    return index

def pick_citations(doc_id, chunks_idx, keywords=None, max_hits=2):
    """Pick up to max_hits chunks from doc_id; prefer those containing keywords."""
    if doc_id not in chunks_idx:
        sys.stderr.write(f"[WARN] doc_id missing in chunks: {doc_id}\n")
        return []
    chunks = chunks_idx[doc_id]
    scored = []
    kw = [k.lower() for k in (keywords or []) if k]
    for r in chunks:
        text = r.get("text","")
        score = 0
        for k in kw:
            if k in text.lower():
                score += 1
        scored.append((score, r))
    # prefer keyword hits, then earlier chunks
    scored.sort(key=lambda t: (-t[0], t[1].get("chunk_no", 0)))
    hits = []
    for _, r in scored[:max_hits]:
        hits.append({
            "doc_id": r.get("doc_id"),
            "chunk_no": r.get("chunk_no"),
            "source_url": r.get("source_url"),
            "title": r.get("title"),
            "snippet": (r.get("text","").strip()[:350] + "…") if r.get("text") else ""
        })
    return hits

def wbs_skeleton(title, scope, discipline, reactor_type, mapped_docs, chunks_idx):
    """
    Build a small but structured WBS with discipline-specific tasks and citations.
    """
    def cite(doc_id, anchors):
        return pick_citations(doc_id, chunks_idx, anchors, max_hits=1)

    tasks = []
    # Discipline-specific task templates
    if discipline == "seismic":
        tasks = [
            {
                "id": "1.0",
                "name": "Classify SSCs by Seismic Design Class",
                "description": "Classify safety-related SSCs and I&C per seismic design classification.",
                "inputs": ["Plant layout", "System P&IDs", "Safety analyses", "SSC list"],
                "methods": ["Map functions to safety categories", "Apply RG 1.29 criteria"],
                "deliverables": ["Seismic classification matrix", "Basis memo"],
                "acceptance_criteria": ["Consistent with RG 1.29, including interfaces and dependencies"],
                "citations": cite("rg-1-29-rev6", ["seismic","classification","RG 1.29"])
            },
            {
                "id": "2.0",
                "name": "Define Site Ground Motion & SSE",
                "description": "Establish SSE/OBE and ground motion spectra based on site hazard.",
                "inputs": ["Site geotech/PSHA", "Regional seismicity data"],
                "methods": ["Probabilistic hazard", "Site response modeling"],
                "deliverables": ["SSE/OBE report", "Ground motion spectra"],
                "acceptance_criteria": ["Meets 10 CFR Part 50 Appendix S criteria"],
                "citations": cite("cfr-10-50-app-s", ["earthquake","SSE","ground motion"])
            },
            {
                "id": "3.0",
                "name": "Seismic Analysis & Qualification",
                "description": "Analyze and qualify structures/components to SSE; define design input.",
                "inputs": ["SSE spectra", "SSCs models", "Material data"],
                "methods": ["Modal/dynamic analyses", "Equipment qualification"],
                "deliverables": ["Seismic analysis reports", "Design input files"],
                "acceptance_criteria": ["Margins per design basis; documentation of assumptions & uncertainties"],
                "citations": cite("cfr-10-50-app-s", ["analysis","qualification"])
            }
        ]
    elif discipline == "qa":
        tasks = [
            {
                "id": "1.0",
                "name": "Establish QA Program",
                "description": "Define and implement QA program for design and construction.",
                "inputs": ["Project Q-List", "Org structure", "Procedures"],
                "methods": ["Program plan per Appendix B", "RG 1.28 alignment"],
                "deliverables": ["QA Program Description", "Implementing procedures"],
                "acceptance_criteria": ["Compliant with 10 CFR 50 App B and RG 1.28"],
                "citations": cite("cfr-10-50-app-b", ["quality assurance"]) + cite("rg-1-28-rev6", ["quality assurance"])
            },
            {
                "id": "2.0",
                "name": "Design Control & Records",
                "description": "Establish design control, document control, and records management.",
                "inputs": ["Engineering procedures", "Configuration mgmt plan"],
                "methods": ["Procedure development", "Training"],
                "deliverables": ["Design control procedure", "Records index"],
                "acceptance_criteria": ["Traceability, verification, independent review"],
                "citations": cite("cfr-10-50-app-b", ["design control"])
            },
            {
                "id": "3.0",
                "name": "Supplier QA & Audits",
                "description": "Qualify suppliers, perform surveillances and audits.",
                "inputs": ["Approved supplier list", "Audit schedules"],
                "methods": ["Audits per App B", "Procurement controls"],
                "deliverables": ["Audit reports", "Supplier qualification files"],
                "acceptance_criteria": ["Procurement/CGD controls established"],
                "citations": cite("cfr-10-50-app-b", ["procurement"])
            }
        ]
    elif discipline == "core_physics":
        tasks = [
            {
                "id": "1.0",
                "name": "Define Nuclear Design Methods",
                "description": "Select and document neutronics/thermal feedback/kinetics methods & code suite.",
                "inputs": ["Design basis", "Fuel data", "XS libraries"],
                "methods": ["Code qualification", "Uncertainty methods"],
                "deliverables": ["Methodology report", "V&V plan"],
                "acceptance_criteria": ["Benchmark basis and validation cases documented"],
                "citations": pick_citations("nureg-0800-index", chunks_idx, ["Chapter 4","reactor"], 1)
            },
            {
                "id": "2.0",
                "name": f"Core Neutronics Model — {reactor_type}",
                "description": "Build steady-state model; compute k-eff, power shapes, reactivity coefficients.",
                "inputs": ["Geometry", "Materials", "Operating conditions"],
                "methods": ["3-D transport or diffusion", "Spectrum-sensitive modeling"],
                "deliverables": ["Analysis report", "Model input decks"],
                "acceptance_criteria": ["Uncertainty quantified; peer review complete"],
                "citations": pick_citations("nureg-0800-index", chunks_idx, ["Chapter 4"], 1)
            },
            {
                "id": "3.0",
                "name": "Validation vs Benchmarks",
                "description": "Compare predictions vs ICSBEP/IRPhEP/plant data; tune uncertainties.",
                "inputs": ["Benchmark specs", "Test data"],
                "methods": ["Code-to-code", "Critical experiments"],
                "deliverables": ["Validation report", "Uncertainty tables"],
                "acceptance_criteria": ["Meets acceptance criteria in SRP nuclear design sections"],
                "citations": pick_citations("nureg-0800-index", chunks_idx, ["nuclear design"], 1)
            }
        ]
    elif discipline == "ep":
        tasks = [
            {
                "id": "1.0",
                "name": "Hazards & EALs",
                "description": "Perform hazards evaluation and define Emergency Action Levels.",
                "inputs": ["Site hazards", "Source terms"],
                "methods": ["Deterministic and PRA inputs"],
                "deliverables": ["Hazards analysis", "EAL technical basis"],
                "acceptance_criteria": ["Meets §50.47 requirements"],
                "citations": cite("cfr-10-50-47", ["emergency plan","EAL"])
            },
            {
                "id": "2.0",
                "name": "Emergency Planning Program",
                "description": "Organization, training, drills, and interfaces with offsite authorities.",
                "inputs": ["Org charts", "Training plans", "MOUs"],
                "methods": ["Program/procedure development"],
                "deliverables": ["EP Program manual", "Training & drill records"],
                "acceptance_criteria": ["Capabilities consistent with §50.47(b) elements"],
                "citations": cite("cfr-10-50-47", ["capability","planning"])
            }
        ]
    elif discipline == "cybersecurity":
        tasks = [
            {
                "id": "1.0",
                "name": "Cyber Security Program",
                "description": "Establish cyber program for digital systems important to safety.",
                "inputs": ["CDAs inventory", "Architecture diagrams"],
                "methods": ["Defensive architecture", "Control selection per RG 5.71"],
                "deliverables": ["Cyber Security Plan", "Implementation procedures"],
                "acceptance_criteria": ["Meets 10 CFR 73.54 and RG 5.71 guidance"],
                "citations": cite("cfr-10-73-54", ["cyber"]) + cite("rg-5-71-rev1", ["cyber"])
            },
            {
                "id": "2.0",
                "name": "Assess, Implement, Verify",
                "description": "Conduct consequence-based assessment; implement controls; verify effectiveness.",
                "inputs": ["Threat model", "System configurations"],
                "methods": ["Assess-controls-verify cycle"],
                "deliverables": ["Assessment report", "Verification records"],
                "acceptance_criteria": ["Controls implemented and verified per plan"],
                "citations": cite("rg-5-71-rev1", ["assessment","verification"])
            }
        ]
    else:
        tasks = [{
            "id":"1.0",
            "name":"Define scope",
            "description":"Discipline not recognized; placeholder WBS.",
            "inputs":[], "methods":[], "deliverables":[],
            "acceptance_criteria":[], "citations":[]
        }]

    wbs = {
        "wbs_id": f"WBS-{discipline}-{reactor_type}",
        "title": title,
        "scope": scope,
        "assumptions": [f"Reactor type: {reactor_type}"],
        "discipline": discipline,
        "tasks": tasks,
        "traceability": {
            "regulatory_basis": list({c['doc_id'] for t in tasks for c in t.get('citations',[]) if c.get('doc_id')}),
            "references": [c for t in tasks for c in t.get('citations',[])]
        },
        "standards_map": mapped_docs  # weak-label docs used
    }
    return wbs

def synth_pair(discipline, reactor_type, map_cfg, chunks_idx):
    # weak-label docs for this discipline
    mapped = map_cfg.get("mappings", {}).get(discipline, {}).get("docs", [])
    prompt = f"Design {discipline.replace('_',' ')} for {reactor_type}"
    title  = f"{discipline.replace('_',' ').title()} — {reactor_type}"
    scope  = f"Develop {discipline.replace('_',' ')} work for a {reactor_type} nuclear power plant."
    wbs = wbs_skeleton(title, scope, discipline, reactor_type, mapped_docs=mapped, chunks_idx=chunks_idx)
    out = {
        "prompt": prompt,
        "target": wbs,
        "weak_labels": {
            "discipline": discipline,
            "reactor_type": reactor_type,
            "docs": mapped
        }
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", default=str(Path(__file__).resolve().parents[2] / "corpus" / "chunks"))
    ap.add_argument("--map",        default=str(Path(__file__).resolve().parents[1] / "configs" / "weak_label_map.yml"))
    ap.add_argument("--reactor-types", default="PWR,BWR,MSR,SFR,HTGR,SMR")
    ap.add_argument("--disciplines", default=",".join(DEFAULT_DISCIPLINES))
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--out",        default=str(Path(__file__).resolve().parents[1] / "data" / "instructions.jsonl"))
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_idx = load_chunks(chunks_dir)
    if not chunks_idx:
        raise SystemExit(f"[FATAL] No chunk files under {chunks_dir}")

    map_cfg = yaml.safe_load(Path(args.map).read_text(encoding="utf-8"))

    disciplines = [d.strip() for d in args.disciplines.split(",") if d.strip()]
    reactors    = [r.strip() for r in args.reactor_types.split(",") if r.strip()]
    seeds = [(d, r) for d in disciplines for r in reactors]
    random.seed(7)
    random.shuffle(seeds)
    if args.num > 0:
        seeds = seeds[:args.num]

    with out_path.open("w", encoding="utf-8") as w:
        for (d, r) in seeds:
            pair = synth_pair(d, r, map_cfg, chunks_idx)
            w.write(json.dumps(pair, ensure_ascii=False) + "\n")
            print(f"wrote: {pair['prompt']}")

    print(f"\n[OK] Wrote {len(seeds)} instruction pairs to {out_path}")

if __name__ == "__main__":
    main()
