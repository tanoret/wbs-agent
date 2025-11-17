# wbs_core/manifest.py
# -*- coding: utf-8 -*-
"""
Manifest utilities for the WBS framework.

This module defines a simple, explicit YAML schema for your corpus manifest and
provides helpers + a CLI to:
  • create a starter manifest from templates
  • add / remove / list documents
  • validate a manifest (static checks)
  • export a lightweight metadata.json for quick inspection

Manifest shape (YAML)
---------------------
version: "1.0"
documents:
  - id: cfr-10-part50-full
    title: 10 CFR Part 50 (full)
    type: cfr            # one of: cfr, rg, nureg, srp, iaea, other
    jurisdiction: nrc    # free text, e.g. "nrc", "us", "iaea"
    section: null        # optional, e.g. "50.36" or "App A"
    url: "https://www.ecfr.gov/current/title-10/part-50"
    mirrors: []          # optional list of fallback URLs
    local: null          # optional absolute/relative path to a local copy
    tags: ["licensing", "requirements"]  # optional
    notes: null          # optional string

Rules
-----
• Each document must have a unique 'id'.
• It must have *either* a 'url' (with optional mirrors) or a 'local' file path.
• 'type' is recommended (downstream uses it for parsing heuristics).
• This module does not perform any network access.

CLI quick start
---------------
# Create a starter manifest
python -m wbs_core.manifest init --out base_files/corpus/manifests/corpus_manifest.yaml --template minimal

# Validate
python -m wbs_core.manifest validate --file base_files/corpus/manifests/corpus_manifest.yaml

# Add a document
python -m wbs_core.manifest add --file base_files/corpus/manifests/corpus_manifest.yaml \
  --id rg-1-29-rev6 --type rg --title "RG 1.29 Rev.6 Seismic Design Classification" \
  --url "https://www.nrc.gov/docs/ML2329/ML23299A123.pdf" --jurisdiction nrc --tags seismic,classification

# Remove a document
python -m wbs_core.manifest remove --file base_files/corpus/manifests/corpus_manifest.yaml --id rg-1-29-rev6

# List
python -m wbs_core.manifest list --file base_files/corpus/manifests/corpus_manifest.yaml

# Export metadata.json (handy for UIs)
python -m wbs_core.manifest export-metadata --file base_files/corpus/manifests/corpus_manifest.yaml --out base_files/corpus/manifests/metadata.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml

# -----------------------------
# Datamodel
# -----------------------------

ALLOWED_TYPES = {"cfr", "rg", "nureg", "srp", "iaea", "other"}

@dataclass
class DocumentEntry:
    id: str
    type: str = "other"                 # cfr|rg|nureg|srp|iaea|other
    title: Optional[str] = None
    jurisdiction: Optional[str] = None  # e.g., "nrc", "us", "iaea"
    section: Optional[str] = None       # e.g., "50.36", "App A"
    url: Optional[str] = None
    mirrors: List[str] = field(default_factory=list)
    local: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # normalize null-ish fields to None explicitly
        for k, v in list(d.items()):
            if v == "" or v == []:
                # keep [] for lists; empty string becomes None
                if isinstance(v, str):
                    d[k] = None
        return d


@dataclass
class Manifest:
    version: str
    documents: List[DocumentEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "documents": [d.to_dict() for d in self.documents],
        }


# -----------------------------
# IO helpers
# -----------------------------

def load_manifest(path: Path) -> Manifest:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "documents" not in data:
        raise ValueError(f"Invalid manifest structure in {path}")
    version = str(data.get("version", "1.0"))
    docs = []
    for raw in data.get("documents", []):
        docs.append(DocumentEntry(
            id=str(raw.get("id", "")).strip(),
            type=str(raw.get("type", "other")).strip() or "other",
            title=raw.get("title"),
            jurisdiction=raw.get("jurisdiction"),
            section=raw.get("section"),
            url=raw.get("url"),
            mirrors=list(raw.get("mirrors", []) or []),
            local=raw.get("local"),
            tags=list(raw.get("tags", []) or []),
            notes=raw.get("notes"),
        ))
    return Manifest(version=version, documents=docs)


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(
        manifest.to_dict(),
        path.open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        indent=2,
    )


# -----------------------------
# Validation
# -----------------------------

def validate_manifest(manifest: Manifest) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    seen_ids = set()

    for i, doc in enumerate(manifest.documents):
        ctx = f"documents[{i}] (id={doc.id or '<missing>'})"

        # id presence + uniqueness
        if not doc.id:
            problems.append(f"{ctx}: missing 'id'")
        elif doc.id in seen_ids:
            problems.append(f"{ctx}: duplicate id '{doc.id}'")
        else:
            seen_ids.add(doc.id)

        # source requirement
        has_net = bool(doc.url) or bool(doc.mirrors)
        has_local = bool(doc.local)
        if not (has_net or has_local):
            problems.append(f"{ctx}: needs at least one of 'url/mirrors' or 'local'")

        # type
        if doc.type not in ALLOWED_TYPES:
            problems.append(f"{ctx}: unsupported type '{doc.type}', allowed={sorted(ALLOWED_TYPES)}")

        # trivial URL sanity
        if doc.url and not (doc.url.startswith("http://") or doc.url.startswith("https://")):
            problems.append(f"{ctx}: url must start with http(s)://")

        for m in doc.mirrors:
            if not (m.startswith("http://") or m.startswith("https://")):
                problems.append(f"{ctx}: mirror '{m}' must start with http(s)://")

    ok = len(problems) == 0
    return ok, problems


# -----------------------------
# Convenience helpers
# -----------------------------

def find_doc(manifest: Manifest, doc_id: str) -> Optional[DocumentEntry]:
    for d in manifest.documents:
        if d.id == doc_id:
            return d
    return None


def add_doc(
    manifest: Manifest,
    *,
    id: str,
    type: str = "other",
    title: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    section: Optional[str] = None,
    url: Optional[str] = None,
    mirrors: Optional[List[str]] = None,
    local: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    overwrite: bool = False,
) -> Manifest:
    existing = find_doc(manifest, id)
    if existing and not overwrite:
        raise ValueError(f"Document with id='{id}' already exists. Use overwrite=True to replace it.")

    entry = DocumentEntry(
        id=id,
        type=type if type in ALLOWED_TYPES else "other",
        title=title,
        jurisdiction=jurisdiction,
        section=section,
        url=url,
        mirrors=mirrors or [],
        local=local,
        tags=tags or [],
        notes=notes,
    )

    if existing:
        # replace
        manifest.documents = [entry if d.id == id else d for d in manifest.documents]
    else:
        manifest.documents.append(entry)
    return manifest


def remove_doc(manifest: Manifest, id: str) -> Manifest:
    before = len(manifest.documents)
    manifest.documents = [d for d in manifest.documents if d.id != id]
    after = len(manifest.documents)
    if before == after:
        raise ValueError(f"No document with id='{id}' found.")
    return manifest


def export_metadata(manifest: Manifest, out_path: Path) -> None:
    """
    Write a compact JSON array, handy for UIs:
      [
        {"id": "...", "title": "...", "type": "cfr", "source": "...", "local": "...", "tags": [...]},
        ...
      ]
    """
    rows = []
    for d in manifest.documents:
        rows.append({
            "id": d.id,
            "title": d.title,
            "type": d.type,
            "jurisdiction": d.jurisdiction,
            "section": d.section,
            "source": d.url or (d.mirrors[0] if d.mirrors else None),
            "mirrors": d.mirrors,
            "local": d.local,
            "tags": d.tags,
            "notes": d.notes,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# Templates (starter sets)
# -----------------------------

def make_minimal_template() -> Manifest:
    """
    A small, practical starter that mirrors the files you already processed.
    Note: URLs are examples; use your own mirrors/local copies if your network blocks eCFR/NRC.
    """
    docs = [
        # CFR Part 50 (full) + key sections + appendices
        DocumentEntry(
            id="cfr-10-part50-full",
            title="10 CFR Part 50 (full)",
            type="cfr",
            jurisdiction="nrc",
            url="https://www.ecfr.gov/current/title-10/part-50",
            tags=["licensing", "requirements"],
        ),
        DocumentEntry(
            id="cfr-10-50-36",
            title="10 CFR 50.36 Technical Specifications",
            type="cfr",
            jurisdiction="nrc",
            section="50.36",
            url="https://www.ecfr.gov/current/title-10/part-50/section-50.36",
            tags=["techspec"],
        ),
        DocumentEntry(
            id="cfr-10-50-47",
            title="10 CFR 50.47 Emergency Plans",
            type="cfr",
            jurisdiction="nrc",
            section="50.47",
            url="https://www.ecfr.gov/current/title-10/part-50/section-50.47",
            tags=["emergency-planning", "ep"],
        ),
        DocumentEntry(
            id="cfr-10-50-55a",
            title="10 CFR 50.55a Codes and Standards",
            type="cfr",
            jurisdiction="nrc",
            section="50.55a",
            url="https://www.ecfr.gov/current/title-10/part-50/section-50.55a",
            tags=["asme", "qualification"],
        ),
        DocumentEntry(
            id="cfr-10-50-app-a",
            title="10 CFR Part 50 Appendix A",
            type="cfr",
            jurisdiction="nrc",
            section="App A",
            url="https://www.ecfr.gov/current/title-10/part-50/appendix-appendix%20a%20to%20part%2050",
            tags=["gdc"],
        ),
        DocumentEntry(
            id="cfr-10-50-app-b",
            title="10 CFR Part 50 Appendix B",
            type="cfr",
            jurisdiction="nrc",
            section="App B",
            url="https://www.ecfr.gov/current/title-10/part-50/appendix-appendix%20b%20to%20part%2050",
            tags=["qa"],
        ),
        DocumentEntry(
            id="cfr-10-50-app-s",
            title="10 CFR Part 50 Appendix S",
            type="cfr",
            jurisdiction="nrc",
            section="App S",
            url="https://www.ecfr.gov/current/title-10/part-50/appendix-appendix%20s%20to%20part%2050",
            tags=["seismic", "site-criteria"],
        ),

        # Other CFR parts used in your corpus
        DocumentEntry(
            id="cfr-10-part20",
            title="10 CFR Part 20 Standards for Protection Against Radiation",
            type="cfr",
            jurisdiction="nrc",
            url="https://www.ecfr.gov/current/title-10/part-20",
            tags=["radiological-protection"],
        ),
        DocumentEntry(
            id="cfr-10-part51",
            title="10 CFR Part 51 Environmental Protection Regulations",
            type="cfr",
            jurisdiction="nrc",
            url="https://www.ecfr.gov/current/title-10/part-51",
            tags=["environmental"],
        ),
        DocumentEntry(
            id="cfr-10-part52",
            title="10 CFR Part 52 Licenses, Certifications, and Approvals",
            type="cfr",
            jurisdiction="nrc",
            url="https://www.ecfr.gov/current/title-10/part-52",
            tags=["licensing"],
        ),
        DocumentEntry(
            id="cfr-10-part100",
            title="10 CFR Part 100 Reactor Site Criteria",
            type="cfr",
            jurisdiction="nrc",
            url="https://www.ecfr.gov/current/title-10/part-100",
            tags=["siting"],
        ),
        DocumentEntry(
            id="cfr-10-73-54",
            title="10 CFR 73.54 Cyber Security",
            type="cfr",
            jurisdiction="nrc",
            section="73.54",
            url="https://www.ecfr.gov/current/title-10/part-73/section-73.54",
            tags=["cyber"],
        ),

        # NRC Regulatory Guide sample (you processed RG 1.199 Rev.1)
        DocumentEntry(
            id="rg-1-199-rev1",
            title="RG 1.199 (Rev.1) Seismic Design Classification (example)",
            type="rg",
            jurisdiction="nrc",
            url="https://www.nrc.gov/reading-rm/doc-collections/reg-guides/power-reactors/rg/01/rg1.199.pdf",
            tags=["seismic"],
        ),

        # NUREG-0800 SRP index (as in your corpus)
        DocumentEntry(
            id="nureg-0800-index",
            title="NUREG-0800 Standard Review Plan (Index)",
            type="srp",
            jurisdiction="nrc",
            url="https://www.nrc.gov/reading-rm/doc-collections/nuregs/staff/sr0800/",
            tags=["srp", "review-plan"],
        ),
    ]
    return Manifest(version="1.0", documents=docs)


TEMPLATES = {
    "minimal": make_minimal_template,
    # You can add "full" later if you want a more exhaustive starter.
}


# -----------------------------
# CLI
# -----------------------------

def _cmd_init(args: argparse.Namespace) -> None:
    if args.template not in TEMPLATES:
        raise SystemExit(f"Unknown template '{args.template}'. Options: {', '.join(TEMPLATES.keys())}")
    manifest = TEMPLATES[argparse.Namespace]( ) if False else TEMPLATES[args.template]()  # keep linters calm
    save_manifest(manifest, Path(args.out))
    print(f"[OK] Wrote starter manifest -> {args.out}")
    ok, probs = validate_manifest(manifest)
    if not ok:
        print("[WARN] Template has validation warnings:")
        for p in probs:
            print("  -", p)

def _cmd_validate(args: argparse.Namespace) -> None:
    m = load_manifest(Path(args.file))
    ok, probs = validate_manifest(m)
    print(json.dumps({"file": args.file, "valid": ok, "problems": probs}, indent=2))
    if not ok and args.fail_on_error:
        raise SystemExit(2)

def _cmd_list(args: argparse.Namespace) -> None:
    m = load_manifest(Path(args.file))
    rows = []
    for d in m.documents:
        rows.append({
            "id": d.id,
            "title": d.title,
            "type": d.type,
            "jurisdiction": d.jurisdiction,
            "section": d.section,
            "source": d.url or (d.mirrors[0] if d.mirrors else None),
            "local": d.local,
            "tags": d.tags,
        })
    print(json.dumps(rows, indent=2, ensure_ascii=False))

def _cmd_add(args: argparse.Namespace) -> None:
    mpath = Path(args.file)
    m = load_manifest(mpath)
    mirrors = [s for s in (args.mirrors or "").split(",") if s.strip()]
    tags = [s for s in (args.tags or "").split(",") if s.strip()]
    m = add_doc(
        m,
        id=args.id,
        type=args.type,
        title=args.title,
        jurisdiction=args.jurisdiction,
        section=args.section,
        url=args.url,
        mirrors=mirrors,
        local=args.local,
        tags=tags,
        notes=args.notes,
        overwrite=args.overwrite,
    )
    save_manifest(m, mpath)
    print(f"[OK] Added/updated document id={args.id} in {args.file}")
    ok, probs = validate_manifest(m)
    if not ok:
        print("[WARN] Manifest now has validation issues:")
        for p in probs:
            print("  -", p)

def _cmd_remove(args: argparse.Namespace) -> None:
    mpath = Path(args.file)
    m = load_manifest(mpath)
    m = remove_doc(m, args.id)
    save_manifest(m, mpath)
    print(f"[OK] Removed document id={args.id} from {args.file}")

def _cmd_export_metadata(args: argparse.Namespace) -> None:
    m = load_manifest(Path(args.file))
    export_metadata(m, Path(args.out))
    print(f"[OK] Wrote metadata -> {args.out}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WBS Corpus Manifest Utilities")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # init
    p = sp.add_parser("init", help="Create a starter manifest from a template")
    p.add_argument("--out", required=True, help="Path to output YAML (corpus_manifest.yaml)")
    p.add_argument("--template", default="minimal", choices=sorted(TEMPLATES.keys()))

    # validate
    p = sp.add_parser("validate", help="Validate an existing manifest")
    p.add_argument("--file", required=True, help="Path to manifest YAML")
    p.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if invalid")

    # list
    p = sp.add_parser("list", help="List documents in JSON")
    p.add_argument("--file", required=True, help="Path to manifest YAML")

    # add
    p = sp.add_parser("add", help="Add or update a document entry")
    p.add_argument("--file", required=True, help="Path to manifest YAML")
    p.add_argument("--id", required=True)
    p.add_argument("--type", default="other", choices=sorted(ALLOWED_TYPES))
    p.add_argument("--title", default=None)
    p.add_argument("--jurisdiction", default=None)
    p.add_argument("--section", default=None)
    p.add_argument("--url", default=None)
    p.add_argument("--mirrors", default=None, help="Comma-separated list of mirror URLs")
    p.add_argument("--local", default=None, help="Path to a local file copy")
    p.add_argument("--tags", default=None, help="Comma-separated tags")
    p.add_argument("--notes", default=None)
    p.add_argument("--overwrite", action="store_true", help="Replace if id already exists")

    # remove
    p = sp.add_parser("remove", help="Remove a document by id")
    p.add_argument("--file", required=True, help="Path to manifest YAML")
    p.add_argument("--id", required=True)

    # export-metadata
    p = sp.add_parser("export-metadata", help="Write a compact metadata.json for UIs")
    p.add_argument("--file", required=True, help="Path to manifest YAML")
    p.add_argument("--out", required=True, help="Path to output metadata.json")

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    if args.cmd == "init":
        _cmd_init(args)
    elif args.cmd == "validate":
        _cmd_validate(args)
    elif args.cmd == "list":
        _cmd_list(args)
    elif args.cmd == "add":
        _cmd_add(args)
    elif args.cmd == "remove":
        _cmd_remove(args)
    elif args.cmd == "export-metadata":
        _cmd_export_metadata(args)
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
