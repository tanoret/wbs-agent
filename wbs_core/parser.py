# wbs_core/parser.py
# -*- coding: utf-8 -*-
"""
Parse & normalize raw source documents (HTML/PDF) into a common JSON schema.

This module is intentionally conservative:
- No network access.
- Heuristics handle eCFR/NRC HTML reasonably, but fall back to a generic mode.
- PDFs are extracted with pypdfium2 (preferred) or PyPDF2 (fallback).

Output JSON schema (one JSON per input doc)
-------------------------------------------
{
  "id": "cfr-10-50-36",
  "title": "10 CFR 50.36 Technical Specifications",
  "type": "cfr",                  # cfr|rg|nureg|srp|iaea|other
  "source_path": "base_files/corpus/raw/cfr_10_50_36.html",
  "metadata": {
    "jurisdiction": "nrc",
    "section": "50.36",
    "url": "https://...",
    "mirrors": []
  },
  "sections": [
    {
      "sec_id": "h1-0",           # synthetic stable id within the doc
      "level": 1,                 # 1..4 from heading structure, 0 for flat text block
      "title": "Main heading",
      "path": ["Main heading"],   # breadcrumb of headings to this section
      "text": "Plain text..."
    },
    ...
  ],
  "stats": {
    "num_sections": 12,
    "fulltext_chars": 15234
  }
}

CLI quick start
---------------
# Parse everything listed in a manifest
python -m wbs_core.parser run \
  --manifest base_files/corpus/manifests/corpus_manifest.yaml \
  --raw-dir base_files/corpus/raw \
  --out-dir base_files/corpus/parsed

# Parse a single file with explicit metadata
python -m wbs_core.parser one \
  --in base_files/corpus/raw/cfr_10_50_36.html \
  --id cfr-10-50-36 \
  --type cfr \
  --title "10 CFR 50.36 Technical Specifications" \
  --out base_files/corpus/parsed/cfr-10-50-36.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import get_logger
from .manifest import load_manifest, find_doc, Manifest, DocumentEntry

log = get_logger(__name__)

# -----------------------------
# Utilities
# -----------------------------

_HEADING_TAGS = ("h1", "h2", "h3", "h4")

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = re.sub(r"[^a-z0-9\-\.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "doc"

def guess_html_parser() -> str:
    # Prefer lxml if available for robustness; else builtin parser
    try:
        import lxml  # noqa: F401
        return "lxml"
    except Exception:
        return "html.parser"

def read_bytes(path: Path) -> bytes:
    return path.read_bytes()

def detect_type_by_suffix(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".html", ".htm"):
        return "html"
    if suf in (".pdf",):
        return "pdf"
    # fallback: try to sniff
    magic = read_bytes(path)[:256]
    if magic.startswith(b"%PDF-"):
        return "pdf"
    if b"<html" in magic.lower() or b"<!doctype html" in magic.lower():
        return "html"
    return "text"

def text_stats(sections: List[Dict[str, Any]]) -> Tuple[int, int]:
    total = sum(len(s.get("text", "")) for s in sections)
    return len(sections), total


# -----------------------------
# HTML parsing
# -----------------------------

def html_extract_sections(raw_html: bytes) -> List[Dict[str, Any]]:
    """
    Generic HTML -> sections.
    - Build sections from heading structure (h1..h4).
    - Any text before the first heading goes into a level=0 section.
    """
    from bs4 import BeautifulSoup

    parser = guess_html_parser()
    soup = BeautifulSoup(raw_html, parser)

    # Remove script/style/nav/aside to reduce noise
    for tag in soup(["script", "style", "nav", "aside", "footer", "noscript"]):
        tag.decompose()

    body = soup.body or soup
    headings = body.find_all(_HEADING_TAGS)
    sections: List[Dict[str, Any]] = []

    def norm_text(node) -> str:
        # Join text with single newlines, squeeze multiple blanks
        txt = node.get_text(separator="\n", strip=True)
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()

    # If there are no headings, return a single flat section
    if not headings:
        txt = norm_text(body)
        if txt:
            sections.append({
                "sec_id": "s0",
                "level": 0,
                "title": None,
                "path": [],
                "text": txt
            })
        return sections

    # Text before first heading
    pre = []
    cur = body.contents[0] if body.contents else None
    first = headings[0]
    # Collect siblings until 'first'
    if first:
        for sib in list(body.children):
            if sib == first:
                break
            pre.append(sib)
        pre_text = norm_text(BeautifulSoup("".join(str(x) for x in pre), parser))
        if pre_text:
            sections.append({"sec_id": "pre", "level": 0, "title": None, "path": [], "text": pre_text})

    # Build a flattened list of heading nodes to segment content
    nodes = list(body.descendants)
    heading_nodes = [n for n in nodes if getattr(n, "name", "") in _HEADING_TAGS]

    def heading_level(tagname: str) -> int:
        try:
            return int(tagname[1:])  # "h2" -> 2
        except Exception:
            return 4

    # Iterate headings, take content until next heading at same or higher level
    for i, h in enumerate(heading_nodes):
        title = norm_text(h)
        lvl = heading_level(h.name)

        # Determine the slice of content for this section
        content_fragments: List[str] = []
        # Go through following siblings until next heading of level <= current level
        # Simpler approach: iterate next_elements
        for el in h.next_elements:
            if el == h:
                continue
            name = getattr(el, "name", None)
            if name in _HEADING_TAGS:
                # encountered next heading -> stop
                break
            # only text-containing tags
            if hasattr(el, "get_text"):
                content_fragments.append(el.get_text(separator="\n", strip=False))

        txt = "\n".join(content_fragments)
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        txt = txt.strip()

        if title or txt:
            sections.append({
                "sec_id": f"{h.name}-{i}",
                "level": lvl,
                "title": title or None,
                "path": _compute_path_from_heading(h),
                "text": txt
            })

    # De-dup empty sections
    sections = [s for s in sections if (s.get("title") or s.get("text"))]
    return sections

def _compute_path_from_heading(hnode) -> List[str]:
    """
    Build a simple breadcrumb from previous headings (h1..h4).
    """
    # Walk back through previous elements and collect headings
    path: List[Tuple[int, str]] = []
    from bs4 import NavigableString

    def heading_level(tagname: str) -> int:
        try:
            return int(tagname[1:])
        except Exception:
            return 4

    cur = hnode
    while cur:
        prev = cur.find_previous(_HEADING_TAGS)
        if prev is None:
            break
        lvl = heading_level(prev.name)
        title = prev.get_text(strip=True)
        if title:
            path.append((lvl, title))
        cur = prev

    path = list(reversed(path))
    out: List[str] = [t for _, t in path]
    # add current heading title as tail (caller already stores title separately)
    return out


# -----------------------------
# PDF parsing
# -----------------------------

def pdf_extract_text_pages(path: Path) -> List[str]:
    """
    Extract text per-page. Prefer pypdfium2 for speed; fallback to PyPDF2.
    """
    # Try pypdfium2
    try:
        import pypdfium2  # type: ignore
        pdf = pypdfium2.PdfDocument(str(path))
        texts = []
        for i in range(len(pdf)):
            page = pdf[i]
            txtpage = page.get_textpage()
            texts.append(txtpage.get_text_range())
            txtpage.close()
        return texts
    except Exception:
        pass

    # Fallback to PyPDF2
    try:
        import PyPDF2  # type: ignore
        texts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
        return texts
    except Exception as e:
        raise RuntimeError(f"Cannot extract text from PDF: {path} ({e})")


def pdf_to_sections(path: Path) -> List[Dict[str, Any]]:
    pages = pdf_extract_text_pages(path)
    sections: List[Dict[str, Any]] = []
    for i, ptxt in enumerate(pages):
        if ptxt:
            txt = ptxt.replace("\x00", "")
            txt = re.sub(r"[ \t]+\n", "\n", txt)
            txt = re.sub(r"\n{3,}", "\n\n", txt)
            sections.append({
                "sec_id": f"page-{i+1}",
                "level": 0,
                "title": f"Page {i+1}",
                "path": [f"Page {i+1}"],
                "text": txt.strip(),
            })
    # If all pages empty, return a single empty section to avoid downstream crashes
    if not sections:
        sections = [{"sec_id": "page-1", "level": 0, "title": "Page 1", "path": ["Page 1"], "text": ""}]
    return sections


# -----------------------------
# Public API
# -----------------------------

@dataclass
class ParseOptions:
    min_section_chars: int = 40    # drop tiny fragments
    drop_empty: bool = True


def parse_file_to_json(
    *,
    input_path: Path,
    doc_id: str,
    doc_type: str = "other",
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    options: Optional[ParseOptions] = None,
) -> Dict[str, Any]:
    """
    Parse a single raw file and return normalized JSON blob (not written).
    """
    options = options or ParseOptions()
    metadata = metadata or {}

    kind = detect_type_by_suffix(input_path)
    log.info(f"[parse] id={doc_id} type={doc_type} kind={kind} src={input_path.name}")

    if kind == "html":
        raw = read_bytes(input_path)
        sections = html_extract_sections(raw)
    elif kind == "pdf":
        sections = pdf_to_sections(input_path)
    else:
        # Plain text fallback
        txt = input_path.read_text(encoding="utf-8", errors="ignore")
        txt = re.sub(r"\r\n", "\n", txt)
        sections = [{
            "sec_id": "s0",
            "level": 0,
            "title": None,
            "path": [],
            "text": txt.strip(),
        }]

    # Filter tiny sections
    if options.drop_empty or options.min_section_chars > 0:
        keep = []
        for s in sections:
            t = (s.get("text") or "").strip()
            if not t and options.drop_empty:
                continue
            if t and len(t) < options.min_section_chars:
                # keep if it's a heading-only section with a title and followed by bigger text?
                # For simplicity: drop tiny text blocks but keep titled sections
                if s.get("title"):
                    keep.append(s)
                continue
            keep.append(s)
        sections = keep

    nsec, nchar = text_stats(sections)
    out = {
        "id": doc_id,
        "title": title,
        "type": doc_type,
        "source_path": str(input_path),
        "metadata": metadata,
        "sections": sections,
        "stats": {
            "num_sections": nsec,
            "fulltext_chars": nchar,
        },
    }
    return out


def write_parsed_json(doc_json: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# CLI commands
# -----------------------------

def _resolve_raw_for_id(raw_dir: Path, doc: DocumentEntry) -> Optional[Path]:
    """
    Heuristics to locate a local raw file for a manifest entry.
    - Prefer 'local' if provided.
    - Else, try by id-based stem: e.g., id 'cfr-10-50-36' looks for
      raw/{id}.html|.htm|.pdf and some tolerant variants.
    - Else, try a forgiving scan for files whose name contains the id slug.
    """
    if doc.local:
        p = Path(doc.local)
        if not p.is_absolute():
            p = raw_dir / doc.local
        if p.exists():
            return p

    # direct {id}.{ext}
    candidates = [
        raw_dir / f"{doc.id}.html",
        raw_dir / f"{doc.id}.htm",
        raw_dir / f"{doc.id}.pdf",
    ]
    for c in candidates:
        if c.exists():
            return c

    # tolerant variants from your earlier naming pattern
    id_variants = {
        doc.id,
        slugify(doc.id),
        doc.id.replace("-", "_"),
        slugify(doc.id).replace("-", "_"),
        doc.id.replace(".", "-"),
    }
    exts = (".html", ".htm", ".pdf")
    for stem in id_variants:
        for ext in exts:
            p = raw_dir / f"{stem}{ext}"
            if p.exists():
                return p

    # final resort: scan raw dir for partial matches
    for p in raw_dir.glob("*"):
        if p.is_file() and any(k in p.name.lower() for k in [doc.id.lower(), slugify(doc.id)]):
            return p

    return None


def _cmd_run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mf: Manifest = load_manifest(manifest_path)
    log.info(f"[parser] Loaded manifest with {len(mf.documents)} documents.")

    parsed_ok = 0
    missing: List[str] = []

    for doc in mf.documents:
        src = _resolve_raw_for_id(raw_dir, doc)
        if not src:
            log.error(f"[MISSING] id={doc.id} no local file found in {raw_dir}")
            missing.append(doc.id)
            continue

        meta = {
            "jurisdiction": doc.jurisdiction,
            "section": doc.section,
            "url": doc.url,
            "mirrors": doc.mirrors,
            "tags": doc.tags,
            "notes": doc.notes,
        }
        try:
            blob = parse_file_to_json(
                input_path=src,
                doc_id=doc.id,
                doc_type=doc.type,
                title=doc.title,
                metadata=meta
            )
            out_name = f"{slugify(doc.id)}.json"
            write_parsed_json(blob, out_dir / out_name)
            log.info(f"[parsed] {src.name} -> {out_name} (sections={blob['stats']['num_sections']}, chars={blob['stats']['fulltext_chars']})")
            parsed_ok += 1
        except Exception as e:
            log.exception(f"[ERROR] Failed to parse id={doc.id} from {src}: {e}")

    if missing:
        log.error("\n[ERROR] Missing raw files for these manifest entries:")
        for mid in missing:
            # show a few patterns we tried
            print(f"  - id={mid}")

    print(f"\n[DONE] Parsed {parsed_ok} / {len(mf.documents)} documents. Output: {out_dir}")


def _cmd_one(args: argparse.Namespace) -> None:
    src = Path(args.inp)
    if not src.exists():
        raise SystemExit(f"Input file not found: {src}")
    meta = {}
    if args.url:
        meta["url"] = args.url
    if args.jurisdiction:
        meta["jurisdiction"] = args.jurisdiction
    if args.section:
        meta["section"] = args.section

    blob = parse_file_to_json(
        input_path=src,
        doc_id=args.id,
        doc_type=args.type,
        title=args.title,
        metadata=meta
    )
    out = Path(args.out)
    write_parsed_json(blob, out)
    print(f"[OK] wrote -> {out} (sections={blob['stats']['num_sections']}, chars={blob['stats']['fulltext_chars']})")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Parse & normalize raw corpus files to JSON")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # run over manifest
    p = sp.add_parser("run", help="Parse all manifest entries from a raw directory")
    p.add_argument("--manifest", required=True, help="Path to corpus_manifest.yaml")
    p.add_argument("--raw-dir", required=True, help="Directory containing raw files")
    p.add_argument("--out-dir", required=True, help="Directory to write parsed JSON files")

    # one-off
    p = sp.add_parser("one", help="Parse a single file")
    p.add_argument("--in", dest="inp", required=True, help="Input file (html/pdf/txt)")
    p.add_argument("--id", required=True, help="Document id")
    p.add_argument("--type", default="other", choices=["cfr", "rg", "nureg", "srp", "iaea", "other"])
    p.add_argument("--title", default=None, help="Human-readable title")
    p.add_argument("--url", default=None)
    p.add_argument("--jurisdiction", default=None)
    p.add_argument("--section", default=None)
    p.add_argument("--out", required=True, help="Output JSON path")

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()
    if args.cmd == "run":
        _cmd_run(args)
    elif args.cmd == "one":
        _cmd_one(args)
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
