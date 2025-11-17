import json, re, os, yaml
from pathlib import Path
from datetime import date
from pdfminer.high_level import extract_text
import trafilatura
from urllib.parse import urlparse

ROOT = Path(__file__).parents[1]

# Look for either manifest name (aligns with your downloader)
CANDIDATES = [ROOT/"manifests"/"corpus_manifest.yaml", ROOT/"manifests"/"manifest.yaml"]
MANIFEST = next((p for p in CANDIDATES if p.exists()), None)
if MANIFEST is None:
    raise SystemExit(f"[FATAL] No manifest found. Looked for:\n  - {CANDIDATES[0]}\n  - {CANDIDATES[1]}")

RAW, PARSED = ROOT/"raw", ROOT/"parsed"
PARSED.mkdir(parents=True, exist_ok=True)

def clean_text(t: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def parse_html(path: Path) -> str:
    html = path.read_text(errors="ignore")
    text = trafilatura.extract(html) or html
    return text

def parse_pdf(path: Path) -> str:
    return extract_text(path)

def infer_ext_from_url(u: str) -> str:
    p = urlparse(u).path.lower()
    return ".pdf" if p.endswith(".pdf") else ".html"

def normalize_items(m):
    """Return unified items with keys: id, title, type, primary_url, mirrors, filename, fmt ('pdf'|'html'|None)."""
    out = []
    if "sources" in m:  # new schema
        for d in m["sources"]:
            out.append({
                "id": d["id"],
                "title": d.get("title"),
                "type": d.get("type"),
                "primary_url": d.get("primary_url"),
                "mirrors": d.get("mirrors", []),
                "filename": d.get("filename"),
                "fmt": (d.get("format") or "").lower() or None,
            })
    if "documents" in m:  # old schema
        for d in m["documents"]:
            out.append({
                "id": d["id"],
                "title": d.get("title"),
                "type": d.get("type"),
                "primary_url": d.get("url"),
                "mirrors": d.get("mirrors", []),
                "filename": d.get("filename"),
                "fmt": (d.get("format") or "").lower() or None,
            })
    return out

def resolve_filename(entry):
    """Decide which file in raw/ to read."""
    sid = entry["id"]
    # 1) explicit filename
    if entry.get("filename"):
        return RAW / entry["filename"]
    # 2) explicit format
    if entry.get("fmt") in ("pdf", "html"):
        return RAW / f"{sid}.{entry['fmt']}"
    # 3) infer from URL
    if entry.get("primary_url"):
        ext = infer_ext_from_url(entry["primary_url"])
        candidate = RAW / f"{sid}{ext}"
        if candidate.exists():
            return candidate
    # 4) common fallbacks
    for ext in (".pdf", ".html"):
        candidate = RAW / f"{sid}{ext}"
        if candidate.exists():
            return candidate
    # 5) scan raw/ for unique match prefix
    matches = list(RAW.glob(f"{sid}*"))
    return matches[0] if len(matches) == 1 else RAW / f"{sid}.missing"

# --- Load manifest & items
m = yaml.safe_load(open(MANIFEST, "r", encoding="utf-8")) or {}
items = normalize_items(m)

if not items:
    raise SystemExit(f"[FATAL] Manifest has no 'sources' or 'documents'. Keys present: {list(m.keys())}")

missing, parsed = [], 0

for d in items:
    sid = d["id"]
    src_path = resolve_filename(d)

    if not src_path.exists():
        # helpful diagnostics with all filename attempts
        exp = [d.get("filename")] if d.get("filename") else []
        if d.get("fmt") in ("pdf","html"): exp.append(f"{sid}.{d['fmt']}")
        if d.get("primary_url"): exp.append(f"{sid}{infer_ext_from_url(d['primary_url'])}")
        exp += [f"{sid}.pdf", f"{sid}.html", f"{sid}*"]
        missing.append((sid, list(dict.fromkeys([e for e in exp if e]))))
        continue

    fmt = src_path.suffix.lower().lstrip(".")
    text = parse_pdf(src_path) if fmt == "pdf" else parse_html(src_path)

    record = {
        "doc_id": sid,
        "source_url": d.get("primary_url"),
        "source_type": d.get("type"),
        "title": d.get("title"),
        "rev_date": str(date.today()),
        "publisher": "NRC" if d.get("type") in {"cfr","regulatory_guide"} else "Other",
        "license": "public-domain",
        "text": clean_text(text),
    }

    out = PARSED / f"{sid}.json"
    out.write_text(json.dumps(record, ensure_ascii=False))
    print(f"parsed: raw/{src_path.name} -> parsed/{out.name}")
    parsed += 1

if missing:
    print("\n[ERROR] Missing raw files for these manifest entries:")
    for sid, tries in missing:
        print(f"  - id={sid}  looked for: {', '.join(tries)}")
    raise SystemExit(1)

print(f"\nDone. Parsed {parsed} documents from {MANIFEST.name}.")
