# scripts/01_download.py
import os, sys, time, re, yaml, requests
from pathlib import Path
from urllib.parse import urlparse

# --- Locate manifest (supports either name) ---
ROOT = Path(__file__).parents[1]
CANDIDATES = [ROOT/"manifests"/"corpus_manifest.yaml", ROOT/"manifests"/"manifest.yaml"]
MANIFEST = next((p for p in CANDIDATES if p.exists()), None)
if MANIFEST is None:
    print("[FATAL] No manifest found. Looked for:", *map(str, CANDIDATES), sep="\n  ")
    sys.exit(1)

RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

S = requests.Session()
S.headers.update({
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
    "Accept-Language":"en-US,en;q=0.9",
    "Accept":"*/*",
})
# Respect corporate CA bundle / proxies if set (Requests already honors env by default)
VERIFY = os.environ.get("REQUESTS_CA_BUNDLE", True)

def blocked(html: bytes) -> bool:
    try:
        text = html.decode("utf-8", errors="ignore")
    except Exception:
        return False
    # FederalRegister / eCFR request-access pages
    return ("Federal Register :: Request Access" in text) or ("<h1>Request Access</h1>" in text)

def infer_ext_from_url(u: str) -> str:
    path = urlparse(u).path.lower()
    if path.endswith(".pdf"): return ".pdf"
    return ".html"

def save_bytes(filename: str, content: bytes):
    out = RAW / filename
    out.write_bytes(content)
    return out

def normalize_items(manifest_obj):
    """
    Supports both schemas:
      - {'documents': [{id, title, type, url, mirrors?}]}
      - {'sources':   [{id, title, type, format, primary_url, mirrors?, filename?}]}
    Returns a list of unified dicts with keys:
      id, title, type, primary_url, mirrors (list), filename (optional), format ('pdf'|'html' or None)
    """
    items = []
    if "documents" in manifest_obj:
        for d in manifest_obj["documents"]:
            items.append({
                "id": d["id"],
                "title": d.get("title"),
                "type": d.get("type"),
                "primary_url": d.get("url"),
                "mirrors": d.get("mirrors", []),
                "filename": d.get("filename"),
                "format": d.get("format"),  # may be None
            })
    elif "sources" in manifest_obj:
        for d in manifest_obj["sources"]:
            items.append({
                "id": d["id"],
                "title": d.get("title"),
                "type": d.get("type"),
                "primary_url": d.get("primary_url"),
                "mirrors": d.get("mirrors", []),
                "filename": d.get("filename"),
                "format": d.get("format"),
            })
    else:
        print("[FATAL] Manifest has neither 'documents' nor 'sources' at top level.")
        sys.exit(1)
    return items

# --- Load manifest ---
m = yaml.safe_load(open(MANIFEST, "r", encoding="utf-8"))
items = normalize_items(m)
if not items:
    print("[FATAL] Manifest contained zero sources/documents.")
    print("Keys present:", list(m.keys()))
    sys.exit(1)

print(f"[INFO] Using manifest: {MANIFEST}")
print(f"[INFO] Found {len(items)} entries\n")

# Optional filter for quick testing: export ONLY_IDS="id1,id2"
ONLY_IDS = os.environ.get("ONLY_IDS")
if ONLY_IDS:
    keep = set(i.strip() for i in ONLY_IDS.split(",") if i.strip())
    items = [d for d in items if d["id"] in keep]
    print(f"[INFO] Filtering to ONLY_IDS={sorted(keep)} -> {len(items)} entries\n")

# --- Download loop ---
success = 0
for d in items:
    sid = d["id"]
    primary = d["primary_url"]
    mirrors = d.get("mirrors", []) or []
    if not primary:
        print(f"[WARN] {sid}: missing primary_url/url in manifest; skipping.")
        continue

    # Determine filename
    if d.get("filename"):
        filename = d["filename"]
    else:
        # If 'format' provided, use it; else infer from URL
        ext = "." + d["format"].lower() if d.get("format") in ("pdf","html") else infer_ext_from_url(primary)
        filename = f"{sid}{ext}"

    out_path = RAW / filename
    if out_path.exists():
        print(f"skip (exists): {sid} -> {filename}")
        success += 1
        continue

    urls = [primary] + mirrors
    tried = []
    ok = False

    for u in urls:
        if not u: 
            continue
        tried.append(u)
        try:
            r = S.get(u, timeout=90, allow_redirects=True, verify=VERIFY)
            r.raise_for_status()
            ct = r.headers.get("Content-Type","").lower()
            # If expecting HTML and got a known block page, try next
            if (filename.endswith(".html") or "text/html" in ct) and blocked(r.content):
                print(f"[BLOCKED] {sid} @ {u} -> trying next mirror…", file=sys.stderr)
                time.sleep(0.8)
                continue
            save_bytes(filename, r.content)
            print(f"downloaded: {sid} <- {u} -> raw/{filename}")
            ok = True
            success += 1
            break
        except Exception as e:
            print(f"[ERROR] {sid} @ {u} -> {e}", file=sys.stderr)
            time.sleep(0.8)

    if not ok:
        print(f"[FAIL] {sid}: exhausted {len(tried)} URL(s). Tried:", *("  - " + u for u in tried), sep="\n")

print(f"\n[SUMMARY] {success}/{len(items)} files present in raw/")
