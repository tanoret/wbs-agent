"""
wbs_core.downloader
===================

Robust, offline-friendly downloader for your corpus.

This module reads the manifest at ``base_files/corpus/manifests/corpus_manifest.yaml``
and downloads each document into ``base_files/corpus/raw``. It mirrors the behavior
of your working `01_download.py`, but with additional safety and utilities:
  • Retries with backoff, custom UA, optional proxies/CA bundle
  • Detects eCFR/FederalRegister "Request Access" block pages and tries mirrors
  • Saves canonical *id-based* filenames (e.g., ``rg-1-199-rev1.pdf``) plus an
    optional URL-slug symlink for traceability
  • Returns a structured report and writes a JSONL download log

Typical use
-----------
>>> from wbs_core.downloader import download_from_manifest
>>> report = download_from_manifest()  # uses default manifest/location
>>> print(report["ok"], report["fail"])

CLI
---
python -m wbs_core.downloader --help
"""
from __future__ import annotations

import re
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local helpers from wbs_core.__init__
from . import paths as _discover_paths, offline_env, run_script, WbsPaths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_PATTERNS = (
    "<title>Federal Register :: Request Access</title>",
    "<h1>Request Access</h1>",
    "Request Access - eCFR",
    "Due to aggressive automated scraping of FederalRegister.gov and eCFR.gov",
)

PDF_CT = ("application/pdf", "application/x-pdf", "binary/octet-stream")

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


@dataclass
class DownloadItem:
    id: str
    url: str
    mirrors: List[str]


@dataclass
class DownloadResult:
    id: str
    url: str
    saved_as: Optional[str]
    content_type: Optional[str]
    size: int
    ok: bool
    status: int
    note: Optional[str] = None


def _blocked(html: bytes) -> bool:
    try:
        text = html.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return any(p in text for p in BLOCK_PATTERNS)


def _slug_from_url(url: str) -> str:
    # Example: https://www.nrc.gov/docs/ML1234.pdf -> nrc.gov_docs_ML1234.pdf
    from urllib.parse import urlparse

    u = urlparse(url)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", (u.netloc + u.path)).strip("_")
    # If no extension, default to .html to match prior behavior
    if not re.search(r"\.[A-Za-z0-9]{2,5}$", slug):
        slug += ".html"
    return slug


def _ext_from_response(url: str, content_type: str | None, body: bytes) -> str:
    if content_type:
        ct = content_type.lower().split(";")[0].strip()
        if ct in PDF_CT or "pdf" in ct:
            return ".pdf"
    if url.lower().endswith(".pdf"):
        return ".pdf"
    # cheap sniff
    head = body[:64].lower()
    if head.startswith(b"%pdf"):
        return ".pdf"
    return ".html"


def _canonical_name(item_id: str, ext: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", item_id).strip("-").lower()
    return f"{safe}{ext}"


def _prepare_session(verify: Optional[str] = None, proxies: Optional[Dict[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    # Robust retries
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    if verify is not None:
        s.verify = verify
    if proxies:
        s.proxies.update(proxies)
    return s


def _load_manifest(manifest_path: Path) -> List[DownloadItem]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")

    # YAML or JSON; prefer YAML if available
    try:
        import yaml  # type: ignore

        y = yaml.safe_load(text)
    except Exception:
        y = json.loads(text)

    docs = []
    for d in y.get("documents", []):
        item = DownloadItem(
            id=str(d["id"]),
            url=str(d["url"]),
            mirrors=[str(u) for u in d.get("mirrors", [])],
        )
        docs.append(item)
    return docs


def _save_bytes(raw_dir: Path, item: DownloadItem, url: str, body: bytes, content_type: Optional[str]) -> str:
    ext = _ext_from_response(url, content_type, body)
    canon = _canonical_name(item.id, ext)
    out_path = raw_dir / canon
    out_path.write_bytes(body)

    # Optional URL slug symlink for traceability
    slug = _slug_from_url(url)
    slug_path = raw_dir / slug
    try:
        if slug_path.exists() or slug_path.is_symlink():
            slug_path.unlink()
        slug_path.symlink_to(out_path.name)  # relative symlink inside same dir
    except Exception:
        # On filesystems without symlink support, skip silently.
        pass
    return out_path.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_from_manifest(
    p: Optional[WbsPaths] = None,
    manifest_path: Optional[Path] = None,
    skip_existing: bool = True,
    rate_limit_s: Tuple[float, float] = (0.5, 1.2),  # min/max sleep between requests
    proxies: Optional[Dict[str, str]] = None,
    verify: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Download all entries in the manifest into base_files/corpus/raw.

    Parameters
    ----------
    p : WbsPaths
        Optional explicit paths object. If None, paths() auto-detects repo root.
    manifest_path : Path
        Optional manifest path. Defaults to base_files/corpus/manifests/corpus_manifest.yaml
    skip_existing : bool
        If True and a canonical id-based file already exists, skip.
    rate_limit_s : (min,max) tuple
        Random jitter sleep between requests (be polite to servers).
    proxies : dict
        Requests-compatible proxies mapping, e.g. {"http":"http://...", "https":"http://..."}
    verify : str or None
        Path to corporate CA bundle. If None, uses requests default or HTTPS verification env.
    log_file : Path or None
        If provided, appends JSONL lines with results.

    Returns
    -------
    dict with keys: ok (int), fail (int), results (List[DownloadResult as dict])
    """
    p = p or _discover_paths()
    manifest_path = manifest_path or (p.corpus_manifests / "corpus_manifest.yaml")
    raw_dir = p.corpus_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (p.corpus_root / "parsed").mkdir(parents=True, exist_ok=True)  # ensure downstream dirs exist

    docs = _load_manifest(manifest_path)
    session = _prepare_session(verify=verify, proxies=proxies)

    if log_file is None:
        log_file = p.corpus_root / "download_log.jsonl"

    ok = 0
    fail = 0
    results: List[DownloadResult] = []

    for d in docs:
        # Skip if we already have id.* (pdf or html)
        if skip_existing:
            already = list(raw_dir.glob(_canonical_name(d.id, ".pdf"))) + list(
                raw_dir.glob(_canonical_name(d.id, ".html"))
            )
            if already:
                res = DownloadResult(
                    id=d.id,
                    url=d.url,
                    saved_as=already[0].name,
                    content_type=None,
                    size=already[0].stat().st_size,
                    ok=True,
                    status=200,
                    note="skipped_existing",
                )
                results.append(res)
                continue

        tried = [d.url] + list(d.mirrors)
        saved_name = None
        last_note = None
        last_status = 0
        last_ct = None
        for u in tried:
            try:
                # politeness jitter
                time.sleep(random.uniform(*rate_limit_s))
                r = session.get(u, timeout=60, allow_redirects=True, stream=True)
                last_status = int(getattr(r, "status_code", 0))
                last_ct = r.headers.get("Content-Type", "")

                body = r.content if getattr(r, "content", None) is not None else b""

                if last_status != 200 or not body:
                    last_note = f"http_{last_status}"
                    continue
                if "text/html" in last_ct.lower() and _blocked(body):
                    last_note = "blocked_captcha"
                    continue

                saved_name = _save_bytes(raw_dir, d, u, body, last_ct)
                size = len(body)
                res = DownloadResult(
                    id=d.id,
                    url=u,
                    saved_as=saved_name,
                    content_type=last_ct,
                    size=size,
                    ok=True,
                    status=last_status,
                )
                results.append(res)
                ok += 1
                break
            except Exception as e:
                last_note = f"error:{type(e).__name__}"
                continue

        if not saved_name:
            fail += 1
            res = DownloadResult(
                id=d.id,
                url=d.url,
                saved_as=None,
                content_type=last_ct,
                size=0,
                ok=False,
                status=last_status,
                note=last_note or "unknown",
            )
            results.append(res)

        # append to log
        try:
            with open(log_file, "a", encoding="utf-8") as w:
                w.write(json.dumps(asdict(results[-1]), ensure_ascii=False) + "\n")
        except Exception:
            pass

    return {"ok": ok, "fail": fail, "results": [asdict(r) for r in results], "log_file": str(log_file)}


def delegate_to_script(p: Optional[WbsPaths] = None, stream: Optional[callable] = None):
    """
    Run your original base_files/corpus/scripts/01_download.py for 1:1 behavior.
    """
    p = p or _discover_paths()
    ret, out = run_script(
        ["python", str(p.corpus_scripts / "01_download.py")],
        cwd=p.repo_root,
        env=offline_env(p),
        stream=stream,
    )
    return ret, out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download corpus from manifest into base_files/corpus/raw")
    ap.add_argument(
        "--manifest",
        type=str,
        default="",
        help="Path to manifest (default: base_files/corpus/manifests/corpus_manifest.yaml)",
    )
    ap.add_argument("--no-skip", action="store_true", help="Do not skip already-downloaded id-based files")
    ap.add_argument("--proxy", type=str, default="", help="HTTPS proxy (e.g., http://proxy:port)")
    ap.add_argument("--verify", type=str, default="", help="Path to corporate CA bundle .pem")
    ap.add_argument("--log", type=str, default="", help="Path to JSONL log file")
    return ap.parse_args()


def main():
    args = _parse_args()
    p = _discover_paths()

    proxies = None
    if args.proxy:
        proxies = {"http": args.proxy, "https": args.proxy}
    verify = args.verify or None
    manifest = Path(args.manifest) if args.manifest else (p.corpus_manifests / "corpus_manifest.yaml")
    log_file = Path(args.log) if args.log else None

    rep = download_from_manifest(
        p=p,
        manifest_path=manifest,
        skip_existing=(not args.no_skip),
        proxies=proxies,
        verify=verify,
        log_file=log_file,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
