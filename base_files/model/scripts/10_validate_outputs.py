#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

# --------- robust JSON extraction (mirrors 08 script) ----------
_FENCE_JSON = re.compile(r"```json\s*(.*?)\s*```", re.S | re.I)
_FENCE_ANY  = re.compile(r"```\s*(.*?)\s*```", re.S)
_TRAILING_BLOCK = re.compile(r"\{.*\}\s*$", re.S)

def _strip_code_fences(txt: str) -> str:
    m = _FENCE_JSON.search(txt)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(txt)
    if m:
        return m.group(1).strip()
    return txt

def _remove_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*(\})", r"\1", s)
    s = re.sub(r",\s*(\])", r"\1", s)
    return s

def _balanced_brace_extract(txt: str) -> list[str]:
    out = []
    n = len(txt)
    i = 0
    while i < n:
        if txt[i] == "{":
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                c = txt[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            out.append(txt[i:j+1])
                            break
                j += 1
        i += 1
    return out

def parse_json_like(raw: str):
    try:
        return True, json.loads(raw.strip()), ""
    except Exception:
        pass
    inner = _strip_code_fences(raw)
    if inner != raw:
        try:
            return True, json.loads(inner), ""
        except Exception:
            cleaned = _remove_trailing_commas(inner)
            try:
                return True, json.loads(cleaned), ""
            except Exception:
                pass
    m = _TRAILING_BLOCK.search(raw)
    if m:
        candidate = m.group(0)
        try:
            return True, json.loads(candidate), ""
        except Exception:
            cleaned = _remove_trailing_commas(candidate)
            try:
                return True, json.loads(cleaned), ""
            except Exception:
                pass
    candidates = _balanced_brace_extract(raw)
    for c in candidates:
        try:
            return True, json.loads(c), ""
        except Exception:
            pass
    for c in candidates:
        try:
            return True, json.loads(_remove_trailing_commas(c)), ""
        except Exception:
            pass
    return False, raw, "could not find valid JSON in output"


# --------- schema helpers ----------
def normalize_citations(c) -> List[str]:
    out = []
    if isinstance(c, list):
        for item in c:
            if isinstance(item, dict) and "id" in item and isinstance(item["id"], str):
                out.append(item["id"])
            elif isinstance(item, str):
                out.append(item)
    return out

def validate_schema(obj: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    if not isinstance(obj, dict):
        return False, "root is not an object", []
    if "tasks" not in obj or not isinstance(obj["tasks"], list) or len(obj["tasks"]) == 0:
        return False, "missing or empty 'tasks' array", []
    all_cites: List[str] = []
    for idx, t in enumerate(obj["tasks"]):
        if not isinstance(t, dict):
            return False, f"task[{idx}] is not an object", []
        for field in ["name", "description", "guidance"]:
            if field not in t or not isinstance(t[field], str) or not t[field].strip():
                return False, f"task[{idx}] missing/invalid '{field}'", []
        if "applicable_codes" not in t or not isinstance(t["applicable_codes"], list):
            return False, f"task[{idx}] missing/invalid 'applicable_codes' (list expected)", []
        if "citations" not in t:
            return False, f"task[{idx}] missing 'citations'", []
        cites = normalize_citations(t["citations"])
        if not cites:
            return False, f"task[{idx}] has no valid citations", []
        all_cites.extend(cites)
    return True, "ok", all_cites

def validate_against_retrieved(citation_ids: List[str], retrieved_ids: Set[str]) -> Tuple[bool, str]:
    if not retrieved_ids:
        return True, "no retrieved set provided"
    missing = [c for c in citation_ids if c not in retrieved_ids]
    if missing:
        return False, f"{len(missing)} citations not in retrieved: {missing[:5]}{'...' if len(missing)>5 else ''}"
    return True, "ok"


# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to JSON or JSONL with records")
    ap.add_argument("--strict-retrieved", action="store_true",
                    help="Fail a record if its citations are not a subset of 'retrieved' ids in that record")
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"[FATAL] File not found: {p}")

    total = 0
    parsed_output_ok = 0
    schema_ok = 0
    retrieved_ok = 0
    problems: List[str] = []

    def get_output_obj(rec: Dict[str, Any]) -> Tuple[bool, Any, str]:
        # Prefer pre-parsed field if present (from new 08 script)
        if isinstance(rec.get("output_json"), dict):
            return True, rec["output_json"], ""
        if isinstance(rec.get("output_json"), str):
            ok, obj, err = parse_json_like(rec["output_json"])
            return ok, obj, err
        # Fall back to legacy 'output' or 'output_raw'
        for key in ("output", "output_raw"):
            val = rec.get(key)
            if isinstance(val, dict):
                return True, val, ""
            if isinstance(val, str) and val.strip():
                ok, obj, err = parse_json_like(val)
                if ok:
                    return True, obj, ""
        return False, None, "no parseable output field found"

    def handle_record(record: Dict[str, Any], line_no: int):
        nonlocal parsed_output_ok, schema_ok, retrieved_ok

        ok_json, obj, err = get_output_obj(record)
        if not ok_json:
            problems.append(f"line {line_no}: cannot parse output JSON: {err}")
            return
        parsed_output_ok += 1

        ok_schema, msg, citation_ids = validate_schema(obj)
        if not ok_schema:
            problems.append(f"line {line_no}: schema invalid: {msg}")
            return
        schema_ok += 1

        retrieved_ids: Set[str] = set()
        if isinstance(record.get("retrieved"), list):
            for r in record["retrieved"]:
                if isinstance(r, dict) and isinstance(r.get("id"), str):
                    retrieved_ids.add(r["id"])

        ok_ret, rmsg = validate_against_retrieved(citation_ids, retrieved_ids if args.strict_retrieved else set())
        if ok_ret:
            retrieved_ok += 1
        else:
            problems.append(f"line {line_no}: citation mismatch: {rmsg}")

    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                total += 1
                try:
                    rec = json.loads(s)
                except Exception as e:
                    problems.append(f"line {i}: invalid JSONL record: {e}")
                    continue
                handle_record(rec, i)
    else:
        total = 1
        try:
            rec = json.loads(open(p, "r", encoding="utf-8").read())
        except Exception as e:
            raise SystemExit(f"[FATAL] JSON parse failed: {e}")
        handle_record(rec, 1)

    print(json.dumps({
        "file": str(p),
        "total_records": total,
        "parsed_output_ok": parsed_output_ok,
        "schema_ok": schema_ok,
        "retrieved_subset_ok": retrieved_ok if not args.strict_retrieved else f"{retrieved_ok} (strict)",
        "problems": problems[:20]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()