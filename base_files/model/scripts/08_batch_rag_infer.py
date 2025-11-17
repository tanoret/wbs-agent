#!/usr/bin/env python3
import argparse, os, json, re, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Avoid bitsandbytes import by forcing PEFT to think bnb is unavailable
import peft
try:
    import peft.import_utils as _peft_import_utils
    _peft_import_utils.is_bnb_available = lambda: False
    import peft.tuners.lora.model as _lora_model
    _lora_model.is_bnb_available = lambda: False
except Exception:
    pass
from peft import PeftModel

# Reuse retrieval helpers from rag_infer.py (same folder)
from rag_infer import load_retriever, build_prompt  # type: ignore


# ---------- JSON extraction helpers ----------
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
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*(\})", r"\1", s)
    s = re.sub(r",\s*(\])", r"\1", s)
    return s

def _balanced_brace_extract(txt: str) -> list[str]:
    """Return all substrings that are balanced JSON-style objects."""
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
    """Try hard to parse JSON from a model output string."""
    # 1) direct
    try:
        return True, json.loads(raw.strip()), ""
    except Exception as e1:
        pass

    # 2) code fences
    inner = _strip_code_fences(raw)
    if inner != raw:
        try:
            return True, json.loads(inner), ""
        except Exception:
            # try cleanup
            cleaned = _remove_trailing_commas(inner)
            try:
                return True, json.loads(cleaned), ""
            except Exception as e2:
                pass

    # 3) trailing block
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

    # 4) balanced brace scan; prefer one that has "tasks"
    candidates = _balanced_brace_extract(raw)
    # try all candidates as-is, then cleaned
    def _try_list(lst):
        for c in lst:
            try:
                obj = json.loads(c)
                return True, obj, ""
            except Exception:
                pass
        return False, None, "no balanced candidate parsed"

    ok, obj, _ = _try_list(candidates)
    if not ok:
        ok, obj, _ = _try_list([_remove_trailing_commas(c) for c in candidates])

    if ok and isinstance(obj, dict) and "tasks" in obj:
        return True, obj, ""
    if ok:
        return True, obj, ""
    return False, raw, "could not find valid JSON in output"


# ---------- I/O helpers ----------
def read_queries(path: Path):
    queries = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    q = obj.get("q")
                    if q:
                        queries.append(str(q))
                except Exception:
                    continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    queries.append(s)
    if not queries:
        raise SystemExit(f"[FATAL] No queries found in {path}")
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="LOCAL base model path (merged or original)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path; omit if merged")
    ap.add_argument("--input", required=True, help="Text file (lines) or JSONL with {'q': ...}")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--corpus-root", default="corpus")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--json_only", action="store_true", help="Append a hard instruction to return ONLY JSON (no prose, no fences)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # retriever
    search = load_retriever(Path(args.corpus_root))

    # model
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval().to(device)

    # inputs
    inp = Path(args.input)
    queries = read_queries(inp)

    # outputs
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_p, "w", encoding="utf-8")

    for q in queries:
        retrieved = search(q, args.k)
        prompt = build_prompt(q, retrieved)
        if args.json_only:
            prompt += "\n\nReturn ONLY the JSON object. Do NOT include code fences or additional text."

        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
            )
        text = tok.decode(out[0], skip_special_tokens=True)

        ok, obj, perr = parse_json_like(text)
        rec = {
            "q": q,
            "retrieved": [
                {
                    "id": f"{r['doc_id']}#{r['chunk_no']}",
                    "score": r.get("score"),
                    "title": r.get("title"),
                    "source": r.get("source"),
                }
                for r in retrieved
            ],
            "output_raw": text,
            "output_json": obj if ok else None,
            "parse_error": "" if ok else perr,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
