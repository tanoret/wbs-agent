# wbs_core/stf.py
# -*- coding: utf-8 -*-
"""
SFT (Supervised Fine-Tuning) dataset preparation utilities + CLI.

This module takes the instruction pairs produced by your supervision step
(e.g., `supervision/data/instructions.jsonl`) and converts them into a clean
JSONL dataset for chat-style SFT, split into train/val.

It is model-agnostic (works with Qwen/Llama/Mistral chat templates) by emitting
examples in the common `"messages"` chat format:

{
  "messages": [
    {"role": "system", "content": "... grounding / schema instructions ..."},
    {"role": "user",   "content": "... the design prompt + retrieved context ..."},
    {"role": "assistant", "content": "{...WBS JSON...}"}
  ]
}

Input JSONL (each line is one instruction pair) is expected to contain at least:
- prompt: str                       # user prompt ("Design X for Y")
- target: dict or str               # WBS JSON target (dict is preferred)
Optional fields:
- retrieved: list[dict]             # subset of chunks used (for teaching citations)
- meta: dict                        # any extra metadata (reactor type, discipline, etc.)

You may optionally pass a JSON Schema to validate `target` before creating an
example (`--schema path/to/schema.json`). Invalid samples are skipped.

CLI examples
------------
# Basic: split 80/20 and write to an output directory
python -m wbs_core.stf \
  --in supervision/data/instructions.jsonl \
  --out-dir model/data \
  --val-ratio 0.2

# With schema validation and max samples
python -m wbs_core.stf \
  --in supervision/data/instructions.jsonl \
  --schema supervision/configs/schema.json \
  --out-dir model/data \
  --val-ratio 0.15 \
  --max-samples 500

# Customize the system prompt and the amount of retrieved context included
python -m wbs_core.stf \
  --in supervision/data/instructions.jsonl \
  --out-dir model/data \
  --retrieved-top 3 \
  --system-prompt "You are a nuclear licensing assistant that emits strict JSON."

Programmatic usage
------------------
from pathlib import Path
from wbs_core.stf import SFTPreparer

prep = SFTPreparer(
    system_prompt="You are a helpful assistant...",
    retrieved_top=4,
)
records = prep.load_instructions(Path("supervision/data/instructions.jsonl"))
valid = prep.filter_and_normalize(records, schema_path=Path("supervision/configs/schema.json"))
messages = [prep.make_example(r) for r in valid]
train, val = prep.split(messages, val_ratio=0.2, seed=42)
prep.write_jsonl(Path("model/data/train.jsonl"), train)
prep.write_jsonl(Path("model/data/val.jsonl"), val)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


DEFAULT_SYSTEM_PROMPT = (
    "You are a nuclear engineering planning assistant. Produce ONLY strict JSON matching the schema:\n"
    "{\n"
    '  "wbs": [\n'
    '    {\n'
    '      "id": "string",                // unique task id\n'
    '      "title": "string",             // concise task name\n'
    '      "discipline": "string",        // e.g., thermal-hydraulics, I&C, seismic, QA\n'
    '      "description": "string",       // what must be done\n'
    '      "inputs": ["string"],          // key inputs the agent must collect\n'
    '      "deliverables": ["string"],    // expected outputs\n'
    '      "codes": [                     // citations to regulations/standards/guides\n'
    '        {"id":"string","title":"string","section":"string","url":"string"}\n'
    '      ],\n'
    '      "steps": ["string"]            // step-by-step execution guidance\n'
    '    }\n'
    '  ],\n'
    '  "assumptions": ["string"],\n'
    '  "notes": ["string"]\n'
    "}\n"
    "Your output must be valid JSON. Do not include explanations or Markdown—JSON only."
)


# -----------------------------
# Utilities
# -----------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                if not isinstance(rec, dict):
                    raise ValueError("record is not a JSON object")
                out.append(rec)
            except Exception as e:
                print(f"[stf] WARN: skipping line {i}: {e}", file=sys.stderr)
    if not out:
        raise RuntimeError(f"No usable records in {path}")
    return out


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _truncate(s: str, max_chars: int) -> str:
    s = s.strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


# -----------------------------
# Schema validation (optional)
# -----------------------------

def _validate_against_schema(data: Any, schema: Dict[str, Any]) -> bool:
    """
    Minimal schema check: verify required top-level keys if present in schema.
    Avoids bringing jsonschema dependency. Extend as needed.
    """
    try:
        if not isinstance(data, dict):
            return False
        req = schema.get("required")
        if isinstance(req, list):
            for k in req:
                if k not in data:
                    return False
        # Optionally check types for a subset of keys
        props = schema.get("properties", {})
        for k, spec in props.items():
            if k in data and "type" in spec:
                t = spec["type"]
                if t == "array" and not isinstance(data[k], list):
                    return False
                if t == "object" and not isinstance(data[k], dict):
                    return False
                if t == "string" and not isinstance(data[k], str):
                    return False
        return True
    except Exception:
        return False


# -----------------------------
# Core preparer
# -----------------------------

@dataclass
class SFTPreparer:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    retrieved_top: int = 4                 # include up to N retrieved chunks in the user content
    retrieved_snippet_chars: int = 700     # snippet length per chunk
    seed: int = 42

    def load_instructions(self, path: Path) -> List[Dict[str, Any]]:
        recs = _read_jsonl(path)
        print(f"[stf] Loaded {len(recs)} instruction records from {path}")
        return recs

    def filter_and_normalize(
        self,
        records: List[Dict[str, Any]],
        schema_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Keep only records with the minimum fields and (optionally) valid target per schema.
        Coerce 'target' to dict when given as string JSON.
        """
        schema = _load_json(schema_path) if schema_path else None
        ok: List[Dict[str, Any]] = []
        for i, r in enumerate(records, 1):
            prompt = r.get("prompt")
            target = r.get("target")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            # normalize target
            if isinstance(target, str):
                try:
                    target = json.loads(target)
                except Exception:
                    # if it's a plain string, wrap into {"notes":[...]} to maintain JSON
                    target = {"wbs": [], "assumptions": [], "notes": [target]}
            if not isinstance(target, dict):
                continue
            if schema is not None and not _validate_against_schema(target, schema):
                continue
            # normalize retrieved
            retrieved = r.get("retrieved")
            if not isinstance(retrieved, list):
                retrieved = []
            ok.append({"prompt": prompt.strip(), "target": target, "retrieved": retrieved, "meta": r.get("meta")})
        if not ok:
            raise RuntimeError("No valid records after filtering/normalization.")
        random.Random(self.seed).shuffle(ok)
        if max_samples is not None:
            ok = ok[: max_samples]
        print(f"[stf] Kept {len(ok)} normalized records (schema={'yes' if schema_path else 'no'})")
        return ok

    def _format_retrieved(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Turn retrieved subset into a compact, citeable context block.
        """
        if not retrieved:
            return ""
        rows = retrieved[: self.retrieved_top]
        lines: List[str] = []
        for j, r in enumerate(rows, 1):
            rid = str(r.get("id") or r.get("row_index") or j)
            title = r.get("title") or ""
            url = r.get("url") or ""
            txt = r.get("text") or r.get("chunk") or ""
            snippet = _truncate(str(txt), self.retrieved_snippet_chars)
            head = f"[{rid}] {title}".strip()
            tail = f"\nURL: {url}" if url else ""
            lines.append(f"{head}{tail}\n{snippet}")
        return "CITED CONTEXT (top-N chunks):\n" + "\n\n".join(lines)

    def make_example(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a normalized instruction record to a chat-style SFT sample.
        """
        prompt: str = record["prompt"]
        target: Dict[str, Any] = record["target"]
        retrieved: List[Dict[str, Any]] = record.get("retrieved", [])

        context_block = self._format_retrieved(retrieved)
        if context_block:
            user_content = (
                f"{prompt.strip()}\n\n"
                "Use ONLY the following context to ground your citations and task mapping.\n"
                f"{context_block}\n\n"
                "Return strict JSON only."
            )
        else:
            user_content = f"{prompt.strip()}\n\nReturn strict JSON only."

        assistant_json = json.dumps(target, ensure_ascii=False)

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_json},
            ]
        }

    def make_examples(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = [self.make_example(r) for r in records]
        print(f"[stf] Built {len(out)} chat examples")
        return out

    def split(
        self,
        examples: List[Dict[str, Any]],
        val_ratio: float = 0.2,
        seed: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rnd = random.Random(self.seed if seed is None else seed)
        ex = list(examples)
        rnd.shuffle(ex)
        n_val = max(1, int(round(len(ex) * float(val_ratio)))) if len(ex) > 1 else 1
        val = ex[:n_val]
        train = ex[n_val:]
        if not train:  # ensure at least one train sample if possible
            train, val = val, train
        print(f"[stf] Split -> train={len(train)} val={len(val)} (val_ratio={val_ratio})")
        return train, val

    def write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> int:
        n = _write_jsonl(path, rows)
        print(f"[stf] Wrote {n} records -> {path}")
        return n


# -----------------------------
# CLI
# -----------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare SFT chat dataset from instruction pairs")
    p.add_argument("--in", dest="inp", type=str, required=True, help="Input instructions JSONL")
    p.add_argument("--schema", type=str, default=None, help="Optional JSON schema to validate targets")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory (will write train.jsonl/val.jsonl)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0..1)")
    p.add_argument("--max-samples", type=int, default=None, help="Limit number of samples after filtering")
    p.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Override default system prompt text")
    p.add_argument("--retrieved-top", type=int, default=4, help="How many retrieved chunks to include in user content")
    p.add_argument("--retrieved-snippet-chars", type=int, default=700, help="Max characters per retrieved snippet")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffles/splits")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    inp = Path(args.inp).resolve()
    out_dir = Path(args.out_dir).resolve()
    schema_path = Path(args.schema).resolve() if args.schema else None

    prep = SFTPreparer(
        system_prompt=args.system_prompt,
        retrieved_top=int(args.retrieved_top),
        retrieved_snippet_chars=int(args.retrieved_snippet_chars),
        seed=int(args.seed),
    )

    records = prep.load_instructions(inp)
    records = prep.filter_and_normalize(
        records,
        schema_path=schema_path,
        max_samples=args.max_samples,
    )
    examples = prep.make_examples(records)
    train, val = prep.split(examples, val_ratio=float(args.val_ratio), seed=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    prep.write_jsonl(out_dir / "train.jsonl", train)
    prep.write_jsonl(out_dir / "val.jsonl", val)


if __name__ == "__main__":
    main()
