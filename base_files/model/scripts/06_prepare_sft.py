#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path

TEMPLATE = """You are an expert nuclear engineering planner. Produce ONLY valid JSON matching the schema, no prose.
Schema keys: wbs_id,title,scope,assumptions,discipline,tasks[id,name,description,inputs,methods,deliverables,acceptance_criteria,citations[doc_id,chunk_no,source_url,title,snippet]],traceability,standards_map.

Ground your output on the provided retrieved context by citing chunk doc_id and chunk_no in each task.
"""

def format_example(prompt, target):
    user = f"{TEMPLATE}\nTask: {prompt}\nOutput JSON only."
    assistant = json.dumps(target, ensure_ascii=False)
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="supervision/data/instructions.jsonl")
    ap.add_argument("--out-dir", default="model/data")
    ap.add_argument("--val-ratio", type=float, default=0.06)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            rows.append(format_example(obj["prompt"], obj["target"]))

    random.seed(13)
    random.shuffle(rows)
    n_val = max(1, int(len(rows)*args.val_ratio))
    val = rows[:n_val]; train = rows[n_val:]

    def dump(path, data):
        with open(path, "w", encoding="utf-8") as w:
            for ex in data:
                w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    dump(out_dir/"train.jsonl", train)
    dump(out_dir/"val.jsonl", val)
    print(f"[OK] train={len(train)} val={len(val)} -> {out_dir}")

if __name__ == "__main__":
    main()
