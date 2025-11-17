import json, re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PARSED, CHUNKS = ROOT/"parsed", ROOT/"chunks"
CHUNKS.mkdir(exist_ok=True)

def split_sections(text):
    # Heuristics: ALL-CAPS headings, CFR § markers, or numbered headings
    return re.split(r"\n(?=[A-Z][A-Z0-9 .()/\-]{5,}\n)|\n(?=§\s*\d)|\n(?=\d+\.\s)", text)

def chunk(section, max_words=1200, min_words=600):
    words, out, cur = section.split(), [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            out.append(" ".join(cur)); cur=[]
    if cur: out.append(" ".join(cur))
    merged=[]
    for c in out:
        if merged and len(c.split()) < min_words:
            merged[-1] += " " + c
        else:
            merged.append(c)
    return merged

total=0
for f in PARSED.glob("*.json"):
    doc = json.loads(f.read_text())
    sections = split_sections(doc["text"])
    with (CHUNKS / f'{doc["doc_id"]}.jsonl').open("w", encoding="utf-8") as out:
        i=0
        for s in sections:
            for piece in chunk(s):
                rec = {
                    "doc_id": doc["doc_id"],
                    "source_url": doc["source_url"],
                    "source_type": doc["source_type"],
                    "title": doc["title"],
                    "rev_date": doc["rev_date"],
                    "text": piece,
                    "chunk_no": i
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                i += 1
                total += 1
print("wrote chunks:", total)
