# supervision/scripts/make_supervision.sh
#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python "$ROOT/supervision/scripts/05_make_supervision.py" \
  --chunks-dir "$ROOT/corpus/chunks" \
  --map "$ROOT/supervision/configs/weak_label_map.yml" \
  --out "$ROOT/supervision/data/instructions.jsonl" \
  --num 40 \
  --reactor-types "PWR,BWR,MSR,SFR,HTGR,SMR" \
  --disciplines "seismic,qa,core_physics,ep,cybersecurity"