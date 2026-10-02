#!/usr/bin/env bash
# One conditional gate-driven adoption. Do not run until the gate summary exists.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TMPDIR=/work/tmp
python3 - <<'PY'
from pathlib import Path
import json
r=Path('candidates/subset/lab');d=next(r.glob('iter6-fullcap-pack8-promoted-gate-*'));s=json.loads((d/'summary.json').read_text())
assert s['all_verified'] and len(s['runs']['A'])==len(s['runs']['B'])==3
assert s['delta_percent']>=4,s
# Install only the source/carrier already prepared and audited, no running arms edited.
w=Path((r/'iter12-pack8-ready-dir.txt').read_text().strip())/'candidates/subset'
for name in ['subset.cu','qsb_carrier_sm89.h']:
 Path('candidates/subset',name).write_bytes((w/name).read_bytes())
# Wrapper0 uses the actual installed production selection. Fresh host build.
p=r/'n24L_iter12final.cu';p.write_text('#define QSB_LOCAL_SM86 1\n#include "../subset.cu"\n')
PY
src=candidates/subset/lab/n24L_iter12final.cu
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter12-pack8-final-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
OUT=candidates/subset/lab/iter12-pack8-final-preflight
mkdir -p "$OUT"
flock /tmp/angel-gpu.lock bash -c '
 set -euo pipefail
 QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/subset/lab/n24L_iter12final.cu" QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=30 QSB_MAX_REL_VAR=none QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset 2>&1 | tee "$1/full.log"
 cp score-subset.json "$1/score.json"
' _ "$OUT"
python3 - <<'PY'
from pathlib import Path
import json
p=Path('candidates/subset/lab/iter12-pack8-final-preflight');x=json.loads((p/'score.json').read_text());assert x['metrics']['verified'] and 'RESULT: PASS' in (p/'full.log').read_text();print('FINAL INSTALLED INTEGRATION PASS',x['score'],x['metrics']['verified_hits'])
PY
