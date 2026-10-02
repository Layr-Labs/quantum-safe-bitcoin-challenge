#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=$(mktemp -d /work/tmp/qsb-it9-leader.XXXXXX)
echo "$W" > candidates/subset/lab/iter9-leader-dir.txt
git archive 2f57d80b8877a9e63b6af220da913236886a7ce5 candidates/subset | tar -x -C "$W"
python3 - "$W" <<'PY'
from pathlib import Path
import sys,json,hashlib
w=Path(sys.argv[1]);d=w/'candidates/subset';m={}
for p in sorted(d.rglob('*')):
 if p.is_file() and p.suffix in ('.cu','.cuh','.h'):m[str(p.relative_to(d))]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('candidates/subset/lab/iter9-leader-source-manifest.json').write_text(json.dumps(m,indent=2))
local = ['#define QSB_LOCAL_SM86 1', '#define QSB_GLV11 0', '#define QSB_Q_P18 0', '#define QSB_Q_MIX 0', '#define QSB_GLV_ZDEC 0', '#define QSB_S3_NM_MASK 0', '#define QSB_S3_NM_SEED 0', '#define QSB_GATHER_ONE_FORM 0', '#define QSB_DECODE_CUT 0']
Path('candidates/subset/lab/n24L_iter9leader.cu').write_text('\n'.join(local) + f'\n#include "{d}/subset.cu"\n')
PY
src=candidates/subset/lab/n24L_iter9leader.cu
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter9-leader-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
sha256sum "${src%.cu}"
