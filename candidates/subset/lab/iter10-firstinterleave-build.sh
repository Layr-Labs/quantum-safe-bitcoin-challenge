#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TMPDIR=/work/tmp
W=$(mktemp -d /work/tmp/qsb-it10-firstinterleave.XXXXXX)
echo "$W" > candidates/subset/lab/iter10-firstinterleave-dir.txt
git archive 3a23ebf candidates/subset | tar -x -C "$W"
python3 - "$W" <<'PY'
from pathlib import Path
import sys
w=Path(sys.argv[1]); d=w/'candidates/subset'
Path('candidates/subset/lab/n24L_iter10control.cu').write_text('#define QSB_LOCAL_SM86 1\n#include "'+str(d/'subset.cu')+'"\n')
Path('candidates/subset/lab/n24L_iter10interleave.cu').write_text('#define QSB_LOCAL_SM86 1\n#define QSB_FIRST_SLOT_INTERLEAVE 1\n#include "../subset.cu"\n')
PY
for name in control interleave; do
 src=candidates/subset/lab/n24L_iter10$name.cu
 /work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > "candidates/subset/lab/iter10-$name-build.log" 2>&1
 python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
done
bash candidates/subset/lab/iter6_ab.sh candidates/subset/lab/n24L_iter10control.cu candidates/subset/lab/n24L_iter10interleave.cu firstinterleave-screen 1 120
