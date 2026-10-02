#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TMPDIR=/work/tmp
W=$(mktemp -d /work/tmp/qsb-it11-budget.XXXXXX)
echo "$W" > candidates/subset/lab/iter11-budget-dir.txt
git archive b5fe417 candidates/subset | tar -x -C "$W"
python3 - "$W" <<'PY'
from pathlib import Path
import sys
w=Path(sys.argv[1]);d=w/'candidates/subset'
Path('candidates/subset/lab/n24L_iter11control.cu').write_text('#define QSB_LOCAL_SM86 1\n#include "'+str(d/'subset.cu')+'"\n')
Path('candidates/subset/lab/n24L_iter11budget.cu').write_text('#define QSB_LOCAL_SM86 1\n#define QSB_LAUNCH_BUDGET 1\n#define QSB_LAUNCH_TEST_MAX_EPOCHS 1048576\n#include "../subset.cu"\n')
PY
for name in control budget; do
 src=candidates/subset/lab/n24L_iter11$name.cu
 /work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > "candidates/subset/lab/iter11-$name-build.log" 2>&1
 python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
done
bash candidates/subset/lab/iter6_ab.sh candidates/subset/lab/n24L_iter11control.cu candidates/subset/lab/n24L_iter11budget.cu launchbudget-screen 1 120
