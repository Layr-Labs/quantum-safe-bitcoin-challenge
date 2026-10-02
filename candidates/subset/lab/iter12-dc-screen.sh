#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TMPDIR=/work/tmp
src=candidates/subset/lab/n24L_iter12dc.cu
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter12-dc-local-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
bash candidates/subset/lab/iter6_ab.sh candidates/subset/lab/n24L_iter11control.cu "$src" legacy-dc-screen 1 120
