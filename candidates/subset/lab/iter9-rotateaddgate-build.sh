#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=$(mktemp -d /work/tmp/qsb-it9-rotateaddgate.XXXXXX)
echo "$W" > candidates/subset/lab/iter9-rotateaddgate-dir.txt
mkdir -p "$W/current"; cp -a candidates/subset/. "$W/current/"
python3 - "$W" <<'PY'
from pathlib import Path
import sys,json,hashlib
w=Path(sys.argv[1]);d=w/'current';m={}
for p in sorted(d.rglob('*')):
 if p.is_file() and p.suffix in ('.cu','.cuh','.h') and 'lab' not in p.relative_to(d).parts:m[str(p.relative_to(d))]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('candidates/subset/lab/iter9-rotateaddgate-frozen-dependencies.json').write_text(json.dumps(m,indent=2))
Path('candidates/subset/lab/n24L_rotateaddgate.cu').write_text(f'#define QSB_LOCAL_SM86 1\n#define QSB_SHA_LEA 1\n#define QSB_SHA_LEA_GATE 1\n#define QSB_LEA_PARTS 0\n#define QSB_S3_FINAL_PREFETCH 0\n#include "{d}/subset.cu"\n')
PY
src=candidates/subset/lab/n24L_rotateaddgate.cu
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter9-rotateaddgate-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}')
PY
for mode in 0 1; do
 /work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -DQSB_SHA_LEA=1 -DQSB_SHA_LEA_GATE=$mode -DQSB_LEA_PARTS=0 -arch=sm_89 -cubin -Xptxas=-v -o "$W/mode$mode.cubin" "$W/current/subset.cu" > candidates/subset/lab/iter9-rotateaddgate-native$mode-build.log 2>&1
 /work/frosty40/cuda128/root/bin/cuobjdump -res-usage "$W/mode$mode.cubin" > candidates/subset/lab/iter9-rotateaddgate-native$mode-resource.txt
 /work/frosty40/cuda128/root/bin/cuobjdump -sass "$W/mode$mode.cubin" > "$W/mode$mode.sass"
done
sha256sum "$W"/mode*.cubin
bash candidates/subset/lab/iter6_ab.sh candidates/subset/lab/n24L_rotateadd.cu candidates/subset/lab/n24L_rotateaddgate.cu rotateaddgate-screen 1 120
