#!/usr/bin/env bash
# Preparation only. No production knob/header overwrite and no automatic submit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=$(mktemp -d /work/tmp/qsb-it9-staged-scoped.XXXXXX)
echo "$W" > candidates/subset/lab/iter9-staged-scoped-dir.txt
F=$(cat candidates/subset/lab/iter9-rotateadd-dir.txt)
mkdir -p "$W/scoped"; cp -a "$F/current/." "$W/scoped/"
python3 - "$W" <<'PY'
from pathlib import Path
import sys
w=Path(sys.argv[1]);p=w/'scoped/subset.cu';s=p.read_text();p.write_text('#define QSB_SHA_LEA 1\n#define QSB_LEA_PARTS 0\n'+s)
PY
(cd "$W/scoped"; NVCC=/work/comps/bitcoin/tools/bin/nvcc CUOBJDUMP=/work/frosty40/cuda128/root/bin/cuobjdump bash build_carrier.sh) > candidates/subset/lab/iter9-staged-scoped-carrier-build.log 2>&1
python3 - "$W" <<'PY'
from pathlib import Path
import sys,re,json,hashlib,base64
w=Path(sys.argv[1]);h=(w/'scoped/qsb_carrier_sm89.h').read_text();sha=re.search(r'qsb_carrier_cubin_sha256\[\] = "([a-f0-9]+)"',h).group(1)
assert sha=='82b9868eb8213657b8f2ade164ccb503afc632ce7018c19cb5ab176b1d8e4c64',sha
x={'native_sha256':sha,'matches_measured_scoped_image':True,'header_sha256':hashlib.sha256(h.encode()).hexdigest(),'production_unchanged':True,'staged_source_sha256':hashlib.sha256((w/'scoped/subset.cu').read_bytes()).hexdigest()};Path('candidates/subset/lab/iter9-staged-scoped-identity.json').write_text(json.dumps(x,indent=2));print(x)
p=Path('candidates/subset/lab/n24L_stagedscoped.cu');p.write_text(f'#define QSB_LOCAL_SM86 1\n#include "{w}/scoped/subset.cu"\n')
PY
src=candidates/subset/lab/n24L_stagedscoped.cu
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter9-staged-scoped-local-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys,hashlib,json
p=pathlib.Path(sys.argv[1]);(p.parent/f'.{p.stem}.build').write_text(f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}');print(hashlib.sha256(p.with_suffix('').read_bytes()).hexdigest())
PY
