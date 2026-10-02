#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=$(mktemp -d /work/tmp/qsb-it9-finaldefault.XXXXXX)
/work/comps/bitcoin/tools/bin/nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -arch=sm_89 -cubin -Xptxas=-v -o "$W/default.cubin" candidates/subset/subset.cu > candidates/subset/lab/iter9-finaldefault-native-build.log 2>&1
python3 - "$W" <<'PY'
from pathlib import Path
import re,json,hashlib,sys
w=Path(sys.argv[1]);actual=hashlib.sha256((w/'default.cubin').read_bytes()).hexdigest();h=Path('candidates/subset/qsb_carrier_sm89.h').read_text();expect=re.search(r'qsb_carrier_cubin_sha256\[\] = "([a-f0-9]+)"',h).group(1);assert actual==expect,(actual,expect);x={'native_sha256':actual,'default_image_byte_identity':True,'probes':{'QSB_S3_FINAL_PREFETCH':0,'QSB_SHA_LEA':0,'QSB_SHA_LEA_GATE':0},'live_tree_sha256':hashlib.sha256(Path('candidates/subset/tests/gpu_epochs/tree.cu').read_bytes()).hexdigest(),'live_GPUHash_sha256':hashlib.sha256(Path('candidates/subset/GPUHash.h').read_bytes()).hexdigest(),'live_gate_sha256':hashlib.sha256(Path('candidates/subset/sha_gate_fma.cuh').read_bytes()).hexdigest()};Path('candidates/subset/lab/iter9-finaldefault-native-identity.json').write_text(json.dumps(x,indent=2));print(x)
PY
