#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
src=candidates/subset/lab/n24L_centerfused.cu
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o "${src%.cu}" "$src" -lcrypto -lm > candidates/subset/lab/iter7-centerfused-build.log 2>&1
python3 - "$src" <<'PY'
import pathlib,sys
s=pathlib.Path(sys.argv[1]); (s.parent/f'.{s.stem}.build').write_text(f'QSB_ZEROS_N=24 {s.stat().st_mtime_ns}')
PY
W=$(mktemp -d /work/tmp/qsb-it7-native.XXXXXX)
echo "$W" > candidates/subset/lab/iter7-native-dir.txt
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -DQSB_K2S_CENTER_FUSED=1 -arch=sm_89 -cubin -Xptxas=-v -o "$W/fused.cubin" candidates/subset/subset.cu > candidates/subset/lab/iter7-centerfused-native-build.log 2>&1
"${CUOBJDUMP:-/work/frosty40/cuda128/root/bin/cuobjdump}" -res-usage "$W/fused.cubin" > candidates/subset/lab/iter7-centerfused-native-resource.txt
"${CUOBJDUMP:-/work/frosty40/cuda128/root/bin/cuobjdump}" -sass "$W/fused.cubin" > "$W/fused.sass"
sha256sum "$W/fused.cubin"
