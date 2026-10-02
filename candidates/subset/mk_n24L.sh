#!/usr/bin/env bash
# mk_n24L.sh <name> <knob>=<val>... — build a TRUE N=24 + sm_86 (GLV12) variant.
#
# Why this exists (2026-10-01): tree.cu sets QSB_LOCAL_SM86(67), QSB_PRE3_ROOT(701),
# QSB_TAIL_STAGGER(770), QSB_PARK128(873) with UNGUARDED #defines, so neither
# `#define` before `#include` nor `-D` on the nvcc line can change them — wrapper
# "variants" silently compile with tree.cu's defaults (= C2). The only honest way
# is to patch tree.cu itself, build, restore. Binary lands in candidates/subset/
# n24L_<name>; a hand-written .n24L_<name>.build stamp makes gpu_wrap.py reuse it
# (gpu_wrap recompiles PLAIN — which would OOM with GLV11 21.1 GiB on this 3090).
# Wrapper .cu is auto-created as a bare #include "subset.cu" if absent: knobs are
# baked into tree.cu by the sed patches, so the wrapper carries no configuration.
set -euo pipefail
cd "$(dirname "$0")"
name=$1; shift
src=tests/gpu_epochs/tree.cu
cp "$src" /work/tmp/tree.backup.cu
trap 'cp /work/tmp/tree.backup.cu "$src"' EXIT
for kv in "$@"; do
  k=${kv%%=*}; v=${kv#*=}
  grep -q "^#define $k " "$src" || { echo "no knob $k in $src"; exit 2; }
  sed -i "s|^#define $k .*|#define $k $v|" "$src"
  grep -n "^#define $k " "$src" | head -1
done
[[ -f "n24L_$name.cu" ]] || printf '#include "subset.cu"\n' > "n24L_$name.cu"
nvcc -O3 -DQSB_ZEROS_N=24 -o "n24L_$name" "n24L_$name.cu" -lcrypto -lm 2>&1 \
  | grep -viE "deprecated|declared here|^ *\^|~|EC_POINT|/usr/include|^ *[0-9]+ \|" || true
[[ -x "n24L_$name" ]] || { echo "BUILD FAILED for n24L_$name"; exit 5; }
echo "QSB_ZEROS_N=24 $(stat -c %.9Y n24L_$name.cu)" > ".n24L_$name.build"
# gpu_wrap compares strings against python's src.stat().st_mtime_ns (integer ns):
python3 -c "import os;open('.n24L_$name.build','w').write(f'QSB_ZEROS_N=24 {os.stat(\"n24L_$name.cu\").st_mtime_ns}')"
grep -q '^QSB_ZEROS_N=24 [0-9]\+$' ".n24L_$name.build" || { echo "stamp format bad"; exit 4; }
echo "built n24L_$name  knobs: $*  stamp: $(cat .n24L_$name.build)"
