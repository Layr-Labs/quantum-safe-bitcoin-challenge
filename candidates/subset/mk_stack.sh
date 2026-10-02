#!/usr/bin/env bash
# mk_stack.sh <name> KNOB=V [KNOB=V ...] — build an n24L_ arm from a knob list.
# Emits the wrapper .cu (QSB_LOCAL_SM86+TAIL_STAGGER always), builds, stamps
# (python-format, see DEAD-ENDS rig-bug row), verifies the knob strings landed.
set -euo pipefail
cd "$(dirname "$0")"
name=$1; shift
out="n24L_$name"
{
  echo "// stack arm generated $(date -u +%FT%TZ): $*"
  echo "#define QSB_LOCAL_SM86 1"
  for kv in "$@"; do echo "#define ${kv%%=*} ${kv#*=}"; done
  echo '#include "subset.cu"'
} > "$out.cu"
nvcc -O3 -DQSB_ZEROS_N=24 -o "$out" "$out.cu" -lcrypto -lm 2>&1 | grep -cE " error:" || true
sync; sleep 1
python3 -c "import os;open('.${out}.build','w').write(f'QSB_ZEROS_N=24 {os.stat(\"${out}.cu\").st_mtime_ns}')"
grep -q '^QSB_ZEROS_N=24 [0-9]\+$' ".${out}.build"
ok=1
for kv in "$@"; do k=${kv%%=*}; v=${kv#*=}; strings -a "$out" | grep -q "$k=$v" || { echo "MISSING $k=$v"; ok=0; }; done
[ $ok = 1 ] && echo "built $out OK: $*"
