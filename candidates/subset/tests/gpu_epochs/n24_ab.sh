#!/usr/bin/env bash
# n24_ab.sh — interleaved A/B on the SCORED path (N=24 fixed_time via benchmark.sh),
# alternating A,B,B,A,... with a cool-down gap, for two prebuilt n24 wrapper binaries.
# Usage: n24_ab.sh <A-name> <B-name> [rounds=4] [seconds=120] [seed=1789110211]
#   <A-name>/<B-name> are wrapper stems, e.g. `stag` for candidates/subset/n24_stag.cu
# Reads: candidates/subset/n24_<name>.cu (+ sidecar .build stamp, so gpu_wrap never compiles).
# Writes: /work/ab/n24_<A>_<B>_r<i>_<X>.json (one score-subset.json copy per arm).
# Parse rule (per harness/score.py): primary = verified_hits * 2^N / 2 / elapsed.
# NOTE the guard in NIGHT.md: only parse r1 files that postdate this job's start.
set -uo pipefail
cd "$(dirname "$0")/../../.."   # repo root
A="$1" B="$2" rounds="${3:-4}" secs="${4:-120}" seed="${5:-1789110211}"
out="/work/ab"; mkdir -p "$out"
jobstart=$(date +%s)
echo "n24 AB: A=n24_$A B=n24_$B rounds=$rounds secs=$secs seed=$seed start=$(date +%H:%M:%S)"
# order-alternating: A B B A A B B A ...
for ((r=1; r<=rounds; r++)); do
  for side in A B; do
    # alternate which arm leads each round pair
    if (( (r % 2) == 0 )); then [ "$side" = A ] && arm=$B || arm=$A; else arm=$([ "$side" = A ] && echo $A || echo $B); fi
    echo "-- round $r arm $arm ($(date +%H:%M:%S))"
    flock -x /tmp/angel-gpu.lock \
      QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/subset/n24_${arm}.cu" \
      QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=$secs QSB_MAX_REL_VAR=none \
      QSB_PROBLEM_SEED=$seed ./benchmark.sh subset >/dev/null 2>&1
    cp score-subset.json "$out/n24_${A}_${B}_r${r}_${side}.json" 2>/dev/null
    sleep 20   # cool-down
  done
done
python3 - "$out" "$A" "$B" "$rounds" <<'PY'
import json, sys, statistics as st
out, A, B, rounds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
def prim(p):
    try: return json.load(open(p))["score"]["primary_throughput_Mps"]
    except Exception: return None
a=[]; b=[]
for r in range(1, rounds+1):
    va=prim(f"{out}/n24_{A}_{B}_r{r}_A.json"); vb=prim(f"{out}/n24_{A}_{B}_r{r}_B.json")
    print(f"r{r}: A={va} B={vb}")
    if va: a.append(va)
    if vb: b.append(vb)
if len(a)>=2 and len(b)>=2:
    print(f"A mean {st.mean(a):.1f} sd {st.pstdev(a):.1f} | B mean {st.mean(b):.1f} sd {st.pstdev(b):.1f} | delta {(st.mean(b)/st.mean(a)-1)*100:+.2f}%")
else:
    print("insufficient rounds parsed")
PY
echo "n24 AB done $(date +%H:%M:%S)"
