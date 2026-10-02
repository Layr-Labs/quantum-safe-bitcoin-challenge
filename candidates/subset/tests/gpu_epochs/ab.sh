#!/usr/bin/env bash
# Interleaved A/B on the local RTX 3090 (dev-only QSB_LOCAL_SM86=1 build; not a
# ranked artifact).  Usage: ab.sh <binA> <binB> <rounds> <secs>
#
# Protocol (learned from the A/A calibration of 2026-10-01): this desktop GPU
# and socket both heat up over back-to-back runs, so plain A,B,A,B gives the
# first side of each pair a systematically cooler machine (A/A measured a
# monotonic 358->331 M/s total drift over six 141 s runs).  Therefore:
#   - ABBA: odd rounds run A then B, even rounds run B then A, so the drift
#     cancels in the mean and in the per-position comparison.
#   - COOLDOWN s of idle before every run (temperature, not just GPU idle).
#   - WARMUP: one untimed throw-away run first, so run 1 does not start cold.
# Every line prints (side, position) so a position effect stays visible.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../../.." && pwd)"   # repo root (tests/gpu_epochs is 3 deep)
cd "$root" || exit 1
binA=$1; binB=$2; rounds=$3; secs=$4
COOLDOWN=${COOLDOWN:-30}
WARMUP=${WARMUP:-1}
prob="$root/benchmark-results/problem"
mkdir -p /work/ab/results
md5a=$(md5sum "$binA" | cut -c1-8); md5b=$(md5sum "$binB" | cut -c1-8)
echo "A=$binA ($md5a)  B=$binB ($md5b)  rounds=$rounds secs=$secs cooldown=$COOLDOWN"
run_one() {  # $1 side  $2 bin  $3 tag(for log name)
  local seed=$(( (RANDOM << 15) ^ RANDOM ^ $(date +%s%N 2>/dev/null || echo 0) ))
  QSB_PROBLEM_DIR="$prob" python3 harness/gen_problem.py --seed "$seed" >/dev/null 2>&1
  local seqv=$(( seed ^ 0x5bf03635 )) ltv=$(( (seed >> 3) ^ 0x1e3779b9 ))
  local log=/work/ab/results/$3.log
  ( flock -x /tmp/angel-gpu.lock timeout $((secs + 20)) "$bin" \
      "$prob/subset.bin" 0 "$seqv" "$ltv" 1 0 single_hash ) >"$log" 2>&1
  local tot cpu
  tot=$(grep -o 'Stopped by signal 15[^;]*' "$log" | tail -1)
  cpu=$(grep 'CPU co-grind' "$log" | tail -1)
  printf '%s: %s | %s\n' "$3" "$tot" "$cpu"
}
bin=$binA
if [[ $WARMUP == 1 ]]; then
  echo "--- warmup (untimed) ---"
  run_one A "$binA" warmup >/dev/null 2>&1 || true
fi
for r in $(seq 1 "$rounds"); do
  if [[ $((r % 2)) == 1 ]]; then order=("A:$binA" "B:$binB"); else order=("B:$binB" "A:$binA"); fi
  for spec in "${order[@]}"; do
    side=${spec%%:*}; bin=${spec#*:}
    sleep "$COOLDOWN" >/dev/null 2>&1 || true
    run_one "$side" "$bin" "r${r}_${side}"
  done
done
echo "--- summary ---"
python3 - "$rounds" <<'PY'
import re, sys, glob, statistics as st
rounds=int(sys.argv[1])
def parse(tag):
    try: t=open(f"/work/ab/results/{tag}.log").read()
    except OSError: return None
    m=re.findall(r"Stopped by signal 15 after draining every launched batch: \d+M in \d+s \(([\d.]+)M/s\)", t)
    c=re.findall(r"CPU co-grind: \d+M candidates, \d+ hits \(([\d.]+)M/s\)", t)
    if not m or not c: return None
    return float(m[-1]), float(c[-1])
def series(side):
    out=[]
    for r in range(1,rounds+1):
        v=parse(f"r{r}_{side}")
        if v: out.append((r,v))
    return out
for side in ("A","B"):
    s=series(side)
    gpu=[v[0] for _,v in s]; cpu=[v[1] for _,v in s]; tot=[g+c for g,c in zip(gpu,cpu)]
    if not tot: print(f"{side}: no data"); continue
    print(f"{side}: n={len(tot)} gpu mean={st.mean(gpu):.2f} sd={st.stdev(gpu) if len(gpu)>1 else 0:.2f} | "
          f"cpu mean={st.mean(cpu):.2f} | total mean={st.mean(tot):.2f}"
          f"{f' sd={st.stdev(tot):.2f}' if len(tot)>1 else ''} runs={[round(x,1) for x in tot]}")
a=series("A"); b=series("B")
if a and b:
    ta=[g+c for _,(g,c) in a]; tb=[g+c for _,(g,c) in b]
    d=st.mean(ta)-st.mean(tb)
    print(f"A-B total delta = {d:+.2f} M/s ({100*d/st.mean(tb):+.2f}% of B)")
    # position-matched comparison: same round, first-vs-second position via ABBA
    firsts=[r for r in range(1,rounds+1,2)]; seconds=[r for r in range(2,rounds+1,2)]
    def val(side,r):
        for rr,(g,c) in series(side):
            if rr==r: return g+c
        return None
    posA=[val("A",r) for r in range(1,rounds+1) if val("A",r) is not None]
    print("note: odd rounds ran A first, even rounds ran B first (drift cancels in the mean)")
PY
