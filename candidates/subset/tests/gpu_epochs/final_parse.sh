#!/usr/bin/env bash
# One-shot parse of chain 1027 (+1029 later): ABBA verdict + N=24 scored-path numbers.
# Guards: refuses to parse stale AB files (freshness rule from NIGHT.md).
set -uo pipefail
cd /work/comps/yukon-glm-night/qsb-subset
echo "== freshness guard =="
w=$(stat -c %Y /work/ab/results/warmup.log 2>/dev/null || echo 0)
r1=$(stat -c %Y /work/ab/results/r1_A.log 2>/dev/null || echo 0)
now=$(date +%s)
echo "warmup $((now-w))s old, r1_A $((now-r1))s old"
if (( r1 <= w )); then echo "STALE: r1_A not newer than warmup — REFUSING to parse"; exit 1; fi
echo
echo "== ABBA C2 (A) vs D2 (B), GPU-only M/s =="
python3 - <<'PY'
import re, statistics as st
def m(f):
    try: t=open(f"/work/ab/results/{f}.log").read()
    except OSError: return None
    g=re.search(r"draining every launched batch: (\d+)M in \d+s \(([\d.]+)M/s\)", t)
    return float(g.group(2)) if g else None
A=[m(f"r{i}_A") for i in (1,2,3,4)]; B=[m(f"r{i}_B") for i in (1,2,3,4)]
if any(x is None for x in A+B):
    print("incomplete:", A, B); raise SystemExit
print("A:", A, "mean %.2f sd %.2f" % (st.mean(A), st.pstdev(A)))
print("B:", B, "mean %.2f sd %.2f" % (st.mean(B), st.pstdev(B)))
print("delta %+.2f%%  pairwise %s" % ((st.mean(B)/st.mean(A)-1)*100,
      [f"{b-a:+.1f}" for a,b in zip(A,B)]))
PY
echo
echo "== N=24 scored-path (score-subset.json, newest run only) =="
for f in score-subset.json benchmark-results/score-subset.json; do
  [ -f "$f" ] && python3 -c "
import json;d=json.load(open('$f'))
s=d.get('score',{});print('$f:', 'prim %.2f M/s' % s.get('primary_throughput_Mps',0),
      'hits', d.get('verified_hits'), 'valid', s.get('valid'), 'elapsed %.1fs' % d.get('elapsed_s',0))"
done
