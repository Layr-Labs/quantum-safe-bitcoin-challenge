#!/usr/bin/env bash
# stack_ab.sh — scored N=24 promotion decision against a promoted-source control.
# Usage: stack_ab.sh <candidate> [pairs=3] [control=promoted] [seconds=120]
# Preserve every log and score in a unique repo-local directory; never parse a
# previous experiment's shared score file after a failed verifier invocation.
set -euo pipefail
cd "$(dirname "$0")/../.."
v=$1; pairs=${2:-3}; base=${3:-promoted}; secs=${4:-120}; S=1789110211
out="candidates/subset/lab/ab-${base}-${v}-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$out"
echo "scored A/B: $base vs $v, $pairs pairs, seed $S; artifacts=$out"
for i in $(seq 1 $pairs); do
  order=("$base" "$v")
  if ((i % 2 == 0)); then order=("$v" "$base"); fi
  for arm in "${order[@]}"; do
    # Acquire the lock separately for each arm; copying while still holding it
    # prevents a following benchmark from replacing the score under our feet.
    flock /tmp/angel-gpu.lock bash -c '
      set -euo pipefail
      bash candidates/subset/n24L_run.sh "$1" "$2" "$3" | tee "$4.log"
      cp score-subset.json "$4.json"
    ' _ "$arm" "$secs" "$S" "$out/p${i}-${arm}"
  done
done
python3 - "$out" "$base" "$v" "$pairs" <<'PY'
import json, pathlib, statistics, sys
out, base, candidate, pairs = sys.argv[1:]
rates = {}
for arm in (base, candidate):
    values = []
    for i in range(1, int(pairs) + 1):
        record = json.loads(pathlib.Path(out, f'p{i}-{arm}.json').read_text())
        assert record['metrics']['verified'], record
        values.append(record['score'] / 1e6)
    rates[arm] = values
    print(arm, values, 'mean', statistics.mean(values))
delta = 100 * (statistics.mean(rates[candidate]) / statistics.mean(rates[base]) - 1)
print(f'AB-DONE verified delta {delta:+.3f}%; submit gate >= +4.000%')
pathlib.Path(out, 'summary.json').write_text(json.dumps({'rates_Mps': rates, 'delta_percent': delta}, indent=2))
PY
