#!/usr/bin/env bash
# Compare an isolated, prebuilt wrapper against a production wrapper using the
# unchanged benchmark.sh + verifier. No GPU calls outside the serial lock.
# Usage: frozen_ab.sh SOURCE_PATH NAME [pairs=1] [control=block128_fill2] [seconds=120]
set -euo pipefail
cd "$(dirname "$0")/../../.."
src=$1; name=$2; pairs=${3:-1}; base=${4:-${BASE:-block128_fill2}}; secs=${5:-120}
S=1789110211
out="candidates/subset/lab/ab-${base}-${name}-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$out"
python3 - "$out" "$src" "$base" "$pairs" "$secs" "$S" <<'PY'
import hashlib,json,os,pathlib,sys
out,source,base,pairs,secs,seed=sys.argv[1:]
arms={'A':pathlib.Path(f'candidates/subset/n24L_{base}.cu'),'B':pathlib.Path(source)}
manifest={'control':base,'pairs':int(pairs),'seconds':int(secs),'seed':int(seed),'arms':{}}
for arm,path in arms.items():
    binary=path.with_suffix('');stamp=path.parent/f'.{path.stem}.build'
    assert path.is_file() and binary.is_file() and stamp.is_file(),f'missing prebuilt arm: {path}'
    assert os.access(binary,os.X_OK),f'prebuilt binary is not executable: {binary}'
    assert stamp.read_text()==f'QSB_ZEROS_N=24 {path.stat().st_mtime_ns}',str(stamp)
    manifest['arms'][arm]={'source':str(path),'source_text':path.read_text(),
        'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'binary_mode':oct(binary.stat().st_mode & 0o777),
        'build_stamp':stamp.read_text()}
pathlib.Path(out,'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
PY
for i in $(seq 1 "$pairs"); do
  arms=(A B); if (( i % 2 == 0 )); then arms=(B A); fi
  for arm in "${arms[@]}"; do
    if [[ $arm == A ]]; then source="candidates/subset/n24L_${base}.cu"; else source=$src; fi
    flock /tmp/angel-gpu.lock bash -c '
      set -euo pipefail
      QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src $1" \
      QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS="$2" QSB_MAX_REL_VAR=none \
      QSB_PROBLEM_SEED="$3" ./benchmark.sh subset 2>&1 | tail -16 | tee "$4.log"
      cp score-subset.json "$4.json"
    ' _ "$source" "$secs" "$S" "$out/p${i}-${arm}"
  done
done
python3 - "$out" "$pairs" <<'PY'
import json,pathlib,statistics,sys
out=pathlib.Path(sys.argv[1]); pairs=int(sys.argv[2]); rates={}
for arm in ('A','B'):
    rates[arm]=[]
    for i in range(1,pairs+1):
        x=json.loads((out/f'p{i}-{arm}.json').read_text())
        assert x['metrics']['verified'],x
        rates[arm].append(x['score']/1e6)
    print(arm,rates[arm],'mean',statistics.mean(rates[arm]))
delta=100*(statistics.mean(rates['B'])/statistics.mean(rates['A'])-1)
print(f'VERIFIED {pairs} paired runs delta {delta:+.3f}%')
(out/'summary.json').write_text(json.dumps({'rates_Mps':rates,'delta_percent':delta},indent=2))
PY
