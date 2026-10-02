#!/usr/bin/env bash
# Frozen current-qualified control versus one default-off next-best experiment.
set -euo pipefail
cd "$(dirname "$0")/../../.."
A=$1; B=$2; NAME=$3; PAIRS=${4:-1}; SECS=${5:-120}
OUT=candidates/subset/lab/iter6-$NAME-$(date +%Y%m%dT%H%M%S)-$$
mkdir -p "$OUT"
python3 - "$OUT" "$A" "$B" <<'PY'
import pathlib,hashlib,json,sys,os
p=pathlib.Path(sys.argv[1]);m={}
for arm,path in zip('AB',sys.argv[2:]):
 s=pathlib.Path(path);b=s.with_suffix('');stamp=s.parent/f'.{s.stem}.build'
 assert os.access(b,os.X_OK),b
 assert stamp.read_text()==f'QSB_ZEROS_N=24 {s.stat().st_mtime_ns}'
 m[arm]={'source':path,'text':s.read_text(),'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'stamp':stamp.read_text()}
(p/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2),flush=True)
PY
for i in $(seq 1 "$PAIRS"); do
 arms=(A B); if ((i%2==0)); then arms=(B A); fi
 for arm in "${arms[@]}"; do
  src=$A; [[ $arm != B ]] || src=$B
  flock /tmp/angel-gpu.lock bash -c '
   set -euo pipefail
   QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src $1" QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS="$2" QSB_MAX_REL_VAR=none QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset 2>&1 | tee "$3.full.log" | tail -16 | tee "$3.log"
   cp score-subset.json "$3.json"
  ' _ "$src" "$SECS" "$OUT/p$i-$arm"
 done
done
python3 - "$OUT" "$PAIRS" <<'PY'
import json,pathlib,sys,statistics
p=pathlib.Path(sys.argv[1]);n=int(sys.argv[2]);m={};scores={};hits={}
for a in 'AB':
 s=[];h=[]
 for i in range(1,n+1):
  x=json.loads((p/f'p{i}-{a}.json').read_text());assert x['metrics']['verified'],x
  s.append(x['score']/1e6);h.append(x['metrics']['verified_hits'])
 scores[a]=s;hits[a]=h;m[a]=statistics.mean(s)
r={'Mps':m,'runs':scores,'verified_hits':hits,'delta_percent':100*(m['B']/m['A']-1),'all_verified':True}
(p/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
PY
