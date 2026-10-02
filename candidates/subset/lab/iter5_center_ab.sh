#!/usr/bin/env bash
# Frozen prebuilt exact-verifier screen, source + dependency hashes retained.
set -euo pipefail
cd "$(dirname "$0")/../../.."
W=$(cat "${WORK_DIR_FILE:-/work/tmp/qsb-it5-center-dir}")
S=1789110211
out="candidates/subset/lab/ab-denbest-${PROBE_NAME:-center}-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$out"
python3 - "$W" "$out" <<'PY'
import pathlib,hashlib,json,sys,os
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);data={}
for a,n in [('A',os.environ.get('CONTROL_NAME','denbest')),('B','center')]:
 p=root/f'candidates/subset/lab/n24L_{n}.cu';b=p.with_suffix('');stamp=p.parent/f'.{p.stem}.build'
 assert b.is_file() and os.access(b,os.X_OK) and stamp.read_text()==f'QSB_ZEROS_N=24 {p.stat().st_mtime_ns}'
 data[a]={'source':str(p),'text':p.read_text(),'sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'stamp':stamp.read_text()}
# Wrapper binaries compile fallback sm_86 only; native header rebuilt after them
# is irrelevant to this screen. Snapshot device implementation separately.
for f in ['tests/gpu_epochs/tree.cu','tests/gpu_epochs/pair_shared.cuh','subset.cu']:
 p=root/'candidates/subset'/f;data[f]=hashlib.sha256(p.read_bytes()).hexdigest()
out.joinpath('manifest.json').write_text(json.dumps(data,indent=2))
PY
for i in $(seq "${FIRST_PAIR:-1}" "${LAST_PAIR:-1}"); do
 arms=(A B); if (( i % 2 == 0 )); then arms=(B A); fi
 for a in "${arms[@]}"; do
  if [[ $a == A ]]; then n=${CONTROL_NAME:-denbest}; else n=center; fi
  src="$W/candidates/subset/lab/n24L_${n}.cu"
  flock /tmp/angel-gpu.lock bash -c '
  set -euo pipefail
  QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src $1" QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=120 QSB_MAX_REL_VAR=none QSB_PROBLEM_SEED="$2" ./benchmark.sh subset 2>&1 | tail -16 | tee "$3.log"
  cp score-subset.json "$3.json"
  ' _ "$src" "$S" "$out/p${i}-$a"
 done
done
python3 - "$out" "${FIRST_PAIR:-1}" "${LAST_PAIR:-1}" <<'PY'
import pathlib,json,sys,statistics
p=pathlib.Path(sys.argv[1]);d={};lo,hi=map(int,sys.argv[2:])
for a in 'AB':
 r=[]
 for i in range(lo,hi+1):
  x=json.loads((p/f'p{i}-{a}.json').read_text());assert x['metrics']['verified'],x;r.append(x['score']/1e6)
 d[a]=statistics.mean(r)
s={'Mps':d,'delta_percent':100*(d['B']/d['A']-1),'verified':True};print(json.dumps(s,indent=2));(p/'summary.json').write_text(json.dumps(s,indent=2))
PY
