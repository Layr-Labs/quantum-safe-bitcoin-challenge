#!/usr/bin/env bash
# One-time prerequisites for the QSB grinding benchmark: check the toolchain,
# generate the synthetic problem instances, and smoke-test the verifier.
# Safe to rerun. Nothing here is inside a measured interval.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

die() { echo "setup.sh: $*" >&2; exit 1; }

(($# == 1)) || die "usage: ./setup.sh {pinning|subset}"
bench="$1"
case "${bench}" in
  pinning|subset) ;;
  *) die "track must be pinning or subset: ${bench}" ;;
esac
src="candidates/${bench}/${bench}.cu"

command -v python3 >/dev/null 2>&1 || die "python3 is required"
python3 - <<'PY' || die "python3 >= 3.9 is required"
import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY

# The harness and verifier are pure stdlib on purpose (they must run in CPU-only
# CI), so there is nothing to pip install.
echo "setup.sh: python3 $(python3 -c 'import platform;print(platform.python_version())')"

# Generate the canonical problem instances. Deterministic from --seed, so this
# is idempotent.
python3 harness/gen_problem.py --seed "${QSB_PROBLEM_SEED:-0}"

# A GPU kernel is optional: the benchmark scores any program that satisfies the
# I/O contract in spec/PROBLEM.md. Report what is available so a submission
# knows what it is building against.
if command -v nvcc >/dev/null 2>&1; then
  echo "setup.sh: nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1)"
else
  echo "setup.sh: warning: nvcc not found — only the CPU reference grinder can run" >&2
fi
# The harness times the grinder process end to end, so compilation must not
# happen inside a ranked run. Pre-build only this track's candidate kernel at
# the ranked N; gpu_wrap.py reuses it instead of invoking nvcc while the clock
# is running.
if command -v nvcc >/dev/null 2>&1; then
  zeros="$(python3 -c "import json;print(json.load(open('harness/config.json'))['leading_zero_bits'])")"
  python3 -c "
import sys; sys.path.insert(0, 'harness')
from pathlib import Path
import gpu_wrap
gpu_wrap.compile_kernel(Path('${src}'), ${zeros})
" || die "failed to build ${src} at N=${zeros}"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^/setup.sh: gpu: /'
else
  echo "setup.sh: warning: nvidia-smi not found — no GPU detected" >&2
fi

# Prove the verifier works on this host before any ranked run depends on it.
# Always the CPU reference grinder: this is a check of the harness, not of the
# GPU, and it must pass on a host with no GPU at all.
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
python3 harness/run_benchmark.py \
    --bench "${bench}" --N 4 --mode fixed_hits --hits 2 --grinder cpu \
    --max-rel-var none --out "${tmp}/smoke.json" >/dev/null \
  || die "verifier smoke test failed"
echo "setup.sh: verifier smoke test passed"
echo "setup.sh: ready — run ./benchmark.sh ${bench}"
