#!/usr/bin/env bash
# n24L_run.sh <variant> [seconds] [seed] — scored-path N=24 measurement of a
# prebuilt n24L_* variant via benchmark.sh + gpu_wrap (stamp-cached binary).
# Single-flight: refuses to run if another scored run is active (they share
# benchmark-results/problem/ and results/ and corrupt each other's artifacts).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
v=$1; secs=${2:-120}; seed=${3:-1789110211}
if ls /tmp/angel-n24-run-* >/dev/null 2>&1; then
  echo "REFUSED: another n24 scored run is active ($(ls /tmp/angel-n24-run-* 2>/dev/null | head -1))"; exit 3
fi
touch "/tmp/angel-n24-run-$v"
trap 'rm -f /tmp/angel-n24-run-$v' EXIT
QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/subset/n24L_$v.cu" \
QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=$secs QSB_MAX_REL_VAR=none \
QSB_PROBLEM_SEED=$seed ./benchmark.sh subset 2>&1 | tail -16
