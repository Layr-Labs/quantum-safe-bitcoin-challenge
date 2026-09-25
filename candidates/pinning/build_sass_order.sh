#!/usr/bin/env bash
# Offline development command. The ranked build does not execute this script.
set -euo pipefail
cd "$(dirname "$0")"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
"${NVCC:-nvcc}" -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1 -arch=sm_89 \
    -cubin -Xptxas=-v pinning.cu -o "$work/control.cubin" 2> "$work/build.log"
python3 sass_order.py "$work/control.cubin" --cuobjdump "${CUOBJDUMP:-cuobjdump}" --out .
cat "$work/build.log"
