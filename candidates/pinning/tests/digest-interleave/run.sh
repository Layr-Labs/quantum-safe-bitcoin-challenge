#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
OUT=$(mktemp -d)
echo "Test artifacts: $OUT"
g++ -O2 host.cpp -lcrypto -o "$OUT/host"
"$OUT/host"
# Select the architecture of the test GPU; sm_120 was used locally.
ARCH=${ARCH:-sm_120}
for I in 0 1; do
  for Z in 0 1; do
    nvcc -O3 -arch="$ARCH" -DQSB_DIGEST_INTERLEAVE_Z="$I" -DQSB_DIGEST_WMIX_Z="$Z" device.cu -lcrypto -o "$OUT/device-$I-$Z"
    "$OUT/device-$I-$Z"
  done
done
