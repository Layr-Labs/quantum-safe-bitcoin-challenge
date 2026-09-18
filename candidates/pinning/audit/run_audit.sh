#!/bin/sh
# Host correctness gate for the pin-merge switches.  No GPU required.
#   arg1: blocks of 128 candidates (400 -> ~50k compared candidates)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
B="$HERE/build"; mkdir -p "$B"
N=${1:-400}
echo "== slicing the shipped device source text out of pin-merge =="
python3 "$HERE/extract.py" "$B/ph.cpp"
for c in 11 10 01 00; do
  m=$(echo $c | cut -c1); s=$(echo $c | cut -c2)
  g++ -std=c++17 -O2 -shared -fPIC -DQSB_FUSE_MULSUB=$m -DQSB_FUSE_SQRADDSUB2=$s \
      "$B/ph.cpp" -o "$B/libph$c.so"
done
g++ -std=c++17 -O2 "$HERE/e2e_recovery.cpp" -o "$B/e2e" -lcrypto -ldl
for c in 11 10 01; do
  echo; echo "== arm $c vs unfused 00, both against an independent secp256k1 reference =="
  (cd "$B" && ./e2e "$N" "./libph$c.so" "./libph00.so")
done
