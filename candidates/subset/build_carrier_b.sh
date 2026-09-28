#!/usr/bin/env bash
# Build variant B's native image for the in-run A/B (QSB_AB) and write qsb_carrier_b_sm89.h.
# Development tool, never run by the ranked harness. B = this same tree with QSB_AB_B_KNOBS applied:
#   QSB_AB_B_KNOBS="-DQSB_Q_MIX=2" NVCC=/usr/local/cuda-12.8/bin/nvcc ./build_carrier_b.sh
# Each -DNAME=VALUE is passed to nvcc; a knob that subset.cu defines unconditionally (e.g.
# QSB_SHA_FMA_ADD, QSB_PAIR_SHA_UNROLL_CONST) is also rewritten in place in the temporary copy's
# subset.cu (same line, so no line number moves). Empty QSB_AB_B_KNOBS = the NULL image (B == A).
# Another tree's image (e.g. the root-inverse package): run mk_b_header.py on that tree's
# qsb_carrier_sm89.h instead (see REPORT.md). The kernel ABI (mangled names) must match image A.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
KN=${QSB_AB_B_KNOBS:-}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir "$T/src"
tar cf - --exclude=./qsb_carrier_b_sm89.h . | tar xf - -C "$T/src"
for kv in $KN; do
    case "$kv" in
        -D*=*) name=${kv#-D}; val=${name#*=}; name=${name%%=*}
               if grep -q "^#define $name " "$T/src/subset.cu"; then
                   sed -i "s|^#define $name .*|#define $name $val|" "$T/src/subset.cu"
                   echo "build_carrier_b: subset.cu: #define $name $val"
               fi ;;
        -D*) ;;
        *) echo "build_carrier_b: QSB_AB_B_KNOBS takes -DNAME=VALUE / -DNAME only (got '$kv')" >&2; exit 1 ;;
    esac
done
( cd "$T/src" && QSB_CARRIER_NVCC_FLAGS="$KN ${QSB_CARRIER_NVCC_FLAGS:-}" ./build_carrier.sh )
python3 "$HERE/mk_b_header.py" "$T/src/qsb_carrier_sm89.h" "$HERE/qsb_carrier_b_sm89.h" --flags "$KN"
