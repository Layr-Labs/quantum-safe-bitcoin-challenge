#!/usr/bin/env bash
# build_sm89_cubin.sh — regenerate pinning_sm89.cubin (v8 driver-API path).
#
# The ranked build line carries no -arch, so the produced binary embeds sm_52
# SASS that cannot run on the RTX 4090; the driver JIT-compiles the PTX inside
# the timed window.  This script builds a native sm_89 cubin from the SAME
# pinning.cu that ships in the archive; the binary loads it at startup through
# the driver API (dlopen libcuda, no -lcuda needed) and launches via
# cuLaunchKernel, eliminating the whole JIT.
#
# MUST be rebuilt whenever pinning.cu changes — the cubin and the source are
# one unit of correctness.  The shipped file must sit next to the produced
# binary: candidates/pinning/pinning_sm89.cubin.
#
# Flags must mirror the ranked build (setup.sh): -O3 and the same -D defines.
# The cubin is compiled with QSB_FUSED=0 defaults; do not add -DQSB_FUSED.

set -euo pipefail
cd "$(dirname "$0")/.."
nvcc -O3 -arch=sm_89 -cubin pinning.cu -o pinning_sm89.cubin

# Sanity: the six launched kernels and the five runtime-uploaded constants
# must all be present, and e_flags must say sm_89 (low byte of flags == 0x59).
readelf -h pinning_sm89.cubin | grep -q "Flags:.*0x59" || {
    echo "cubin is not sm_89" >&2; exit 1; }
for s in \
  _Z19kernel_build_gtablePKmS0_Ph \
  _Z23kernel_pinning_pipelineILb1ELi0EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_ \
  _Z23kernel_pinning_pipelineILb1ELi2EEvPKjPKhiiiijjPKmS5_S5_S5_S5_PhPjS7_iiiP10ulonglong2PmSA_ \
  _Z22qsb_root_group_preparePKmiPmS1_ \
  _Z22qsb_invert_super_rootsPmi \
  _Z21qsb_root_group_finishPmiPKmS1_ \
  pin_u2rx_words pin_u2ry_words pin_u2rk_words pin_recovery_c pin_tail_words
do
    readelf -sW pinning_sm89.cubin | grep -q " $s\$" || {
        echo "missing symbol $s" >&2; exit 1; }
done
echo "pinning_sm89.cubin OK: $(stat -c%s pinning_sm89.cubin) bytes, sm_89, all symbols present"
