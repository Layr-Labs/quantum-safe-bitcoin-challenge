# Paired public-key SHA unroll probe

This experiment starts from the exact promoted pinning commit `067302c` and
changes only `QSB_PK_UNROLL` from 0 to 1. It tests whether ptxas can interleave
the two independent recovered-public-key SHA chains without damaging register
allocation, spills, occupancy, or native instruction shape.

This is primarily a calibration experiment for the larger two-chain fixed-base
idea. It removes no arithmetic and is not expected to meet the project's large
improvement target by itself. A regression or spill increase is evidence against
doubling the much larger XYZZ live state; a clean code-generation result only
justifies an NVIDIA timing experiment.

Commands:

```sh
python3 candidates/pinning/research/pk_unroll/prepare.py
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl start qsb-cuda --tty=false
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/pk_unroll/control --default-build --report candidates/pinning/research/pk_unroll/control-native.json
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/pk_unroll/candidate --default-build --report candidates/pinning/research/pk_unroll/candidate-native.json
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl stop qsb-cuda
```

The reports establish native CUDA 12.8.93 compilation and static resource/code
differences only. They are not GPU execution or throughput measurements.

## Result

Both exact-source builds pass for `sm_89` and the default CUDA target. On the
full finish path, unrolling keeps 80 registers while changing 24 bytes of stack
and 20/20 bytes of spill stores/loads to zero. Static SASS slots increase from
4,696 to 7,336. On the FastTail finish path, registers fall from 80 to 76,
spills again become zero, and slots rise from 3,312 to 4,600. Prepare kernels
are byte-identical at the native-resource level.

This is favorable register evidence but ambiguous performance evidence. The
compiler may simply have emitted two sequential SHA bodies; static duplication
also increases instruction-cache risk. Do not submit this switch alone without
NVIDIA timing, and do not extrapolate finish headroom to the 128-register
prepare kernel or a second live XYZZ accumulator.
