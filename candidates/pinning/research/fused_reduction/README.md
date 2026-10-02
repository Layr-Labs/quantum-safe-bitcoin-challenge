# Pending fused-reduction native comparison

This experiment independently stages public validation PR 219
(`f7f588c7`, scalar-pipeline base) and pending PR 225 (`211f74dc`, submission
`f297b0f`). The candidate adds fused modular multiply/subtract and
square/add/subtract reducers to the mixed XYZZ addition.

The pending submission already carries extensive host/PTX semantic evidence.
This directory adds exact-source CUDA 12.8.93 `sm_89` resource and SASS-shape
comparison before its official RTX 4090 result is available. It does not claim
GPU execution or throughput.

```sh
python3 candidates/pinning/research/fused_reduction/prepare.py
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl start qsb-cuda --tty=false
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/fused_reduction/control --default-build --report candidates/pinning/research/fused_reduction/control-native.json
python3 candidates/pinning/research/compile_local.py --source candidates/pinning/research/fused_reduction/candidate --default-build --report candidates/pinning/research/fused_reduction/candidate-native.json
LIMA_HOME=/tmp/qsb-lima-state /tmp/qsb-lima-bin/bin/limactl stop qsb-cuda
```

## Result

Both immutable trees pass `sm_89` and default-flag builds. The full prepare
specialization moves from 126 to 128 registers with no spills and from 7,400
to 7,424 SASS slots. The FastTail prepare specialization has the same +2
registers and +24 slots. Finish kernels are identical. `IMAD.WIDE.U32` falls
655 to 653, `IMAD.WIDE.U32.X` remains 1,224, and `IADD3.X` rises 1,108 to
1,150 on the full prepare path.

Thus the fusions change dependency structure but do not create an obvious
static work reduction; they also consume the remaining register headroom.
Official runtime is decisive. Do not duplicate the pending submission.
