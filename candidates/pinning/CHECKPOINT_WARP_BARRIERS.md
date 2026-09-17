# Warp-scoped synchronization in pinning checkpoint trees

This is an unbenchmarked pinning optimization candidate. It has not been
compiled with CUDA, evaluated by Yukon, or promoted. No speed gain is claimed.

## Change and provenance

Starting point: upstream commit
`df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634` in
[Layr-Labs/quantum-safe-bitcoin-challenge](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/tree/df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634).

The promoted subset tree already chooses warp synchronization when the next
level is entirely contained in warp zero:
`candidates/subset/tests/gpu_epochs/tree_inverse.cuh`, especially its packed
level implementation. This candidate applies that existing scheduling idea to
the pinning track's external-root checkpoint helpers. It is attributed reuse
and a narrow adaptation, not a new batch-inversion algorithm. All existing
contributors' implementation and license notices remain intact. The candidate
continues to be GPLv3-governed through its existing dependencies.

Only the checkpoint helpers' synchronization choices change. The initial
publication barriers remain block-wide. In the upward pass, a warp barrier is
used once the number of writers is at most 32; the next level reads only those
writers. In the downward pass, a warp barrier is used only when twice the
current number of writers is at most 32. A block barrier precedes every level
whose consumers reach another warp, including the final leaf read.

All threads reach every selected barrier; the choice depends only on the
uniform level width. Warp zero performs every small-level producer/consumer
access. Other warps have no work at those levels and rejoin at the next block
barrier or kernel completion. Tree indices, field arithmetic, normalization,
identity substitution, candidate enumeration and hit output are unchanged.

For the default 128-lane candidate tree, prepare changes from seven block
barriers to two block and five warp barriers. Finish changes from seven block
barriers to three block and four warp barriers. These are source-level counts,
not timing measurements. Fewer block barriers may help but compiler behavior,
occupancy and the fraction of runtime spent here determine the actual result.

## Preflight completed on Apple Silicon, without CUDA

`python3 candidates/pinning/audit_checkpoint_barriers.py`

The audit extracts the actual helper function bodies from `pinning.cu`, compiles
them as C++17, emulates block and per-warp barriers with CPU threads, and checks
secp256k1 field roots and leaf inverses with OpenSSL. All 27 cases pass: 4,032
inverses across widths 64, 128 and 256; identity and partial-tail lanes; nonzero
checkpoint block offsets. The same audit also passes on the unmodified upstream
helper bodies using `--source <baseline-pinning.cu>`.

The host emulation uses OpenSSL in place of the unchanged PTX field multiplier.
It validates actual tree control flow, indexing, data dependencies under the
emulator and modular results. It does not establish CUDA compilation, device
memory ordering, GPU race freedom, or throughput.

A ThreadSanitizer build crashes before useful diagnostics on this host; a
trivial instrumented `int main(){return 0;}` also exits with signal 11. There
is no successful sanitizer result to claim.

The official CPU-reference smoke path also passed with a fresh synthetic
problem: seed 1789668001, N=6, fixed_hits=3, three of three hits independently
verified. This checks the harness, not the modified GPU candidate. The harness
prints the configured RTX_4090 label even in CPU-reference mode; that label is
not a measurement of this Mac and its CPU score must never be used as a GPU
result.

## GPU checks prepared but not run

`tests/audit_checkpoint_gpu.cu` includes the actual candidate and checks its
actual device multiplier and checkpoint helpers against OpenSSL over widths
64/128/256, three checkpoint blocks and ten active-tail cases per width. The
source has not been compiled locally because this host has no CUDA toolkit.

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/qsb-checkpoint-audit \
  candidates/pinning/tests/audit_checkpoint_gpu.cu -lcrypto -lm
/tmp/qsb-checkpoint-audit
compute-sanitizer --tool synccheck /tmp/qsb-checkpoint-audit
compute-sanitizer --tool racecheck /tmp/qsb-checkpoint-audit
```

For full candidate behavior, use the unchanged official setup and benchmark
commands on a CUDA host, then submit through Yukon's authorized pinning flow.
Yukon's own runner performs ranked compilation and evaluation, so local NVIDIA
hardware is optional. Official evaluation and promotion are the only reward
qualification evidence; no Taskmarket evidence should be submitted for these
local preflight results.
