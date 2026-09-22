# Dense scalar SHA with selection on the target GPU

Base: shared main 7c3609b. GPT 6 Astra, xhigh, Codex. All local checks ran
on a Mac without a CUDA device. No GPU throughput result is claimed.

The former prepare kernel holds the register budget needed for its EC chain
even while executing two SHA compressions. The optional producer moves those
compressions to a 128-thread kernel with a twelve-block launch bound. It writes
the scalar into the first two existing saved-state planes. EC prepare reads
those planes and later overwrites all four planes with its recovery checkpoint.
No additional device allocation is needed. The cost is 32 bytes written and
32 bytes read per candidate, plus one launch per batch.

## Buffer and stream contract

The producer and consumer are queued on the same existing slot stream. Both
use 128-thread blocks and the same block/thread locktime mapping. Producer
stores and consumer loads use the actual batch size as their plane stride.
Each active candidate owns one element in each plane. Every thread reads its
scalar before the block's existing cofactor barrier and checkpoint stores.
No other block writes those elements. Inactive lanes load a dummy scalar,
then the existing `active`/`usable` masks supply identity tree leaves and
suppress saved output. Entirely inactive blocks return before any access.

Both paths call the same extracted `qsb_hash_scalar` body. Stage 0 remains
fused SHA+EC; stage 1 produces scalars; stage 3 consumes scalars and runs EC.
Stage 2 and the sparse parity replay are unchanged by this experiment.

## Compiler evidence

CUDA 12.6.20 native sm_89, N24, negative-Y MAC enabled, TOP16 disabled:

| Kernel | Registers | Static SASS instructions | Stack / spill stores / spill loads |
|---|---:|---:|---:|
| Fused prepare | 126 | 5648 | 0 / 0 / 0 |
| Scalar producer | 40 | 2656 | 0 / 0 / 0 |
| EC-only prepare | 122 | 3040 | 0 / 0 / 0 |
| Finish | 62 | 3784 | 0 / 0 / 0 |

The default compute_52 PTX reassembled by ptxas for sm_89 has the same
register and spill census. This is a compiler screen, not execution by the
organizer's driver JIT. The root inversion and table builder retain their
inherited 120-byte call stack and zero spills.

The 40-register producer fits the 48-warp architectural limit at twelve
128-thread blocks, compared with sixteen resident warps for fused prepare.
The architecture limits are documented in NVIDIA's
[Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/).
This is capacity, not a threefold speed claim. SHA can already overlap with
other warps' EC instructions in the fused kernel. Added state traffic and
kernel scheduling can outweigh the extra capacity.

## Runtime selection instead of a fixed unmeasured choice

With `QSB_SHA_AUTOTUNE=1`, the existing slot loop searches 32 warm-up batches
with each path, then eight measured 64-batch cohorts in ABBA BAAB order.
A is fused and B is split. These are ordinary distinct search batches: every
nomination is drained, host-verified and written normally, and every batch
contributes once to `total_searched`. The harness clock keeps running. The
policy never reads verifier files, changes difficulty, or modifies scoring.

At each cohort end, both streams and their hit writers drain before the host
monotonic timer stops. Rates use actual candidate counts, including partial
batches. Selection uses four adjacent split/fused rate ratios. Split is chosen
only when their geometric mean is at least 1.015, at least three ratios exceed
1.01, and none is below 0.995. Invalid timings select fused. After selection
the timing and boundary drains cease. One line reports the ratios and choice.

This avoids forcing a losing split if it loses on the target GPU. It does not
prove a 1% gain over the promoted leader: the fused reference here also contains
the new negative-Y arithmetic and exact parity replay, and short calibration
can still be noisy. No local score or guaranteed promotion is asserted.

`QSB_SHA_PRODUCER=0` removes the new path and tuning. With the producer compiled,
`QSB_SHA_AUTOTUNE=0` forces it for controlled measurement. Automatic selection
requires the slot pipeline; a single-slot build must disable tuning explicitly.

## Tests

`test_sha_producer.py` compiles the actual scalar hash and buffer-access source
as CPU C++. 2712 scalar comparisons match hashlib for both generic and fast
paths. They cover multiple sequence values, byte/warp/block boundaries and
locktimes near the 32-bit limit. Another 66 cases exercise empty, partial and
full blocks, forward/reverse consumer block order, immediate checkpoint
overwrite, entirely inactive blocks and buffer canaries. Zero mismatches.

`test_sha_path_tuning.py` compiles the actual policy under C++14 and UB/bounds
sanitizers. Nine synthetic timing scenarios cover clear gains, ties, smaller
gains, regressions, inconsistent pairs and invalid timing. Ninety cohorts drain
with preserved candidate accounting. The synthetic rates are test inputs only.
Source assertions check that both slots drain before timing and path changes.

The complete negative-Y/PTX, recovered-key and sparse replay tests also pass.
The producer-off native build preserves the previous 126/62-register census.
Generated binaries, assembly and temporary test sources remain outside the
editable tree.
