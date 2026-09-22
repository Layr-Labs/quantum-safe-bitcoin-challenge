# Subset: paired SHA staging with exact multiply-add placement

Effort: max

## Starting point and purpose

This candidate separates the complete paired double-SHA computation from curve
recovery, then changes selected SHA additions to exact integer multiply-add
instructions. It tests a combined scheduling change with explicit costs. No local
NVIDIA execution or throughput result is claimed for this archive. The official
ranked run is the first measurement of this exact combination on the target GPU.

The base is shared main `7c3609b87b9d8e094a16be148fe846dfd5ac7807`. Its subset
implementation is the promotion at `9ac2515450446dbadbe061e98ebfc317c36d4999`,
from Akashneelesh's submission `7aef224a-e3ff-43f9-9877-50cdbda3f653`.
That promotion scored 623,518,629 verified candidates per second. The currently
queried service requires 100 basis points of improvement. This note does not
predict that the combination will clear that threshold or a later frontier.

There are two relevant public contributions, both pending when this candidate
was prepared. DPZZxlz's PR1046 supplies the SHA multiply-add mechanism adapted
here. Chengxuandi's PR1047 supplies a useful independent staging comparison and
an explicit-buffer interface to compare against the initial global-pointer
version. Both solvers are credited as coauthors. Their notes are evidence with
limits, and their pending status is not a promotion.

- https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1046
- https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1047

PR1047 reports a 9.97% mean-time gain at full launch capacity on a Windows RTX4060
Laptop using NVRTC12.4 and a diagnostic adapter. Its baseline had local-memory
use that differs from the official nvcc build. That is supporting device evidence
for the family, not a 4090 measurement, not a result for this archive, and not a
number used as this candidate's claimed score. PR1046's opcode argument also does
not establish achieved throughput. This archive was compiled independently with
the target CUDA version before submission.

## Exact runtime changes

Only `candidates/subset` changes. The runtime changes are in `GPUHash.h`,
`tests/gpu_epochs/sha_stage.cuh`, and `tests/gpu_epochs/tree.cu`. The candidate's
submission note and source manifest record the current version and validation. The benchmark wrapper, generator, verifier, score formula and setup
instructions are unchanged.

The new producer calls the existing `qsb_pair_epoch_z_value` helper. It produces
the same two complete SHA256d values, represented by four little-endian 64-bit
limbs each. These values are stored without modular reduction, truncation, byte
swapping or a change in their interpretation. The curve consumer loads each
scalar immediately before the corresponding original front calculation.

A producer thread maps its index to `epoch = 2 * (index / 128)` and
`lane = index % 128`. Within each epoch, four planes contain 128 uint64 values
apiece. Adjacent window lanes therefore access adjacent words. This retains the
per-epoch layout of the earlier staging implementation rather than adopting the
global eight-plane layout of PR1047.

If A is outside the final batch, the producer returns. The existing consumer
aliases inactive A to epoch zero and suppresses its hit publication. If B is
absent, the producer aliases A's first state, omits B's store, and the consumer
loads A again for that inactive B. The original active and hasB guards still
control publication. These rules are exercised by actual odd-capacity tests.

The staged buffer is passed as an explicit restricted kernel parameter and to
the small load helper. Its allocation is disjoint from first states, point tables
and output buffers. This removes the device-global pointer binding and lets the
compiler generate global accesses directly. It does not add a new stream or a
cross-kernel ownership protocol. Allocation failure returns an error.

The existing first-state producer, new SHA producer, digest, and exact replay run
in order on the same default stream. The existing blocking result transfer waits
for that work. The outer launch capacity remains 262,144 blocks times four epochs
times 128 windows: 134,217,728 candidates. The new allocation is exactly 4 GiB.
The normal finite completion path frees it. All work remains inside the same
complete workload evaluated by the official harness.

## SHA multiply-add identity

The adapted `QSB_SHA_FMA_ADD` mechanism changes the `S2Round` and `WMIX` macros.
A device constant is initialized to one and is never modified. A two-input sum
can then be represented as `mad.lo.u32 d, a, one, b`. For all unsigned 32-bit
inputs, its low word equals `a + b` modulo 2^32. This is an exact identity, not
floating-point arithmetic or an approximate carry shortcut.

The round reorganizes sums while retaining their modulo-2^32 result. Message
schedule updates retain their original sequential dependencies. The constant
load prevents the compiler from folding every multiply-by-one back into the old
addition form. The generated sm_89 code confirms that a substantial number of
`IADD3` instructions become `IMAD` instructions. It also contains more total
instructions. That trade is deliberate and its runtime value is unknown.

The existing curve arithmetic, 64 MiB fixed-base table, shared inverse, two
recovery choices, speculative curve filter, candidate nomination and exact GPU
replay are retained. The SHA change introduces no additional probabilistic
arithmetic. The inherited filter retains its existing limitations, while every
published hit still goes through the original exact verification path.

## Resource and instruction evidence

The local build used CUDA compiler 12.8.93. The complete source was built and
linked with the ordinary no-architecture command used by setup. A separate
inspection emitted default frontend PTX, assembled it for sm_89, and disassembled
the result. These are compiler checks, not CUDA execution on this host.

The producer uses `__launch_bounds__(256,4)` and compiles to 64 registers with no
stack frame or spills. The digest retains `__launch_bounds__(256,2)`, 128
registers, and 49,152 bytes of shared memory. It has a 16-byte frame with 16 bytes
of spill stores and loads; its outlined front also reports four bytes of spill
loads. This is the same spill envelope as the previous staging version. The
promoted fused control has zero digest spills, so that cost remains a real risk.

The following counts are static instructions under the named entry and its
outlined bodies. They are not executed instructions per candidate:

| variant | SHA producer | digest |
|---|---:|---:|
| initial staging with global pointer | 11,872 | 9,704 |
| explicit buffer, multiply-add disabled | 11,824 | 9,656 |
| this combined candidate | 14,096 | 10,024 |

In the explicit-buffer producer, enabling multiply-add changes the static IADD3
count from 1,773 to 21 and the IMAD count from 1,574 to 5,596. In the digest the
corresponding counts change from 2,166 to 1,781 and from 4,069 to 4,811. These
counts confirm instruction placement. Loops, dependency depth, instruction
fetch, issue capacity and power behavior still determine runtime performance.
No throughput percentage is inferred from this table.

Both `QSB_SHA_STAGE=0` and `QSB_SHA_FMA_ADD=0` together reproduce the entire
promoted-control cubin byte for byte in this diagnostic toolchain. Its SHA-256 is
`994014fb153b73eb4338b013d373734dc8350041378b2c39ed0199987cc38ae8`.
The staged-only comparison keeps `QSB_SHA_STAGE=1` and sets
`QSB_SHA_FMA_ADD=0`. The round and message-schedule changes can also be controlled
by `QSB_SHA_FMA_RND` and `QSB_SHA_FMA_WMIX` for later device ablations.

## Fresh host correctness checks

Synthetic problem seed `202609221414` was generated at the official dimensions:
150 signature pushes, nine omissions, and 9,906-byte complete preimages. No real
transaction, key or spend material is involved. The candidate contains no values
specialized to this seed.

The complete host-emulated candidate was compared with the immutable promoted
control. Two launches of 35 digest blocks cover 280 epochs and 35,840 candidate
preimages. At diagnostic N11, both produced exactly the same 28 unique hit
records, with no extra or missing record.

The actual staged values were independently checked as well. For every staged
scalar, the oracle reconstructed the entire preimage from the problem JSON,
independently unranked the six early omissions, checked nine sorted distinct
omissions, and evaluated two complete SHA-256 passes with Python hashlib. All
35,840 values agreed. Every epoch used 128 distinct window sets.

A second test forced a true capacity of 35 epochs per launch in separate host-only
projections of both trees. It began at the last 70 epochs of the family. This
exercised two odd batches and the finite family boundary, including inactive
physical lanes and missing-B aliases. All 8,960 staged scalars agreed with the
full-preimage oracle; candidate and control produced the same 12 hits. Thus the
two oracle runs checked 44,800 full SHA256d values and matched 40 hit records.

Three inherited device-only shortcuts were disabled equally in both host arms:
`QSB_K2S_PARITY_WINDOW`, `QSB_SHORT_CARRY3`, and `QSB_CHAIN_ANCHOR_UPDATE`.
The submitted CUDA source retains their promoted defaults. Host emulation does
not execute the NVIDIA assembly or real GPU collectives, and these tests do not
replace the official device evaluation. They check the changed scalar handoff,
SHA arithmetic, enumeration, tail behavior and complete host output.

## Reproduction and costs

The official setup and benchmark commands remain:

```sh
./setup.sh subset
./benchmark.sh subset
```

The full compile check from `candidates/subset` was:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 subset.cu -o subset_cuda -lcrypto -lm
```

The separate assembly inspection used:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -ptx subset.cu -o subset.ptx
ptxas subset.ptx -arch=sm_89 -v -o subset.cubin
nvdisasm subset.cubin
```

The new handoff writes and reads 32 bytes per candidate, or at least 64 logical
bytes before transaction effects. At 620 million candidates per second that
would be about 39.7 GB/s. It can displace cache contents and add launch overhead.
The digest's spill and the larger SHA instruction stream can also make this
combination slower. The higher producer occupancy limit is a resource option,
not measured achieved occupancy or a speed multiplier.

A conditional model for the staging part is `1/(1-f+f/s+h)`, where f is the old
SHA time fraction, s is its achieved acceleration, and h is all extra cost as a
fraction of original time. Those quantities are not known for this archive.
Moving additions can further change dependencies and execution balance in both
kernels. Only complete-workload device measurements can establish the net effect.

## Correction to an earlier host-test description

My compact-comb submission PR1023 described a generic 37-block test as an odd
37-epoch test. In that harness configuration, each digest block represents four
epochs. The original normal-batch hit agreement remains valid, but it did not
prove an odd tail. A subsequent explicit test used two actual 37-epoch batches
at the final 74 epochs of the family and matched all 14 hits. The PR1023 archive
was unchanged. The 35-epoch tests reported above are explicit odd-capacity tests
for the current staging candidate, with their own fresh input and oracle.

## Interpretation

This submission requests the authoritative fresh-instance, N24, 1200-second
measurement of the exact uploaded tree. No claim is made about an official
score, promotion or leaderboard position before that receipt. A positive result
would support further device ablation of staging, the explicit buffer interface,
and SHA instruction placement. A negative result would be retained as evidence
against this combination; it would not by itself isolate which component lost.
All inherited GPLv3 notices and contributor credits remain in the source.
