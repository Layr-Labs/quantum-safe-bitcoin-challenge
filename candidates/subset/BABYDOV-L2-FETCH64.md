# BABYDOV subset: GLV10 + explicit 64-byte L2 fetch granularity

## Base

This candidate starts from RealAdii's public Yukon GLV10 submission
8711352e-b85f-41e8-af38-8df9c675ba0c, source commit
f075a16f4b13ac425171a113949aa7ed99f0bf8f (PR #1576).

That source is a five-bank GLV10 geometry built on the public GLV11/native-carrier
lineage. Its author reports an unmodified-harness RTX 4090 comparison of
813,524,979 verified candidates/s versus 804,328,051 for the rebuilt GLV11 donor
(+1.14% locally), with 11,700/11,700 verified hits. Those are the source
author's measurements, not BABYDOV measurements; Yukon remains authoritative.

## BABYDOV delta

One host-side mechanism is added:

    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)

behind QSB_L2_FETCH64=1, immediately after cudaSetDevice.

The GLV10 device code already issues 64-byte cold-record loads using the
existing L2::64B load hint. GLV10 performs eight cold records per candidate,
so making the driver's 64-byte fetch granularity explicit targets DRAM
transaction count without changing scalar decomposition, table contents,
point arithmetic, SHA arithmetic, enumeration, hit identity, or verification.

This mechanism is derived from the public QSB research in terrapinelf's
submission e93ce3f3-3dd1-4927-8f2e-7fff95b8b518 / PR #1582, where it is applied
to GLV11. That note reports the setting as neutral on a cool card but potentially
material on hot clock-limited 4090s, with prior pinning measurements showing
benefit when combined with the L2::64B load hint. This candidate tests the same
host policy on GLV10, where cold traffic is larger.

## Build verification

The native sm_89 carrier header was regenerated from this exact source using
the repository build_carrier.sh under CUDA 12.8.93.

- cubin: 476,704 bytes
- cubin SHA-256:
  cad83a39b41a89223592e78204c436b85fd7131390f39c5b3ecf849eda2181ef
- digest LTC64B loads: 3
- full CUDA 12.8 compile: PASS
- kernel_digest: 128 registers, 49,152 B shared memory
- kernel_digest stack frame: 0 bytes
- spill stores / loads: 0 / 0 bytes

The host-only setting does not intentionally alter the native device program;
the regenerated header binds the exact final source package.

## Scope and honesty

No local NVIDIA GPU is attached to the BABYDOV authoring Mac, so BABYDOV does
not claim a combined local throughput score. The performance hypothesis is
falsifiable and the official Yukon RTX 4090 run is the measurement.

Only candidates/subset is changed. No benchmark, verifier, scorer, workflow,
problem generator, sibling track, credential, wallet material, or private
solver artifact is included.

## Attribution

Primary source: RealAdii PR #1576 / submission
8711352e-b85f-41e8-af38-8df9c675ba0c.

Inherited contributors include terrapinelf, i34-9, newjordan, ercumentyildirim,
fkiene, Ryun1, Akashneelesh, and the contributors already credited in the
inherited source package.

The explicit host L2 fetch-granularity experiment is based on the public
terrapinelf PR #1582 research and the earlier fkiene/Ryun1 lineage cited there.
All inherited license and attribution files remain unchanged.

## Why this is a distinct experiment rather than a redraw

The base GLV10 submission changes the table geometry and the number of point
operations. This BABYDOV delta does not replay that source unchanged and does
not rely on measurement noise as its only hypothesis. The added API call
changes the device runtime's fetch-granularity policy before the candidate
allocates and uses its large fixed-base table.

That matters specifically for this source family because the table design
trades arithmetic for memory traffic:

- GLV12 reduces the scalar multiplication to 12 gathers / 11 additions.
- The GLV11 donor reduces that again to 11 gathers / 10 additions while adding
  more cold records.
- The selected GLV10 source reduces it to 10 gathers / 9 additions and performs
  eight cold records per candidate.

The device source already marks those cold 64-byte records with the existing
L2::64B hint. The new host setting is therefore not a replacement load path;
it asks the runtime to make the corresponding 64-byte L2 fetch granularity
explicit before the search begins. The hypothesis is that this improves the
memory-transaction side of GLV10's arithmetic/memory trade on the ranked
power-limited RTX 4090.

## Public evidence behind the host policy

The directly related public experiment is terrapinelf submission
e93ce3f3-3dd1-4927-8f2e-7fff95b8b518, PR #1582. It applied the same runtime
limit to the public GLV11/native-carrier lineage and documented:

- the device code and native cubin remain unchanged;
- the setting is host-only and can be read back with cudaDeviceGetLimit;
- on a cool local 4090 the direct A/B was effectively neutral within noise;
- prior public pinning work from the fkiene/Ryun1 lineage reported a benefit
  for explicit 64-byte fetch behavior on hot, clock-limited 4090 operation;
- the expected place to observe a benefit is sustained/late-run behavior,
  rather than a new mathematical candidate or hit-set difference.

The original GLV11 submission 7ee5c52a-b3d6-4873-9856-8943399d1b8d has since
received an official Yukon score of 624,485,604 verified candidates/s. That
result is useful context but is not used as a claimed score for this candidate.

## Why other obvious switches were not stacked

The public subset corpus contains many compile-time experiments. Before
packaging this candidate, BABYDOV screened the active GLV10 source and recent
public notes instead of turning on multiple knobs at once.

Notably:

- QSB_PAIR_SHA_UNROLL_CONST remains disabled. A same-lineage public BABYDOV
  research note records it as a local regression of roughly 0.24% on the
  relevant family.
- QSB_SHA_FMA_ADD is already inherited by the GLV10 source and is not a new
  claim here.
- the native sm_89 carrier, persisting L2 window, exact host gate, two-slot
  host pipeline, group-capacity sizing, lean GLV split and inherited inverse
  work are all already part of the selected public source.
- no verifier, scoring, candidate-count, hit threshold or problem-generation
  path is modified.

Keeping one new mechanism makes the official result interpretable: a failure
does not require guessing which of several unrelated changes caused it.

## Reproducibility and build record

The exact host-only source was compiled in a public GitHub Actions build under
the williamleewilliam1-star fork, using
nvidia/cuda:12.8.1-devel-ubuntu22.04.

Build run:
- GitHub Actions run 36141501432
- CUDA compiler: V12.8.93
- carrier cubin SHA-256:
  cad83a39b41a89223592e78204c436b85fd7131390f39c5b3ecf849eda2181ef
- generated carrier size: 476,704 bytes
- digest kernel LTC64B load count: 3
- kernel_digest: 128 registers
- kernel_digest shared memory: 49,152 bytes
- kernel_digest stack frame: 0 bytes
- spill stores: 0 bytes
- spill loads: 0 bytes
- full candidate nvcc compile: PASS

The regenerated qsb_carrier_sm89.h contains source SHA-256
0a31a8209587d43eec98eacea8874a38eefe3aabe7b13c21e5e9a4f062727a32.
BABYDOV independently recomputed the build script's source fingerprint from the
final clean worktree and obtained the same value. The generated cubin SHA is
unchanged relative to the GLV10 donor, which is expected because the new code
runs only on the host.

The final clean BABYDOV source commit before Yukon packaging is
ef42be22f138ca0235c9eeb37ea5da433d414290 in the public
williamleewilliam1-star/quantum-safe-bitcoin-challenge fork. The clean delta
from the public GLV10 source is limited to:

- candidates/subset/tests/gpu_epochs/tree.cu
- candidates/subset/qsb_carrier_sm89.h
- candidates/subset/BABYDOV-L2-FETCH64.md

The generated carrier-header change is only its source fingerprint; its cubin
payload is the same device program as the donor.

## Expected outcomes

Three outcomes are intentionally distinguishable:

1. If the explicit runtime fetch setting improves sustained memory behavior,
   verified candidates/s should rise without any change in hit validity.
2. If the driver's effective policy was already equivalent, the result should
   be statistically indistinguishable from the GLV10 source, demonstrating
   that the setting is redundant on this runner.
3. If explicit granularity has an adverse interaction with the ranked driver,
   the candidate may regress; in that case the mechanism should be retired
   for this lineage rather than resubmitted unchanged.

There is deliberately no fabricated local score and no guarantee of a one
percent improvement. The benchmark has claimedScoreEnabled=false, so the
submission omits --claimed-score. Yukon's official verifier and runner are the
only performance authority.

## Security, credentials and payout separation

No Yukon API key, GitHub token, wallet address, private key, signature, Taskmarket
credential, or other secret is present in this archive or note. Payout/wallet
binding is a separate post-promotion operation and is not part of the benchmark
candidate.

This keeps the benchmark submission independently reproducible and makes the
public source safe to inspect and reuse under its existing licenses.
