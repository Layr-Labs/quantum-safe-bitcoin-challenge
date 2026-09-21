# Pinning: exact chain schedule + merged TOP16

Effort: xhigh. No local NVIDIA device; the official 1,200 s RTX 4090 run is
the throughput measurement. Local evidence is host algebra plus CUDA 12.6
`sm_89` compile/SASS inspection.

## Base and target

Base: promoted commit `94abdd0`, submission `07009ac3`, official
**797,446,582/s**. Promotion floor: **805,421,048/s**.

Our prior `7c159de9` was rejected at **791,873,384/s** (113,407 verified hits,
94.40 hits/s). It combined this chain schedule with a large fused deferred
point-add. The fused body is completely absent here; this branch stops before
it at `44b3c32` and adds only independently measured TOP16.

The base's PR827 field arithmetic, parity window, `xR=+-1` isomorphism, Y-offset
table, OpenSSL publication gate, table/SHA/launch geometry, and recovery
equations are unchanged.

## Measured components

1. fkiene submission `f7e4ddef-a698-4127-b1fa-b6ee257637da`, commit
   `344cd8fa`, scored **792,667,656/s**. Its note identifies the nearest public
   measured predecessor at **791,077,271/s** and introduces the two-buffer
   chain, paired digit reads, and plane addressing: ratio **1.0020104**
   (+0.2010%). This conservative ratio is used instead of the +0.4634%
   crown-to-crown ratio because that donor archive also carried arithmetic.
2. ercumentyildirim submission `8740a30d-3674-47e4-a0ed-7ad3e98561e2`,
   commit `caf7f8c0`, scored **804,598,773/s** directly on `94abdd0`
   (+0.8969%). Its public note says the only production delta was
   `cofactor_checkpoint.h`. The wave schedule originated in @EvanYan1024's
   public submission `58005ee5`.

They run in disjoint phases: fixed-base work per candidate versus cofactor work
once per block. Conservative multiplicative model:

```
804,598,773 * (792,667,656 / 791,077,271) = 806,216,342/s
promotion floor                               805,421,048/s
modeled margin                                    795,294/s
modeled gain from 797,446,582                       1.0997%
```

This is a projection, not a local benchmark. The composition assumption is
only that the chain saving survives the separately scheduled cofactor top.

## Exact chain schedule

`pinning.cu` defaults:

```
QSB_DIGIT_WINDOW32=1   QSB_DIGIT_SIGN_FOLD=1
QSB_DIGIT_SEED_REG=1  QSB_CHAIN_ROT2=1
QSB_DIGIT_PAIRLDS=1   QSB_CHAIN_PTR=0
```

The scalar uses 32-bit funnel windows. Seed chunks 0/1 stay in registers.
Remaining codes are arranged so each pair is one aligned 64-bit shared read.
Chunk 2 is peeled; `(3,4)` through `(13,14)` rotate two affine buffers, removing
the four-limb Y-anchor copy after every mixed add. The same thirteen
`_PointAddXYZZT<true>` calls receive the same table points, signs and anchors.
`QSB_CHAIN_PTR=0` is deliberate: pointer addressing spills on this 128-register
base; indexed planes do not. No field identity or candidate semantics change.

## Merged TOP16

The ordinary cofactor top takes seven dependent warp waves: up-sweep activity
8/4/2/1, then exclusion activity 4/8/16. `QSB_TOP16=1` combines them into four
waves: spare lanes compute the next parents while other lanes extend exclusion
products.

For tree size `N`, retained indices are:

```
x=2*N-32  p2=2*N-16  p4=2*N-8  p8=2*N-4  e=N-32
```

The ordinary up-sweep stops at 16. Four warp-synchronous multiplies publish
`p2`, `p4`, `p8`, the root, and sixteen top exclusions. The down-sweep resumes
at `count=32`, `offset=2*N-64`. Each exclusion still contains exactly the other
fifteen leaves. `QSB_TOP16_SC=1` keeps the base short-carry multiplier; the
exact host gate re-derives every reported hit.

Kill switch: `-DQSB_TOP16=0` restores the promoted traversal while retaining
the chain schedule. Every chain switch is also independently guarded.

## Static and correctness evidence

Candidate and `-DQSB_TOP16=0` control both compile in `qsb-build:latest` for
`sm_89`, `-O3 -DQSB_ZEROS_N=24`:

| stage 0 | control | + TOP16 |
|---|---:|---:|
| registers/thread | 128 | 128 |
| static shared | 12,288 B | 12,288 B |
| stack | 0 B | 0 B |
| spill stores/loads | 0/0 | 0/0 |
| static SASS instructions | 7,800 | 8,216 |

Stage 2 is 66 registers, zero stack/spills. The +416 static instructions are
the once-per-block merged top; runtime removes three serial dependent waves.
`BAR`, `LDG`, `SHF`, occupancy and the per-candidate chain are unchanged by the
TOP16 toggle.

Passing host checks:

```
python3 candidates/pinning/test_chain_schedule.py
python3 candidates/pinning/test_top16_compose.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_carry62.py
python3 candidates/pinning/test_sha_interleave.py
```

Chain audit: 3,000,000 digit comparisons against the inherited extraction,
sign/index packing, all 1,920 arena slots, and switch liveness. TOP16 audit:
5,000 random sixteen-leaf vectors; merged root equals the ordinary product and
all sixteen exclusions equal the product of the other fifteen leaves. The host
gate checks both recovery IDs and benchmark SHA; carry and SHA reference tests
also pass. These checks establish semantics, not speed.

## Files and credit

Production deltas from `94abdd0`:

- `pinning.cu`: exact chain schedule
- `cofactor_checkpoint.h`: merged TOP16

Tests are not part of the ranked binary. `SOURCE-MANIFEST.json` records all
production hashes. Ranked build:

```
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

Credit: fixed-base schedule by fkiene (`344cd8fa`); measured TOP16 port by
ercumentyildirim (`caf7f8c0`); original TOP16 schedule by @EvanYan1024
(`58005ee5`). The promoted base and inherited notices remain. This archive
keeps the measured 804.60M near-winner, removes the rejected fused point-add,
and adds only the disjoint exact-chain scheduling cut.
