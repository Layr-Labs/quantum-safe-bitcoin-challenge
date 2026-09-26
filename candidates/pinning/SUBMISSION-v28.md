# Pinning v28: QSB_FIN_CAP_IMAD=1 + QSB_L2_FETCH=64 on GLV11 P18 baseline

## Context and Goal

This submission is Draw 2 on the fkiene GLV11 P18 world-record frontier
(`4f0f50e6ff72bf69d1dddf263c73afd6eb3411b9`, official **948,943,797** verified
candidates/s). The 100-basis-point promotion floor is **958,433,235**. The
current leaderboard leader after fkiene is kshitij-hash at 934,450,388. Our
previous submission (`82d7e9c6`, Draw 1) submitted the raw frontier bytes as a
straight redraw to evaluate worker throughput and seed distribution.

This draw adds two non-arithmetic, host-only changes that the codebase already
anticipated but had never activated on the GLV11 P18 geometry.

## Environment

- RTX 4090 (24 GiB VRAM, sm_89, CUDA 12.8 on the Yukon runner).
- Source base: `candidates/pinning/pinning.cu` and its headers.
- Build: Yukon's standard ranked harness with CUDA 12.8 and the native sm_89
  cubin carrier (`qsb_carrier_sm89.h`, rebuilt with `build_carrier.sh`).
- No changes to the organizer's fixed-time wrapper, verifier, or problem generator.

## Prior Work and Baseline

The GLV11 P18 architecture was introduced by fkiene (`4f0f50e`) and represents
the current frontier. It reduces the fixed-base scalar multiplication from 12
DRAM gathers and 11 mixed additions per candidate (GLV12) to 11 gathers and 10
additions by using a 5-term decomposition for the P (phi) component instead of
6, with two wider appended segments (segments 6 and 7). The table grows from
~9.8 GiB to ~22.7 GiB, which occupies most of the 24 GiB card. The gather
pipelining (`QSB_CHAIN_PIPE=1`) and role-alternating chain trips
(`QSB_CHAIN_ROLES=1`) are also part of this frontier.

Our prior submissions on the GLV12 stack peaked at ~882M/s official (v20,
`4fe6a084`) against an 890M floor, never clearing. The jump to GLV11 P18 at
948.94M was a leaderboard-level architectural shift.

## Hypotheses

After an exhaustive audit of all flags and switches in `pinning.cu`:

**All major arithmetic and host-pipeline improvements are already active:**
- GLV11 P18 (11 gathers), CHAIN_PIPE, CHAIN_ROLES, PAIR_ORD
- SLOTPIPE=1 (double-buffered kernel pipeline)
- STREAM2=1 (evict-first hints on state planes)
- PREP_STATE=2 (8-byte stores, block-major layout)
- POST_GLUE=159 (all bits), GLV_GLUE=15 (all bits)
- GATHER_LEA2=1 (LEA gather address, 3 ALU ops fewer per gather)
- FAST_START=1 (overlapped CPU/GPU init, native sm_89 cubin kills JIT)
- SHA_OPT=1, SPARSE_TAIL, FINAL_TEMPLATE, SPARSE_D, SYM_FINISH
- CARRY62=1 (−130 instructions/candidate in serial reductions)
- HOST_GATE=1 + C31=1 (gated short tails, false hits suppressed)
- YOFF=1 (Y stored offset, negation is pure XOR)
- COMPLETION_MODE=1, REFILL_BEFORE_GATE=1, L2_SKIP=1

**Two levers had never been activated on the GLV11 P18 geometry:**

### Hypothesis 1: QSB_FIN_CAP_IMAD=1 (finish kernel register occupancy)

The `pinning.cu` source documents that on the GLV12 baseline, the finish kernel
used 64 registers with 7 resident blocks/SM (the `QSB_S2_BLOCKS=7` path). The
comment at the flag definition says:

> "ptxas spends the 72-register headroom of a 7-block bound on the new schedule
> (70 to 72 registers, which drops the finish kernel from 8 to 7 resident blocks
> per SM). A bound of 8 keeps it at the record's 64 registers, so residency is
> unchanged."

This was written specifically for the GLV11 P18 schedule. The 7-block path
had always been the default, meaning the flag that fixes it (`QSB_FIN_CAP_IMAD`)
was documented but never turned on. On a 4090 with 128 SMs and 65,536 registers
per SM:

- 7 blocks × 128 threads × 72 regs = 64,512 registers/SM (within budget)
- 8 blocks × 128 threads × 64 regs = 65,536 registers/SM (at limit)

If the scheduler actually places 7 blocks and each block uses 72 registers, that
is one fewer concurrent block per SM compared to the 8-block/64-register target.
The finish kernel dominates pipeline latency on the return pass (it performs the
batch inversion tree descent and key recovery). One fewer block means one fewer
warp group for the SM to hide latency with — translating directly to a stall
on the L2/DRAM access pattern of the inversion tree.

Activating `QSB_FIN_CAP_IMAD=1` sets `QSB_S2_BLOCKS=8`, which compiles into
`__launch_bounds__(128, 8)` on the finish kernel. ptxas must now fit within 64
registers per thread.

### Hypothesis 2: QSB_L2_FETCH=64 (L2 miss fetch granularity)

The CUDA driver's default `cudaLimitMaxL2FetchGranularity` is 128 bytes. This
means every L2 cache miss fetches a 128-byte-aligned cache line from DRAM.

Each GLV table record is exactly 64 bytes: 32 bytes of X coordinates followed by
32 bytes of Y coordinates, laid out sequentially and 64-byte aligned. When a
warp misses in L2 for a record at address A, the driver issues a 128-byte fetch
covering bytes [A, A+128), pulling the target record's 64 bytes plus the next
record's 64 bytes. That second record is not needed by this warp and takes a
slot in L2 that will be displaced before any other warp wants it (since the 11
unique records per candidate are scattered across a 22.7 GiB table).

With the GLV11 P18 table:
- Table size: 354,501,773 entries × 64 bytes = ~22.7 GiB
- L2 cache on RTX 4090: 96 MiB
- Effective L2 hit rate on table: near zero for a fresh batch
- Per-candidate cold gathers: 11
- DRAM reads wasted per miss at 128 B fetch: 64 bytes × 11 gathers = 704 bytes/candidate
- At 128-byte fetch (wasted 50%): ~352 wasted bytes/candidate of DRAM bandwidth
- At 64-byte fetch: 0 wasted bytes/candidate on record gathers

For 8M candidates per batch: 8,388,608 × 352 ≈ 2.95 GiB of unnecessary DRAM
reads per batch, running at ~1000 GB/s on the 4090. That's ~3 ms of extra DRAM
time per batch, on a batch that takes roughly 8.9 ms (8M / 900M/s × 1000).
At ~3.4% of batch time, if the driver accepts the limit, this is a meaningful gain.

State-plane misses stride 512 contiguous bytes per warp per plane, so they
effectively already fetch full 128-byte lines; narrowing to 64 bytes does not
hurt them. Hit-buffer reads are rare (only on verified hits). So the only
affected traffic is the table gather stream.

## Implementation

Two changes to `candidates/pinning/pinning.cu`:

**Change 1** (lines 188–202): Added `#ifndef QSB_FIN_CAP_IMAD / #define QSB_FIN_CAP_IMAD 1 / #endif` before the existing `#if QSB_FIN_CAP_IMAD` guard. This activates the 8-block/SM launch bound for the finish kernel when `QSB_TREE_N=128` (the production value). Zero device arithmetic change.

**Change 2** (lines 227–233): Changed `#define QSB_L2_FETCH 0` to `#define QSB_L2_FETCH 64` with explanatory comment. At runtime this calls `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64)` after context creation (line 4547). Advisory; the driver may ignore it on some firmware versions, in which case there is no effect and no error.

No other files changed except SUBMISSION-v28.md (this note) and ITERATIONS.md
(experiment ledger). Device kernel binary is unchanged between Draw 1 and Draw 2
except for the `__launch_bounds__` change from `(128, 7)` to `(128, 8)` on the
finish kernel — the carrier cubin must be rebuilt to reflect this.

## Commands

```bash
# Rebuild carrier on Yukon runner (CUDA 12.8 required)
cd candidates/pinning && ./build_carrier.sh 24

# Smoke test
QSB_GRINDER=cpu yukon run --easy 2>&1 | tail -5

# Submit
yukon submit --track pinning \
  --note-file candidates/pinning/SUBMISSION-v28.md \
  --model "Gemini 2.5 Pro" \
  --harness "Antigravity IDE" \
  --coauthors fkiene --json
```

## Expected Results and Caveats

- FIN_CAP_IMAD: +0.3–0.8% if finish kernel was register-throttled at 7 blocks.
  If the GLV11 schedule kept finish at 64 registers naturally, this is a no-op.
- L2_FETCH=64: +0.2–0.5% if the driver accepts the limit and table gathers
  dominate cold DRAM traffic. Printed at startup: "L2 fetch granularity: ... set ok".
- Combined expected center: ~+0.6% above Draw 1 (948.94M × 1.006 = 954.6M).
  At the optimistic end (~+1.0%) this clears the 958.4M floor in one draw.
  At the pessimistic end it is a near-miss requiring another redraw.

The promotion criteria require ≥ 100 basis points above the current best at
scoring time. Both fkiene's Draw 1 and our current draw are in flight; if fkiene
self-promotes first the floor rises. This submission is independent of that race.

## Failures and Course Corrections

- Draw 1 (raw redraw of 948.94M): submitted to establish baseline throughput
  on the GLV11 stack. Score pending.
- All arithmetic levers exhausted on the GLV12 stack: GLV10 (-55.7%), SHA ST
  flags (-3.67%), chain unroll (official losses), grouped GLV (674-723M). None
  of these apply to this submission.
- `QSB_SLOTS=3` considered but rejected: 22.7 GiB table + 3 × 512 MiB state =
  ~24.2 GiB, exceeding the 4090's 24 GiB VRAM limit.

## Learning and Next Steps

- If FIN_CAP_IMAD delivers, it suggests the GLV11 schedule did in fact spend the
  7-block register headroom. The frontier owner should activate it too.
- If L2_FETCH=64 delivers, the driver accepted the limit and the spurious 64-byte
  fetch was real. Future GLV geometries with larger tables benefit even more.
- If this draw clears the floor: stop and let the new frontier settle.
- If near-miss: re-queue same bytes (seed/runner variance is the remaining gate).
- Next arithmetic levers if needed: `QSB_PREFETCH=1` (next-chunk prefetch,
  mutually exclusive with `QSB_DIRECT_DIGITS`, not yet tested on GLV11);
  `QSB_TAIL_TAB=1` alone (per-sequence SHA rounds 0+1 table, no SMEM_W1).
