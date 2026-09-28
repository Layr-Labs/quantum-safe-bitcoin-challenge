# v23 — GLV10: five chunks per component (−2 point-adds, −2 DRAM loads)

## What this package is

The corrected v22 init stack + GLV10 geometry. Each GLV component decomposes
into 5 table terms instead of 6 → **10 serial point-adds per candidate
instead of 12**, and 10 random 64-byte table reads instead of 12. On the
measured workload (~72% of SASS in point-adds, table reads streamed from
DRAM) this is the largest remaining structural cut: −16.7% of the serial
add chain, −16.7% of table traffic.

## Geometry (candidate A — minimum-table)

- Shifts `[0,24,50,76,102]`; widths {24 raw, 26,26,26 signed windows}, plus
  bounded top.
- Top bound **T = 42,639,943** (the proven split bound `BOUND>>102 =
  42,639,942` — margin exactly one).
- Bias `K = 42,639,944·2^101 − 2^23` — telescopes the three 26-bit window
  corrections `(2^26−1)(2^23+2^49+2^75) = 2^101−2^23` plus `T·2^101`.
- Entries: 2^24 + 3·2^25 + 21,319,972 = **138,760,484 records =
  8,880,670,976 bytes (~8.27 GiB)** — fits the RTX 4090's 24 GiB with the
  ~1.5 GiB pipeline buffers.
- `GT_HI` 4096→16384: the 26-bit odd-multiple range needs `hi = m>>12`
  up to 16383. The `m>>12`/`m&4095` ladder split is unchanged.
- Physical order is linear `[0,1,2,3,4]` — no segment fits L2, so the
  GLV12 dense-first permutation buys nothing.

## Verification

- Recode oracle: 400,052 magnitudes × 2 signs re-derived exactly against the
  GLV10 digit semantics; telescoping bias verified to cancel identically.
- Three clean nvcc builds (GLV10 on/off + non-BIGTBL), hot kernel 106 regs /
  0 spills; hot .text shrank 384 B (the two removed decode terms).
- Geometry stamp `qsb_geom_stamp = 0xA10` baked into both the runtime image
  and the shipped sm_89 cubin; `cubin_init` refuses a mismatched module, so a
  stale-geometry cubin can never lay out the wrong table.

## What changed vs v22c

v22c scored (838.79M official, 120,089 verified hits) after the
`qsb_table_offset_y` restoration. Two deltas on top:

1. **GLV10 enabled** (`QSB_GLV10 1`): the table/kernels move to the 10-term
   geometry above. Table build is ~9–18 s of GPU work on the private stream,
   still overlapped with init; sampled spot-check (mandatory at this size)
   unchanged.
2. **`cudaMallocAsync` reverted to blocking `cudaMalloc`** for the pipeline
   state planes: pooled memory was the one remaining suspected steady-state
   drag in v22c's numbers, and the ~0.3 s init saving never justified it.

## Failure history this stack already survived

- v22/v22b both failed Benchmark after full windows — root cause was the
  dropped `qsb_table_offset_y` launch (raw ordinates → zero verifiable hits),
  fixed in v22c and retained here.
- v22b added the end-of-window hardening (SIGTERM drain re-raises with
  SIG_DFL, SA_RESTART, watchdog, term-checked fatal paths, bounded host
  fallback, cubin build relaunch) — all still in place.

## Risk analysis

- **Table size**: 8.88 GiB leaves ~13 GiB of device memory headroom next to
  the pipeline planes — no pressure. The per-chunk ladders grow (`GT_HI`
  16384 high entries) but stay threaded and overlapped.
- **Build latency**: the ~9–18 s GPU build runs on `gt_stream` under the
  whole remaining init, so the phase-2 event sync adds ~0–5 s of wait —
  far less than the serial build plus full readback v20 paid.
- **Fallback path**: if the sampled spot check rejects the table, the cubin
  build gets one runtime relaunch, then the OpenSSL host builder — bounded
  by the window-signal check so it can no longer hang past the timeout.
- **Memory behaviour**: blocking `cudaMalloc` restores the exact v20 memory
  profile for the streaming planes; the 8.88 GiB table itself has always
  been a plain `cudaMalloc`.
- **What is deliberately unchanged**: the hit path, the gate, the slot
  pipeline choreography, and every launch signature are byte-identical to
  the configuration that produced v20's verified 882.10M run (modulo the
  geometry flag and the two restored/removed calls documented above).

## Expected effect

GLV14→GLV12 removed two terms for +6.6%. GLV12→GLV10 removes two more on a
larger base; conservatively +6–7% self-rate (~960M on a fast worker, ~925M
on a mid draw) → ~940M+ official at a ~0.98 yield — comfortably over the
890.09M floor on either worker class, and within reach of the ~925M goal
the field is converging on.

## How to read the result

The artifact's hit-derived candidate count (`verified_hits × 2^24 / 2`)
divided by the harness wall clock is the only scored quantity; the binary's
own `searched` figure is advisory. Because GLV10 trades DRAM traffic for a
bigger table, the score is dominated by the term-loop speedup rather than
init time — the expected signature is a materially higher verified-hits
count for the same 1200 s window, i.e. throughput well above the 882M the
12-term version measured on this runner.

## Provenance

v22c init-cut stack (native sm_89 cubin, two-phase table build, sampled
readback, threaded ladders, blocking allocs, SIGTERM drain, restored
ordinate-offset pass) + the promoted GLV12 union + our LEAN/SFC2 arithmetic.
GLV10 geometry derived internally; algebra verified by the recode oracle.
