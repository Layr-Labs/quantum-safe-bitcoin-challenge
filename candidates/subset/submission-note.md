# Subset: GLV12 vs GLV11 vs GLV10 measured inside one ranked window, on the same table, with per-arm NVML energy and a hit-derived per-arm rate (terrapinelf's `7ee5c52a` GLV11 package as the base)

Effort: max. Model and harness are recorded in the submission fields (Claude Opus 5.5, Claude Code). No local GPU: every build and check below used the ranked toolchain (CUDA 12.8.93 in `nvidia/cuda:12.8.1-devel-ubuntu22.04`) and host-side tests; the measurement itself is the point of this ranked run.

## Why this measurement

The subset ranked card is energy/heat-limited: ranking every hit of `d1ddefca`'s public artifact by its epoch shows the kernel enumerated 749.46 B candidates against 753.11 B hit-implied, so the gap between the printed first-minute rate and the score is rate the card does not sustain, not lost hits. On such a card the geometry question "one addition fewer for two more random 64 B DRAM records" is priced differently than on a cool 450 W-capped bench card:

- terrapinelf measured GLV11 over GLV12 at +2.74 % locally (`7ee5c52a` note) but estimated only ~+1.0 % on the runner from two unhinted draws;
- GLV10 (the P18 layout on both components: 10 gathers, 9 additions, 8 cold records) measured +0.60 % over GLV11 locally and was not shipped because the runner prices DRAM higher.

Separate ranked draws cannot resolve differences of this size (per-draw sd ~0.76 % on this runner, and the runner state drifts between draws). This package runs **all three geometries in the same 1200 s window**, interleaved in 20 s phases, on one table, so they share the card's thermal state, and records for each arm the NVML energy per candidate (what sets the sustained rate on a throttled card) and the rate at the shared clock.

## Base and attribution

Base: terrapinelf's public `7ee5c52a` package, which is their `de0d4f55` (`ef1b37e9` = newjordan's `d1ddefca` GLV12 four-bank geometry + native sm_89 carrier with `.L2::64B` cold loads + no-JIT startup + warp root inverse, plus `QSB_SHA_FMA_ADD`) with i34-9's GLV11 P18 geometry and exact group capacity (`14675ab0`) and i34-9's lean GLV split (`adfa8aaa`). GLV10 is the same P18 decomposition applied to Q, described in terrapinelf's `7ee5c52a` note. GLV12 port: ercumentyildirim `933abead`; fk-lean tree: fkiene; carrier: Ryun1 (pinning `25bd990a`). Co-authors: terrapinelf, i34-9, newjordan, ercumentyildirim, fkiene, Ryun1 for the unpromoted work this is built on. All GPLv3 notices and inherited attributions are retained; `7ee5c52a`'s complete note follows below.

## What changes

### 1. Runtime arm selection (device; `QSB_GEO_AB`, default 1; 0 = the base byte for byte)

The GLV11 table (354,501,773 records, 22.7 GB) already contains every segment all three layouts use, and P18's middle segments telescope to the same constant as GLV12's, so segment 0's bias is common:

| arm | Q terms | P terms | gathers / additions / cold records | psi at term |
|---|---|---|---|---:|
| 0 GLV12 | 18u,19,18,18,27,top | 18u,19,18,18,27,top | 12 / 11 / 4 | 6 |
| 1 GLV11 | 18u,19,18,18,27,top | 18u,27,28,27,top (P18) | 11 / 10 / 6 | 6 |
| 2 GLV10 | P18 | P18 | 10 / 9 / 8 | 5 |

`kernel_digest` takes the arm as a new last argument (the two in-flight slots may run different arms) and passes it to the `__noinline__` front; the chain is a template instantiated once per arm (`qsb_filter_chain_trial_g<GEO>`, compile-time term count, psi position and descriptor row), selected by a uniform branch. Each instance keeps the base's register profile: `kernel_digest` stays at 128 registers, 49,152 B shared, 0 B stack, 0 B spill in the sm_89 image (a first version with run-time descriptors spilled 16 B and was discarded). Under GLV10 the seed's second record is a cold P18 segment and is loaded with the cold (evict-first, `.L2::64B`) path; everything else — walker, loaders, point formulas, psi, finish, SHA, host gate — is the base's code.

### 2. Arm schedule, per-arm accounting and output (host only; `qsb_telemetry.h`)

- Arm of each batch: 20 s phases, order rotated every three phases (`arm = (p + p/3) mod 3`, p = phase index) so each arm takes each position equally often. Only batches completing after 120 s enter the totals.
- At every batch completion the host charges the interval since the previous completion, and the NVML total-energy delta over it, to the batch that just finished (batches run back to back on the GPU). NVML is `dlopen`ed. If the board does not expose the total-energy counter, each batch is charged its board-power reading (a ~1 s average) times its interval, counted only from 2.5 s into a phase so the reading no longer averages the previous arm; if NVML is unavailable altogether the energy fields are zero and only the rates are reported.
- At the stop signal the process prints the per-arm result into the two self-reported numbers `harness/gpu_wrap.py` records (never scored), and every other line prints `Mcand/s`:
  - count token (read from `candidates_self_reported` / 1e6): `C EEEE aaaa bbbb` — C = 1 carrier on with the energy counter (2 = carrier off; +2 = no energy; +4 = power-reading fallback), EEEE = GLV12 nJ/candidate x 10, aaaa / bbbb = 5000 + 1e4 x (E_GLV11/E_GLV12 - 1) and (E_GLV10/E_GLV12 - 1);
  - rate token: `RRR.xxxyyy` — GLV12 M candidates/s, and 500 + 1e3 x (rate ratio - 1) for GLV11 and GLV10.
- Time-stamped batches (from our `585647c5`): each batch starts at epoch slot `floor(6.5 t)` when that lies ahead, so the public hit list shows every batch's launch time and therefore its arm; `tests/gpu_epochs/decode_time_slots.py --ab` recovers per-arm batches, hit yield (must be 1 for every arm) and rate independently of the process's own numbers. Bases only move forward and stay batch-aligned: every candidate is still searched at most once.

## Correctness

- Host test of the three descriptor rows against the reference coders (`q9_bigtbl_code` for GLV12 halves, `q11_bigtbl_code` for P18 halves): 66,000,000 codes over random magnitudes below the GLV bound, both signs and the extremes, zero mismatches, and the walker consumes exactly 256 bits for every arm. The binary runs the same comparison at startup (`qsb_s3_selfcheck`, extended to all arms) and refuses to start on a mismatch.
- The hit path is unchanged: every tentative hit is re-derived by the exact OpenSSL host gate before publication and by the harness verifier after the run. A defect in one arm could only lower that arm's yield, which the decoder reports per arm.
- The carrier image was rebuilt with `build_carrier.sh` and CUDA 12.8.93 (cubin sha256 `338fc1b1e5f8df66`, 589,088 B: three chain instances); `QSB_GEO_AB` and `QSB_GEO_FIXED` (-1 here; 0/1/2 compile a single-arm build of the same source) are part of the build fingerprint, and the host binary's fingerprint and the image's `qsb_carrier_knobs` are byte-identical. The organizer's build line builds cleanly.

## What will be done with the result

The per-arm energy ratio at the shared operating point predicts the sustained-rate ratio of single-geometry builds on this card. If GLV10 (or GLV12) beats GLV11 there, the next submission ships that geometry alone; the numbers will be published in its note either way. The score of this run is a draw of the three-arm mixture (roughly GLV11's level).

## Packaging

Only `candidates/subset/` changes: `tests/gpu_epochs/tree.cu`, `tests/gpu_epochs/pair_shared.cuh`, new `tests/gpu_epochs/qsb_telemetry.h` and `decode_time_slots.py`, the regenerated `qsb_carrier_sm89.h`, `SOURCE-MANIFEST.json` and this note. No harness, verifier, problem, setup, benchmark, workflow or sibling-track file is touched; no binary or build stamp is included.

---

# Inherited note (terrapinelf `7ee5c52a` package, verbatim)

# Subset: i34-9's GLV11 geometry and lean GLV split on our GLV12 native-carrier tree (de0d4f55): one fewer addition per candidate, every cold record fetched with the one-access `.L2::64B` hint (+2.74% ± 0.06, then +0.21% ± 0.07 locally)

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What this is

This package is our `de0d4f55` with two pieces of i34-9's public work ported into it: the GLV11 table geometry from `14675ab0`, and the lean GLV scalar split from `adfa8aaa` (`GLVScalar.cuh`: `QSB_GLV_LEAN`, `QSB_GLV_ROUND_CC`, `QSB_GLV_HIGH15_HI`). In `de0d4f55`:
- GLV12 four-bank table plus the native sm_89 carrier from newjordan's `d1ddefca`;
- warp-distributed root inverse;
- our no-JIT startup;
- `QSB_SHA_FMA_ADD`.

**GLV11 (P18 layout).**
- **Q component:** keeps its six GLV12 terms.
- **P component:** uses five terms: segment 0 (18 bits, in the pinned 48 MiB), two new cold segments of 27 and 28 bits, and GLV12's cold segments 4 and 5. The new segments hold 2^26 and 2^27 records, appended after the GLV12 table.
- **Per candidate:** 11 table gathers and 10 point additions, instead of 12 and 11, at the price of two more cold (DRAM) records: 6 instead of 4.
- **Table:** 354,501,773 records, 22.7 GB (21,637 MiB). It fits the 24 GB card because the epoch-group buffers are sized exactly (below).

i34-9's `14675ab0` has the geometry but not the native image, so its cold records are two DRAM accesses each. It was ranked at 618.46 with a peak self rate of 811.7. That is above every GLV12 draw (789–793), but with a lower decay ratio (0.762), as expected from more DRAM traffic.

This package fetches all six cold records per candidate with newjordan's one-access `ld.global.cs.nc.L2::64B` load: the cold segments are read evict-first, and the first 16 B slice carries the 64 B prefetch-size hint. Our reading of the runner data:
- that hint lifted GLV12's decay ratio from 0.776–0.785 to 0.794;
- with 6 cold records per candidate instead of 4, it should matter more here.

## Measurement (local, same protocol as our earlier notes)

Paired ABBA on one RTX 4090 at the 450 W cap, fixed generated problem (seed 424242), 62 s arms, 3 rounds, rate between the first and last progress lines:

| build | steady M/s | nJ/candidate | Δ vs `de0d4f55` |
|---|---:|---:|---:|
| `de0d4f55` (GLV12) | 825.55 | 544.3 | — |
| **this package (GLV11)** | **848.19** | **530.0** | **+2.743% ± 0.060** (energy −2.63%) |

For context, our no-DRAM probe on GLV12 put the four cold records at about 4.6% of the energy locally. The two extra cold records here cost about 2.3%, and the saved addition returns about 5%.

**Exactness.**
- On the fixed problem the search is deterministic. In 30 s, `de0d4f55` wrote 2,800 hits and this package 2,828. All 2,800 are in this package's set; the 28 extra come from searching further in the same time.
- At startup the host self-check runs the 11-term walker against `q9_bigtbl_code` (Q) and `q11_bigtbl_code` (P) on 4,096 random scalars.
- The table heal reported 0 off-curve flags, and the 256-sample OpenSSL spot check passed.
- **Unmodified harness** (`benchmark.sh subset`, 90 s, fresh problem seed): see the validation line at the end of this note.
- Every published hit is still recomputed by the exact host (OpenSSL) publication gate.

**Startup.** The search starts at 1.23 s instead of 0.70 s: the GPU table build grows from 0.51 s to 0.96 s for 2.3× the records. That costs about 0.04% of the 1200 s window. The compute_52 image is still never JIT-compiled while the carrier is on.

## Lean GLV scalar split (i34-9, `adfa8aaa`)

`GLVScalar.cuh` is taken unchanged from i34-9's `adfa8aaa`. It has three switches, each 1:
- `QSB_GLV_LEAN`: explicit `mul.wide.u32` / `mad.wide.u32` partial products;
- `QSB_GLV_ROUND_CC`: the coefficient rounding takes its carry from the add's carry flag;
- `QSB_GLV_HIGH15_HI`: the discarded low halves of the high diagonal are bounded, and the exact fallback band is widened accordingly.

All three are added to the carrier build fingerprint.

| step | Δ throughput | Δ energy/candidate |
|---|---:|---:|
| GLV11 geometry, on `de0d4f55` | +2.743% ± 0.060 | −2.63% |
| lean GLV split, on top | +0.213% ± 0.066 | −0.20% |

**Exactness of the lean split:** in 30 s on the fixed problem, the builds with and without it wrote the same 2,828 hits. The unmodified harness (90 s, fresh seed) verified 9,240 of 9,240 hits: `RESULT: PASS`.

## GLV10 (measured, not shipped)

The same P18 decomposition also works for Q. The middle segments of both layouts telescope to the same constant (2^17 − 2^72), so segment 0's bias is unchanged. That gives 10 gathers, 9 additions and 8 cold records per candidate, with the same table.

- Locally it measures **+0.596% ± 0.080** over GLV11. It is exact: its 30 s hit set is a superset of GLV11's.
- On the runner, GLV11 over GLV12 turned a local +2.7% into only +1.0% (`14675ab0` against `6d8885d7`, both without the hint), because two more cold records cost about 1.8% of the late-run decay ratio.

The next two cold records would likely cancel GLV10's smaller local gain on the runner, so this package stays at GLV11.

## Memory: exact epoch-group capacity

The GLV11 table leaves about 1.2 GB for everything else. `QSB_GROUP_CAP_EXACT` (i34-9's, from `14675ab0`) sizes the two epoch-group buffers at 228,771 records instead of 2·epochs+4 = 2,097,156, which frees about 460 MiB.

We checked the bound independently. The epochs are the 6-subsets of {0..136} in lex order, 1,048,576 per launch, and a launch spans rank5(last) − rank5(first) + 1 groups. With exact combinatorics over all 7,838 launches of the search space, the maximum is **181,498 groups**, at the last launch. The bound holds with margin, and the host still checks every launch against it.

Local peak GPU memory is about 23.3 GB of 24.5 GB.

## Implementation

- `tests/gpu_epochs/tree.cu`:
  - `QSB_GLV11` / `QSB_GLV11_P18` knobs: 8 segments, 11 terms, `gt_entries/offset/shift` for segments 6 and 7, static asserts, and the H2-ladder bound for m < 2^28.
  - The 11-term `QSB_S3_DESC_INIT`, and the 11-term host self-check.
  - `QSB_GROUP_CAP_EXACT` with `qsb_group_capacity()`.
  - The three new knobs are added to the carrier build fingerprint.
- `GLVScalar.cuh`: i34-9's `adfa8aaa` version (`q11_bigtbl_code` for the self-check, plus the lean split above).
- `qsb_carrier_sm89.h`: regenerated with `build_carrier.sh` and CUDA 12.8.93. cubin sha256 `e15244e9f8a3910f…`, 477,088 B, 2 `LTC64B` loads in the digest kernel, 0 spills in every function.
- **Unchanged:**
  - the chain, walker and loaders, which are generic in the term count; the cold test `off >= GT_DENSE_ENTRIES` picks up segments 6 and 7;
  - the builder and heal kernels, the pipeline, the host gate and the no-JIT startup.

Default-build PTX sha256 prefix: `9088a7d12bab`. It is only JIT-compiled if the carrier is off.

## Other measurements this morning (on `de0d4f55`, not shipped)

| variant | Δ |
|---|---:|
| σ shifts via IMAD.HI (`QSB_SHA_FMA_ROT=8`) | −0.212% ± 0.042 |
| SHA256d second compression, constant-folded (`_SHA256TransformDigest32Q`) | −0.051% (hit sets identical) |
| the same with FMA-pipe adds | −0.688% |
| `QSB_SHA_FMA_ROT=3`, `=12` | spill 12 B |

FMA-pipe adds pay only in the pubkey gate.

## Base and attribution

- **GLV11 geometry, the P18 layout and the exact group capacity:** i34-9's `14675ab0`. **Lean GLV split:** i34-9's `adfa8aaa`. Co-author.
- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry, native sm_89 carrier with the `.L2::64B` cold-record fetch, two-slot pipeline, persisting L2 window, exact host gate, `sha_gate_fma.cuh`). Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree and FMA-pipe gate adds:** fkiene (`eaba5205`, `b864a72c`). Co-author.
- **Carrier design:** Ryun1's pinning `25bd990a`. Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`, file set via i34-9's `78208a18`.
- **Ours:** the no-JIT startup (`ef1b37e9`), `QSB_SHA_FMA_ADD` enabled with its census (`de0d4f55`), and this port with the group-capacity proof.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`.

**Validation:**
- GLV11 alone: the unmodified harness (`benchmark.sh subset`, 90 s, fresh problem seed) verified 8,953 of 8,953 hits, `RESULT: PASS`.
- This exact package, with the lean split: 9,240 of 9,240 hits, `RESULT: PASS`.
