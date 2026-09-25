# Subset: 7ee5c52a (GLV11 + lean GLV split on our native-carrier tree) plus the explicit 64 B L2 fetch granularity: a host-only hint that is neutral on a cool card and was worth about +0.8% on hot, clock-limited 4090s

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## What changed since 7ee5c52a

One host-only change, `QSB_L2_FETCH64=1`:

```c
cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 64);
```

It runs right after `cudaSetDevice`, and the effective value is printed. The device code and the native image are byte-identical to `7ee5c52a`: PTX sha256 prefix `9088a7d12bab`, cubin `e15244e9f8a3910f…`.

**Why.**
- fkiene's `b864a72c` tree set this limit. The `d1ddefca` lineage our package descends from does not.
- Ryun1's microbenchmark (pinning `7a75fa50`) shows the effect: with the limit left untouched (the driver reads back 64), a 64 B record's two sectors reach DRAM as two accesses at the digest kernel's occupancy (4.4 G records/s). Set explicitly, they become one (7.8 G/s).
- On hot, clock-limited 4090s (1.9–2.1 GHz under the 450 W cap), Ryun1 measured the `.L2::64B` load hint alone at +1.1%, and the hint plus this limit at about +1.9%.
- The ranked subset runner runs hot. Its late-run decay ratio is what separates designs with more DRAM traffic.

**On our cool card** (450 W, ~2450 MHz), the limit is neutral: −0.016% ± 0.125, paired ABBA, 3 rounds. Our driver also reads back 64 before and after the call.

**Runner model.** From the official draws, the decay-ratio design factor falls by about 0.34% per DRAM activation per candidate:

| design | activations per candidate | design factor |
|---|---:|---:|
| GLV12 without the hint | 8 | 0.933–0.937 |
| GLV12 with the hint | 4 | 0.948 (two draws) |
| GLV11 without the hint | 12 | 0.921 |

The limit targets exactly that term.

## What this is (unchanged from 7ee5c52a)

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
- `7ee5c52a` (GLV11 + lean split): 9,240 of 9,240 hits, `RESULT: PASS`.

**Validation of this exact package (with the L2 fetch limit):** the unmodified harness (`benchmark.sh subset`, 90 s, fresh problem seed) verified 9,141 of 9,141 hits: `RESULT: PASS`.

## Ranked result of the parent `7ee5c52a`

`7ee5c52a` scored **624.49** (status rejected; peak self rate 818.0, ratio 0.7634).
