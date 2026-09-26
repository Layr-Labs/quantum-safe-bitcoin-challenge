# Subset: terrapinelf's `82d8493f` (host-built epoch producers, warp-uniform root) with `QSB_SHA_FMA_ADD=0` and our faster host-CPU co-grinder: signed-digit memory-sized host table, ±C folded into the last window, shorter IFMA reduction, no table re-read, short prefetch, precomputed SHA schedules, and run-time calibrated 16-lane AVX-512 hashing / SMT hybrid

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed and counted with Intel SDE 9.48 (`-spr`), and its Zen 4 scheduling estimated with `llvm-mca-18 -mcpu=znver4`. Our host has no AVX-512.

## Starting point

The base is terrapinelf's `82d8493f` (queued at the time of writing), taken from its public submission branch:
- its host-built epoch producers: 3 host threads with SHA-NI replace 3 small per-epoch GPU kernels, +0.93% GPU rate and −0.93% GPU energy per candidate, self-checked at start-up;
- the warp-uniform root inverse (`QSB_ROOT_UNIFORM_WARP`);
- the whole `de5739c9` GPU tree beneath them.

We change one GPU knob, `QSB_SHA_FMA_ADD` 1 → 0 (below), and replace its co-grinder (`CpuGrindSubset.h`) with ours. Ours is the second stage of our co-grinder work; our `889742ab` (promoted at 651.26 M/s) was the first. It also keeps what `82d8493f`'s producers need from that header (`qsha_x4`, `qsha_iv`, `qsha_k`, `qsha_supported`).

The co-grinder sizes itself from the process's CPU set as it was before `main()`, because the producers pin the GPU host thread to one core. Threads inherit that one-core mask, so the co-grinder's builder thread first widens its own mask to every CPU but that core; the table build and the workers inherit the wide mask. It reserves the host core and places its workers on every other logical CPU, below the producer threads (`SCHED_IDLE`). Without the widening, the table build took 10.9 s instead of 0.8 s and the unpinned workers shared two CPUs; we caught this in our gate below.

## The GPU side: `82d8493f` with `QSB_SHA_FMA_ADD=0`

- One line differs from `82d8493f` on the GPU side: `#define QSB_SHA_FMA_ADD 0` in `subset.cu` (theirs: 1). Every `.cu`/`.cuh` file and every other device-visible header is byte-identical to `82d8493f`, including `host_producers.h`, `tree.cu` and `tree_inverse.cuh`.
- `QSB_SHA_FMA_ADD=1` emits each two-input add of the pubkey-hash compression as an `IMAD` on the FMA-heavy pipe, for pipe balance, at +3 instructions per round. At 0 they are plain three-input `IADD3`s again, as in Meganpark980320's queued `bb2a3eb7`, which flips the same knob on `de5739c9` and describes it.
- Why: the ranked card runs throttled (about 317 W, about 1.7 GHz, 90 °C after the first seconds), so energy per candidate sets its sustained rate. Meganpark980320 measured the flip at −0.41% rate but +0.5% SM clock at a 450 W cap, and +0.41% after their clock normalization. In the public ranked metrics, `QSB_SHA_FMA_ADD=1` GPU-only draws sit about 0.011 lower in sustained/peak ratio than the comparable 0 draws. The direction agrees, although both estimates are small and noisy.
- `build_carrier.sh` with CUDA 12.8.93 regenerates the native sm_89 image: cubin sha256 `070afc8c84113a07…`, 0 spills. That is byte-identical to `bb2a3eb7`'s image, because the digest kernel is the same (`82d8493f`'s host producers change only host code and the small per-epoch kernels).

## Why the co-grinder

The ranked metrics expose `candidates_self_reported`, the GPU grinder's own peak-rate estimate, which excludes CPU candidates. So `verified_hits · 2^23 / self_reported` is the GPU's sustained ratio plus the CPU's share:
- GPU-only draws of the GLV12 class sit at 0.776–0.794 (mean 0.784).
- `de5739c9` drew 0.7957, and RealAdii's re-draw of the same tree (`43667cf5`, 641.01) drew 0.8037.
- The IFMA co-grinder therefore adds roughly 1–2.5% of hits on the 32-thread runner. Each extra CPU M/s is about one point, and it costs the GPU nothing.

## Where the co-grinder's time went (per candidate, measured)

Dynamic instructions per candidate on the 8-lane IFMA path, one worker, from SDE `-mix`:

| part | `de5739c9` | this package |
|---|---:|---:|
| window additions | 2,745 (15 additions, 694 `vpmadd52*`) | 1,375 (9 additions) + 361 (last window, both recids, C folded) |
| total `vpmadd52*` | 764 | 576 (the shorter reduction trades 23 shift/and/add for 4 IFMA per multiplication) |
| SHA-256 | 1,989 (≈ 816 are register-spill moves) | 1,048 with SHA-NI, no spills (or the 16-lane AVX-512 path) |
| per-candidate glue in `worker` | 1,952 | ≈ 990 |
| final step (both recids, canonicalization) | 503 | folded into the last window, vector canonicalization |
| **total** | **7,678** | **4,080** (SHA-NI mode) / **3,900** (16-lane mode) |

## Changes (all in `CpuGrindSubset.h`)

### 1. A signed-digit wide host table, sized to the host at run time
z·A used sixteen 16-bit windows: 64 MiB, 15 additions. Now:
- The table has L lookups of signed digits of ⌊257/L⌋ or ⌈257/L⌉ bits, holding the 2^(w−1) positive multiples. Negative digits negate y at load.
- L = 11 (23–24-bit digits) takes 4.3 GiB with the folded copies.

**Memory budget:**
- A third of the smaller of `MemAvailable` and the headroom of every memory cgroup on this process's `/proc/self/cgroup` path (v2 `memory.max`/`memory.high`, v1 `limit_in_bytes`), capped at 6 GiB, so 11 lookups at most.
- One 2 MiB-aligned `mmap` with `MADV_HUGEPAGE`. We verified the whole table lands in huge pages.
- Any failed allocation drops to more lookups, down to the 16-window, 38 MiB floor.
- The whole builder thread is exception-safe. A failure anywhere switches the co-grinder off; the process never aborts.

**Build:** a parallel doubling-round builder, 0.8 s for 4.3 GiB on 22 threads, while the GPU starts.

**Recoding:** branch-free signed recoding, with the top window absorbing the carry (its value never exceeds 2^(w−1)).

The start line reports the choice, e.g. `table 11 lookups (23..24-bit signed digits, C folded into the last window), 4352 MiB (budget 17009 MiB), built in 0.75s`.

### 2. ±C folded into the last window
Recovery adds ±C (recid 0/1) after z·A. The table also stores the last window as T+C and T−C, so the last lookup produces both recovered keys directly:
- Q₀ = S + (s·T[a] + C), Q₁ = S + (s·T[a] − C). For s < 0 these are −(T−C)[a] and −(T+C)[a].
- Both recids share one batch inversion.
- This replaces the last window's addition plus the separate ±C step: 10 multiplications + 2 squarings instead of 12 + 3 per 8 candidates.

### 3. Shorter IFMA reduction, no table re-read and a short prefetch (after Meganpark980320's `bb2a3eb7` and newjordan's `2a1f43c5`)
Both ideas come from Meganpark980320's queued `bb2a3eb7`, which describes them precisely in its public note. Its code was not restorable; we re-implemented both from the description and tested them independently.

- **Reduction.** The high product columns c5..c9 are no longer normalized by a serial carry chain before the fold. Each is split into its low 52 bits and the rest (< 2^5):
  - both parts are folded with 2^260 ≡ R = 0x1000003D10 (lo·R as a lo/hi IFMA pair, rest·R < 2^42 as one lo IFMA);
  - the part landing at 2^260 again (from c9) is folded once more;
  - the bits of column 4 at and above 2^256 are folded with 0x1000003D1;
  - one carry chain finishes.

  That is 18 IFMA and 24 shift/and/add instead of 14 IFMA and about 47, with no serial chain before the fold. Meganpark980320 measured an 8-lane multiplication at 20.3 → 16.8 ns on a Zen 4 EPYC. We checked all bounds: every IFMA input stays below 2^52, and outputs are normalized (limbs 0..3 < 2^52, limb 4 < 2^49).
- **No table re-read.** The forward pass of each window keeps D = tx − X and the signed TY = ±ty − Y. The backward pass computes λ = TY · dinv and x3 = λ² − D − 2X (`fe8_sub3`, one carry pass), so it never loads or transposes table rows again. The C-folded last window does the same per recid.
- **Short prefetch.** With no table loads in the backward pass, newjordan's queued `2a1f43c5` measured the row-prefetch distance on a Zen 4 SMT host: 3 groups ahead was fastest (2: 1.610, 3: 1.615, 4: 1.586, 6: 1.589, 8: 1.575 M candidates per CPU-second). Our windows prefetch 3 groups ahead too (`QSB_CPU_PF`, was 8). Prefetches are hints, so the hit set cannot change; the exactness gates below were rerun anyway.

### 4. SHA-256 schedules are precomputed (SHA-NI path)
- Every epoch leaves the same 8 bytes after its midstate, so blocks 1..5 of a candidate depend only on its window pattern. Their W+K are computed once and deduplicated; blocks 2..5 are identical for all 158 patterns.
- Block 0 has only 77 distinct variants among the 158 patterns, so each epoch computes 77 block-0 states.
- Per candidate only SHA rounds run (`qsha_x4_run`, chaining states in registers, no spills).
- The SHA-256d outer block and the key hashes take their message words straight from state words and field elements (`pk_words`).

### 5. A 16-lane AVX-512 SHA-256 path, chosen at run time
- `sha16_*` computes 16 candidates per `__m512i` lane set: each epoch's 158 digests in 10 chunks (block 1 from lane-transposed W+K tables, blocks 2..5 broadcast), and the key hashes 16 at a time.
- Why it may win: llvm-mca's znver4 model puts `sha256rnds2` at one per 2 cycles on all four FP pipes, while 16-lane rounds cost about 44 cycles per block.
- **It is not assumed.** A calibration picks it only if it measures ≥ 2% faster on the actual host (below).

### 6. An SMT hybrid, chosen at run time
- llvm-mca's znver4 model shows the IFMA window loop bound on FP0/FP1 and the scalar 5×52 loop bound on the integer multiplier, which are disjoint units. Two IFMA threads on one core therefore mostly compete.
- The workers are pinned core by core (`thread_siblings_list` within the affinity mask), leaving one whole core for the GPU host thread.
- In hybrid mode the second thread of each core runs a scalar batch-affine path built on libsecp256k1's 5×52 field code (`fe_mul_inner`/`fe_sqr_inner`, MIT, notice in `COPYING-secp256k1`), with four interleaved Montgomery chains.

### 7. Run-time calibration
Starting 1.5 s after the table is ready, the builder thread measures CPU throughput in ABBA order, 3 s per phase:
1. SHA-NI vs 16-lane hashing;
2. then all-IFMA vs the hybrid.

A mode is adopted only if it is ≥ 2% faster. It prints e.g. `calibration all-IFMA … M/s, SMT hybrid … M/s` and `using …`. Mode switches take effect at batch boundaries, and each epoch's hashing state is rebuilt on demand, so no candidate is dropped or repeated.

### 8. Smaller items
- `fe8_canon_words` (vector canonicalization);
- `fe8_sub2` (λ² − x₁ − x₂ in one carry pass);
- explicit vector moves where struct copies became `rep movsq`;
- backward-pass row prefetch;
- exception-safe threading: a failed `std::thread` or allocation stops a worker or switches the co-grinder off, instead of aborting the process.

## Exactness

The CPU path can still only lose hits, never publish a wrong one: every CPU hit is re-derived by the exact OpenSSL gate `qsb_hv_check` before it is written. We also checked each piece directly:

| check | result |
|---|---|
| `qsha_sched`/`qsha_x4_run` vs OpenSSL `SHA256_Transform`, 1.1 M lane-blocks | 0 mismatches |
| `sha16_*` (schedule, per-lane and broadcast blocks) vs OpenSSL, 320 k lane-blocks (SDE) | 0 mismatches |
| `fe8_canon_words`, 3.2 M values incl. p, p..p+4, 2^256−1, 2^257−1 (SDE) | 0 mismatches |
| `fe8_sub2` vs two `fe8_sub`, 2.4 M values incl. maximal limbs (SDE) | 0 mismatches, outputs normalized |
| `fe8_sub3` (λ² − D − 2X) vs three `fe8_sub`, 2.4 M values incl. maximal limbs (SDE) | 0 mismatches, outputs normalized |
| new `fe8_mul`/`fe8_sqr` (shorter reduction) vs scalar `fe_mul`, 2 M lanes incl. maximal limbs (SDE) | 0 mismatches, outputs normalized |
| table and folded entries vs OpenSSL at 38 MiB / 1.1 / 4.3 GiB, and under `ulimit -v` fallbacks | 0 mismatches |
| hit set, `QSB_ZEROS_N=12`, one worker, first 131,072 candidates: scalar 5×52 path (native) vs IFMA path (SDE) in SHA-NI mode, 16-lane mode, and with the calibration switching modes mid-run; 16 and 11 lookups | identical to each other and to the pre-change code (66 = 66) |
| 4 pinned workers under SDE, forced hybrid + 16-lane hashing, and live calibration | no duplicates; hit rate within Poisson noise |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, fresh seeds (our host runs the scalar path) | 150 s: 14,408 / 14,408 verified, 164 from the CPU file (promoted code, 180 s: 129) — `RESULT: PASS` |

We found one bug on the way, and SDE caught it before any run: a `std::vector` buffer used with aligned 512-bit stores (`vmovdqa64`) was not 64-byte aligned. It is now allocated 64-byte aligned, and every aligned vector access was audited.

## Validation of this exact package (82d8493f tree + this co-grinder)

| check | result |
|---|---|
| co-grinder hit set, one worker, first 131,072 CPU candidates, `QSB_ZEROS_N=12`: scalar path natively, IFMA path under SDE in SHA-NI and 16-lane modes | identical in all three runs (66 = 66 hits), with the 3-group prefetch |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 11,684 / 11,684 verified, 98 from the CPU file (our host runs the scalar path), `RESULT: PASS` |
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills, byte-identical to `bb2a3eb7`'s image; the start line prints `Native sm_89 carrier: on` |
| `82d8493f`'s host producers next to this co-grinder (their start-up self-check, then `[HP]` host-built vs GPU-fallback batches) | self-check passed (batch 0: 1,048,576 descriptors + 8,388,608 first-block states bit-identical); in 40 s, 235 host-built batches and 1 GPU-built after start-up (of 242); co-grinder table built in 0.81 s beside them |
| the same tree with `QSB_SHA_FMA_ADD=1` (`82d8493f`'s GPU side exactly) vs our previous co-grinder stage on `de5739c9`'s GPU side, interleaved 60 s runs on our host | GPU 818.2 vs 811.3 M/s (+0.85%, the producers), co-grinder 6.94 vs 7.84 M/s (the 3 producer threads share its CPUs); total +6.0 M/s |

## Measured rates on our host (no AVX-512: scalar EC path)

| build | CPU co-grinder, real run with the GPU grinding (60 s) |
|---|---:|
| `de5739c9` as promoted | 5.32 M/s |
| first stage (our `889742ab`: wide table + precomputed SHA) | 7.36 M/s |
| second stage, without items 3 (our v8) | 8.06 M/s (+52%) |
| this package | the scalar path is unchanged by item 3 (it runs only the 5×52 path here); item 3 is IFMA-only |

The GPU rate is unchanged within noise: 808.8–811.3 M/s across 8 interleaved 60 s runs with the co-grinder off, as promoted, and new.

## Caveats

- The runner's CPU model and memory are not published. Our AVX-512 cycle estimates come from llvm-mca's znver4 model, not from Zen 4 hardware. That is why the hashing choice and the hybrid are calibrated on the host instead of hard-coded.
- The calibration measures only the co-grinder's own rate. Pinning leaves a whole core free for the GPU host thread, and on our host the GPU rate is unchanged within noise with the co-grinder on or off. The GPU host thread's behavior on the runner is not measured.
- The table uses at most 4.3 GiB (11 lookups), and less on a host with less than 13 GiB available.
- On the promoted base the host producers' 3 threads run at normal priority and the co-grinder workers are `SCHED_IDLE`. This follow-up keeps the same disjoint CPU placement but runs the co-grinder workers at normal priority, because they are pinned away from the GPU host thread and producer lane. It also reserves one logical CPU instead of two, leaving one host lane for the driver.
- `QSB_CPU_DEVBENCH` / `QSB_CPU_DEVCAND` (CPU-only rate, deterministic hit-set runs) and the `QSB_CPU_MODE` / `QSB_CPU_TABLE_MB` / `QSB_CPU_NOPIN` / `QSB_CPU_NOFOLD` / `QSB_CPU_NOFAST` environment switches are dev-only. The first two are compiled out of the ranked build, and the rest are never set by the harness.

## Base and attribution

- **Base: terrapinelf's `82d8493f`** (co-author): the host-built epoch producers, the warp-uniform root inverse, its package. Beneath it is the promoted `de5739c9`, also by terrapinelf: the GLV12xc GPU tree, the 8-lane IFMA co-grinder path, `fe8_sqr`, the shared inversion, the 4-lane SHA-NI routine, and the runner-CPU inference. We build on that promotion.
- **Through that base:**
  - i34-9: the lean GLV split;
  - fkiene: fk-lean, the L2 fetch granularity and `QSB_S3_HALF_WALK`;
  - Ryun1: the carrier design and the `CpuGrind.h` co-grinder design, table and batch-affine code;
  - our own `933abead` GLV12 port;
  - Akashneelesh's crown `7aef224a` and every contributor it credits.
- **libsecp256k1** (MIT): `fe_mul_inner`/`fe_sqr_inner` (field_5x52_int128_impl.h), `normalize`/`normalize_weak`, and the inversion addition chain.
- **Meganpark980320** (`bb2a3eb7`, queued): `QSB_SHA_FMA_ADD=0` on this GPU tree, the shorter IFMA reduction and the forward-pass `ty − Y` / no-re-read idea (co-author).
- **newjordan** (`2a1f43c5`, queued): the Zen 4 prefetch-distance measurement (co-author); beneath the base, the `d1ddefca` carrier tree and the warp root inverse.
- **Ours:** the signed memory-sized table and builder, the budget and cgroup logic, the C fold, the precomputed-schedule and 16-lane SHA paths, the 5×52 scalar batch path, the SMT pinning, hybrid and calibration, `fe8_canon_words`/`fe8_sub2`, exception-safe threading, and the SDE/llvm-mca/hit-set test method.

All inherited source, GPLv3 notices and attributions are kept.

## Follow-up scheduling change

This submission is the promoted `196861248169aed40f0b0ec051f74c29174d3d5f` tree with one host-only scheduling change in `CpuGrindSubset.h`:

- `QSB_CPU_RESERVE` changes from 2 to 1, so one additional logical CPU can participate in the co-grinder while one lane remains available for the GPU host thread and driver.
- The two `SCHED_IDLE` calls are guarded by `QSB_CPU_IDLE`, whose ranked default is 0. Workers remain pinned away from the host lane, so normal priority avoids the idle scheduler penalty without putting work on the host core.
- No device code, table geometry, carrier knob, arithmetic, hit publication, verifier, or benchmark harness code changes. The embedded CUDA 12.8 sm_89 carrier remains compatible because its build-knob fingerprint is unchanged.

The change is intentionally isolated to the CPU co-grinder. The exact host CPU model is selected by the ranked runner, so the official fixed-time result is the authority for the gain. All existing exactness gates and the promoted base validation remain applicable.
