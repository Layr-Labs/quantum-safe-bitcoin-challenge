# Subset: terrapinelf's `82d8493f` (host-built epoch producers, warp-uniform root) with `QSB_SHA_FMA_ADD=0` and the third stage of our host-CPU co-grinder: fused field operations (one carry pass for signed rows, subtractions folded into the product columns), an unrolled 16-lane SHA-256 schedule, digit recoding specialized per table size, one more worker on the GPU host thread's core, 16-lane AVX-512 hashing for the host producers chosen at start-up, a 10-lookup table behind a sacrificial memory probe on hosts with the memory, and each window's table rows prefetched during the previous window's backward pass (after terrapinelf's 97f347a8), on top of the signed memory-sized table, ±C fold, precomputed SHA schedules and run-time calibrated 16-lane AVX-512 hashing / SMT hybrid

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 + i7-13700K host (CUDA 12.8.93). AVX-512 code was executed and counted with Intel SDE 9.48 (`-spr`), and its Zen 4 scheduling estimated with `llvm-mca-18 -mcpu=znver4`. Our host has no AVX-512.

## Starting point

The base is terrapinelf's `82d8493f` (queued at the time of writing), taken from its public submission branch:
- its host-built epoch producers: 3 host threads with SHA-NI replace 3 small per-epoch GPU kernels, +0.93% GPU rate and −0.93% GPU energy per candidate, self-checked at start-up;
- the warp-uniform root inverse (`QSB_ROOT_UNIFORM_WARP`);
- the whole `de5739c9` GPU tree beneath them.

We change one GPU knob, `QSB_SHA_FMA_ADD` 1 → 0 (below), and replace its co-grinder (`CpuGrindSubset.h`) with ours. This is the third stage of our co-grinder work: our `889742ab` (promoted at 651.26 M/s) was the first, and our `b539d6dc` (this tree with the second stage; promoted at 665.13 M/s, GPU 630.65 + CPU 34.48 M/s) the second. This package is `b539d6dc` with `CpuGrindSubset.h` (sections 9 to 12, 14 and 15) and the host-only `tests/gpu_epochs/host_producers.h` (section 13) changed, plus the regenerated carrier header's source-hash comment. The device code and the carrier image are unchanged. It also keeps what `82d8493f`'s producers need from that header (`qsha_x4`, `qsha_iv`, `qsha_k`, `qsha_supported`).

The co-grinder sizes itself from the process's CPU set as it was before `main()`, because the producers pin the GPU host thread to one core. Threads inherit that one-core mask, so the co-grinder's builder thread first widens its own mask to every CPU but that core; the table build and the workers inherit the wide mask. It reserves the host core and places its workers on every other logical CPU, below the producer threads (`SCHED_IDLE`). Without the widening, the table build took 10.9 s instead of 0.8 s and the unpinned workers shared two CPUs; we caught this in our gate below.

## The GPU side: `82d8493f` with `QSB_SHA_FMA_ADD=0`

- One line differs from `82d8493f` in the device code: `#define QSB_SHA_FMA_ADD 0` in `subset.cu` (theirs: 1). Every `.cu`/`.cuh` file and every other device-visible header is byte-identical to `82d8493f`, including `tree.cu` and `tree_inverse.cuh`. `host_producers.h` gains a host-only 16-lane hashing path (section 13); its GPU producer kernels and everything the device sees are unchanged.
- `QSB_SHA_FMA_ADD=1` emits each two-input add of the pubkey-hash compression as an `IMAD` on the FMA-heavy pipe, for pipe balance, at +3 instructions per round. At 0 they are plain three-input `IADD3`s again, as in Meganpark980320's queued `bb2a3eb7`, which flips the same knob on `de5739c9` and describes it.
- Why: the ranked card runs throttled (about 317 W, about 1.7 GHz, 90 °C after the first seconds), so energy per candidate sets its sustained rate. Meganpark980320 measured the flip at −0.41% rate but +0.5% SM clock at a 450 W cap, and +0.41% after their clock normalization. In the public ranked metrics, `QSB_SHA_FMA_ADD=1` GPU-only draws sit about 0.011 lower in sustained/peak ratio than the comparable 0 draws. The direction agrees, although both estimates are small and noisy.
- `build_carrier.sh` with CUDA 12.8.93 regenerates the native sm_89 image: cubin sha256 `070afc8c84113a07…`, 0 spills. That is byte-identical to `bb2a3eb7`'s image, because the digest kernel is the same (`82d8493f`'s host producers change only host code and the small per-epoch kernels).

## Why the co-grinder: ranked ground truth

Every ranked subset run publishes its verified hits. Each hit's window pattern says whether the GPU (the 128-pattern set) or the co-grinder (the other 158) found it, so the two rates can be separated exactly:

| ranked run | GPU M/s | CPU M/s | CPU share of hits |
|---|---:|---:|---:|
| `de5739c9` (promoted, 634.72) | 608.51 | 26.21 | 4.13% |
| `43667cf5` (a re-draw of `de5739c9`) | 615.30 | 25.70 | 4.01% |
| our `889742ab` (promoted, 651.26) | 618.46 | 32.80 | 5.04% |

- The co-grinder is about 5% of the score now, and our first stage lifted it 1.25× on the ranked host.
- A busy host CPU costs the thermally limited GPU about 2% whatever the co-grinder's speed: a 6 M/s scalar lane paid the same as a 26 M/s IFMA lane. So co-grinder work per candidate, not raw activity, is what pays.

## Where the co-grinder's time goes (per candidate, measured)

Dynamic instructions per candidate on the 8-lane IFMA path, one worker, from SDE `-mix`. The build is the ranked `QSB_ZEROS_N=24`, so no OpenSSL hit checks pollute the counts. The table is 11 lookups:

| part | `de5739c9` | `b539d6dc` (stage 2) | this package |
|---|---:|---:|---:|
| window additions (`ec8_window`) | 2,745 (15 additions) | 1,376 (9 additions) | 1,288 |
| last window, both recids, C folded | 503 (final step) | 362 | 351 |
| `vpmadd52*` | 764 | 576 | 568 |
| 16-lane SHA-256 (`epoch16`, `sha16_*`, `keyhash16`) | – | 1,184 | 950 |
| per-candidate glue in `worker_body` (digit recoding, bookkeeping) | 1,952 | 695 | 387 |
| **total, 16-lane hashing** | – | **3,753** | **3,112** |
| **total, 4-lane SHA-NI hashing** | **7,678** | **3,934** | **3,525** |

Which hashing mode is faster on the ranked Zen 4 is decided on the host at run time (section 7). In llvm-mca's `znver4` model, `sha256rnds2` holds one FP pipe for 8 cycles, which would make 4-lane SHA-NI the dominant cost. The ranked split of our `b539d6dc` (second stage, where the 16-lane path and the hybrid are calibrated on the host) says otherwise: CPU 34.48 M/s next to the producers, against 32.80 M/s for the first stage without them. So hashing was not the large hidden cost. terrapinelf's Zen 4 measurement in `97f347a8` points at table-row stalls instead (about 11% of SMT-loaded time), which section 15 attacks. The calibration stays, and the other items of this stage cut instructions and vector work, which pays either way.

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
2. then all-IFMA vs the hybrid;
3. then (this package) the table-row prefetch distance: 3 vs 8 groups of 8 lanes ahead. 3 was measured fastest on a desktop Zen 4 (newjordan's `2a1f43c5`), but the ranked host's memory latency is not known, so it is measured there.

A mode is adopted only if it is ≥ 2% faster. It prints e.g. `calibration all-IFMA … M/s, SMT hybrid … M/s` and `using …`. Mode switches take effect at batch boundaries, and each epoch's hashing state is rebuilt on demand, so no candidate is dropped or repeated.

### 8. Smaller items
- `fe8_canon_words` (vector canonicalization);
- `fe8_sub2` (λ² − x₁ − x₂ in one carry pass);
- explicit vector moves where struct copies became `rep movsq`;
- backward-pass row prefetch;
- exception-safe threading: a failed `std::thread` or allocation stops a worker or switches the co-grinder off, instead of aborting the process.

### 9. Fused field operations (this package)
Most non-IFMA vector work in a batch-affine addition is carry passes after subtractions. Three are removed:
- **Signed rows in one pass.** The forward pass computed `ty' = −ty` for the lanes with a negative digit (a full subtraction and carry pass), then `TY = ty' − Y`. It now computes `8p + ty − Y`, or `8p − ty − Y` in the negative lanes (a masked subtraction), with one carry pass (`fe8_sub_sgn`). Every limb stays non-negative, because 8p's limbs exceed the sum of two normalized limbs.
- **`y3 = λ·t − Y` with one reduction.** The product columns 0..4 start at `4p − Y` instead of zero (`fe8_mul_sub`). They stay below 2^56, within the reduction's 2^57 column bound, so the separate subtraction and its carry pass are gone.
- **`x3 = λ² − D − 2X` with one reduction.** `12p − D − 2X` is added to the square's columns 0..4 before the reduction (`fe8_sqr_sub3`); the columns stay below 2^56.4.
- **No limb-4 fold where it is not needed.** `D = tx − X`, `TY` and `t = X − x3` only feed multiplications (and D the fused subtraction above). IFMA reads 52 bits per limb, and the reduction's column bounds depend only on the limbs being below 2^52. So these skip the fold of limb 4's bits ≥ 2^48 (one shift, one mask, one IFMA each): limb 4 stays below 2^51.3, and the bound for `12p − D − 2X` still holds (D₄ + 2X₄ < 2^51.1 < 12p₄).

The window loop drops from 1,376 to 1,288 instructions per candidate. The removed work is all vector (IFMA, shifts, masks, adds), which is what bounds the co-grinder when both SMT threads run IFMA code.

### 10. Unrolled 16-lane SHA-256 schedule (this package)
The 16-lane message schedule (`sha16_sched`, used for the per-epoch first blocks, the SHA-256d outer block and the key hashes) was a rolled loop over a 16-entry ring. The compiler kept the ring on the stack and spent 34 instructions per schedule word. Fully unrolled, the ring indices are constants, the ring stays in registers, and each word costs 13: 380 → 148 instructions per candidate.

### 11. Digit recoding specialized per table size (this package)
Each candidate's 256-bit scalar is recoded into signed window digits. The loop read the window layout (start bit, width) from the context and computed shifts, masks and bounds at run time: about 32 instructions per window, 11 windows per candidate. The layout depends only on the number of lookups L. `digits_L<L>` takes it from the same formula at compile time (`win_w`/`win_s`, checked against `layout()` for L = 10..16), for L = 10..16, so every shift, mask and half-width is a constant and the scalar stays in registers. Other L keep the generic loop. With the smaller bookkeeping, glue drops from 695 to 387 instructions per candidate.

### 12. One more worker on the GPU host thread's core (this package)
`82d8493f`'s producers pin the GPU host thread to one core. It spins on one CPU of that core while it waits for the GPU (`cudaEventSynchronize` under CUDA's default spin schedule), and until now the core's second CPU stayed idle.
- When `start()` finds the host thread pinned to one core, it adds one worker. `smt_plan` pins it to that core's other CPU as an IFMA worker.
- It is `SCHED_IDLE` like the others, so the host thread always runs first. Unpinned workers now avoid only the host thread's own CPU.
- Checked under SDE on our 24-CPU host: 23 workers, one on each CPU except the host thread's, the extra one on the host core's sibling.
- `de5739c9` and our `889742ab` ran their host thread next to a worker the whole time (unpinned, 30 workers on 32 CPUs); `889742ab`'s GPU drew 618.46 M/s.
- Off with `QSB_CPU_NOEXTRA=1`, and only when the IFMA path and pinning are on.

### 13. 16-lane AVX-512 hashing for the host producers, chosen at start-up (this package, `host_producers.h`)
The producers hash about 3.6 suffix blocks and 8 first-block states per epoch, 4 lanes at a time with SHA-NI, and keep 3 threads 55–65% busy at the ranked rate (terrapinelf's figure). On Zen 4 that is the hashing mode that holds the FP pipes longest (section "Where the co-grinder's time goes"), so the producers get the same 16-lane alternative as the co-grinder:
- `flush16`/`produce16` hash 16 epochs per SHA-256 pass. Suffix blocks go in lockstep, and a lane whose suffix has ended keeps its state (masked update). First-block states run class-major: one pass per class for all 16 epochs, message words 2..15 broadcast. Descriptors and states are written exactly as `flush()` writes them (non-temporal stores, same offsets).
- **Chosen by measurement, on the host.** Batch 0 is built by the host anyway, for the start-up self-check. Its 64 chunks now alternate between the SHA-NI x4 path and the x16 path, and the per-chunk times decide: x16 is used afterwards only if it is at least 2% faster. The start line prints both times and the choice. Until then, and whenever AVX-512F/BW is missing, the producers run exactly as in `82d8493f`.
- **Both paths are verified.** The self-check compares the whole of batch 0 (both halves) with the GPU producers' copy before any host batch is used. Under SDE (`-spr`) the mixed batch 0 passed: 1,048,576 descriptors + 8,388,608 first-block states bit-identical.
- Off with `QSB_HP_NO16=1`.

### 15. Each window's table rows prefetched during the previous window's backward pass (this package, after terrapinelf's `97f347a8`)
terrapinelf's queued `97f347a8` measured about 11% of SMT-loaded time on a Zen 4 host still waiting on table rows, with rows prefetched only a few groups ahead in the short forward pass. Moving the prefetch into the previous window's long backward pass gained +9 to +12% there. We ported the scheme to our window code:
- Each batch keeps two row-pointer buffers, the current window's and the next one's (for the C-folded last window, 16 rows per group: both recids).
- The first window fills window 1's rows while it loads its own.
- Each window's backward pass fills the next window's rows of group `G − 1 − h` as it walks back over group h, so the next forward pass finds its first groups first. It prefetches half of them before and half after the group's two chain multiplications.
- The forward pass loads from the precomputed pointers and keeps a short prefetch `pf` groups ahead as a backstop.
- The pointers and sign masks are exactly those the forward pass computed before, only computed earlier, so the hit set cannot change: identical in every checked configuration (below).

### 14. Ten lookups when memory is plentiful, behind a sacrificial memory probe (this package)
With 10 lookups (25–26-bit signed digits, 19,456 MiB with the folded copies) a candidate needs 9 window additions instead of 10. On our host that runs the co-grinder 8.1% faster CPU-only (scalar path, 8.52 vs 7.88 M/s, two runs each, 22 threads).
- It is used only when all of these hold:
  - the memory available under every visible limit (`MemAvailable` clamped by each cgroup level's `memory.max`/`memory.high`, v2 or v1) is at least **4×** the table;
  - no visible cgroup has `memory.oom.group = 1`;
  - transparent huge pages are not `never`;
  - a **sacrificial probe** passes.
- **The probe.** The co-grinder re-executes its own binary (`posix_spawn` of `/proc/self/exe`) with `QSB_CPU_PROBE_MB` set. In that child a static initializer runs before `main()`: it touches no CUDA state, sets `oom_score_adj` to 1000, faults the whole table size in (huge pages) and exits. If a memory limit the process cannot see is smaller than the table, the kernel kills that child (the highest OOM score), not the grinder. Any failure (spawn refused, killed, non-zero exit, over 40 s) keeps the 11-lookup table. The start line reports the probe result.
- Tested on our host:
  - The probe passes and 10 lookups are built (the table build takes 4.9 s including the probe).
  - A child that kills itself halfway, as an OOM kill would, makes the grinder fall back to 11 lookups and carry on.
  - The 10-lookup hit set is identical to the reference: native scalar path, and IFMA path under SDE.
- Off with `QSB_CPU_NOPROBE=1`. On a host with less than 4× the table available, nothing changes.

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
| `fe8_mul_sub`, `fe8_sqr_sub3`, `fe8_sub_sgn` vs the unfused sequences (`fe8_mul`+`fe8_sub`, `fe8_sqr`+`fe8_sub3`, `fe8_cneg`+`fe8_sub`), 2.4 M lanes incl. 0, p, 2^256−1, 2^257−1 and all mask patterns (SDE) | 0 mismatches, outputs normalized |
| unfolded `fe8_sub_m` / `fe8_sub_sgn` values as multiplication inputs and as `fe8_sqr_sub3`'s subtrahend vs the normalized path, 2.4 M lanes (SDE); their limb-4 bounds | 0 mismatches; limb 4 < 2^51 and < 2^52 |
| unrolled `sha16_sched` + 16-lane compressions vs OpenSSL, 320 k lane-blocks (SDE) | 0 mismatches |
| `digits_L<L>` vs the generic recoding loop, L = 10..16, 400 k scalars each incl. all-zero, all-one, single-bit and alternating patterns; `win_w`/`win_s` vs `layout()` | 0 mismatches; layouts equal |
| table and folded entries vs OpenSSL at 38 MiB / 1.1 / 4.3 GiB, and under `ulimit -v` fallbacks | 0 mismatches |
| hit set, `QSB_ZEROS_N=12`, one worker, first 131,072 candidates: scalar 5×52 path (native) vs IFMA path (SDE) in SHA-NI mode, 16-lane mode, and with the calibration switching modes mid-run; 16 and 11 lookups | identical to each other and to the pre-change code (66 = 66) |
| 4 pinned workers under SDE, forced hybrid + 16-lane hashing, and live calibration | no duplicates; hit rate within Poisson noise |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, fresh seeds (our host runs the scalar path) | 150 s: 14,408 / 14,408 verified, 164 from the CPU file (promoted code, 180 s: 129) — `RESULT: PASS` |

We found one bug on the way, and SDE caught it before any run: a `std::vector` buffer used with aligned 512-bit stores (`vmovdqa64`) was not 64-byte aligned. It is now allocated 64-byte aligned, and every aligned vector access was audited.

## Validation of this exact package

| check | result |
|---|---|
| co-grinder hit set, one worker, first 131,072 CPU candidates, `QSB_ZEROS_N=12`: scalar path natively, IFMA path under SDE in SHA-NI and 16-lane modes | identical in all three runs (66 = 66 hits) |
| unmodified harness (`benchmark.sh subset`, `QSB_GRINDER=cmd:… gpu_wrap.py`), N = 24, 120 s, fresh seed | 11,803 / 11,803 verified, 97 from the CPU file (our host runs the scalar path and the SHA-NI producers; it has less than 4× the 10-lookup table, so 11 lookups), `RESULT: PASS`. With the probe forced (dev switch `QSB_CPU_PROBE_TEST`): 11,622 / 11,622, `RESULT: PASS` |
| `build_carrier.sh`, CUDA 12.8.93 | cubin sha256 `070afc8c84113a07…` (462,496 B), 0 spills, byte-identical to `bb2a3eb7`'s image; the start line prints `Native sm_89 carrier: on` |
| `82d8493f`'s host producers next to this co-grinder (their start-up self-check, then `[HP]` host-built vs GPU-fallback batches) | self-check passed (batch 0: 1,048,576 descriptors + 8,388,608 first-block states bit-identical); 1–4 GPU-built batches after start-up of 241–242 per 40 s run, the same range as with stage 2 (1–7) on our 24-thread host; co-grinder table built in 0.82 s beside them |
| the 10-lookup path next to the GPU grind (probe forced, 50 s) | `memory probe for 19456 MiB (available 50786 MiB): passed`, 10-lookup table built in 5.32 s while the GPU started; `[HP]` 1 GPU-built batch after start-up of 302; GPU 816.1 M/s as usual |
| 10-lookup hit set, one worker, first 131,072 CPU candidates, `QSB_ZEROS_N=12`: native scalar path, IFMA path under SDE in SHA-NI and 16-lane modes | identical in all three runs (66 = 66 hits) |
| hit sets with the backward-pass row prefetch (section 15), one worker, 131,072 candidates: native, SDE SHA-NI and 16-lane modes with 11 lookups, SDE without the C fold, SDE with 10 lookups | identical in all five runs (66 = 66 hits) |
| live calibration of all three rounds (hashing, hybrid, prefetch distance), 4 workers under SDE, 3,014,656 candidates at `QSB_ZEROS_N=12` | all three calibration lines printed; 1,455 hits against 1,472 expected (z = −0.44), no duplicates |
| the producers' x16 path and the co-grinder's host-core worker, whole binary under SDE (`-spr`, AVX-512 on) | batch 0 built with alternating SHA-NI x4 / x16 chunks: self-check passed (1,048,576 descriptors + 8,388,608 first-block states bit-identical), calibration line printed; 23 co-grinder workers on our 24 CPUs, the extra one pinned to the host thread's sibling CPU. A test-only build forcing x16 (longer fallback wait for SDE's speed) then built full batches with it that the GPU used (`[HP] host-built batches 2`), with no fault |
| (stage 2, unchanged here) the same tree with `QSB_SHA_FMA_ADD=1` (`82d8493f`'s GPU side exactly) vs our first co-grinder stage on `de5739c9`'s GPU side, interleaved 60 s runs on our host | GPU 818.2 vs 811.3 M/s (+0.85%, the producers), co-grinder 6.94 vs 7.84 M/s (the 3 producer threads share its CPUs); total +6.0 M/s |

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

- **Base: terrapinelf's `82d8493f`** (co-author): the host-built epoch producers, the warp-uniform root inverse, its package. Also terrapinelf's queued `97f347a8`: the backward-pass row prefetch scheme and its Zen 4 measurement (section 15). Beneath it is the promoted `de5739c9`, also by terrapinelf: the GLV12xc GPU tree, the 8-lane IFMA co-grinder path, `fe8_sqr`, the shared inversion, the 4-lane SHA-NI routine, and the runner-CPU inference. We build on that promotion.
- **Through that base:**
  - i34-9: the lean GLV split;
  - fkiene: fk-lean, the L2 fetch granularity and `QSB_S3_HALF_WALK`;
  - Ryun1: the carrier design and the `CpuGrind.h` co-grinder design, table and batch-affine code;
  - our own `933abead` GLV12 port;
  - Akashneelesh's crown `7aef224a` and every contributor it credits.
- **libsecp256k1** (MIT): `fe_mul_inner`/`fe_sqr_inner` (field_5x52_int128_impl.h), `normalize`/`normalize_weak`, and the inversion addition chain.
- **Meganpark980320** (`bb2a3eb7`, queued): `QSB_SHA_FMA_ADD=0` on this GPU tree, the shorter IFMA reduction and the forward-pass `ty − Y` / no-re-read idea (co-author).
- **newjordan** (`2a1f43c5`, queued): the Zen 4 prefetch-distance measurement (co-author); beneath the base, the `d1ddefca` carrier tree and the warp root inverse.
- **Ours:** the signed memory-sized table and builder, the budget and cgroup logic, the C fold, the precomputed-schedule and 16-lane SHA paths, the 5×52 scalar batch path, the SMT pinning, hybrid and calibration, `fe8_canon_words`/`fe8_sub2`, the fused field operations, the specialized digit recoding, the host-core worker, the producers' 16-lane path and its batch-0 calibration (on terrapinelf's producer design), exception-safe threading, and the SDE/llvm-mca/hit-set test method.

All inherited source, GPLv3 notices and attributions are kept.

## Follow-up scheduling canary

This submission is the promoted `a137e289b236c3622eba80f1ad5e9a0c8a91eb67` tree with one host-only scheduling change in `CpuGrindSubset.h`:

- `QSB_CPU_RESERVE` changes from 2 to 1, allowing one additional logical CPU to participate in the co-grinder while one lane remains available for the GPU host thread and driver.
- The two `SCHED_IDLE` calls are guarded by `QSB_CPU_IDLE`, whose ranked default is 0. Workers remain pinned away from the host lane, so normal priority avoids the idle scheduler penalty without putting work on the host core.
- No device code, table geometry, arithmetic, hit publication, verifier, producer algorithm, carrier knob, or benchmark harness code changes. The embedded CUDA 12.8 sm_89 carrier remains compatible because its build-knob fingerprint is unchanged.

The official fixed-time runner is the authority for the gain. All inherited exactness gates, producer self-checks, memory probe safeguards, and verifier checks remain unchanged from the promoted base.
