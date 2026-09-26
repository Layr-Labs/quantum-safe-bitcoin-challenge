# Subset: our de5739c9 line plus host-built epoch producers (+0.93% GPU rate) and a co-grinder doing ~63% less CPU work per candidate with memory-aware scheduling (12-window signed table, safegcd inversion, spread prefetch, L2-sized batches) — all bit-identical

**What changed against our promoted `de5739c9`:**
1. **Host-built epoch producers (`QSB_HOST_PRODUCERS`, new `tests/gpu_epochs/host_producers.h`).** The three small per-epoch producer kernels (`kernel_epoch_groups`, `kernel_build_epochs_inc`, `kernel_build_first_flat`) now run on 3 host threads with SHA-NI; the GPU only runs the digest kernel. Locally **+0.930% ± 0.011 rate and −0.927% energy per candidate** (paired A/B, 5 warm rounds), bit-identical outputs, self-checked at start-up.
2. **A leaner co-grinder (`CpuGrindSubset.h`).** Same candidates, same gate, bit-identical results, **about 63% fewer CPU instructions per candidate** (7,269 → 3,674 from the lean hashing and field code, → 2,974 with the wider signed-digit table, → 2,658 with the inversion and field changes below). On the ranked host the co-grinder's load costs the GPU a roughly fixed ~2.3% through chassis heat (measured on four GLV12 draws with scalar, AVX-512F and IFMA lanes alike), so candidates per second from the same CPUs is what decides the net.
3. **The warp-uniform root inverse on the GPU** (`QSB_ROOT_UNIFORM_WARP`, +0.06%, described below).
4. The co-grinder now sizes itself and its workers from the process's CPU set as it was before `main()`, because the host producers pin the GPU's host thread to its own core; the co-grinder's workers run on every other CPU at `SCHED_IDLE`, below the producers.

The GPU digest kernel is unchanged from `de5739c9` apart from item 3: cubin sha256 `4e1b6d4c8fc9f8ed…`, default-build PTX prefix `c19c9c840e1c`.

## Host-built epoch producers

Before each digest batch (2^20 epochs × 128 candidates), three small kernels built the batch's epoch descriptors (the SHA-256 midstate of each epoch's fixed prefix: every push below the window except the epoch's 6 early omissions) and the 8 first-block states per epoch that the digest kernel's window hashing starts from. That work is 145–228 instructions per candidate on the GPU (0.6–0.9% of the digest's), and on the thermally limited ranked card GPU energy per candidate is effectively the score.

- **Measured ceiling first.** A probe that skipped the producers (wrong math, measurement only) gave +0.98% rate / −0.98% energy per candidate; adding back the realistic host→GPU copies (320 MiB per batch from pinned memory) still gave +0.96% / −0.95%.
- **Host algorithm.** Epochs are walked in lexicographic order with one SHA-256 stream context per omission level, so each epoch hashes only its own suffix (3.6 blocks per epoch instead of 7.0 for the GPU kernel), four epochs in lockstep with 4-lane SHA-NI (`qsha_x4`); first-block states share the first `sha256rnds2` across classes. Outputs are written with non-temporal stores into 4 pinned slots and uploaded on the slot's stream before its digest launch. Without SHA-NI an OpenSSL path produces the same bytes.
- **Exactness.** Batch 0 is built on both host and GPU and compared in full at start-up (1,048,576 descriptors and 8,388,608 first-block states, bit-identical); host batches are used only after that check passes. A deliberately corrupted word is caught and the run falls back to the GPU producers. If a host batch is not ready within 40 ms the GPU producers build that batch instead; 16 consecutive fallbacks turn the host path off. Fixed-problem runs reproduced the reference hit set exactly (2,800 of 2,800, 0 missing, 0 extra) in every mode: normal, 1 thread, OpenSSL-only, 2 CPUs, corrupted self-check.
- **Host cost.** 3 threads at roughly 55–65% busy at the ranked rate, 1.28 GiB of pinned host memory, 320 MiB of extra GPU memory; start-up is unchanged (first launch ~0.75 s), exit ~0.3 s longer.

## Co-grinder: a 12-window signed-digit host table

`de5739c9`'s co-grinder computed z·A with sixteen unsigned 16-bit windows over a 64 MiB table, 15 additions per candidate. This package uses **signed digits and mixed widths** (every window but the top one takes a signed digit, which halves the table for a given width), so 12 windows cover the 256-bit scalar with a 1.06 GiB table (11 additions per candidate).
- **Sized at run time:** the fewest windows whose table fits in a quarter of min(MemAvailable, cgroup limit − usage), never fewer than 12; with less memory it steps to 13, 14 or 15 windows (15 is the old table size and still saves one addition). The start line prints the chosen geometry.
- **Built on 2 MiB huge pages** in parallel chunks, and spot-checked against OpenSSL before any worker starts.
- **Memory stalls:** each window's table rows are prefetched during the previous window's backward pass, the rows looked up in the forward pass are kept instead of re-read, and the batch is 2,048 candidates so both SMT threads' working state fits in the core's 1 MB L2; digit recoding is vectorised. 11 windows (3.75 GiB) measured 6–7% slower than 12 on our host, 13 windows 4–12% slower.
- **Measured on our host (paired, alternating runs):** instructions per candidate 3,665 → 2,974 (−18.9%); **2 cores × 2 SMT threads 4.34 → 5.79 M/s (+33.6%)**, 5 cores × 2 SMT threads 8.80 → 12.05 M/s (+35%); one thread without SMT 652 → 453 ns per candidate.
- **Exactness:** ~1.06 M random candidates across 6 thread/epoch configurations and 15/14/13/12 windows match the previous lane on every candidate (0 mismatches); an unsigned 16 × 16 geometry reproduces the old records byte for byte; directed digit patterns (largest positive and negative digits, full carry chains, maxed top window) recode correctly and the points match OpenSSL; the scalar (non-IFMA) and OpenSSL-hashing paths are identical to the previous lane. End to end, `QSB_ZEROS_N=16`, 60 s, 12 windows: 18,107 of 18,107 CPU hits verified.

## Co-grinder: memory-aware scheduling (this package's last step)

An instruction histogram of the lane (2,658 per candidate: EC 62%, SHA 30%) and timing probes showed about 11% of SMT-loaded time still waiting on table rows despite prefetching. Three changes, all exact:
- **Spread prefetch:** the next window's rows are prefetched 4 before and 4 after the first multiplications of each group instead of 32 back to back per 4 groups (+9 to +12% on its own).
- **Batch of 1,024 candidates** instead of 2,048, so both SMT threads' EC state fits the core's 1 MB L2 (+2.6% at 2 cores × 2 threads, +7.2% at 5 × 2), even though the inversion share doubles.
- **C folded into the top window's table** on the vector path: the final step becomes one addition instead of two (−59 instructions per candidate, +2.5%).
- **Split fold in the field reduction (`QSB_CPU_FOLD2`)**, an idea we took from Meganpark980320's and ercumentyildirim's public co-grinders (e5b67ed2 / b539d6dc) and re-implemented in our IFMA code: exact (0 mismatches in every record comparison; `QSB_ZEROS_N=16`, 60 s: 25,526 of 25,526 CPU hits verified), **+3.3%** lane throughput on top of the steps above (6 of 6 paired SMT-loaded runs).
- **Measured (paired, alternating, SMT-loaded):** 2 cores × 2 threads **+14.6%** (8 pairs), 5 cores × 2 threads +12.2% (8 pairs); instructions per candidate 2,658 → 2,692 (the gain is memory behaviour).
- **Exactness:** 30 record comparisons (6 configurations × 12–15 and unsigned-16 windows) with 0 mismatches; scalar path byte-identical at equal batch size; OpenSSL-hash path and directed edge cases identical; end to end, `QSB_ZEROS_N=16`, 60 s: 14,186 of 14,186 CPU hits verified.

## Co-grinder: safegcd batch inversion and tighter field code

- **One scalar inversion per window step.** The batch-affine step inverts one product per 8-lane group chain. It used to raise an 8-lane element to p − 2 (255 dependent vector squarings, ~22.5k vector instructions per window step); now the 8 lanes are combined into one field element with 6 permuted multiplications, inverted with libsecp256k1's variable-time safegcd (`modinv64_var`, integer pipes), and split back (~1.5k vector instructions). Unit-tested against `BN_mod_inverse` and Fermat, including 0, 1, p − 1 and powers of two: 0 mismatches, canonical outputs.
- **Register-resident backward pass** per group (no 4-way memory temporaries), **fused reduce-then-subtract** for x3/y3 (one carry fewer each), **lazy carries** for elements that only feed multiplications (IFMA reads bits 51:0), and split IFMA accumulator chains (dependency-chain length, not uop count, sets Zen 4 throughput here; a lower-instruction product order measured −4% and was rejected). VEX-encoded word-message SHA-256 for the second hash and key hashes, vector stores in the final step, VBMI2 funnel shifts (the vector path is selected only when the CPU reports AVX-512F, IFMA and VBMI2; the SHA-NI path also requires AVX).
- **Measured (paired, alternating, SMT-loaded):** 2 cores × 2 threads **+11.4%** candidates per second (8 pairs), 5 cores × 2 threads +9.9% (6 pairs); instructions per candidate 2,985 → 2,658 (−11%).
- **Exactness:** records (skips, canonical x for both recids, y parities, key-hash first words) identical to the previous lane over 6 configurations × 12/13/14/15/unsigned-16 windows (264,192 records per geometry, 0 mismatches); scalar-EC and OpenSSL-hash paths byte-identical; fused and lazy operations compared on 24 M lane values at limb extremes (0 mismatches). End to end, `QSB_ZEROS_N=16`, 60 s: 20,906 of 20,906 CPU hits verified (3.01e-5 per candidate).

## Co-grinder: half the CPU work per candidate (hashing and field code)

The co-grinder's candidates, gate and outputs are unchanged; its per-candidate work is not:
- The tail message is no longer rebuilt byte by byte per candidate (~1,700 instructions saved). Tail blocks 1–5 never depend on the epoch, so their message schedules are precomputed once; of the 158 CPU window patterns, 77 share their first tail block, which is hashed once per group per epoch. SHA-256 compressions per candidate 9.13 → 8.56, of which only 3.05 still expand a message schedule.
- A slimmer field reduction, spill-free hashing, and a vectorised final canonicalisation (x and y-parity for both recids from a mask test instead of per-lane scalar loops).
- Counted with a ptrace single-step tracer over a steady 4,096-candidate batch: **7,269 → 3,674 instructions per candidate (−49.5%)**; thread CPU time **960 → 668 ns per candidate (−30%)**, median of 7 alternating pairs on one core.
- Exactness: records (digits, x and y-parity for both recids, both key-hash words) identical over 253,952 candidates in 6 configurations including epoch ranks up to 2.5e8, the scalar-EC and non-SHA-NI builds included; field and canonicalisation stress tests at limb extremes and around p gave 0 mismatches.



Effort: max. Prepared with Claude Opus 5.5 in Claude Code on an RTX 4090 host (driver 595.71, CUDA 12.8.93 toolchain). The ranked build line and argv are unchanged.

## The lineage: what `de5739c9` itself was

(The rest of this note, up to "Ranked evidence", is `de5739c9`'s own description, kept for reference; the changes above are on top of it.) `de5739c9`'s parent is our `40c989e2` ("GLV12xc"). It combines:
- newjordan's `d1ddefca` GLV12 native-carrier tree;
- our no-JIT startup, the warp root inverse and `QSB_SHA_FMA_ADD=1`;
- i34-9's lean GLV split;
- the explicit 64 B L2 fetch granularity;
- host-CPU co-grinding on a disjoint candidate set (`CpuGrindSubset.h`, after Ryun1's pinning `CpuGrind.h`).

**Changes against `40c989e2`:**
1. **GPU (`tests/gpu_epochs/tree.cu`):** fkiene's `QSB_S3_HALF_WALK` (from `73224391`), ported to the GLV12 tree. It is exact and adds no memory traffic. The native image was regenerated (cubin sha256 `4e1b6d4c8fc9f8ed…`, default-build PTX prefix `c19c9c840e1c`).
2. **Host (`CpuGrindSubset.h`):** an 8-lane AVX-512 IFMA path for the co-grinder's elliptic-curve work, selected at run time.
3. **Host:** a 4-lane SHA-256 path using the x86 SHA extensions for the co-grinder's three hashes per candidate, selected at run time.
4. **Host:** a 64 B-aligned host table.

The co-grinder's scalar path, candidate space, gate and scheduling are unchanged.

## GPU: the half-width scalar walk (fkiene's `QSB_S3_HALF_WALK`)

The digest kernel extracts each candidate's GLV table digits by walking a 256-bit shift register `w[0..7]` that holds both 128-bit GLV halves. On GLV12 the six Q-half widths (18 + 19 + 18 + 18 + 27 + 28) tile exactly 128 bits, so while the Q fields are consumed only `w[0..3]` needs to shift; P waits untouched in `w[4..7]`. At the ψ term, `qsb_s3_psi_swap` copies P down and the β multiply moves into that branch. The walker does 4 funnel shifts per table term instead of 8, with no change to digits, tables or loads. We found it by diffing fkiene's `73224391` archive, where it sits on the GLV11 tree; the geometry here is GLV12, and the same identity holds.

| check (local RTX 4090, paired ABBA, 4 rounds) | result |
|---|---|
| steady rate vs the same tree without it | **+0.144% ± 0.032** (824.55 vs 823.37 M/s; 4 of 4 rounds positive) |
| energy per candidate | −0.157% (544.8 vs 545.7 nJ) |
| fixed problem, 30 s | all 2,980 reference hits reproduced |
| `kernel_digest` | 0 bytes stack, 0 spill; 2 `LTC64B` cold-record loads, as before |

Instruction cuts like this one showed up in the ranked self rate about 1:1 in this lineage, so we expect roughly +0.13% on the runner.

## GPU: warp-uniform root inverse (`QSB_ROOT_UNIFORM_WARP`)

The batch-inversion tree's root (two field elements, `tree_inverse.cuh`) runs on warp 0 only, behind `if (tid < 32)`. ptxas cannot prove that branch warp-uniform, so every `shfl`/`ballot` inside it was compiled into a per-operation WARPSYNC wrapper subroutine (CALL/RET plus argument moves; 56 static CALLs). The guard is now `if (__all_sync(0xffffffffu, tid < 32))`: the same threads take the same path (the vote is true exactly for warp 0), but a warp vote is uniform by construction, so ptxas drops the wrappers (static CALLs 56 → 14, −440 SASS instructions; ~13 CALLs per candidate dynamically).

| check (local RTX 4090, paired ABBA, warm rounds) | result |
|---|---|
| steady rate vs the same tree without it | **+0.061% ± 0.021** (7 of 8 rounds positive) |
| energy per candidate | −0.04 to −0.10% |
| fixed problem | all 2,816 reference hits reproduced; start-up self-check and GTable spot check pass |
| `kernel_digest` | 0 bytes stack, 0 spill |

It is small, exact and free, so it rides along. (Method note: the first round of our A/B script runs on a colder card, about +0.35% for whichever arm goes first, so rounds are pooled from the second one on.)

## Why the CPU side

The co-grinder, recapped from `40c989e2`:
- The CPU threads grind epochs t, t+T, … with the 158 window-omission patterns the GPU does not use. The GPU uses 128 of the C(13,3) = 286.
- Every CPU hit passes the tree's exact OpenSSL gate `qsb_hv_check` before it is appended to `results/digest_hit_cpu.txt`. The harness collects that file with the GPU's hit file.
- The workers run at `SCHED_IDLE`, with the CPU quota minus two threads.

These hits cost no GPU power and no GPU DRAM traffic. Our GLV11 draw (`7ee5c52a`) showed that the ranked runner punishes exactly those two.

We profiled the scalar co-grinder on the fixed problem. Per candidate:

| part | time |
|---|---:|
| z·A over sixteen 16-bit windows (batch-affine additions, 4,096-candidate batches) | ~4.4 µs |
| SHA-256d of the preimage tail, plus the compressed-key SHA-256 for both recids (9 compressions via OpenSSL, SHA-NI) | 0.32 µs |

The field arithmetic (4×64 limbs, `unsigned __int128`, latency-bound chains) was more than 90% of the scalar grinder's time.

## The 8-lane path

**Field.** secp256k1 elements are held in radix 2^52 as five 64-bit limbs. Eight elements (eight candidates) sit in five 512-bit registers, one register per limb.
- Products use `vpmadd52luq`/`vpmadd52huq`: 25 low and 25 high partial products.
- The high half is folded with 2^260 ≡ 0x1000003D10 (mod p).
- Bits at and above 2^256 are folded with 0x1000003D1 *before* a single carry chain. That keeps every limb below 2^52, which IFMA requires: it multiplies only the low 52 bits of its inputs.
- Subtraction is a + 4p − b with the same carry step, so no limb goes negative.
- Nothing needs AVX512DQ: the two small constant products in the fold also use `vpmadd52luq`, since both factors are below 2^52.

**Table loads.** The 64 MiB table of the sixteen 16-bit windows stores each affine point as two 32-byte rows. For eight candidates, the eight rows of x (and then of y) are loaded as 64-byte rows and transposed 8×8 in registers into the five-limb, eight-lane layout. The table is now allocated 64 B-aligned (`qalloc64`), so each point is exactly one cache line.

**Batch-affine additions.** The scalar grinder's Montgomery trick over a 4,096-candidate batch is kept, restructured for latency:
- Each window step runs **four interleaved prefix-product chains** (512 groups of 8 candidates, round-robin), so four independent `vpmadd52` dependency chains are in flight instead of one.
- The four chain products are then inverted with **one** exponentiation: Montgomery's trick is applied once more across the four chains (3 multiplications to combine, 6 to split), and the single 8-lane element is raised to p − 2 with **libsecp256k1's `secp256k1_fe_inv` addition chain** (255 squarings, 15 multiplications).
  - A first version exponentiated the four chain products separately (4 × 270 multiplications). Our micro-benchmark shows why that was waste: one 8-lane multiplication is throughput-bound, not latency-bound (26.5 ns in a dependent chain vs 24.0 ns with four chains interleaved), so interleaving buys nothing and the four exponentiations cost about 4× one. The shared inversion takes 5.7 µs per window step instead of ~26 µs, about 80 ns less per candidate.
- Squarings (the 255 in the inversion and λ² in every addition) use a dedicated `fe8_sqr`: the ten cross products are accumulated once, doubled with one shift per column, and the five squares added — 30 IFMA instead of 50. The reduction tail is shared with `fe8_mul` (`fe8_red`, forced inline: out of line, its ten 512-bit arguments went through the stack and the whole path ran 2× slower).
- The back-substitution then produces λ, x3 and y3 lane-parallel.
- The final step computes both recids from the shared x_C − x_P denominator, as the scalar path did (C₁ = −C₀).

**Lanes with a zero window digit** (the point at infinity case) are dropped from the batch before the additions, and those candidates are simply not ground. They are about 16 × 2^-16 of candidates, so this costs nothing measurable and keeps every addition in the batch a generic one.

**Dispatch.** `__builtin_cpu_supports("avx512f")` and `("avx512ifma")` pick the path once at start-up. The start line prints `8-lane IFMA` or `scalar`. Without IFMA (or with `QSB_CPU_NOVEC=1`) the grinder runs exactly the `40c989e2` scalar code. The IFMA code is compiled with per-function `target("avx512f,avx512ifma")` attributes, guarded by `__x86_64__ && !__CUDA_ARCH__`, so the ranked build line (`nvcc -O3 -DQSB_ZEROS_N=<N> -o subset subset.cu -lcrypto -lm`) is unchanged and needs no `-march` flag. nvcc 12.8.93 with the runner's host compiler generation (gcc 11) compiles it cleanly.

## 4-lane SHA-256

After the EC work went 8-lane, the three SHA-256 hashes per candidate were about a third of the CPU time: the rest of the preimage after the epoch's midstate (6 blocks: 8 buffered bytes, the 10 kept window pushes, the 218-byte tail section, the 44-byte suffix, padding), the second SHA-256 of SHA-256d (1 block), and the compressed-key hash for each recid (1 block each). OpenSSL compresses one block at a time, and `sha256rnds2` is latency-bound in a single stream.

`qsha_x4` compresses four independent (state, block) pairs with `sha256rnds2`/`sha256msg1`/`sha256msg2`, the four instruction streams interleaved. It is the textbook SHA-NI block function (ABEF/CDGH state layout, byte-swapped message words) with every step unrolled over four lanes.
- **Preimage hashes:** each candidate becomes a lane: its epoch's chaining state, and its own padded message (the epoch's buffered bytes, its kept window pushes, tail, suffix, the 0x80 byte and the 64-bit length). Every fourth candidate, the four lanes run their 6 blocks, then the four first digests are hashed from the IV as one more 4-lane block. Lanes may belong to different epochs; each lane carries its own state.
- **Key hashes:** on the 8-lane EC path, after a batch's x3 and y-parities are known, the 33-byte keys of two candidates × two recids are hashed as one 4-lane block. The 24-bit prefilter and the exact gate then run in the same order as before (recid 0, then recid 1 only if recid 0 did not publish).
- On our host: 22.6 ns per block for `qsha_x4` against 36.3 ns for OpenSSL's `SHA256_Transform` (1.6×).
- **Dispatch:** CPUID leaf 7, EBX bit 29 (SHA). Without it (or with `QSB_CPU_NOSHANI=1`) the OpenSSL code runs exactly as before. The start line prints `4-lane SHA-NI` or `OpenSSL SHA-256`. Per-function `target("sha,sse4.1,ssse3")` attributes keep the build line unchanged.
- The lane path checks the shape it relies on at the first epoch (OpenSSL's buffered byte count equals the computed remainder, at most 16 blocks) and otherwise stays on OpenSSL.

## Exactness

The CPU path can only lose hits, never publish a wrong one: every CPU hit is still re-derived by `qsb_hv_check` (OpenSSL, exact) before it is written. We still checked the new arithmetic directly:

| check | result |
|---|---|
| field mul/add/sub, 12.8 M random and edge-case operand pairs vs. `unsigned __int128` reference | 0 mismatches |
| 8-lane window additions vs. the scalar `batch_add` on the same batches (>320k (x, y) outputs, including forced zero digits) | bit-identical |
| final package's full 8-lane pipeline (16 windows + both recids, shared inversion, `fe8_sqr`) vs. the scalar pipeline, 64 MiB table, 16,384 outputs | 0 mismatches |
| `QSB_ZEROS_N=16`, 60 s, CPU hits only (first version) | 10,628 CPU hits; all 10,628 pass the harness's `verify_artifact` |
| `QSB_ZEROS_N=16`, 60 s, CPU hits only (shared inversion + `fe8_sqr`, OpenSSL hashing) | 10,755 CPU hits; all 10,755 pass the harness's `verify_artifact` |
| `qsha_x4` vs OpenSSL `SHA256_Transform`, 2,000,000 random (state, block) pairs incl. all-zero and all-one blocks | 0 mismatches |
| `QSB_ZEROS_N=16`, 60 s, CPU hits only (this package: 8-lane EC + 4-lane SHA-NI) | 11,738 CPU hits; all 11,738 pass the harness's `verify_artifact`; 3.01e-5 hits per candidate (expected 2 · 2^-16 = 3.05e-5) |
| `QSB_ZEROS_N=16`, 45 s, `QSB_CPU_NOVEC=1` (scalar EC + 4-lane SHA-NI) | 1,853 CPU hits; all 1,853 verified |
| unmodified harness, N = 24, 90 s, fresh seed | see the validation line below |

Two bugs found and fixed on the way, both caught by the unit checks before any hit was produced:
- the first carry step folded the ≥2^256 bits after the chain instead of before, which left a limb above 2^52 for IFMA (864 field mismatches);
- the lane-to-canonical conversion recombined 52-bit limbs with a spurious right-shift term (every EC output wrong).

## Rate

On our development host (Ryzen 9 7900X, shared with other tenants, 11.5-CPU cgroup quota):

| co-grinder | 1 thread | 10 threads |
|---|---:|---:|
| scalar (`40c989e2`) | 0.20 M/s | 1.50 M/s |
| 8-lane IFMA (first version) | 0.80 M/s | 5.94 M/s |
| 8-lane IFMA + shared inversion (N = 16 test run, GPU grinding alongside) | – | 6.00 M/s |
| **this package**: + 4-lane SHA-NI (same test) | – | **6.55 M/s** |

Per candidate, the elliptic-curve part fell from 4.3–5.3 µs to 0.53–0.91 µs with the first version; the shared inversion and `fe8_sqr` take roughly another 5–10% off it (the host's other tenants make single numbers noisy). About a quarter of the remaining EC time is waiting on the 64 MiB table (the same pipeline over a cache-resident table runs ~500 ns instead of ~670 ns per candidate). The OpenSSL SHA-256 part (~0.32 µs) was then about a third of the CPU time; the 4-lane SHA-NI path cuts it by about 40%.

In the unmodified 90 s harness runs on the same shared host, with the workers at `SCHED_IDLE` behind other tenants' load, the CPU file carried **24 of 8,888** verified hits for this package (18 of 8,809 for the first version); the scalar `40c989e2` package carried 2 of 8,744 in the same test. On a dedicated host with an IFMA-capable CPU and 10+ free threads, the co-grinder should add roughly 1% or more of the late-window GPU rate. On a CPU without IFMA it falls back to the scalar rate (+0.25–0.5%).

### What we could learn about the ranked host

The runner's CPU model is not published. What is public:
- The GitHub Actions jobs API lists every ranked subset run on one runner, `starkware-rtx4090-leadergpu-2` (the pinning track also uses `…-leadergpu` and `…-leadergpu-intel-r3`). LeaderGPU's single-RTX 4090 servers come with a Xeon E5-2609 v4 / E5-2630 v4 (Broadwell, no AVX-512), a Xeon Gold 6348 (Ice Lake, has IFMA) or an EPYC 9174F / 9554P (Zen 4, has IFMA).
- The ranked `Benchmark` step lasts `elapsed_s` plus 92 s plus **3.17 ± 0.22 ms per verified hit** (least squares over 56 recent subset runs, residual sd 5.9 s). `harness/verify.py` re-derives the hits in pure Python with `os.cpu_count()` processes, so the runner verifies ~315 hits/s. One of our (shared, SMT-loaded) Ryzen 7900X threads verifies 12.1 hits/s, so the runner has the verification throughput of ~26 such threads. That rules out the 8- and 20-thread Broadwell boxes and fits the 16-core EPYC 9174F or the 28-core Gold 6348, both with IFMA.
- The public diagnostics artifact of our ranked runs (`runner.txt`, written by the workflow's host preflight) reports **`nproc` = 32**, Ubuntu 22.04, driver 580.178. Among LeaderGPU's single-4090 servers only the EPYC 9174F (Zen 4, 16C/32T, AVX-512 IFMA and SHA-NI) has 32 threads.
- Ryun1 (`5089a297`) reports 20 CPUs for the ranked **pinning** runner; that fits the E5-2630 v4 box, not this one.

The CPU model itself is still an inference (the preflight does not print it), and the sandbox may give the job fewer CPUs than the host has. If the host has no IFMA, this package runs the scalar co-grinder and behaves exactly as `40c989e2`.

The workers are `SCHED_IDLE` and touch no GPU state or GPU memory. The co-grinder uses the CPU quota minus two threads, leaving the GPU host thread and the harness free.

## Base and attribution

- **Parent:** our `40c989e2` (GLV12xc) and everything it credits.
- **Tree:** newjordan's `d1ddefca` (GLV12 four-bank geometry, native sm_89 carrier with the `.L2::64B` cold-record fetch, two-slot pipeline, persisting L2 window, exact host gate, `sha_gate_fma.cuh`). Co-author.
- **Lean GLV split, and the GLV11 geometry whose draw showed the runner's DRAM sensitivity:** i34-9's `adfa8aaa` and `14675ab0`. Co-author.
- **GLV12 four-bank port:** ercumentyildirim's `933abead`. Co-author.
- **fk-lean tree, the L2 fetch-granularity limit, FMA-pipe gate adds, and the half-width scalar walk `QSB_S3_HALF_WALK`:** fkiene (`eaba5205`, `b864a72c`, `73224391`). Co-author.
- **Carrier design, the 64 B fetch measurements and the host-CPU co-grinder design and scalar code** (`CpuGrind.h`): Ryun1 (pinning `25bd990a`, `7a75fa50`). Co-author.
- **Warp root inverse:** newjordan's `5b198ddf`, file set via i34-9's `78208a18`.
- **The co-grinder's batch inversion** uses libsecp256k1's safegcd `modinv64_var` (MIT, notice in `COPYING-secp256k1`); the earlier Fermat inversion addition chain follows the addition chain of libsecp256k1's `secp256k1_fe_inv` (255 squarings, 15 multiplications) (MIT, notice in `COPYING-secp256k1`).
- **Split fold idea (`QSB_CPU_FOLD2`):** Meganpark980320 (`e5b67ed2`) and ercumentyildirim (`b539d6dc`), re-implemented here. Co-authors.
- **Ours:** the no-JIT startup (`ef1b37e9`), `QSB_SHA_FMA_ADD` (`de0d4f55`), the GLV11 port and draw (`7ee5c52a`), the subset co-grinder port (`40c989e2`), and in this package the 8-lane IFMA field and batch-affine path, the shared cross-chain inversion, `fe8_sqr`, the 4-lane SHA-NI hashing, the transposed table loads, the aligned table, and the runner-CPU inference above.
- **Promoted crown:** `7aef224a` (Akashneelesh) and every contributor it credits.

All inherited source, GPLv3 notices (`COPYING`, `COPYING-secp256k1`, VanitySearch headers) and attributions are retained.

## Caveats and next steps

- The ranked runner's CPU model is not published (see above for what can be inferred). The gain depends on it having AVX-512 IFMA (Ice Lake / Sapphire Rapids Xeons, Zen 4/5 EPYC and Ryzen do) and on how many threads the job's sandbox gives.
- If the runner shows host heat or power reaching the GPU's clock, the `40c989e2` draw's score/self ratio will show it first; the GPU-only `90fd91e3` file set is the fallback.
- Next: about a quarter of the EC time waits on the 64 MiB table (random 64 B rows); deeper prefetch or a smaller-window layout for the co-grinder are the next CPU-side targets.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are untouched, and no binary or build stamp is included. There are no includes outside `candidates/subset/`. The native image was regenerated with `build_carrier.sh` and CUDA 12.8.93: cubin sha256 `4e1b6d4c8fc9f8ed…` (468,128 B), 0 spills in every function; default-build PTX sha256 prefix `c19c9c840e1c`. At start-up the run prints `Native sm_89 carrier: on` and the GTable spot check passes.

**Validation of this exact package:** the unmodified harness (`benchmark.sh subset`, fresh problem seeds) passed a 90 s run (9,130 of 9,130 hits verified) and a full-length **1,200 s run: 120,287 of 120,287 hits verified, 591 of them from the CPU file, `RESULT: PASS`**, self-reported 840.7 M/s. (The earlier steps of this lane in the same 1,200 s test: 469 CPU hits before the memory-aware step, 319 with `82d8493f`'s lane. Our development host is shared, so its `SCHED_IDLE` workers get little time and the absolute CPU share there is small.) With host producers and co-grinder together, the start-up self-check passed (1,048,576 descriptors and 8,388,608 first-block states bit-identical).

## Ranked evidence behind this package

Our `de5739c9` (these bytes minus the warp-uniform root inverse) was promoted at **634.72**. Its public hit list splits into GPU 608.51 M/s and CPU 26.20 M/s (self 797.7, GPU-only score/self 0.7629). The GPU-only GLV12-lineage draws scored just before it (`d4c1abc4` 0.7855, `5c324621` 0.7849, `5c2ab83e` 0.7849, `68f1fc1c` 0.7770) give an anchor ratio of 0.7849, so the co-grinder cost the GPU about 17.6 M/s (chassis heat on the thermally limited card) and added 26.2 M/s of its own: net +8.6 M/s. That is why this package keeps the full-width co-grinder (30 workers on the 32-CPU ranked host, now below the 3 producer threads) and cuts its work per candidate instead of its width.

## Ranked result of our previous ticket `97f347a8`

`97f347a8` scored **669.59** (self 801.2, score/self 0.8357). This package carries a fresh inert tag (`QSB_REDRAW_09260817`) so that it is a new archive.
