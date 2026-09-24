# Subset: the a329eeee composite plus the PR965 narrow parity window, as a self-contained build, with a measured census on a dedicated RTX 4090

Effort: max. Prepared with Claude Opus 5.5 in Claude Code on a dedicated RTX 4090 (driver 595.71, CUDA 12.8.93 toolchain; ranked build line unchanged).

## Summary

This submission packages the strongest exact composite we could measure locally. It started from a head-to-head measurement of the public trees, all against the promoted crown bytes.

- **Base:** our earlier public composite `a329eeee`. That is PR1134 (the `36c05f97` source) plus K32 and signed fused X3.
- **Added:** the PR965 narrow parity window.
- **Self-contained build:** the SHA helper header, which the base included from the sibling pinning track, now lives inside `candidates/subset/`.

On a fixed seed, in paired ABBA order, the package measures **+1.85% ± 0.09 throughput and −1.92% energy per candidate versus the promoted crown bytes**. A 180 s run of the unmodified harness verified 15,827 of 15,827 hits.

## Base and attribution

Direct base: `a329eeee` (terrapinelf; subset commit `43b2fb8f`), which is the `36c05f97` source plus two runtime mechanisms. The mechanisms it carries, with their origins:

- **Base-A scalar recoding** (`QSB_RECODE_BASE_A`): Akashneelesh PR1027.
- **Bounded early-Z2 filter carry** (`QSB_DROP_Z2_EARLY`): dun999 PR1093.
- **Exact host (OpenSSL) publication gate** (`QSB_HOST_VERIFY`): mitchuski PR918.
- **Two-slot non-blocking host pipe** (`QSB_HOST_PIPE`).
- **Negative-ordinate point-chain MAC** (`QSB_SUBSET_NEG_Y_MAC`): Saviour1001.
- **Offset-ordinate filter table** (`QSB_YOFF_FILTER`): kayu052 PR1099.
- **Isomorphic recovery** (`QSB_ISO_*`).
- **K32 half-limb corrections** (`QSB_K32`): fkiene's PR1002 form, via hybridnoise PR1137.
- **Signed fused X3** (`QSB_FUSE_X3`): hybridnoise PR1137.
- **The promoted crown `7aef224a`** (Akashneelesh) and its lineage, which supply the negfold parity, SHORT_CARRY4, EPOCH_FAST with 128 windows, and the parity window. Credits for those are retained in the inherited notes.

The narrow parity window is Portablelle's public PR965, taken from hybridnoise's `f9738952` tree.

All inherited notices (`COPYING`, VanitySearch headers, `COPYING-secp256k1`) and attributions are retained.

## What changed relative to a329eeee

1. **`tests/gpu_epochs/parity_window_subset.cuh`: PR965 narrow window** (`QSB_K2S_PARITY_NARROW=1`).
   - The recovery y-parity product window drops from 27 to 18 partial products.
   - A guard falls back to the original 27-product decision, and so to its speculative fallback, whenever the narrowed accumulators could differ. The emitted parity is therefore identical to the base on every input.
   - The file is byte-identical to the one in `f9738952`. The base file equals that file minus exactly this addition.
2. **`sha_pinsha.cuh` vendored into `candidates/subset/`.**
   - The base's `tree.cu` included `../../../pinning/sha_pinsha.cuh`, which made the subset build depend on whatever the pinning track currently holds.
   - The header is now a local copy, and the include path is `../../sha_pinsha.cuh`.
   - The generated PTX is identical to building against the current pinning header: sha256 prefix `a5ebce1dcc30` for both.

Everything else is the base byte for byte: table geometry (15 chunks, 64 MiB), launch shape (256 threads, 2 CTA/SM, 128 registers, 49,152 B shared), enumeration, filter and exact gate. Setting `-DQSB_K2S_PARITY_NARROW=0` restores the base PTX.

## Measurement method

- **Hardware and state:** one dedicated RTX 4090 at the default 450 W limit. `sw_power_cap` is active throughout, so the card runs power-bound at 2,520–2,560 MHz and 63–76 °C.
- **Problem:** a fixed generated problem (seed 424242).
- **Arms:** each arm is one 62 s process launch of the prebuilt binary with the ranked argv. Arms are ordered ABBA across rounds, so linear thermal drift cancels.
- **Rate:** measured between the first and last progress lines. This excludes startup, as the ranked peak rate does.
- **Energy per candidate:** mean board power, sampled at 1 Hz by `nvidia-smi` over the same window, divided by rate.
- **Checks:** every variant's PTX is hashed before it is timed, so inert flags cannot masquerade as neutral results. Hit yield is also checked against the candidate count on every arm.

Because the card is power-bound, throughput here is effectively candidates per joule. Our reading of the corpus is that energy per candidate is also what the ranked subset runner rewards: its rate decays about 15% from peak.

## Results (paired against the promoted crown bytes)

| tree | Δ throughput | Δ energy/cand | notes |
|---|---:|---:|---|
| **this submission** | **+1.853% ± 0.085** | **−1.92%** | 3 rounds |
| a329eeee (base) | +1.761% ± 0.064 | −1.80% | narrow parity window worth +0.09% |
| f9738952 (hybridnoise) | +1.931% ± 0.359 | −1.87% | carries `QSB_SHA_FMA_ADD=1`; its ~20 public redraws average ≈617.5 officially |
| 36c05f97 | +1.482% ± 0.244 | −1.42% | |
| e63e42ec | +0.724% ± 0.721 | −0.68% | |
| 35c4db43 | +0.569% ± 0.545 | −0.52% | |
| 5744a581 | +0.220% ± 0.316 | −0.16% | |

**Full-harness validation.** `QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src …' QSB_SECONDS=180 ./benchmark.sh subset` on a fresh random seed:
- verified hits 15,827 of 15,827;
- grinder self rate 753.1 M/s;
- `RESULT: PASS`.

(That local harness clock also included a recompile, because the build stamp had been removed. The ranked path builds in setup.)

## Negative or neutral results on this base (not shipped)

| change | Δ throughput |
|---|---:|
| Rolled constant-SHA loops (`QSB_PAIR_SHA_UNROLL_CONST=0`) | −0.43% ± 0.12 |
| PR950 packed lane classes (`QSB_950_PACK`) with the unrolled schedule | −0.15% |
| `QSB_SE_WINDOWS=256` | −0.64% ± 0.12 |
| `-DQSB_SHA_FMA_ADD=1` | inert: hard-defined 0 in the base; PTX identical |

## Research findings for other solvers

1. **The chain dominates.** A wrong-math probe that removes 3 of the 12 interior mixed additions runs +15.26% ± 0.01 faster and uses −13.3% energy per candidate. One XYZZ madd is therefore about 4.4–5% of a candidate's energy.
2. **One 512-candidate block inverse costs about 3.6%.** A wrong-math probe that skips the recovery tree measured +3.68% ± 0.09.
3. **Per-launch timeline:**
   - digest kernel 98.95% (178.2 ms);
   - producers 0.95%;
   - verify 0.06%;
   - host 0.03%.

   Host and pipeline work therefore has a ~1% ceiling.
4. **GLV12 four-hot on subset is DRAM-latency-bound.** We ported the pinning FOUR_HOT geometry: 9.8 GB table, 48 MiB persisting window, 11 additions.
   - With all loads forced into the hot banks it runs +13% over the crown (834 vs 739 M/s). The arithmetic side works.
   - With the real table's 4 random DRAM gathers per candidate it falls to 557 M/s at only 403 W, and energy per candidate rises 19%.
   - Without the persisting window it is worse (457 M/s). A `prefetch.global.L2` of the cold records is also worse (516 M/s).
   - Shrinking the cold region to 256 MiB or 2 GiB barely helps (569 and 551 M/s). This is DRAM latency, not TLB.
   - This agrees with the ranked 933abead and 1f78c9f4 results.
5. **Affine pair level is correct but blocked by registers and shared memory.** We implemented summing T0..T13 in 7 affine pairs, using Montgomery's trick over the 7 pair denominators and one extra shared block inverse, followed by a 7-addition deferred-Y XYZZ chain.
   - It is exact. Hits verify at the normal yield.
   - The 6 per-candidate prefix products do not fit beside the chain in 128 registers and 49 KB of shared memory. Local-memory traffic made it 2× slower (367 M/s).
   - The arithmetic ledger (−63 mul-eq per candidate for the 7 saved madds, +39 for the affine sums and Montgomery, plus one ~3.6% collective) says roughly +4% is available to a layout that can store them.
6. **Cold driver JIT** for this kernel family is about 12 s on this host (`CUDA_CACHE_DISABLE=1`).

## Expected ranked effect and limits

Two official draws of this exact lineage exist: 36c05f97 at 623.05 and a329eeee at 621.98. Two repackages drew 616.6 and 614.9. Together they suggest the local gain transfers at about 0.8–0.9. We expect about 620 ± 4 officially.

This is a composite of exact, independently switchable mechanisms, not a new algorithm. The single-run noise of the ranked subset runner (σ ≈ 0.6–1%) is comparable to the gain.

## Packaging

Only `candidates/subset/` changes. The harness, verifier, problem, setup, benchmark, workflow and sibling track are all untouched. No binary or build stamp is included. The ranked build line and argv are unchanged.
