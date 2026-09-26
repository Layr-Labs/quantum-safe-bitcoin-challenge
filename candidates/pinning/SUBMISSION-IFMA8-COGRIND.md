# Pinning: 8-Lane AVX-512 IFMA Co-Grinder (+158.7% CPU Throughput, +3.9% Net Score)

This candidate integrates a native 8-lane AVX-512 IFMA (`vpmadd52luq` / `vpmadd52huq`) elliptic-curve batch-affine co-grinding engine into the pinning track host solver (`candidates/pinning/cg_ifma8.h` and `cpu_cogrind.h`). The GPU pipeline and native sm_89 carrier remain completely untouched and identical to the promoted frontier (`e892e6e` / `0c9471e`).

## 1. Background & Rationale

The pinning leader (`e892e6e`, scored at 979,222,732 candidates/s) currently relies on a 4-lane AVX2 EC co-grinder (`cpu_cogrind_vec.h`) using libsecp256k1's 10x26 radix representation. On 30 host worker threads, AVX2 yields ~24.2 M cand/s (4.8 M/s on r5 to 18.0 M/s on 356).

Meanwhile, in the subset track, solvers achieved 48.3 M/s host throughput by utilizing an 8-lane 5x52-bit AVX-512 IFMA arithmetic engine (`_mm512_madd52lo_epu64` and `_mm512_madd52hi_epu64`). Because the competition runners are AMD EPYC Zen 4 instances with native 512-bit IFMA pipelines, porting this 8-lane IFMA architecture to pinning provides an immediate, large-magnitude performance edge on host cores without introducing any GPU risk or launch bubbles.

## 2. Mathematical & Algorithmic Implementation

All modifications are confined strictly to `candidates/pinning/`:

1. **`candidates/pinning/cg_ifma8.h`**:
   - **Radix $2^{52}$ Vector Representation**: Each field element is represented as 5 64-bit limbs across 8 vector lanes (`struct fe8 { __m512i l[5]; }`).
   - **Fused Multiply & Square**: `fe8_mul` completes 8 parallel field multiplications in ~50 IFMA instructions; `fe8_sqr` computes 8 squarings in 30 IFMA instructions (cross-products doubled once).
   - **One Exponentiation Chain Batch Inversion**: `fe8_inv1` inverts 8 elements simultaneously using libsecp256k1's 255-square + 15-multiply addition chain. Across a batch of 2048 candidates, a single inversion pass per window takes ~1.5 µs (<0.7 ns per candidate).
   - **Zero-Cost Handling of Signed & Zero Recoded Digits**:
     - Negative digits (`QCG_NEG`): Handled by branchless negation `fe8_neg` and `_mm512_mask_blend_epi64` using `get_neg_mask`.
     - Zero digits (`QCG_ZCODE`): Handled by masking $\Delta x = 1$ in the Montgomery product chain and keeping old $(X, Y)$ points via `_mm512_mask_blend_epi64(zm, x3, px)`.
   - **Vector Canonical Reduction**: `fe8_canon_scatter_arr` fully reduces $x$ modulo $p$ into canonical 4x64 little-endian integers and derives exact $y$ parity for compressed pubkey formatting.
2. **`candidates/pinning/cpu_cogrind.h`**:
   - Added runtime detection of `avx512f` and `avx512ifma`.
   - Dispatches Mode 3 (`v8::ec_batch`) on IFMA-capable hosts with graceful fallback to Mode 2 (AVX2), Mode 1 (MULX), or Mode 0 (C portable).
   - Automatic calibration benchmark evaluates all supported backends on startup and selects the fastest path.

## 3. Verification & Benchmark Proof

- **Bit-Exact Correctness**: A rigorous standalone verification suite (`test_exactness.cpp`) compared 2,048 full candidates across all windows between AVX2 (v4) and AVX-512 IFMA (v8). All 2,048 candidates produced **100% bit-identical** canonical $(X, Y)$ affine coordinates.
- **Exact Gate Verification**: Every tentative hit nominated by the CPU grinder is validated by the unchanged OpenSSL exact gate (`qsb_host_exact_hit`) before being written to `results/pinning_hit_cpu.txt`. In test runs over 40,960 candidates, **123 out of 123** tentative hits were verified by OpenSSL (100.0% verification rate, 0 false positives).
- **Throughput Measurement**:
  - `v4` (AVX2 4-lane): 0.809 M cand/s per worker (2.531 ms/batch) -> ~24.27 M/s across 30 workers.
  - `v8` (AVX-512 IFMA 8-lane): 2.093 M cand/s per worker (0.978 ms/batch) -> ~62.80 M/s across 30 workers.
  - **Net Speedup**: **2.587x (+158.7%)** on CPU EC throughput.
  - **Net Expected Score Gain**: **+38.53 M cand/s (+3.93%)**, well above the +1.0% (+9.79 M/s) promotion threshold.
