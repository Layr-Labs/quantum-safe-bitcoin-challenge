# AVX2 live-column preparation attempt

Baseline: `efef868ab78ff8d1229cdc1591b90797ee3be197` (official QSB source ref).
Production change: host-only `message_live_column` and `hash_record` in
`candidates/pinning/pinning.cu`. The GPU source algorithms, embedded carrier,
OpenSSL publication gate, candidate enumeration, and judge are unchanged.

Run the reproducible check with:

```
python3 candidates/pinning/tests/hostsha/check_live_words.py
```

The test extracts both versions' actual host functions, compiles them together
with GCC 13.3.0 at `-O3`, and links OpenSSL for an independent digest reference.
The local CPU exposed AVX2 and SHA-NI. The benchmark deliberately selects AVX2,
which the source documents as the expected ranked-host backend. It runs on one
pinned logical CPU; it does not use CUDA.

Validation passed:

- 32,768 independent SHA256 compressed-public-key H0 checks against OpenSSL,
  including zero/all-one x limbs and both 02/03 prefixes. Unused padding rows
  were filled with arbitrary values to verify the structured compressor only
  consumes message rows 0..8.
- 196,608 baseline/modified `hash_record` comparisons across scalar, AVX2, and
  SHA-NI paths, covering inactive lanes and short batches at an easier 6-bit
  gate so hit lists are exercised frequently. Hit indices and recid choices
  matched exactly.
- `git diff --check` passed.

Twelve 0.75-second alternating ABBA/BAAB microbenchmark epochs measured:

| Epoch | Path | Candidate/s |
|---:|---|---:|
| 0 | Baseline | 9,890,875.815 |
| 1 | Modified | 11,086,686.807 |
| 2 | Modified | 11,153,889.443 |
| 3 | Baseline | 10,094,997.328 |
| 4 | Modified | 10,981,066.116 |
| 5 | Baseline | 9,974,192.676 |
| 6 | Baseline | 10,074,077.155 |
| 7 | Modified | 10,922,101.639 |
| 8 | Baseline | 9,931,241.905 |
| 9 | Modified | 10,739,407.539 |
| 10 | Modified | 10,719,730.695 |
| 11 | Baseline | 9,742,238.028 |

Median: 9,952,717.290 -> 10,951,583.877 candidate/s, **+10.0361%**
host AVX2 hashing throughput in this microbenchmark.

This is not an overall QSB speedup. A CUDA compiler and RTX 4090 were not
available. The full GPU program has not been compiled, run, or evaluated by the
official judge. The gain may translate into more completed offloaded work only
when host hashing limits that path; GPU throughput, host-offload admission,
thread contention, and official hit-based scoring still need measurement.
