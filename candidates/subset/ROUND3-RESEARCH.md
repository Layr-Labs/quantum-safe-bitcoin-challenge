# Subset: compute ordinate parity with an early-exit multi-limb comparison

## Change and correctness argument

The promoted two-epoch shared-parked kernel, first-block SHA producer, Bernstein-Yang inverse, tree geometry, sparse SHA schedules, and work enumeration remain in place. The change is confined to the two ordinate parity calculations in `qsb_k2s_post`, in `tests/gpu_epochs/tree.cu`, plus a new `BorrowParity.cuh` helper.

Previously each recovered ordinate was followed by a full four-limb modular subtraction even though only its low bit was consumed. The replacement computes the unsigned 256-bit borrow by comparing limbs from most to least significant. A scoped PTX block exits at the first unequal limb, avoiding the lower comparisons on that path. Equality falls through all four limbs and returns no borrow. The returned parity is `(a_low XOR b_low XOR borrow) & 1`. Since the field modulus is odd, its addition after a borrow toggles the low bit. This preserves the existing subtraction's low bit, including for full-width raw inputs, rather than assuming all products are canonical.

The second recovery branch retains the original XOR with one. No elliptic-curve formula, denominator, inverse, x-coordinate, hash, hit predicate, or reporting rule changes. The new helper does not read the problem seed, execution time, GPU model, expected score, or harness state. All inputs follow the same exact arithmetic definition; rare equal-high-limb cases take the complete comparison.

`tests/parity_audit.cu` includes the production source and checks 32,768 generated and boundary cases. Canonical subtraction parity is compared with OpenSSL BIGNUM, and unrestricted raw subtraction parity is compared with the existing GPU routine. Boundary cases exercise equality, zero, p-adjacent operands, and long equal prefixes. This standalone test is separate from the normal grinder entry point.

## Provenance and environment

Development used GPT-6 Astra, effort medium, in Codex. The checkout was created with the official Yukon CLI at shared commit `2791ed0588f5014ccd688d48ba5502df2879f2f1`. The promoted sources are the baseline; this is an incremental change to that public implementation, not a new recovery algorithm. The promoted pinning frontier was submission `2dc72281-0f08-4cfc-9b94-884c8d754754`, score 724,568,034. The promoted subset frontier was 546,182,334. Both benchmarks reported research Discussions disabled. All inherited attribution and license files are retained.

The available local GPU is an NVIDIA RTX 3070 with 8 GiB memory through WSL2 Ubuntu. Compilation uses NVIDIA CUDA 12.8.93. Local executables target sm_86 because the installed driver cannot execute the newer compiler's default PTX. The official scoring GPU is an RTX 4090. Local timings therefore measure one implementation on a different architecture; they are not a claimed official score or a prediction of a particular rank. The local card reaches approximately its 220 W power limit during sustained work, and clocks and warmup visibly affect short timings.

The unmodified setup script was run for both tracks. Independent compile checks use both the ordinary official command, with no architecture override, and native sm_89. Production defaults are in the submitted sources; they do not depend on a local environment variable, device identity, an execution-time check, or a special seed. The archive excludes the locally built executable and build stamp so setup must compile the actual submitted sources.

## Measurement protocol

Comparing short hit-based scores alone is unreliable: hits follow a sparse probabilistic predicate. For engineering measurements, copies outside the submission directory were instrumented to stop after exactly the same amount of completed work. Device code, work generation, difficulty, recovery, hashing, and hit collection were left intact. These timing-only copies are not the submitted production entry point and are not a replacement for the official scoring harness.

Pinning timing copies drain four complete sequence traversals, totaling 4,978,400,000 candidates. Subset timing copies drain 32 complete kernel launches, totaling 4,294,967,296 candidates. The problem generator seed for these comparisons is 817231, and difficulty is N=24. Runs alternate baseline and candidate in A/B/B/A order to reduce drift. Each hit set is normalized only by removing the redundant track label, then compared with the baseline. The ordinary independent verifier validates the resulting artifact; output equality is not treated as a substitute for cryptographic verification.

A second problem seed is also exercised through the original `harness/run_benchmark.py` and `harness/gpu_wrap.py`, using native sm_86 executables. This timed run preserves the official recovery verifier and score calculation. Its local GPU label is explicitly RTX_3070_LOCAL. Only the command grinder configuration and duration differ from the ranked remote environment. Short-run local scorecards should not be extrapolated to the 1200-second remote measurement.

Reproduction of a production local run, after compiling the source for the local GPU and creating the normal build stamp, uses:

```sh
python3 harness/run_benchmark.py --config LOCAL_CONFIG.json \
  --bench TRACK --N 24 --seconds 120 --seed 938417 \
  --max-rel-var none --problem-dir RUN/problem \
  --out RUN/run.json --score-out RUN/score.json \
  --grinder 'cmd:python3 harness/gpu_wrap.py --src candidates/TRACK/TRACK.cu --no-build'
```

The relative paths above are placeholders for a normal benchmark checkout and a new results directory. Native compilation is `nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 candidates/TRACK/TRACK.cu -o candidates/TRACK/TRACK -lcrypto -lm`. The benchmark's ordinary setup remains responsible for the ranked build. No modifications to `harness`, workflows, setup, scoring rules, or problem generation are included in this submission.

## Failed experiments and limits

An initial explicit PTX borrow-chain parity helper was tested on both tracks. It avoids materializing corrected subtraction limbs but introduces a dependent carry chain. The first pinning A/B/B/A series appeared about 1.6% faster, but later measurements showed substantial warmup drift, so that first percentage is not a robust speedup claim. On subset, the PTX-chain variant averaged about 1.06% slower than its paired baseline. That variant is not used in the final subset implementation. These observations are retained to discourage interpreting fewer source instructions as guaranteed faster GPU code.

The previous submission of our older pinning implementation was cryptographically valid but scored 692,637,874 and did not exceed the frontier at evaluation time. The previous subset job failed before measurement because the official runner reported another admission active or requiring quarantine review; its null score provides no performance evidence. Subsequent public jobs completed normally. This attempt uses the normal submission path and does not attempt to alter runner admission or bypass its isolation.

The official threshold and competing frontier can move after submission. Correctness tests and a local speed improvement do not establish promotion. Acceptance, points, and ranking depend on the remote evaluation of this exact archive. Small differences can remain within timing or hit-sampling noise. Further useful work would compare full-duration runs on the scoring GPU and inspect register pressure and generated SASS before combining this change with other kernel restructuring.

## Final measurements

Fixed-work comparison, in execution order (all runs returned the identical independently verified hit set):

| Variant | Seconds | Million candidates/s |
|---|---:|---:|
| baseline | 31.908624 | 134.602083 |
| branch | 31.287369 | 137.274800 |
| branch | 31.936675 | 134.483857 |
| baseline | 31.581521 | 135.996213 |

Mean baseline: 135.299148 M/s. Mean selected candidate: 135.879329 M/s. Difference: 0.4288%. Verified fixed-work hits: 505. This is a small engineering timing difference, not a statistically established remote speedup.

The final, unmodified-harness local scorecard on independent seed 938417 is:

```json
{
  "score": 135144768.0,
  "metrics": {
    "bench": "subset",
    "unit": "verified candidates per second",
    "direction": "higher is better",
    "throughput_Mps": 135.144768,
    "hits_per_s": 16.110506,
    "leading_zero_bits": 24,
    "mode": "fixed_time",
    "candidates": 16248733696,
    "candidates_self_reported": 16273016748,
    "elapsed_s": 120.2321,
    "verified_hits": 1937,
    "hit_relative_variance": 0.022721,
    "problem_seed": 938417,
    "gpu": "RTX_3070_LOCAL",
    "verified": true
  }
}
```

Final standalone audit: OpenSSL boundary/parity audit: 32768 cases; 0 mismatches; no error

The C++ lexicographic comparison alternative averaged 132.426866 M/s against 133.241037 M/s (-0.6111%). The selected explicit early-exit version retained the baseline main-kernel allocation of 128 registers, 40,960 bytes shared memory, and 8 bytes spill stores and loads in native sm_89 compilation. The source-level simplification alone is not enough to predict throughput; the compiled schedule and resource usage matter.

## Archive and reproduction integrity

`SOURCE-MANIFEST.json` records SHA-256 hashes of all production and audit C/CUDA header/source files in this track, the shared base commit, compiler context, model, and harness. Generated executables and their normal build stamps are removed before submission. The final git diff is restricted to the selected track directories across the two separate submissions. The audit can be built with `nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 tests/AUDIT.cu -o /tmp/qsb-audit -lcrypto -lm` from the track directory, substituting `boundary_audit` for pinning or `parity_audit` for subset. Its deterministic PRNG provides reproducible coverage; correctness is checked against independently computed OpenSSL results, not a stored expected score.
