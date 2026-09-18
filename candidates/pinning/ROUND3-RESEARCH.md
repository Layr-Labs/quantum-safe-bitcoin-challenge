# Pinning: borrow-only parity and bounded raw-product addition

## Change and correctness argument

The implementation retains the promoted two-stream host pipeline, signed-digit preparation, 128-leaf cofactor recovery, weighted root inversion, squarefree finish, and SHA predicates. It changes two local operations in `PackedRecovery.cuh` and adds `BorrowParity.cuh`.

First, parity computation uses a scoped PTX block with a four-limb subtraction carry chain. The discarded subtraction limbs stay in a temporary register, and only the final borrow is returned. Its low bit is XORed with the input low bits. Adding the odd field modulus on borrow flips parity, so this gives exactly the same bit as the previous corrected subtraction. Equality and near-boundary operands follow the same full comparison; no input distribution assumption is needed for correctness. This replaces the previous nested C++ magnitude comparison with an explicit borrow chain.

Second, the two x-coordinate products use the existing full-width raw multiplication and existing `qsb_add_boundary` helper before `_ModAdd256`. Let B=2^256 and p=B-K, K=2^32+977. The raw product is below B. If the canonical addend's high limb is not UINT64_MAX, it is below B-2^192, so the raw sum is below 2B-2^192 < 2p. The following modular addition therefore needs at most one correction. For the extreme upper addend range, the product is normalized first, exactly retaining the original conservative path. This removes two unconditional product normalizations in the common case without dropping the rare boundary handling.

`tests/boundary_audit.cu` contains a standalone GPU audit against OpenSSL BIGNUM. It checks 32,768 generated cases, including zero, equality, p-adjacent values, full-width raw values, and addends at the normalization boundary. It also compares the parity helper against the existing full subtraction for unrestricted raw operands. The test includes the production source so it exercises the actual helpers used by the grinder. It is not compiled or invoked by the normal benchmark entry point.

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
| baseline | 29.691151 | 167.672853 |
| candidate | 30.015912 | 165.858695 |
| candidate | 29.732017 | 167.442390 |
| baseline | 30.409799 | 163.710388 |

Mean baseline: 165.691621 M/s. Mean selected candidate: 166.650543 M/s. Difference: 0.5787%. Verified fixed-work hits: 585. This is a small engineering timing difference, not a statistically established remote speedup.

The final, unmodified-harness local scorecard on independent seed 938417 is:

```json
{
  "score": 167755952.0,
  "metrics": {
    "bench": "pinning",
    "unit": "verified candidates per second",
    "direction": "higher is better",
    "throughput_Mps": 167.755952,
    "hits_per_s": 19.99807,
    "leading_zero_bits": 24,
    "mode": "fixed_time",
    "candidates": 20166213632,
    "candidates_self_reported": 20739354239,
    "elapsed_s": 120.2116,
    "verified_hits": 2404,
    "hit_relative_variance": 0.020395,
    "problem_seed": 938417,
    "gpu": "RTX_3070_LOCAL",
    "verified": true
  }
}
```

Final standalone audit: OpenSSL boundary/parity audit: 32768 cases; 0 mismatches; no error

Ablations: the raw-product-only variant averaged 166.274703 M/s against its paired 167.970669 M/s baseline (-1.0097%). The early-exit comparison plus raw addition averaged 169.759903 M/s against 169.916788 M/s (-0.0923%) and produced 32 bytes of spill stores and loads in the sm_89 fast finish specialization. It was rejected. The submitted borrow-chain version has zero spill stores and loads in that specialization and uses 72 registers, versus 70 in the baseline. Both fit the same allocation granularity. The selected combination is the version in the fixed-work table above; the negative ablations are not omitted from the evidence.

## Archive and reproduction integrity

`SOURCE-MANIFEST.json` records SHA-256 hashes of all production and audit C/CUDA header/source files in this track, the shared base commit, compiler context, model, and harness. Generated executables and their normal build stamps are removed before submission. The final git diff is restricted to the selected track directories across the two separate submissions. The audit can be built with `nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 tests/AUDIT.cu -o /tmp/qsb-audit -lcrypto -lm` from the track directory, substituting `boundary_audit` for pinning or `parity_audit` for subset. Its deterministic PRNG provides reproducible coverage; correctness is checked against independently computed OpenSSL results, not a stored expected score.
