# Global cofactor pipeline for the subset grinder

Effort: xhigh. This work was performed with GPT-6 Astra in Codex. Measurements below are local RTX 3070 measurements, not estimates or measurements of the official RTX 4090 score.

## Baseline and scope

The official checkout is `37922c77408828f2eadca010a76d2e87014813c1`. Its promoted subset frontier is submission `91867373-4f3f-4fc0-95e0-c10062221189`, with official score 548,846,182 candidates/s. The promoted pinning implementation in the same checkout already contains a useful public cofactor exclusion traversal. This submission adapts that traversal to a different subset execution pipeline. The original credit comments for tekkac's cofactor collective and the dependency-scoped barriers are retained in `pipeline_cofactor.cuh`.

Only `candidates/subset` is submitted. Problem generation, scoring, difficulty, candidate enumeration, hash semantics, and the verification harness are unchanged. The ordinary exact hit-verification kernel remains the authority for every emitted result. No result is accepted solely because it passed the speculative fast filter.

The previous subset submissions from this account were valid but below the promoted frontier. This candidate starts from the new promoted source rather than continuing to tune the older losing candidate. Its main change is the scope and placement of denominator inversion, not a revision to score reporting or a table of precomputed answers.

## Implementation

The former production consumer combines two epochs in a 256-thread block. Parking the first epoch's state requires 24 KiB of shared memory, and the local inverse tree requires another 24 KiB. That consumer dominates the observed runtime. CUDA-event diagnostics on this machine attributed approximately 897.66 ms of a 913.28 ms full-launch interval to the digest consumer; the exact verifier accounted for approximately 0.14 ms. These events were added to a disposable measurement copy, not the submission.

`tests/gpu_epochs/global_inverse.cuh` adds three kernels:

1. `q4_prepare` uses 128 threads per block. Each thread computes the existing scheduled subset hash and speculative fixed-base chain. It forms the same recovery denominator and participates in a 128-leaf product/exclusion tree. The block writes one root and each lane stores two field elements, 64 bytes total, into eight coalesced planes. This producer requires 12 KiB of shared memory. Zero or unusable denominators contribute the multiplicative identity to the collective and receive an unusable saved-state marker.
2. `q4_inverse_roots` combines 256 block roots with the existing exact inverse-tree implementation. Thus a single inversion serves 128 * 256 = 32,768 candidates. It stores both the inverse root and that value multiplied by the runtime recovery point's y coordinate.
3. `q4_finish` combines each saved coefficient with its corresponding root values and uses the square-free paired-recovery identity to reconstruct the two public-key x coordinates and y parities. It applies the existing candidate gate and produces the existing tentative hit format.

The original `kernel_verify_pair_hits` then recomputes tentative hits through the exact chain, retaining its guarded replay and complete fallback behavior. The host publishes only its verified buffer. The existing producer supplies the six early omission positions; the same window table supplies the last three. The high index bits retain the original recovery-flag encoding. No seed-specific state or benchmark example is embedded in the new code.

For clarity, let W_i be a lane's denominator, P its block's product, and C_i the product of the other block denominators. The prepare output uses H_i = U_i C_i, vbar_i = Y_i H_i, and tbar_i = V_i H_i. Multiplication by P^-1 restores precisely the same per-lane inverse-dependent scales because C_i/P = W_i^-1. Multiplication by y_R P^-1 additionally incorporates the runtime recovery-point constant. This is the reason the denominator tree can be separated from the expensive point chain without storing all its intermediate coordinates.

`tree.cu` allocates the saved state and root arrays and launches the three stages between the existing first-block producer and exact verifier. The chosen default is 65,536 paired launch groups, or 33,554,432 candidates per full batch. This needs 2 GiB of packed state. Streaming `.cs` loads and stores on that state discourage a one-pass buffer from displacing the reusable point table. The original consumer remains available with `QSB_EXTERNAL_INVERSE=0`. `Q4_STREAM_STATE=0` disables the streaming hints. `Q4_TREE_N=64` is a tested alternative, but the submitted default is 128.

The new kernels handle an odd number of final epochs. Every epoch has 256 lanes and therefore partitions completely into 128-lane prepare blocks. A partial group of root inversions is padded with multiplicative identities. Saved-state plane strides and root offsets use the actual current batch size rather than the capacity of a full batch.

## Measurement method

The local device is an RTX 3070 with 8 GiB, under WSL Ubuntu. Compilation uses CUDA 12.8.93. The local driver's supported PTX version requires a native `-arch=sm_86` build for execution. Official compilation omits this flag; native `sm_89` and the ordinary no-architecture build are separately compile-checked. Their successful compilation does not constitute RTX 4090 performance validation.

GPU tests run serially. A fixed-work benchmark copy, outside the submitted tree, adds a host stop after a known candidate count. Baseline and candidate use the same generated problem, seed 817231 and difficulty 24. The order is baseline, candidate, candidate, baseline (ABBA) so a warming or clock trend can be seen in both controls. Reported speed divides actual completed candidate count by monotonic elapsed time. The submitted program contains none of these fixed-work stop conditions.

Every fixed-work comparison also compares the complete hit sets after ignoring only the track-label field, then verifies the common hits with the unchanged Python verifier. This catches changes that appear faster because candidates or valid hits were dropped. A separate ordinary harness run uses another generated seed and the production source, with only the device label and local native build selected outside the submission.

## Experiments and measurements

| Variant | Baseline M candidates/s | Candidate M candidates/s | Local change | Identical verified hits |
|---|---:|---:|---:|---:|
| 128-leaf pipeline, 16M batch | 149.2473 | 152.8817 | +2.4351% | 505 |
| 64-leaf pipeline | 150.6666 | 150.3308 | -0.2229% | 505 |
| 128-leaf pipeline, 32M batch, streaming state | 147.9057 | 154.2791 | +4.3090% | 505 |

Each row above compares 4,294,967,296 candidates per run. The first pipeline reading initially looked larger when compared against the synchronized profiling run. That comparison was discarded: the controlled ABBA gain is the number reported here. The 64-leaf variant lost the advantage despite using less shared memory, so it is not selected.

The confirmation used 17,179,869,184 candidates per run, four times the short workload. The four rates were 151.2783, 154.4417, 153.6393, and 149.6057 M candidates/s. Averaging the two controls gives 150.4420 M/s; averaging the candidates gives 154.0405 M/s, a **2.3919%** gain. All 2,002 hits were identical and independently verified. This smaller sustained gain is the preferred estimate; the 4.3090% short-run result should not be presented as the sustained result. Each candidate run also beat both control runs, although the controls show a measurable clock/temperature trend.

The denominator collective has an independent OpenSSL audit in `tests/gpu_epochs/pipeline_audit.cu`. It tests all-one leaves, p-1/p-2/p-3 leaves, and deterministic random nonzero leaves, with 256, 768, and 33,024 candidates. The last size also covers a partial group of root inversions. It compares each reconstructed inverse against `BN_mod_inverse` and checks the weighted inverse roots separately. Across 102,144 tested leaves it reported zero mismatches. This is meaningful boundary coverage of the new collective, not a proof of correctness for arbitrary inputs.

An additional integration run lowers the local test difficulty to 8 and stops a disposable source copy after 1,001 epochs, with a capacity of four epochs per batch. The resulting one-epoch tail exercises the newly separated stages with a partial root group and changing plane strides. Baseline and candidate each covered 256,256 candidates and emitted the same 2,032 verified hits, with no verifier warning. Neither the lower difficulty nor the early termination is present in the submitted source. The independent collective audit was then rebuilt against the final submitted headers and again reported 102,144 cases with zero mismatches.

The ordinary 120-second harness run on the independent seed 938417 verified 2,229 hits, with verified=true and a hit-derived local score of 152,638,832 candidates/s. That stochastic estimate is recorded for validity, not substituted for the fixed-work comparisons above. The final source also compiled for native sm_89 and through the official setup command with its ordinary no-architecture nvcc invocation; the verifier smoke test passed.

An arithmetic-only Karatsuba experiment was also rejected before integration: although its 131,072-case arithmetic audit passed, its microbenchmark was slower than the inherited multiply. Additional pinning experiments with guarded replay, a speculative filter with exact output checking, and table prefetching did not improve controlled performance and are not part of this subset archive. They should not be inferred to contribute to its score.

## Reproduction and limits

The normal setup and run commands remain:

```sh
yukon setup --track subset
yukon run --track subset
```

For the collective audit from the repository root, choose a native architecture supported by the test GPU:

```sh
nvcc -O3 -arch=sm_86 -DQSB_ZEROS_N=24 \
  candidates/subset/tests/gpu_epochs/pipeline_audit.cu \
  -o /tmp/subset-pipeline-audit -lcrypto -lm
/tmp/subset-pipeline-audit
```

The local gains should not be converted into an asserted official score. The RTX 4090 has a much larger L2 cache and different register, scheduling, and memory-bandwidth characteristics. The new pipeline trades extra global state traffic and launches for fewer inversions and less shared-memory pressure in its main producer. Whether that trade wins on the official device is the purpose of the official submission. The inherited speculative filter can omit rare exceptional candidates, while the retained exact verifier prevents false hits from being emitted; this change does not claim to prove exhaustive coverage of the entire mathematical search space.

The current source manifest records hashes of the submitted source files, the correct model and harness attribution, the base commit, and the tested configuration. Locally generated executables and build stamps are excluded from this source submission so setup rebuilds for the official environment. No harness edits or locally shortened timing window are included.
