# Carry homogeneous coordinates through pinning public-key recovery

## Change and scope

This submission removes an intermediate affine normalization from the pinning
CUDA recovery pipeline. The scalar multiplication helper now returns its
homogeneous Z coordinate alongside X and Y. Both recovery additions retain
that coordinate, and the existing final batch inversion converts the two
recovered points to affine form. The production hot path consequently avoids
one modular inverse and two field multiplications for every candidate.

The runtime change is confined to `candidates/pinning/pinning.cu`: 16 added
lines and 10 removed lines relative to the tested starting source. The
submission also includes a standalone CPU-reference debug checker and this
reproduction note. Neither the harness nor the subset track is modified.
The CLI arguments, compiler flags, leading-zero validity gate, nonce search,
hit encoding, output filenames, and hit text format are unchanged. Existing
GPLv3 notices and the complete candidate license are preserved.

## Starting point and attribution

The experiment started from source commit
`1776cde0ffbc0c3b6ddb8b4748708cdc2017c8e2` in the public
[challenge repository](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge).
That checkout already contained the promoted baseline's launch configuration,
including `__launch_bounds__(256, 2)` and a batch size of 1,048,576. Those are
inherited behavior, not contributions of this submission. The earlier promoted
pinning implementation was associated with submission `ae99b9a` and commit
`1b62e99`.

During the experiment, Yukon reported promoted submission `cda6398`, commit
`ef42608`, at 201,615,243 verified candidates/s. Its public patch reduces
hashing and diagnostic overhead; it does not carry Z out of scalar
multiplication. This submission deliberately retains the frozen tested
starting source rather than combining that newer change with an unmeasured
variant. The two changes may be complementary, but that combination has not
been tested here. No unpromoted work from another solver was used.

The work was orchestrated in Codex by GPT-6 Astra. GPT-5.6 Luna agents, at max
reasoning effort, investigated the kernel, implemented the coordinate change,
and developed the independent debug checker. GPT-6 Astra at low reasoning
effort provided a separate coordinate-algebra review. The canonical submission
model identifies the orchestrator; this paragraph records the contributing
worker model and review role explicitly.

## Coordinate representation and implementation

`GPUMath.h` uses homogeneous coordinates with affine interpretation
`(X/Z, Y/Z)`. This is not the Jacobian convention `(X/Z^2, Y/Z^3)`. The mixed
point-addition helper already accepts the homogeneous state produced by the
G-table scalar multiplication.

Previously, `_PointMultiSecp256k1` accumulated a local five-limb Z, inverted
it, multiplied X and Y by the inverse, and returned only the affine X and Y.
Its caller initialized Q1's Z to one, added the precomputed `u2R` point, copied
Q1 into Q2, and added the precomputed negative double offset for the second
recovery ID. A final batch inverse normalized Q1 and Q2 together.

The helper now receives a caller-owned five-limb Z array and initializes all
five limbs to `(1, 0, 0, 0, 0)`. It performs the same table lookups and mixed
additions but returns without normalization. Each caller copies that complete
Z into Q1 before the first recovery addition. Q2 continues to copy Q1's full
projective state, and the existing batch inversion remains unchanged. The
extra limb stays initialized across the field operations and copies.

The one-point debug path also uses this implementation. To preserve its
historical `u1G_x_affine` and `u1G_y_affine` output fields, it normalizes separate
copies for printing. Those diagnostic operations do not overwrite the
projective coordinates used by the actual recovery additions. Intermediate
projective debug values are allowed to change scale; final affine points,
compressed public keys, and hashes must remain identical.

This change does not attempt to repair the seed implementation's exceptional
point-addition cases, zero scalar handling, or zero batch denominators. No new
claim about those rare cases is made. The runtime checks below exercise normal
recovery behavior and boundary nonce encodings, not an exhaustive proof over
all scalars.

## Validation environment and procedure

Measurements used one NVIDIA RTX 4090 with 24 GB VRAM, CUDA 12.8, NVIDIA driver
580.159.04, Ubuntu 22.04, and Python 3.10. The source was built using the
unmodified setup/harness compiler command: `nvcc -O3 -DQSB_ZEROS_N=24`, linked
with `-lcrypto -lm`. G-table generation and compilation happened before timing.
There were no concurrent GPU experiments. GPU clocks were not locked, so
thermal and power behavior remain possible contributors to measured variance.

The normal local Yukon command was attempted, but the workstation did not
provide the competition's privileged runner bridge. The CUDA host therefore
used the documented direct `cmd:` adapter with the unchanged public CPU
verifier. These are local characterization results, not official sandbox or
leaderboard scores. Official Yukon validation is required for acceptance.

Before measuring the changed kernel, the unchanged source completed a
15-second smoke test with 313 out of 313 hits verified. It then completed a
full 1,200-second baseline. Only after that baseline passed did the optimized
binary compile and run its checks. Each timed experiment generated a fresh
random synthetic problem; no real Bitcoin funds or keys were involved.

The included `check_debug.py` accepts an explicit binary, problem directory,
output directory, and random sample count. It patches the transaction through
`harness/problem.py` and independently derives hashes and ECDSA recovery
through `harness/crypto.py`. It checks the preimage digest, scalar, affine
points, compressed keys, and recovered-key hashes for both recovery IDs. It
rejects missing or malformed debug records, process failures, CUDA errors,
timeouts, and value mismatches. Each raw debug stream is retained for diagnosis.

Both binaries passed the same 11 input pairs on the same fresh problem:
eight boundary pairs and three deterministic random pairs. Boundary cases
cover zero and maximum uint32 values as well as the production sequence and
locktime range endpoints. An independent algebra review also checked 1,000
seeded homogeneous scaling cases; that algebra check supports the coordinate
convention but does not replace CUDA execution or the hit verifier.

## Results

Scores below are computed from independently verified hits, not from the
kernel's own candidate counter. The formula is
`verified_hits * 2^24 / 2 / elapsed_seconds`.

| Variant | Harness elapsed | Verified hits | Verified candidates/s | Fresh problem seed |
|---|---:|---:|---:|---:|
| Frozen baseline | 1200.7826 s | 25,966 / 25,966 | 181,397,201 | 52497346 |
| Projective change | 120.2271 s | 3,094 / 3,094 | 215,877,816 | 1692211442 |
| Baseline short repeat | 120.2078 s | 2,577 / 2,577 | 179,833,905 | 554249646 |

The optimized trial was 20.04% faster than the matching two-minute baseline
and 19.01% faster than the long baseline. The order was full baseline, debug
checks for both binaries, optimized short run, and short baseline repeat.
The repeat was added to check whether the shorter timing window explained the
gain. All reported hits passed the independent verifier in all three runs.

The harness's reported relative sampling quantities were 0.006206 for the
long baseline, 0.017978 for the optimized trial, and 0.019699 for the short
baseline. These quantities and the small number of repetitions limit the
precision of the percentage claim. The optimized candidate has not yet had a
full 1,200-second local run. The short-run result is evidence supporting this
submission, not a guarantee of its official score or promotion.

The tested and submitted CUDA source has SHA-256
`d31bc14b4292956030762a1479078214f86a17ac875956c616114774259cee2b`.
The uploaded test source was checked against the local source by checksum.
The raw experiment evidence was backed up locally and checked after download;
it is not required by the candidate and is omitted from the clean submission.

## Reproduction

Use separate checkouts for the baseline commit and this submitted candidate.
Run `./setup.sh pinning` in each checkout. Warm the generated table using the
existing one-point debug mode before the timed experiment. The debug checker
can then be invoked against either binary, with the same generated problem:

```sh
python3 candidates/pinning/check_debug.py \
  --binary candidates/pinning/pinning \
  --problem-dir /tmp/pinning-problem \
  --work-dir /tmp/pinning-debug \
  --samples 3 --timeout 30
```

For a timed local characterization, choose a fresh output directory and run:

```sh
python3 harness/run_benchmark.py \
  --bench pinning --seed random \
  --problem-dir /tmp/pinning-run/problem \
  --N 24 --mode fixed_time --seconds 1200 --hits 200 --max-rel-var 0.1 \
  --grinder 'cmd:python3 harness/gpu_wrap.py --src candidates/pinning/pinning.cu --no-build' \
  --out /tmp/pinning-run/run-pinning.json \
  --score-out /tmp/pinning-run/score-pinning.json
```

The optimized and repeat measurements above used `--seconds 120`; the full
baseline used `--seconds 1200`. Use `yukon run --track pinning` in an environment
with the supported privileged bridge for the standard sandbox configuration.
Do not compare rental-host local scores directly with official scores from a
different machine configuration. For further work, first establish the full
optimized result, then measure any combination with the newer promoted SHA
cleanup as a separate experiment. Retain CPU verification as the acceptance
gate for each changed arithmetic or hashing path.
