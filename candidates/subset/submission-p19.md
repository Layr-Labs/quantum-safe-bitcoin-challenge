# Subset: shared five-bank GLV10 with a 32 MiB hot bank

## Context, attribution and scope

This table-geometry experiment builds on terrapinelf's unpromoted GLV11
submission `7ee5c52a-b3d6-4873-9856-8943399d1b8d`, candidate commit
`a8c9a7d26be27db0878566e41428d3400862bc97`. Substantial unpromoted contributors
are terrapinelf, i34-9, newjordan, ercumentyildirim, fkiene and Ryun1, credited
as coauthors. Existing licenses, notices and research files are retained.
The inherited `submission-note.md` describes the donor, not this experiment.
This file is the current submission note.

Implementation and independent review used GPT 6 Astra at high effort through
separate Codex subagents. Codex coordinated isolated worktrees, GPU builds,
differential tests and end-to-end measurements. A different agent reviewed
the table arithmetic and test oracles before GPU validation.

The inspected production frontier was Akashneelesh's 623,518,629 verified
candidates/s, on shared source `d59a969`. An unchanged local control was
measured before edits. Only `candidates/subset/` is changed. The harness,
verifier, scorer, problem generator, track configuration and pinning candidate
are unchanged. We do not claim authorship of the inherited GLV split, field
kernels, enumeration, inverse, exact host gate or native carrier mechanism.

## Hypothesis and tradeoff

The donor has eleven fixed-base terms: six for Q and five for P. It performs
eleven point gathers and ten additions, with 22,688,113,472 bytes of records
and a 48 MiB hot region. The proposed layout gives both signed GLV components
the same five banks. It removes one gather and one addition while retaining
the exact 128-bit component bounds and existing endomorphism.

The price is more cold traffic: eight cold records instead of six. The new
32 MiB initial bank and proportional L2 set-aside avoid reserving 48 MiB for
a smaller region. This differs from applying the donor's unchanged P18 banks
to both components. A smaller allocation and fewer operations do not guarantee
a gain on a power-limited GPU.

No 60% gain is claimed. That was an exploratory target, not a measured result.
SHA, scalar splitting, inversion and recovery remain substantial costs.

## Exact geometry

| Bank | Shift | Field | Records | Offset |
| --- | ---: | --- | ---: | ---: |
| 0 | 0 | 19-bit unsigned | 524,288 | 0 |
| 1 | 19 | 27-bit signed odd | 67,108,864 | 524,288 |
| 2 | 46 | 27-bit signed odd | 67,108,864 | 67,633,152 |
| 3 | 73 | 27-bit signed odd | 67,108,864 | 134,742,016 |
| 4 | 100 | bounded signed top | 85,279,885 | 201,850,880 |

Let `T=170559769` and `K=(T+1)*2^99-2^18`. Bank zero stores `(K+i)*A`.
Middle banks store `(2*i+1)*2^(shift-1)*A`; the top digit is `2*field-T`.
The biases telescope to the original component magnitude. Component signs
are preserved. The endomorphism applies after five Q terms, before P starts.
The packed walker still consumes exactly 128 bits per component.

Total size is 287,130,765 records, or 18,376,368,960 bytes, approximately
17.114 GiB. The initial bank is exactly 33,554,432 bytes. Source-level
fixed-base arithmetic falls from approximately `68M+20S` to `61M+18S`,
including the endomorphism but excluding unchanged recovery work. These are
operation counts, not GPU instruction latency measurements. Total logical
payload falls from 704 to 640 bytes, while cold payload rises from 384 to 512.

## Implementation and reproducible build

`GLVScalar.cuh` adds bounded P19 geometry and a portable direct recoder.
`tests/gpu_epochs/tree.cu` selects the new descriptor sequence, updates both
host bias constructions, changes the endomorphism boundary and fingerprints
the new option. The second seed is cold and receives the same native 64-byte
hint as subsequent cold loads. L2 persistence is capped to the 32 MiB hot
region, subject to device limits.

`subset.cu` enables `QSB_GLV10_P19=1`. The shared implementation defaults to
zero. Compiling both executable and carrier with this option zero restores
the donor geometry and cache policy. The existing speculative filter and
exact OpenSSL publication gate are unchanged. Host verification rejects
false positives but cannot prove the absence of GPU false negatives.

The native sm_89 image was rebuilt from source using CUDA 12.8.93. Its SHA256
is `cad83a39b41a89223592e78204c436b85fd7131390f39c5b3ecf849eda2181ef`,
size 476,704 bytes. The digest build reports three LTC64B loads and no stack
frame or spill loads/stores. Its source fingerprint is
`4af90bea174c936de0d6eb2fb0a0f9b0259f54a3065136b96ba7d18bf5a3d0a6`.
The host and carrier use matching wrapper settings. Rebuild the carrier after
any device-source edit; otherwise a stale image can hide an experiment.

From the repository root with CUDA on PATH:

```sh
bash candidates/subset/build_carrier.sh 24
bash ./setup.sh subset
python3 candidates/subset/tests/glv10_p19_audit.py
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 \
  candidates/subset/tests/glv10_p19_gpu_audit.cu \
  -o /tmp/qsb-p19-audit -lcrypto -lm
/tmp/qsb-p19-audit
QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build" \
QSB_SECONDS=120 QSB_PROBLEM_SEED=424242 QSB_VERIFY_WORKERS=8 \
QSB_OUTPUT_DIR=p19-validation bash ./benchmark.sh subset
```

The CPU audit requires Python cryptography and a C++17 compiler. These are
development-test dependencies, not ranked dependencies. The official runner
uses normal `yukon setup --track subset` and `yukon run --track subset`, without
development timing overrides. No profiling or rejected optimization is included.

## Independent correctness evidence

The CPU test compiles actual portable helpers, descriptors, startup checks and
host table-coefficient expressions. Independent Python integer arithmetic
checks complete signed reconstruction. It passed 22,306 exact recodings,
76 sparse ladder records and 192 sparse OpenSSL point sums under varying
bases and isomorphic scalings. Both donor and new startup checks passed.

A second agent independently derived the bias, checked 20,812 signed geometry
cases and verified the GLV lattice and actual beta constant against lambda*G.
It reviewed builder bounds, both bias constructors, signs, pointer arithmetic,
fallbacks and load hints without changing implementation files.

The native CUDA audit executes actual device split, packed walker and direct
P19 helpers. It passed 13,059 raw scalars and 5,872 signed magnitude pairs
against independent OpenSSL BN arithmetic. Coverage includes zero, order
boundaries, power/window boundaries, component bounds, all sign pairs, seeded
random values and 2,286 rounding-boundary cases. Every output magnitude, sign,
packed code and direct code was compared.

That standalone audit does not allocate the full table or execute the device
point chain. Normal candidate startup and the full unmodified verifier run
provide integration evidence, not an exhaustive proof of every table record
or speculative intermediate.

## Environment and measurements

All GPU tests used one dedicated RTX 4090 with 24 GiB, driver 580.95.05,
CUDA 12.8.93 and the reported 450 W limit. GPU tests were serialized. The
official bridge was unavailable on the rental, so the documented command-mode
override was used. The unmodified harness owned elapsed time and verified hits.

Same machine, N=24, seed 424242, 120-second windows:

| Candidate | Verified hits | Verified candidates/s | Diagnostic peak only |
| --- | ---: | ---: | ---: |
| Unchanged production control | 9,608 / 9,608 | 669,656,336 | 730.2M/s |
| Rebuilt public GLV11 donor | 11,560 / 11,560 | 804,328,051 | 830.4M/s |
| This GLV10 P19 candidate | 11,700 / 11,700 | 813,524,979 | 836.3M/s |

This candidate passed at approximately 120.6 seconds harness elapsed, with
reported hit relative variance 0.009245. The local difference is about +1.14%
versus the donor and +21.48% versus the original local control. The larger
control-to-donor gain belongs to the credited prior work, not this table change.

These are short sequential measurements, not an alternating long-window
confidence interval. An unrelated CPU compilation overlapped the first donor
test; no GPU timings overlapped. Thermal drift, power behavior, startup and
hit-count variation can be comparable to the incremental gain. Official
1200-second evaluation may reject or rank this candidate differently. Local
scores are not claims of official promotion.

## Rejected experiments and lessons

Isolated SHA tests tried constant-block loop unrolling, the existing specialized
SHA256d helper and a newly paired specialized pubkey hash. Direct or end-to-end
correctness checks passed, but short verified rates were approximately 800.20M,
800.59M and 801.69M respectively. No improvement was established; none is added.
The specialized SHA256d GPU audit checked all eight words on 101,024 messages.
The paired pubkey GPU audit checked 202,048 public keys against OpenSSL.

An exact one-level Karatsuba prototype passed 104,601 GPU input pairs in eight
modes but increased registers from 38 to 60. Standalone throughput fell to
about 56% of schoolbook. It is disabled in a separate worktree. Fewer algebraic
products did not outweigh carry work and register pressure. Moreover, that exact
helper was not the donor's normal speculative fixed-base hot path.

Jacobian alternatives were counted rather than assumed superior. The donor's
affine-anchor-deferred XYZZ form already uses fewer squarings. Saving coordinate
registers would need a real resource-threshold benefit. Repeated affine batch
inverses also have substantial synchronization costs. These ideas are not mixed
into this candidate.

Hardware-counter profiling was denied by the rental host. Separate software
timing work is measuring coarse phase costs and is excluded here. Future work
should repeat the geometry comparison in alternating order and use measured
bottlenecks to guide larger pipeline changes. Preserve exact tests, credit,
runtime knobs and verified-throughput evidence when building on this result.
