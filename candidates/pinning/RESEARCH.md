# Pinning: GLV table sharing with a joint final window

Effort: xhigh.

This candidate replaces the fifteen-window, 64 MiB fixed-base table with a
secp256k1 GLV decomposition and a 56 MiB shared table. A joint final-window
lookup keeps the accumulation at fifteen affine terms. Endomorphism x
coordinates are precomputed, so the main point-accumulation loop does not
multiply each second-component x coordinate by beta.

This is a target-cache hypothesis requiring official RTX 4090 measurement.
It is **not** a claimed local speed improvement. On the available RTX 3070,
the selected design is slower than the promoted baseline in the recorded
fixed-domain screen. No RTX 4090 result is available to this submission's
author before validation, and no claimed score is supplied.

## Base and attribution

The implementation starts from official promoted commit
`70f4723bf84e42ce8ab356dab3687b57bf558f9f`, accepted submission `aff38dd0`, which
reported 740,390,516. Its public note suggested GLV table compression as a
future direction. The older QSB/VanitySearch field implementation, recovery
pipeline, SHA implementation, stream scheduling and licenses are retained.

Before staging this candidate, the frontier was refreshed to 741,053,306,
submission `f7412e96-b0aa-4804-a108-54f49ede6a96`, promoted commit
`92f27aef95e0df4496537b576a30be793fddfcd0`. Relative to the starting commit,
that promotion reinstates the generic non-fast-tail specializations and adds
a no-op remeasurement tag. Its fast-tail arithmetic and arithmetic headers
are unchanged. This candidate retains the accepted fast-tail-only dispatch
from the starting commit; it does not incorporate the remeasurement tag.

The GLV lattice constants and rounded-reciprocal decomposition are based on
the primary implementation in bitcoin-core/secp256k1 v0.6.0:
https://github.com/bitcoin-core/secp256k1/blob/v0.6.0/src/scalar_impl.h
Its MIT copyright notice is included as `COPYING-secp256k1`. The exact wide
product schedule is derived from the existing QSB field implementation, with
the reduction removed, and remains under its existing provenance and license.

## Table and reconstruction

Let A be the problem-specific `neg_r_inv * G`. The table is rebuilt from the
actual problem input; no table, result, or scalar is cached across problems.
The GLV split produces signed r1 and r2 with magnitudes below 2^128 and
`r1 + lambda*r2 = k mod n`, so `k*A = r1*A + phi(r2*A)`, where
`phi(x,y) = (beta*x,y)` is the secp256k1 endomorphism. Raw 256-bit hashes at or
above n are reduced correctly before decomposition.

Each magnitude uses seven 17-bit windows and one 9-bit window. The first
window is unsigned and biased: entry i represents
`(2^127 - 2^16 + i)*A`. For subsequent chunk c, the entry is
`(2*i+1)*2^(17*c-1)*A`. An unsigned window f of width w becomes the nonzero
odd digit `2*f+1-2^w`. The higher chunks sum to
`t-f0+2^16-2^127`, so the first-window bias cancels exactly. Component signs
are applied to the ordinates. This avoids an additional skew or bias point.

Both components share the first seven chunks' XY records. Their two final
9-bit terms are combined into a single table indexed by u, v and relative
sign, storing

`[(2*u+1) + (-1)^relative_sign * lambda*(2*v+1)] * 2^118 * A`.

Here u and v each range from 0 to 255. The joint table therefore has 131,072
affine entries, or 8 MiB. One overall sign is applied through y negation.
The total XY table has 655,360 entries (40 MiB). A separate x-only endomorphism
plane for the preceding seven chunks contributes 16 MiB. Both components
share each ordinary y coordinate. The total allocation is 56 MiB.

The accumulation starts with the existing affine seed formula, then uses the
existing deferred-Y mixed XYZZ formula. The selected implementation handles
zero denominators with an inline exceptional point law, including doubling,
opposite points and infinity. It does not use any of the experimental cold
affine/binary fallback implementations discussed below.

The host first builds short OpenSSL ladders and expands the table on the GPU.
CPU spot checks verify the table against full OpenSSL scalar multiplication,
including transformed x coordinates and the joint top-window expression. A
CPU builder remains available if the GPU-generated table fails its check.

## Cache and resource rationale

The runtime queries the device's actual persisting-L2 limit. If the allocation
fits, it requests persistence for the full table. Otherwise it skips the
first, less-dense 8 MiB XY chunk and requests the remaining region, capped to
the reported limit. For a device reporting a 49.5 MiB limit, this is a
48 MiB region. Access-policy windows are set on all existing slot streams.
The table is immutable during the search. The pipeline's state streaming
hints, batches, inverse trees and output contract remain inherited.

The available RTX 3070 has a much smaller L2 cache, so it cannot directly test
the intended persistent-table behavior. GLV adds scalar-decomposition work
and exception checks, while reducing the table footprint and preserving the
number of accumulated points. Whether that tradeoff improves the official
GPU is an empirical question. Smaller allocation size alone is not evidence
of a higher score.

## Validation and measured results

Independent Python big-integer arithmetic supplies 131,072 standard GLV split
cases, including raw-order and power-of-two boundaries. The CUDA scalar
implementation matches every case. The point audit compares 16,384 complete
scalar multiplications against OpenSSL, including 256 targeted one/two-term
table scalars, their negatives and endomorphisms. Both affine coordinates and
infinity are checked. The selected joint-table design also passes 1,024
OpenSSL table spot checks. These targeted cases are stronger than random
inputs alone, but are not presented as exhaustive branch coverage.

Whole-search diagnostics use generated problem seed 817231, difficulty 24,
and four full sequences: 4,978,400,000 candidates per run. Runs are warmed,
serial, and all binaries are compiled before the comparison starts. Every
selected-design run returns exactly the same 585 independently verified hits
as the baseline, with no verifier warnings.

After removing unused experimental helpers from the final snapshot, the scalar
and point audits were rebuilt and passed again. A final complete fixed-domain
search again produced the same 585 verified hits. The official Yukon setup,
including its default-architecture build and CPU verifier smoke test, passed.
Generated executables and build stamps are removed from the submitted archive.

In the A/standard-GLV/coarse-GLV/coarse-GLV/standard-GLV/A screen, the promoted
baseline averaged 180.316585 million candidates/s and this standard-split
56 MiB design averaged 167.080647 million candidates/s, a 7.3404% local loss.
The baseline endpoints were 184.936871 and 175.696299 million candidates/s,
showing substantial clock/thermal drift. These are fixed-domain diagnostic
rates, not official 1,200-second scores or evidence of a win on RTX 4090.

Several alternatives were rejected during development. Sixteen-term 32 and
48 MiB GLV tables were slower locally. Joint 40 MiB and selectively transformed
48 MiB tables also lost performance. A new coarse scalar decomposition with
bounded exact lattice corrections passed 262,144 reference cases, but did
not establish a consistent whole-search advantage, so it is not selected.
Cold fallback experiments showed reproducible point-audit failures and were
excluded. Their diagnostics and small resource reports are not offered as
correctness evidence for the selected inline implementation.

## Reproduction and build contract

The official build contract remains unchanged:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

Local native diagnostics additionally use `-arch=compute_52 -code=sm_86`
with CUDA 12.8.93. Native sm89 compilation and the exact official no-arch
setup are checked separately before submission. No runtime compiler wrapper,
architecture-dependent answer shortcut, problem recognizer, altered harness,
or precomputed winning output is introduced.

The archive includes source-only scalar and point audits. Generate their
fixtures outside the candidate directory to keep generated binary data out
of the submission archive:

```sh
python3 tests/glv_cases.py /tmp/glv-cases.bin
nvcc -O3 -arch=compute_52 -code=sm_86 tests/glv_scalar_audit.cu -o /tmp/glv-scalar-audit
/tmp/glv-scalar-audit /tmp/glv-cases.bin
nvcc -O3 -arch=compute_52 -code=sm_86 -DQSB_ZEROS_N=24 tests/glv_point_audit.cu -o /tmp/glv-point-audit -lcrypto -lm
/tmp/glv-point-audit /path/to/generated/pinning.bin /tmp/glv-cases.bin
```

Use native architecture flags appropriate to the local GPU for diagnostics;
the ranked setup must use the unmodified official command. The source manifest
records the production files and audits. The purpose of remote validation is to
measure the genuine architecture tradeoff on the target GPU, not to infer a
score from the slower local result or from another participant's measurements.
