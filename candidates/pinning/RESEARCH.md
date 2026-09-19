# Pinning: grouped GLV in 40 MiB with a regular-prefix proof

Effort: xhigh.

This candidate replaces the promoted 64 MiB fixed-base table with a 40 MiB
GLV table while retaining fifteen affine terms. It accumulates the ordinary
second component first, applies the secp256k1 endomorphism to that accumulator
once, then appends the first component and the joint final-window point.
Only one additional field product is needed for the endomorphism. There are
no separately stored endomorphism-x planes and no per-term beta products.

A lattice argument proves that all additions in the first fourteen terms
are nonsingular for every signed 128-bit component pair. The final joint
addition retains explicit cancellation and doubling handling. Its exceptional
doubling uses the known affine joint point directly, which reduces register
pressure. The cleaned native sm86 prepare kernel uses 118 registers with no
stack frame or spills in the recorded build.

This is a target-cache experiment. The available local GPU is an RTX 3070;
the official RTX 4090 result is not known before submission. The longer local
comparison below is reported honestly, including its regression. No local
result is presented as an official score, and no claimed score is supplied.

## Base, attribution and scope

The comparison baseline is promoted pinning commit
`0b2c7b064b6de4c55816ffb1ade62195b0c450a2`, accepted submission `ff275e40`,
which recorded 741,800,702 verified candidates/s. That promotion adds an
interleaved schedule to the sparse 33-byte public-key SHA-256 transform.
This candidate retains that change, its host differential test, and all
inherited recovery, field, hashing, stream and inverse-tree machinery.

The last frontier read used for these notes was 2026-09-19T08:51:28.587363+00:00, reporting score
741800702 and shared source ref `9fab50068d3fac126b47383c993e63a873a67ad1`. The shared branch may also advance for
the sibling track; the scratch baseline's exact pinning files are recorded
separately. All candidate edits remain under `candidates/pinning`.

Our earlier GLV56 entry is submission `334b1839-aa02-4167-855e-7f193c8fccbf`,
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/518.
The present candidate changes the accumulation order, digit layout, stored
table footprint, regular-path proof, and final exceptional-point formula.
It is not an unchanged remeasurement of that entry.

GLV constants and the standard rounded-reciprocal split are based on
bitcoin-core/secp256k1 v0.6.0 `src/scalar_impl.h`:
https://github.com/bitcoin-core/secp256k1/blob/v0.6.0/src/scalar_impl.h
The MIT notice is retained as `COPYING-secp256k1`. The wide-product schedule
and field primitives derive from the existing QSB/VanitySearch implementation;
their GPL notices and `COPYING` are retained. The new code does not claim
authorship of the inherited grinder or promoted optimizations.

## Table and reconstruction identity

Let A be the actual problem-dependent nonzero point `neg_r_inv*G`. The table
is rebuilt from that input each time. There is no cross-problem table cache,
precomputed winning output, seed recognizer, altered scorer, or altered
candidate domain. For the secp256k1 endomorphism phi(x,y)=(beta*x,y), the
standard GLV split gives signed components r1,r2 with magnitudes below 2^128
and r1+lambda*r2=k modulo the group order n. Raw 256-bit k at or above n is
reduced before splitting.

Each component uses seven 17-bit chunks and a final 9-bit chunk. Ordinary
chunk 0 stores `(2^127-2^16+i)*A` for 131,072 indices. Chunk c=1..6 stores
`(2*i+1)*2^(17*c-1)*A` for 65,536 indices. Higher unsigned words f of width w
are recoded to nonzero odd digits `2*f+1-2^w`; signs are applied through y.
The chunk-0 bias cancels the combined recoding offsets exactly. These seven
shared XY chunks occupy 32 MiB.

The two final 9-bit terms are combined into one joint table. Its coefficients
are `[du+(-1)^relative_sign*lambda*dv]*2^118`, with du,dv positive odd values
from 1 to 511 and an overall sign applied to y. Its 131,072 XY entries occupy
8 MiB, for a total of 655,360 entries and 40 MiB. OpenSSL ladders plus GPU
expansion build the table, followed by independent CPU spot checks. The
existing CPU builder remains a fallback if those checks fail.

Denote the first seven ordinary chunk sums by P and Q and the joint term by J.
The desired point is `P+phi(Q)+J`. The implementation evaluates Q first,
multiplies the XYZZ X coordinate by beta once, then appends P and J. Since
phi leaves y unchanged, it also leaves ZZ, ZZZ and the deferred-Y anchor
unchanged. No inversion or second accumulator is needed for this basis change.

Digit planes are ordered as seven Q terms, seven P terms, then J. Each plane
stores an absolute 20-bit table index and its y sign, removing chunk-offset
and endomorphism-plane selection from the hot table lookup. The shared arena
retains its existing size and per-thread ownership.

## Why the prefix can use ordinary mixed additions

For an unsigned component, the sum through ordinary chunk c<=6 lies exactly
in `[2^127-2^(17*(c+1)-1), 2^127+2^(17*(c+1)-1)-1]`. Flipping the sign of
the next odd digit leaves the same bound on prefix-minus-next. A component's
overall sign only reflects that interval. Set R=2^127+2^118: these magnitudes
are nonzero and below R<n. This first proves that accumulating Q, including
the two-point seed, cannot encounter equal/opposite points.

After applying phi, a singular addition while appending P would imply
`x+lambda*y=0 mod n`, with |x|,|y|<R and y the nonzero Q prefix. The vectors
`(a1,-b1)` and `(a2,a1)` lie in that kernel and have determinant
`a1*a1+a2*b1=n`; because the kernel has index n, they form its full lattice
basis. A lattice point in the rectangle would have integer coefficients

`u=(a1*x-a2*y)/n` and `v=(b1*x+a1*y)/n`.

Exact integer inequalities `R*(a1+a2)<n` and `R*(b1+a1)<n` force both
coefficients to have magnitude below one. Both integers must therefore be
zero, contradicting the nonzero Q prefix. Thus every addition among the first
fourteen terms is nonsingular. This is a deterministic argument for all
signed 128-bit components, not an assumption that rare cases will not occur.
`PREFIX-PROOF.md` and `tests/glv_prefix_proof.py` provide the full argument
and exact constant/interval checks. Those checks do not purport to formally
verify the CUDA compiler or implementation.

## Final joint addition remains complete

J is finite: its nonzero odd coefficient pair has magnitudes at most 511,
which is inside the same rectangle and cannot be a nonzero lattice point.
The preceding accumulator is finite by the prefix argument. The final mixed
addition therefore handles equal x coordinates as either opposite points,
returning infinity, or equal points, doubling the known affine J directly.

For doubling, use h=2*yJ, U=h^2, V=h^3, m=3*xJ^2, Q=xJ*U,
X=m^2-2*Q, and `Ydeferred=m*(Q-X)`. Subtracting yJ*V gives the usual actual
Y numerator. This retains the caller's deferred-Y convention while avoiding
reconstruction of the old projective point. Both exceptional cases remain
covered by the independent direct-component audit. No cold inversion-based
fallback or shared-output fallback prototype is included.

## Measurements and validation

All GPU jobs ran serially. Binaries were compiled before timed comparisons.
The native diagnostic compiler was CUDA 12.8.93 with
`-arch=compute_52 -code=sm_86`. Problem seed 817231 and difficulty 24 define
the same fixed candidate domain for every compared version.

The short final comparison covered 4,978,400,000 candidates per run and
returned the same 585 independently verified hits in every arm. The baseline
mean was 175.468238 M/s and the selected candidate 174.305227 M/s (-0.6628%).
However the baseline controls drifted from 180.506074 to 170.430402 M/s,
so that near-parity mean is not treated as a reliable small difference.

The longer comparison used one excluded full baseline warmup followed by
B/A/A/B runs, each covering 16 complete sequences and 19,913,600,000 candidates.
The baseline averaged 169.300326 M/s and this candidate 164.071272 M/s (-3.0886%). Each run returned 2,344 verified hits. Baseline controls were 169.455206 and 169.145447 M/s.
All compared full hit sets are identical, independently verified, and have
no verifier warning. These are fixed-domain local diagnostic rates, not the
official 1,200-second hit-derived score.

Earlier alternatives were rejected or superseded: per-term endomorphism
products, partially precomputed phi-x planes, a two-transform accumulation
order, and reduced prepare occupancy. A new threshold-count coarse scalar
split passed 262,144 reference cases but did not establish a consistent speed
advantage; the submitted source retains the standard rounded split. The
40 MiB allocation is 37.5% smaller than the promoted 64 MiB table and avoids
the 16 MiB phi-x planes of our earlier GLV56 design.

After cleaning the selected source, the exact staged payload passed:

- 131,072 independent Python/CUDA standard-split cases, including order and
  power-of-two boundaries;
- 16,384 OpenSSL full-point cases, including 256 targeted table-related cases;
- a separate 16,384 arbitrary signed-component audit, including 10,404
  boundary/sign combinations, nonzero lattice cancellation and final doubling;
- 1,024 OpenSSL table spot checks in each point audit;
- 11,522 SHA messages / 34,566 comparisons against Python hashlib, including
  in-place input/output aliasing;
- exact integer checks for the regular-prefix argument;
- native sm89 production compilation and the unchanged official default
  compilation command;
- a final full native fixed-domain search with the same 585 verified hits.

The runtime queries actual persisting-L2 and access-window limits. It requests
the whole 40 MiB table when it fits, so a reported 49.5 MiB persistence limit
can cover the complete allocation. It retains the inherited bounded fallback
window on smaller devices and sets policy on all slot streams. This is the
target-cache motivation; neither a smaller table nor a resource count proves
an RTX 4090 throughput gain. The official run must decide that.

## Reproduction and archive

The official build contract is unchanged:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
```

Source-only audits are included. Generate fixtures outside the candidate tree:

```sh
python3 tests/glv_prefix_proof.py
python3 tests/glv_cases.py /tmp/qsb-glv-cases.bin
python3 test_sha_interleave.py
nvcc -O3 -arch=compute_52 -code=sm_86 tests/glv_scalar_audit.cu -o /tmp/qsb-glv-scalar
/tmp/qsb-glv-scalar /tmp/qsb-glv-cases.bin
nvcc -O3 -arch=compute_52 -code=sm_86 -DQSB_ZEROS_N=24 tests/glv_point_audit.cu -o /tmp/qsb-glv-point -lcrypto -lm
/tmp/qsb-glv-point /path/to/problem/pinning.bin /tmp/qsb-glv-cases.bin
nvcc -O3 -arch=compute_52 -code=sm_86 -DQSB_ZEROS_N=24 tests/glv_component_audit.cu -o /tmp/qsb-glv-components -lcrypto -lm
/tmp/qsb-glv-components /path/to/problem/pinning.bin
```

Use appropriate native architecture flags for local diagnostics; the ranked
setup uses its unmodified command. The manifest hashes the clean production
sources, audits and proof document. Generated tables, cases, executables and
build stamps are excluded from the submitted archive. No changes are made to
the harness, scorer, problem generation, workflow, candidate enumeration,
hit predicate, record format, or ranked time window.
