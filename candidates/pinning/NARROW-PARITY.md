# Pinning: an 18-product bounded parity window

Effort: ultra. Development and independent reviews used GPT 6 Astra through
Codex. This is a source optimization developed on a CPU-only machine. No local
GPU throughput is claimed; the remote ranked evaluation is the performance
experiment.

## Starting point and attribution

The starting checkout is promoted commit
`94abdd0d72847b780c7d4f99da4f367e6f9f0fd1`, submission
`07009ac3-94a4-428e-b030-1f6ce317ccb7`, with a recorded score of
797,446,582 verified candidates per second. The local schema-v2 manifest
requires a 100-basis-point improvement. At this starting score that means
approximately 805,421,048 candidates/s; this note does not predict that the
new implementation will clear that bar.

The inherited implementation is already substantially optimized. Its
`SUBMISSION.md` documents the PR827 field arithmetic from @stffinfcti, the
bounded parity window from @EvanYan1024's PR885, and @terrapinelf's
problem-wide recovery-coordinate isomorphism. These mechanisms are part of
the promoted source and remain attributed to their authors. This submission
narrows that promoted parity window; it does not introduce the original
window, field arithmetic, recovery equations, or isomorphism.

The recent public submission list and notes were checked before selecting
this experiment. In particular, the top-16 traversal in public submission
`8740a30` and the remeasurement discussion in `ed148db` were read as context.
Their changes are not included here. Public numerical claims are not treated
as a substitute for measurements of this candidate. The source manifest's
100-basis-point threshold also takes precedence over a conflicting threshold
claim in a public note.

## Problem and proposed change

The packed recovery finish computes two compressed public keys for each
candidate. It needs a full x coordinate but only one bit of each corresponding
y coordinate. The promoted parity helper computes selected columns of a
256-by-256-bit multiplication and uses a bound to decide whether those
columns suffice. Ambiguous cases use the inherited full multiplication and
sum-parity implementation.

The previous fast window evaluates 27 word cross-products: columns 5, 6, and
7 for the middle window, and columns 12, 13, and 14 for the high window.
The new implementation omits columns 5 and 12. Those columns contain six and
three cross-products respectively. The remaining 18 products suffice under
a slightly stricter acceptance guard. The helper is used twice per candidate,
so the ordinary fast path removes 18 32-by-32-bit word multiplications across
the two parity computations, together with their accumulation instructions.
This is an operation-count reduction, not a measured end-to-end speedup.

Only `ParityWindow.cuh` changes executable behavior. The compile-time switch
`QSB_PARITY_WINDOW_NARROW` defaults to 1. Defining it as 0 restores the previous
27-product PTX schedule and its original guards, providing a direct control
without changing the other optimization switches. The full multiplication
fallback and the exact OpenSSL host publication gate are retained.

## Bound and acceptance conditions

Let `B = 2^32`, let `a_i` and `b_j` be unsigned base-B limbs, and define
`D_k = sum(a_i*b_j for i+j=k)`. Consider the accumulator before the common
column-8 parity XOR. The original and narrowed middle accumulators are

```
M_old = D7 + floor((D6 + floor(D5/B))/B)
M_new = D7 + floor(D6/B)
```

Since column 5 has six products and each is at most `(B-1)^2`,
`0 <= M_old - M_new <= 6`. These accumulator expressions may be reduced
modulo `B^2` by the PTX instructions. Only the low 33 bits matter. Requiring
`low32(M_new) < B-7` prevents the omitted carry from crossing the low-word
boundary and ensures that the original low word is not `B-1`. Thus both
windows have the same middle bit 32, including after the identical column-8
XOR, and the original helper's middle-word guard is satisfied.

The corresponding high accumulators are

```
H_old = D14 + floor((D13 + floor(D12/B))/B)
H_new = D14 + floor(D13/B)
```

Column 12 has three products, so `0 <= H_old-H_new <= 3`. There is no
64-bit overflow in either high accumulator: the expressions are monotone
in the nonnegative limbs, and the all-maximum-limb input gives
`H_old = B^2-1`. This matters because the next expression also reads the
high 32 bits of H.

Both helpers then form, with the same fixed recovery parameter beta,

```
Q = H + 977*(H >> 32) + low32(M) + (beta[3] >> 32)
```

Under the middle-word guard, the increase from the new Q to the old Q is
at most `3 + 977 + 6 = 986`. The 977 term covers the single possible carry
into H's high limb. The original helper accepts only when
`low32(Q_old) < 0xfffff859`, or `B-1959`. The new requirement is
`low32(Q_new) < 0xfffff47f`, or `B-2945`. It guarantees that adding at most
986 neither crosses the low-word boundary nor violates the original guard.
The two Q values therefore have the same bit 32. A wrap at bit 64 is
irrelevant because the result uses only bits 0 through 32.

The final parity expression uses the same operand low bits, beta low bit,
negation bit, middle bit 32, and Q bit 32. Consequently, whenever the new
fast path accepts, the original window also accepts and returns the same
bit. All other inputs use the retained full-product fallback. The stricter
guard trades a small additional fallback region for fewer ordinary-path
operations. It does not introduce an unguarded carry truncation.

## Validation method and environment

The development host has Python 3.13, a C++ compiler, and no NVIDIA GPU.
`yukon setup --track pinning` passed its CPU verifier smoke test.
The requested unmodified `yukon run --track pinning` could not produce a
score: this host lacks the organizer's bridge executable and GPU runtime.
The harness, problem specifications, setup script, ranked configuration,
and sibling track were not edited to work around that limitation.

For compilation checks, CUDA 12.8.93 was obtained from NVIDIA's official
redistributables into temporary storage. A portable GCC 13 toolchain and
compatible glibc development headers were needed because the VPS's system
GCC 15 and glibc 2.42 headers exceed that CUDA release's supported host
environment. None of those toolchain files are in the submission archive.
Compilation can establish syntax, PTX assembly, resource usage, and linkage;
it cannot establish GPU execution correctness or throughput.

The added `test_parity_window.py` checks the actual selected PTX source
through a host translation and compares it with independent integer models.
Its directed cases target the tightened guards and omitted-column carries;
the original 27-product branch remains available as a differential control.
Existing carry, host-gate, and SHA audits provide regression checks for the
surrounding candidate. The new test translates the actual device
`_ModMultCore` PTX and `qsb_sum_parity` for the fallback comparison, instead
of relying on the different host branch of the multiplier. The generated
host C++ is built with UndefinedBehaviorSanitizer and fail-on-error enabled.

All local checks passed on the final executable source:

| Check | Result |
| --- | --- |
| New extracted-PTX differential | 2,161,035 rows; 8,644,140 parity comparisons; zero mismatches |
| New differential coverage | 2,000,000 random rows plus one-hot, extreme, and guard-boundary constructions; both negation bits |
| Narrow-window paths | 2,084,428 fast accepts; 76,607 fallbacks; 41,281 additional fallbacks versus control |
| Observed omitted-column deltas | maximum middle delta 6; maximum high delta 3 |
| Independent bigint proof audit | 172,064 rows, including 72,000 guard-boundary constructions; zero failed implications |
| Existing carry audit | exhaustive reduced-width cases, 680 carry62 boundaries, 1,000,000 random rows per carry62 site, and C31 checks passed |
| Existing exact host-gate audit | 64 SHA256d midstate cases, binary layout, EC recovery, and source gate checks passed |
| Existing SHA audit | 11,522 vectors and 34,566 digest comparisons passed, including in-place aliasing |

Directed cases deliberately concentrate on the fallback regions. Therefore
the fallback fraction in this test table is not an estimate of the GPU's
ordinary fallback rate. The two million random rows and the adversarial
constructions serve different validation purposes.

Five complete CUDA builds passed: original and candidate at default sm52,
original and candidate at native sm89, and the candidate with the narrow
switch disabled at sm89. The finish-kernel compiler observations were:

| Architecture | Original static instructions | Candidate static instructions | Registers | Spill loads/stores |
| --- | ---: | ---: | ---: | ---: |
| sm52, default ranked compile target | 10,020 | 9,822 | 72 in both | 0 in both |
| sm89, native RTX 4090 diagnostic | 4,088 | 4,056 | 66 in both | 0 in both |

Other kernels' resource use and static instruction counts were unchanged.
The entire sm89 SASS dump for `QSB_PARITY_WINDOW_NARROW=0` was byte-for-byte
identical to the original dump. Existing OpenSSL deprecation warnings were
present; there were no build errors. The only edit after those builds was
an attribution comment clarifying which window had prior GPU validation;
the executable and preprocessor source was unchanged.

The smaller finish kernel is evidence that compilation preserves a real
change. Neither its static instruction reduction nor zero spills constitutes
a GPU timing result, particularly because the ranked default target may
undergo driver JIT compilation on the runner.

Reproduction from the benchmark work directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/test_parity_window.py
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/test_carry62.py
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/test_host_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 candidates/pinning/test_sha_interleave.py
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/pinning-narrow candidates/pinning/pinning.cu -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_PARITY_WINDOW_NARROW=0 -o /tmp/pinning-control candidates/pinning/pinning.cu -lcrypto -lm
```

If reusing a previously built checkout, explicitly rebuild after this header
edit. The existing harness compilation cache checks the `.cu` modification
time and difficulty, rather than included-header modification times. A fresh
remote checkout builds the selected header normally.

## Limits and interpretation

The inherited device field arithmetic contains deliberate approximations.
The change establishes a narrower sufficient condition relative to the
promoted window; it does not prove universal correctness or perfect hit
recall for the entire inherited kernel. The CPU transcription of the field
multiplier is also not automatically an oracle for GPU arithmetic: the
source documents differences between its host and device reduction tails.
The exact host gate still rejects invalid GPU nominations, but cannot
reconstruct candidates that the GPU did not nominate.

There is no local GPU A/B timing or hit-set comparison for this submission.
Reduced operation count can be offset by register allocation, scheduling,
branch behavior, or the share of total time spent in this helper. A remote
run is needed to determine the actual effect. A single ranked score also
contains sampling and machine-state variation; it should not be described
as a precise matched speedup measurement.

The next decisive check is the unchanged ranked pinning workflow on its
RTX 4090 runner, followed by comparison with the current promoted score.
If the result does not improve, retain the bounded-window proof and local
validation as a reproducible experiment rather than claiming a performance
win. No claimed local score is supplied, and no benchmark condition is
relaxed for this submission.
