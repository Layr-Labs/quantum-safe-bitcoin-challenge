# Pinning: four cached GLV banks on the promoted pipeline, with sparse table readback

Effort: xhigh. Track: pinning. Prepared using GPT 6 Astra through Codex.
There is no NVIDIA GPU on the development host. This source package has CPU
correctness checks and CUDA compiler evidence; its own RTX 4090 throughput is
unmeasured. The official verified-hit score will decide whether it improves.

## Baseline and attribution

The base is promoted commit `1fe5a8e40008befcd917668ea9b1a23c6ee590c4`,
submission `2c7a195e-48d6-4497-8530-ecea28b042df`, at 881,273,403 verified
candidates/s. It includes odinfree's six-bank GLV12 table, anamdongparkjinhyeong's
promotion, the current field/recovery arithmetic, register seed handoff,
sequence-overlap pipeline, refill-before-gate ordering, and exact host gate.
The 100-bip qualifying integer at preparation is 890,086,138 candidates/s.

The four-bank geometry comes from 0xCramJam's public submission
`c13f3832-8bc0-4111-b75d-c0f361541a17`, commit
`90f89008c599bd046906e61794218cc2a22edbcd`, PR #1370. The mathematical geometry
is retained exactly, then implemented in the promoted geometry helpers with an
independent control. This is substantial unpromoted work; 0xCramJam is a coauthor.

The direct device sample readback follows terrapinelf's public submission
`86f500cc`, commit `8f5b6e124a47a6c07e23ea9fc10bb0be5dad474e`.
Terrapinelf is a coauthor. The source here adds an independent control, retains
the promoted modulo sample selection for the non-power-of-two top segment,
and includes a CPU test of the actual check's copy/error handling. The donor's
pubkey-SHA unroll change is not part of this package.

All inherited notices and COPYING files remain. The historical SUBMISSION.md
records the earlier promoted donor. This document describes this new package.

## Rejection that changed the direction

Our previous `0746a6d2`, PR #1369, scored 840,542,283/s and was rejected.
All 120,394 emitted hits passed verification over 1,201.5315 seconds; this was
a speed rejection. It used anchor parking, a 104-register preparation cap,
coefficient changes and a seed rewrite. Static compilation showed no spills,
but no measured GPU timeline established the intended overlap.

That run used the intel-r3 runner, driver 580.178.04, with a reported 863.2M/s
peak. The last verified hit implies at least 1.01094 trillion enumerated
candidates, closely agreeing with the hit-derived count. This is evidence of
a throughput deficit, not evidence that a large fraction of correct hits was
lost. Different runners and seeds prevent attributing all of the score gap to
one source edit. The related public cap104 candidate also rejected at
835,336,697/s. The present package starts again at the promoted source; it does
not retain those preparation lifetime or register-cap edits.

## Why the cache geometry is worth testing

The original six-bank layout keeps three banks, 48 MiB total, in the dense
prefix and streams the other three. Each GLV component visits all six banks,
so the ordinary two-component path makes six gathers outside the dense prefix.
Four small banks in the same 48 MiB prefix reduce that count to four, while
retaining twelve terms and eleven point additions. The price is a larger
streaming suffix: 9,803,211,584 bytes for the complete table.

The public geometry donor reports matched, independently verified RTX 4090
measurements using CUDA 12.8.93, the default organizer build, and a 450 W cap:

| Duration | Reference candidates/s | Four-bank candidate/s | Paired change |
| --- | ---: | ---: | ---: |
| 300 seconds | 851,948,858 | 874,938,566 | +2.70% |
| 1,200 seconds | 851,902,211 | 875,228,938 | +2.74% |

These are the donor's reported local measurements, not measurements of this
port. The donor subsequently scored 876,067,427/s officially on the
leadergpu-3568275 runner and was rejected. Its reported peak was about
910.5M/s. That result establishes valid hits for the donor's complete package;
it does not guarantee this package clears the current record.

This port preserves the promoted pipeline and arithmetic. The donor's
coefficient/residual rewrites, older host loop, and huge-page allocation are
not imported. The changed table layout itself is the search-throughput
hypothesis. Sparse readback removes a concrete startup cost of the larger
layout without claiming a modeled score gain from saved bytes.

## Exact geometry and implementation

`QSB_FOUR_HOT=1` selects these six logical/physical banks:

| Bank | Shift | Entries | Record offset |
| --- | ---: | ---: | ---: |
| 0 | 0 | 262,144 | 0 |
| 1 | 18 | 262,144 | 262,144 |
| 2 | 37 | 131,072 | 524,288 |
| 3 | 55 | 131,072 | 655,360 |
| 4 | 73 | 67,108,864 | 786,432 |
| 5 | 100 | 85,279,885 | 67,895,296 |

There are 153,175,181 records of 64 bytes. Banks 0..3 total 786,432 records,
exactly 48 MiB. The existing persisting-window setup and load policy remain.
The index fits below bit 31; bit 31 remains the Y sign. Table byte offsets use
64-bit size_t/uint64_t, including the Y-offset pass and spot-check reads.

The current rounded-reciprocal GLV splitter has the conservative component
bound `0xa2a8918ca85bafe22016d0b917e4dd77`. At shift 100 its maximum field is
170,559,768. The top center is the next odd integer, 170,559,769. Thus the top
digit `2*f-center` is always odd and nonzero, and its magnitude is bounded by
the center. Signed table selection uses `(abs(d)-1)/2`.

The biased first bank has coefficient
`K + index`, where `K = 170559770 * 2^99 - 2^17`.
The later odd coefficients and this bias telescope to the original signed
component. Both components still pass through the original endomorphism and
point-chain code. Zero-component handling is inherited unchanged.

The builder ladder radix becomes 16,384. The maximum high index is 10,410,
below its 16,384 capacity. Host ladder generation, device splitting and the
OpenSSL scalar helper all use the same geometry constants. The original
4,096-radix builder is restored by `QSB_FOUR_HOT=0`.

## Sparse readback and failure behavior

`QSB_GT_SPARSE_CHECK=1` reads only each selected 64-byte record directly from
device memory. The 216-sample schedule is unchanged: all four corners per
bank plus the existing deterministic random samples, using modulo selection
for the non-power-of-two bank. OpenSSL still independently derives each
expected point and compares all 64 bytes.

A CUDA copy failure or any mismatched byte fails the check. The existing host
builder fallback remains; its full-size allocation is delayed until fallback
is actually necessary. The successful path does not allocate or copy a
9.8 GB host mirror. Copy count is 216 and requested data is 13,824 bytes.
This can still be slower than a batched gather on some systems; no timing is
claimed here. The Y-offset pass continues after the check, as in the baseline.
Setting `QSB_GT_SPARSE_CHECK=0` restores full readback for a host-side control.

## Verification and its scope

`test_fourhot.py` compiles the production geometry/recoder helpers with
undefined-behavior sanitizer. It independently checks the reciprocal error
bound using rational arithmetic, samples 50,000 full scalars and reconstructs
their GLV relation, and checks both signs of bounded random and boundary
magnitudes. Each geometry passes **402,572 signed-component reconstructions**,
**2,415,432 decoded digits**, contiguous table coverage, sign-bit separation,
and ladder bounds. Every reconstructed integer equals the original signed
component exactly, before any reduction modulo the curve order.

`test_sparse_check.py` extracts the production OpenSSL scalar helper, point
conversion and spot-check function into a CPU executable. A CUDA-copy stub
checks actual source offsets against sparse virtual memory. Both geometries
pass all 216 samples; each rejects six injected faults, covering copy errors
and corrupt bytes at early, middle and final samples. The large-layout check
reaches byte offset 9,803,211,520, exercising addressing above 4 GiB.
This is a host-source test, not a CUDA table build or runtime memory test.

CUDA 12.8.93 was used in a linux/arm64 Docker container without a CUDA device.
The organizer-default PTX was assembled for sm_89, a native sm_89 executable
was built, and the complete default executable was linked:

| Build | Preparation registers | Finishing registers | Spill store/load bytes |
| --- | ---: | ---: | ---: |
| Default PTX assembled for sm_89 | 122 | 64 | 0 / 0 |
| Native sm_89 | 122 | 64 | 0 / 0 |
| Default executable sm_52 | 97 | 72 | 0 / 0 |

All eleven emitted entry functions are spill-free in each build. Preparation
shared memory remains 12,288 bytes. Default-PTX sm_89 preparation has 6,688
static instructions versus 6,680 in the promoted control; finishing retains
4,040. Disabling both switches restores every parsed sm_89 function's
instruction list identically to the promoted baseline. Enabling the changes
alters preparation, table construction and the table Y-offset pass; all other
parsed function listings remain identical.

The ranked driver can JIT different machine code. Static counts do not
establish cache hit rates, sustained throughput, or the promotion margin.
The changed point-add ordering can interact with inherited approximate field
filters; CPU scalar equality does not prove GPU hit-set identity. The exact
host publication gate remains enabled, and the official verifier is authoritative.

## Reproduction and submission decision

```bash
python3 -B candidates/pinning/test_fourhot.py
python3 -B candidates/pinning/test_sparse_check.py
nvcc -O3 -DQSB_ZEROS_N=24 candidates/pinning/pinning.cu \
  -o /tmp/pinning-fourhot -lcrypto -lm
nvcc -O3 -DQSB_ZEROS_N=24 -ptx candidates/pinning/pinning.cu \
  -o /tmp/pinning-fourhot.ptx
ptxas -arch=sm_89 -v /tmp/pinning-fourhot.ptx -o /tmp/pinning-fourhot.cubin
# Promoted control: add -DQSB_FOUR_HOT=0 -DQSB_GT_SPARSE_CHECK=0
python3 -B candidates/pinning/submission_preflight.py
git diff --check
```

The one official evaluation is motivated by the donor's measured cache-layout
benefit, its valid official hit result, this isolated port with an identical
baseline control, exact source-derived reconstruction checks, and removal of
the full-table successful-path readback. No official score is projected for
this archive. The larger table can increase DRAM translation/cache pressure,
startup, power demand or allocation cost, and the public paired gain may not
transfer to this port or runner. A rejection requires diagnosis, not an
unchanged re-upload.

Only candidates/pinning is packaged. The source manifest covers all on-disk
files except itself; binaries, PTX, cubins, SASS, logs and downloaded references
are stored outside the editable directory. Harness, enumeration, candidate
accounting, output format, verifier and scorer are unchanged.
