# Pinning: PR827 square and fused-square z9-lane schedule on promoted e876

This source-only candidate starts from promoted pinning commit
`e876032f79e6f4f3af2732bbba39403e29f0e227` and ports one runtime file,
`GPUMath.h`, from public PR #827 by @stffinfcti, head `87a770a`. The host
publication gate, `pinning.cu`, table, recovery, SHA, and other runtime files
are byte-for-byte unchanged. `SOURCE-MANIFEST.json` records every source
hash. No generated binary or problem-specific file is included.

## Source, mechanism, and attribution

The promoted e876 source scored 789,011,576 on Yukon, leaving a 796,901,692
one-percent floor when this candidate was prepared. PR827 trims the bit-288
`z9` carry/borrow lane from active short-square `_ModSqr` and fused
`_ModSqrAddSub2` reductions. The promoted source already removed an
analogous multiply tail. The field header is copied in full from PR #827
without editing its inline PTX schedule; the original donor credit and
`COPYING` remain. This isolated audit and packaging were performed with
Codex. The promoted source's C31 field arithmetic and exact OpenSSL host
publication gate remain intact.

This is an **approximate device arithmetic** optimization. Omitting `z9`
is wrong on some top-carry inputs. The exact host gate verifies every GPU
nomination before publication, so those false nominations cannot become
invalid published hits. The host cannot recover a real hit that wrong GPU
arithmetic failed to nominate. The source option
`-DQSB_SAS_Z9SUB_ALL=0` restores the omitted PTX operations.

The normal ranked build is `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu
-lcrypto -lm` via `setup.sh pinning`. Local tests used the default sm52
target on an RTX 4090. Their stop-after-N and counter instrumentation exists
only in scratch copies, not this package.

## Static build evidence

Both the promoted and PR827 headers compiled and linked with identical N24
diagnostic source. Stage0 remained 101 registers per thread, 12,288 bytes
shared per CTA, no stack and no spill. Stage2 remained 72 registers and its
inherited 24-byte stack/spill. Stage0 static SASS changed from 20,508 to
20,502 instructions; stage2 remained 9,408. These counts narrow the source
change but do not, by themselves, prove speed or correctness.

## Matched search speed

On published seed 892809653, ranked `single_hash` N24, each finite8 arm
completed exactly 9,956,800,000 candidates and published the same 1,226
unique `(sequence, locktime, recid)` hits, zero symmetric difference. All
arms exited zero. The search-loop `Done` times, in A/B/B/A order, were:

| Source | Seconds |
| --- | ---: |
| promoted e876 A1 | 12.118000 |
| PR827 field B1 | 12.024029 |
| PR827 field B2 | 12.030045 |
| promoted e876 A2 | 12.122187 |

Mean e876 12.1200935 seconds versus PR827 12.027037 seconds gives
**+0.77373% completed-candidate throughput** in this short screen. Raw logs
and hit files are in `/tmp/qsb-pin-pr827-audit/run-base`, `run-variant`,
`run-variant2` and `run-base2`. Setup/table time was not included in these
printed search-loop times.

A separate, longer finite32 six-arm comparison on the same e876 source and
seed processed 39,827,200,000 candidates and exactly the same 4,757-hit
set per arm. The promoted control mean was 48.6840455 seconds; field-only
PR827 mean 48.486439 seconds, **+0.40755% throughput**. A pointer-plus-PR827
composite mean was 48.6245565 seconds, only +0.12234% over control; it was
therefore not included here. The longer result is a more conservative speed
estimate than finite8. The local result is not an official Yukon score.

## Device field-vector differential

An independent CUDA harness compiled `_ModSqr` and `_ModSqrAddSub2` with
the promoted and PR827 headers under the active C31 and short-carry flags.
It tested 2,871 directed canonical field triples, including values near
`p−1`, top carries and constructed 2-adic square boundaries, plus 100,000
pseudorandom canonical triples. It also tested square output aliasing and
each fused-square operand alias. PR827 differs from e876 in both primary
outputs and all alias modes for 2,786 directed inputs. **Zero of 100,000
random inputs differed.** Each arm matched exact bigint residues on all
100,000 random inputs.

For directed inputs, exact-residue errors rose from 581 to 2,856 in square
and from 401 to 2,837 in fused square. Those directed inputs deliberately
cluster near carry boundaries, so the high fraction is not a natural-input
error-rate estimate. A concrete input `a=p−2, e=q=0` produces a different
field residue under PR827. The source must not be described as universally
exact. Local raw vector input is
`/tmp/qsb-rpsqr-field-audit/input.bin`; base and variant outputs are in
`/tmp/qsb-pin-pr827-audit`.

## High-hit recall and publication gate

An N20 diagnostic processed 64 complete sequences on seed 892809653,
exactly **79,654,400,000** candidates. PR827 generated 151,948 tentative
GPU nominations, copied all of them, published 151,947 after exact OpenSSL
checking, and rejected one. Maximum tentative hits in any batch were 32,
under the 64-hit host copy cap and device buffer cap; no batch overflow was
observed. Promoted e876 generated and published 151,947, rejecting zero.
The two published 151,947-member tuple sets have **zero symmetric
difference**. The same-range PR600 reference set previously passed an
independent CPU verifier, 151,947/151,947 and zero failures.

The one extra PR827 nomination rejected by the host proves a device
arithmetic divergence reached the hit filter. No reference hit was lost in
this finite range, and no invalid hit was published. This does not exclude
rarer false negatives on longer searches or other seeds. The N20 diagnostic
log is `/tmp/qsb-pin-pr827-audit/run-recall64/recall64.log`; the CPU reference
summary is `/tmp/qsb-pin-recall-n20/summary.json`. These local files are
review evidence, not package dependencies.

## Production-source smoke test

The source-only package passed `./setup.sh pinning` on CUDA 12.8 and an RTX
4090. The local 30-second benchmark used the unchanged verifier and GPU
wrapper through the supported `cmd:` grinder, because the ranked root bridge
is unavailable on this machine. On seed 892809653 at N24, it independently
verified **2,909/2,909** published hits and rejected zero. The resulting
810.867540 M/s score has 1.8541% hit-sample relative variance, so it is a
smoke test rather than a prediction of the 1,200-second official score.
Generated binary, build stamp, score and benchmark output were removed or
stored outside this source package after the test; the ten-file manifest
and 374,184-byte source count still match.

## Decision boundary

This package contains only the measured field-header change. The longer
same-work estimate is +0.408%, below the current +1% promotion floor from
the promoted baseline by itself. The official ranking may differ with seed,
runner and duration. Submission should be decided from the live floor and
queue, with the residual rare false-negative risk explicit. Any future
composition needs its own equal-work speed and hit validation.
