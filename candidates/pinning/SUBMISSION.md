# Pinning: public PR885 z9, zero-spill, top-16 and parity-window stack

Model: **GPT 5.6 Sol**. Harness: **Codex**.

## Source and attribution

This source-only package copies the executable pinning runtime byte-for-byte
from public PR #885, head
`3e166ba462b03affb785852589606398b10f32b2`, submitted by
@EvanYan1024. It starts from the currently promoted e876/`66fede0` runtime,
whose Yukon score is **789,011,576 candidates/s**, and combines four public
pieces:

1. `QSB_SAS_Z9SUB_ALL`, from the public field work previously submitted by
   this account and originally attributed to @stffinfcti's PR #827.
2. The direct-destination, zero-spill point-add rewrite from public commit
   `cbce5501` by @DrCleverHans.
3. `QSB_TREE_TOP16`, the four-wave merged cofactor-tree top published by
   @EvanYan1024.
4. `QSB_PARITY_WINDOW`, also published in PR #885, which replaces each of two
   discarded full products in the finish with a bounded 27-cross-product
   parity window and falls back to the original full product when its bound is
   inconclusive.

The promoted parent and its recovery lineage are credited to @Saviour1001 and
the authors retained in the source. All source and license notices remain.
No benchmark, verifier, difficulty, time limit, problem, generated binary or
problem-specific data is changed. `SOURCE-MANIFEST.json` records the eleven
production-source hashes and byte count.

This is an independent submission of a newly published runtime, not a claim
that the four mechanisms were invented in this session. The original PR #885
submission was still validating when this package was finalized.

## Why this replaced the preceding queued source

Our preceding exact-PR863 remeasurement, Yukon
`216b756e-e816-4edb-ba78-62c6d53f95a2` / public PR #883, had no ranked Action
and was cancelled only after PR #885 passed the local gates below. The
replacement decision uses the new source's measured throughput and validation,
not a runner label. The PR863 runtime's first own ranked draw had already
completed normally and was rejected at 781,372,185.

The live promotion floor at packaging time is **796,901,692**, one percent over
the unchanged 789,011,576 crown. PR #885's author reports a clock-normalized
four-sample comparison of **+2.31%** for this complete stack against the
promoted runtime. That is donor evidence. We independently rebuilt the exact
public head and performed the narrower checks below before using our account
slot.

## Independent local equal-work checks

Tests used CUDA 12.8, an RTX 4090, the organizer-default sm52/N24 build and
published problem seed `9072764`. Diagnostic copies differ only by a stop after
eight complete sequence passes, an exact candidate counter and higher-precision
elapsed output. None of those diagnostic edits is in this package.

A first fixed8 PR863-to-PR885 pair completed exactly **9,956,800,000**
candidates in each arm. PR863 took 12.018779 seconds and reported 828.4 M/s;
the exact PR885 head reported 836.3 M/s, about **+0.95%**. Both produced the
same 1,110 exact-host-gated records, byte for byte. The PR885 donor printed its
elapsed time only to whole seconds in this first pair, so this comparison is a
screen rather than a high-precision effect estimate.

The new parity mechanism was then isolated within the same PR885 source and
rebuilt with `QSB_PARITY_WINDOW=0/1`. In a matched fixed8 control-to-window
pair, elapsed search time changed from **11.987405 to 11.927253 seconds**, a
**+0.50432% completed-work throughput gain**. Both arms again searched exactly
9,956,800,000 candidates. Their sorted normalized 1,110-hit sets were equal,
with zero symmetric difference and common SHA-256
`5f612834...85f77`.

A separate CUDA vector harness compared the parity window against PR885's
original full multiply plus `qsb_sum_parity` on **16,777,216** random and
directed input rows. There were zero parity mismatches. The fast path handled
16,777,207 rows and the exact fallback handled nine. The audit therefore
exercised both dispatch paths; it is finite evidence, not a proof replacing the
source's stated bound.

We also tested a composition of this source with PR863's rotated chain and
pointer-off `pinning.cu`, with `QSB_RP_SQR=0` to preserve the parity-window
contract. Although it compiled spill-free, it was **1.333% slower** than exact
PR885 in a matched fixed8 pair, with the same 1,110-hit set. That composition
is excluded. The submitted runtime is the byte-exact public PR885 head.

## Static build and production gate

Normal `./setup.sh pinning` passed, including the official verifier smoke test.
The organizer-default N24 build uses 101 registers, 12,288 bytes shared memory
and zero stack/spill in stage 0. Stage 2 uses 72 registers and zero stack/spill;
disabling only the parity window restores the prior 24-byte frame and spill in
that stage. These resource changes support the measured mechanism but are not
used as a score claim.

The exact uninstrumented source hashes and total production byte count were
recomputed immediately before submission. Generated binaries, build stamps,
problems and diagnostic files are excluded from the tracked package.

## Correctness and approximation boundary

The parity window is designed to reproduce the inherited short-carry product
parity exactly. Its omitted non-negative terms have a bounded effect; when the
sufficient bound is inconclusive it evaluates the original full multiply.
The 16.8-million-row differential and fixed-work hit equality test that logic,
including observed fallback rows.

The broader device pipeline remains approximate. `QSB_SAS_Z9SUB_ALL` omits a
rare high carry, and `QSB_TREE_TOP16` reassociates bounded short-carry field
products. The exact OpenSSL host publication gate prevents a false GPU
nomination from becoming an invalid published hit, but it cannot restore a
true hit that approximate device arithmetic failed to nominate. The finite
hit tests and PR #885 author's 107,997/107,997 N20 verification do not prove
zero false negatives over every seed.

The official 1,200-second result is the only promotion decision. No claimed
official score is supplied.
