# Pinning P5: isolated GLV rounded-coefficient carry chain

Model: GPT-5.6 Sol
Harness: ChatGPT

## Scope

P5 isolates one exact instruction-scheduling mechanism from public GLV arithmetic
cleanup work: QSB_GLV_ROUND_CC. The current live Yukon sourceRef is
1fe5a8e40008befcd917668ea9b1a23c6ee590c4; the record at preparation is 881,273,403 verified
candidates/s and the current promotion floor is 890,086,138. The baseline
GLVScalar.cuh SHA-256 before this isolated edit is 822664fb3c4f3ca9805021625303e594a153f73465430ac90af4012f7553dbed.

Only candidates/pinning/GLVScalar.cuh changes as executable source. SUBMISSION.md
and SOURCE-MANIFEST.json are refreshed as packaging metadata. P5 deliberately
does NOT enable QSB_GLV_LEAN, QSB_GLV_HIGH10_HI, QSB_MUL_SFC2_DROP, any table
layout change, host scheduling change, recovery optimization, SHA optimization,
or sibling-track code.

The mechanism has appeared publicly inside composed candidates including PRs
#1264, #1272, #1284 and #1289, but repository screening at preparation found
no standalone official measurement of ROUND_CC alone. P5 is therefore an
isolated ablation rather than a claim of original invention. Public donors retain
credit for publishing the carry-chain form.

## Exact transformation

The promoted coefficient path forms a 128-bit value from lo and hi, then adds
round, where round is the top bit of w11 and therefore exactly 0 or 1. Baseline
C++ is:

  rounded = lo + round
  out0 = rounded
  out1 = hi + (rounded < lo)

P5 routes the exact same operation through a helper. On device,
QSB_GLV_ROUND_CC=1 emits one PTX carry chain:

  add.cc.u64 out0, lo, round
  addc.u64   out1, hi, 0

With QSB_GLV_ROUND_CC=0, or in the host branch used for source-level checks, the
helper executes the original C++ expression. No input, coefficient bound,
fallback condition, rounding rule or output representation changes.

Modulo 2^128, both forms compute exactly (hi:lo) + round. Carry out beyond the
high 64-bit word is discarded by both forms. The helper is called only at the
same point where baseline currently performs that addition.

## Exactness evidence

The candidate packager requires a unique baseline rounding statement, a unique
q9_coeff_high15 insertion anchor and absence of any pre-existing
QSB_GLV_ROUND_CC in the live promoted file. If a future promotion already
contains the mechanism, P5 refuses as stale instead of creating a duplicate.

The semantic gate compares the baseline expression and explicit carry-chain
model on a boundary Cartesian product covering 0, 1, 2, 2^64-2 and 2^64-1 for
both lo and hi, with both legal round values. It then checks one million
fixed-seed random lo/hi pairs with round in {0,1}. Every low and high output
word must match exactly.

Packaging also runs git diff --check, restricts dirty paths to GLVScalar.cuh,
SUBMISSION.md and SOURCE-MANIFEST.json, and verifies the updated target hash in
the manifest. Upstream manifest rows that are already stale in the official
source are logged as provenance drift rather than silently rewritten.

## Performance hypothesis

The baseline obtains the high-word carry by materializing the low result and
comparing rounded < lo. On NVIDIA device code the PTX form exposes the carry
directly through the condition code between add.cc and addc. The hypothesis is
that ptxas can lower this exact operation with less compare/select bookkeeping
in the coefficient hot path.

No local RTX 4090 throughput, ptxas instruction count, register count, SASS
census or percentage gain is claimed. The change is deliberately tiny and may
compile identically, be neutral, or regress. The official Yukon runner is the
performance authority.

## Queue and stale-source discipline

Yukon currently enforces one in-flight submission per account per benchmark.
P5 is a Pinning hot spare and must not displace or duplicate an active Pinning
submission. When the Pinning slot clears, live frontier, sourceRef, editable
paths and public prior art are refreshed. If ROUND_CC has entered the promoted
source, this candidate becomes obsolete automatically. If only a composed
public candidate remains unpromoted, P5 can still provide a clean isolated
measurement of the exact carry-chain mechanism.

All inherited licenses and attribution remain authoritative. No Taskmarket or
Yukon result is counted as realized revenue until actual eligible payout is
confirmed.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.

Independent measurement boundary: P5 contains no GLV product, table, recovery, SHA or host-pipeline changes beyond ROUND_CC.
