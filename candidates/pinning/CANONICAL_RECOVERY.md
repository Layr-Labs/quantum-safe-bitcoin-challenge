# Pinning: canonical point arithmetic and boundary-safe recovery

Historical candidate7 record. GROUPED_INTEGRATION.md describes candidate8;
the arithmetic and recovery functions tested here are unchanged there.
The references below to current source/status describe the earlier snapshot.

Model: GPT-6. Harness: Codex.

Prepared follow-up research, not a submission or a measured GPU improvement.
Our earlier PR29 remains in flight and its artifact is preserved. This work
repairs a reproducible coordinate-encoding defect in the preceding carry-fixed
candidate while retaining the specialized pipeline, short tree carry fold and
batched runtime table ladders. See LADDER_INTEGRATION.md for that integration's
history; its arithmetic-readiness conclusion is superseded by this document.

## Source and attribution

The base is current promoted main
`4d39b5f0a881653d6332a7801dd84bc14175fa61`, pinning PR17 by nullforest8200.
Its official score is 644,546,620. That is the parent's score, not this work's.
The specialized deferred-Y pipeline is from alvaroborras PR24; host ladder
batch normalization is from jacklightChen PR53, acknowledging MakiRH4 PR46.
Our preceding integration is commit
`0240ec1c6941f07d59efcd043b45fa4b7bfce023`.

The canonical conditional subtraction follows hybridnoise's subset PR60,
`65fb673d-5014-4ee8-866d-97d972e7b1f0`, source
`f31dcba6b5a3de04a28e9dcf47b336bad8278fa2`:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/60 .
We inspected and compiled that source and reproduced its arithmetic/pipeline
CPU audits. Its normalization prompted the independent valid-curve witness
below. This pinning port changes only GPUMath.h and associated audits/docs;
the candidate6 pinning.cu and GPUHash.h remain byte-identical. Existing GPL,
copyright and earlier research attribution are retained. Relative to promoted
main, all changes remain within candidates/pinning/.

## Why congruence was insufficient

Let p=2^256-2^32-977. The corrected raw multiply/square returned the right
residue in [0,2^256), but that representative could be p+t rather than t.
Compressed public keys require canonical x bytes and the canonical y parity.
Comparing only modulo p masked this difference. The earlier CPU pipeline also
used OpenSSL for add/sub, which silently canonicalized intermediate values.

An independent affine-curve construction finds three collinear secp256k1
points with slope -k and a small recovered x coordinate. For example k=2,
recovered x=1 gives the old finish output p+1. Both input points are valid,
the prepared denominator is nonzero, and neither recovered point is infinity.
This is a real recovery-helper counterexample, not a hypothetical raw residue.
It does not establish a failure frequency in the ranked input distribution.

The new standalone audit constructs 45 such valid point pairs and tests six
projective scales for each, including 1, p-1 and p-65537. It executes the actual
host multiplication/squaring, the actual add/sub function bodies with emulated
fixed-width PTX carry/borrow macros, and actual prepare/finish helpers. Python
affine formulas and inversion are the independent oracle. The previous
candidate fails all 270 cases; the new candidate passes all 270, checking both
x coordinates and both parity bits exactly, not modulo p.

Both hot products now apply the same conditional subtraction after their
host/device branches. For a raw value below 2^256, raw>=p iff all upper limbs
are UINT64_MAX and the low limb is at least 0xFFFFFFFEFFFFFC2F. One subtraction
therefore gives a canonical result without a general multiprecision routine.
The tree multiplier retains its distinct residue contract, with the existing
normalization at inverse roots and returned leaves.

This repair does not turn the inherited mixed-add formulas into complete
elliptic-curve formulas. Exceptional/infinite intermediate-point handling and
CUDA execution remain separate limitations; no universal correctness proof
or device runtime result is claimed.

## Validation

From the benchmark root, the self-contained arithmetic/recovery checks are:

```sh
python3 candidates/pinning/audit_exact_field.py
python3 candidates/pinning/audit_canonical_recovery.py
```

- 50,580 boundary/random input pairs: 202,320 host/emulated-device point
  outputs agree canonically with Python integers; 50,580 separate tree outputs
  agree modulo p. All 2,500 in-place alias checks pass. The device model runs
  the literal production PTX followed by the actual C++ normalization postlude,
  not a test-inserted oracle reduction. UBSan enabled.
- 45 valid curve pairs, six projective scales, 270 recoveries pass the exact
  coordinate/parity regression. The old candidate's failure report and first
  witness are preserved outside the repository under qsb-work/build/.
- Four new generated problems, seeds 2026091821..2026091824, 128 candidate
  pairs each: source-extracted fast/generic pipeline paths agree with the
  independent harness; all 65 emitted hits pass the unchanged official CPU
  verifier at diagnostic N4. Actual recoder: 6,275 cases. Actual host ladders
  and sampled table-builder CPU threads: 2,880 entries; fallback prefix: 180.
  The collective is replaced by a scalar inverse, referenced table entries
  use OpenSSL, and add/sub outside the separate boundary audit use OpenSSL.
  This is not GPU execution or synchronization validation.
- Official setup passes kernel compilation and CPU verifier smoke; all six
  harness tests pass. CUDA13 N24/sm89 compilation passes: ranked prepare126
  registers and finish80, both zero spill loads/stores and zero stack frame.
  Generic prepare has192 bytes stack, no spills; generic finish has no spills.
  Compiler resource changes are not a throughput measurement. The shared
  C++ normalization adds work even where the branch is rarely taken.

Workspace reports: build/pinning-canonical-field-check.json,
build/pinning-canonical-recovery-check.json,
build/pinning-canonical-pipeline-check.json, and matching logs.
The previous all-point host ladder comparison remains applicable to the
unchanged host functions; it is not rerun merely for this header repair.

## Submission decision

At the last live check both tracks are open with a 100-bip (1%) minimum;
pinning's nominal next integer target is650,992,087. The machine has no GPU,
and no qualifying speedup or official candidate score exists. No hardware was
rented. This candidate supersedes candidate6 as the correctness-preserving
conservative integration; candidate4/5/6 remain historical artifacts and must
not be submitted as-is with the now-known encoding defect.

Pending pinning PR29 bb4d7f81 and the independent subset3b1bba4b remain
untouched. After PR29 finishes, recheck the live frontier and PR24/PR53 results
before selecting a measured, substantive follow-up. Do not duplicate pending
work or use the old 249M frontier as a comparison baseline.
