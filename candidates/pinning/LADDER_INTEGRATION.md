# Pinning: exact field carries and batched runtime table ladders

Historical candidate6 integration record. The later canonical-coordinate
repair and current validation/limitations are in CANONICAL_RECOVERY.md.
The device/header identity statements below describe the earlier commit.

Model: GPT-6. Harness: Codex.

Prepared research candidate; not submitted while PR29 remains in flight. This
combines exact field-carry handling, the public specialized deferred-Y pipeline,
and batched affine normalization of the runtime table's host ladders. CPU
correctness checks and CUDA compilation pass. No GPU speedup or qualifying
official score is claimed.

## Source and attribution

The worktree starts at current promoted main
`4d39b5f0a881653d6332a7801dd84bc14175fa61`. Its pinning tree is byte-identical
to nullforest8200's PR17 original `647698377478bf4898f86679791c651e683cf5c3`:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/17 .
That source scored **644,546,620** on the production runner, with all
92,297/92,297 hits verified, N24, elapsed1201.2216s, seed1219282414, workflow
35125858345. This is an official result for the parent, not this candidate.
The earlier development provenance attributed to saucegodbased is retained
in the inherited research ledger; that older repository was inaccessible here.

The deferred-Y intermediate/final specialization comes from alvaroborras PR24,
head `6e76a74fed8e6e5b8439e64ec20f586085f37d52`:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/24 .
Our previously tested candidate4, commit
`25c258335fccb74fdbe4c83347ab3c099f9e8cff`, adds exact final carries to
point multiplication/squaring and shortens the tree multiplier's correction
from eight 32-bit limbs to three. Its arithmetic and device pipeline are
preserved byte-for-byte here.

The new host ladder helper and calls come from jacklightChen PR53,
`9274883051636def6db5add0d3ba0e02314813f0`:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/53 .
That work acknowledges the batched normalization approach in MakiRH4 PR46:
https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/46 .
Only that host improvement is ported; our corrected arithmetic is retained.
Existing copyright, GPLv3 notices, and research attribution remain in place.

Only `candidates/pinning/` differs from the current promoted main. The harness,
verifier, generator, benchmark configuration, and subset track are unchanged.
The candidate retains a 64MiB table and the existing memory/batch geometry.

## Changes and rationale

The runtime table uses 15 signed windows [18,17,...,17] and two small ladders
per window. The serial builder makes 12,002 individual points affine. The new
builder keeps each ladder in projective form during its addition walk, then
normalizes its points together with OpenSSL `EC_POINTs_make_affine`. It emits
the same coordinates, byte order, identity slots, and unused padding as before.
The table remains problem-dependent and is rebuilt from runtime `neg_r_inv`.

The inherited point-field correction handles the final carry in the second
secp256k1 reduction fold. For p=2^256-C, C=2^32+977, the remaining low value
when that carry is set is below C^2. Adding C is exact and affects at most
three 32-bit limbs, since C^2+C < 2^96. For example, a=b=p-65537 previously
produced 0x1fc30 instead of the correct residue 0x100020001. The accompanying
`audit_exact_field.py` and candidate4 ledger document that correction and its
instruction-level CPU validation.

The specialized XYZZ chain and shorter tree correction are the hot-path
performance hypotheses. Correct point carries also add work; the net GPU
effect remains unmeasured. The host initialization change alone is far too
small to establish a 1% improvement over a 1200-second run.

## Validation performed

Workspace evidence is under `/home/traverse/crypto/qsb-work/`:

- `tests/check_batch_ladders.py`, `build/batch-ladder-check.json`: actual serial
  and batched builders, four fresh generated problems (2026091801..1804).
  All 48,008 real points match both the old builder byte-for-byte and separate
  OpenSSL generator multiplications. All identity/padding slots are checked.
  UBSan enabled. Host build times were 0.0160..0.0192s batched versus
  0.0936..0.0980s serial; these are CPU diagnostic observations, not a GPU or
  ranked benchmark. About 0.08s saved per startup cannot establish the 1% gate.
- `tests/check_ladder_pipeline_cpu.py`,
  `build/pinning-ladder-pipeline-check.json`: actual source recoder checked on
  6,275 edge/random scalars; actual host ladders and sampled device table-builder
  threads checked on 2,880 entries, plus 180 CPU-fallback prefix entries.
  The padding-thread guard is exercised. The device code is executed as CPU
  source with emulated literal PTX products, not on a GPU.
- Four fresh problems x128 sequence/locktime pairs: actual prepare/finish code
  in both fast and generic paths agrees with the independent harness. All
  45 emitted hits pass the unchanged official CPU verifier at diagnostic N4.
  Referenced table entries use OpenSSL; the collective inverse is replaced by
  a scalar inverse. This does not validate CUDA synchronization or runtime.
- Candidate4 device source and `GPUMath.h` identity are checked. Its prior
  252,900 arithmetic results and 2,500 alias checks apply to the unchanged
  functions; they were not rerun merely for a host-only edit.
- Official `./setup.sh pinning` passes, including canonical kernel compilation
  and CPU verifier smoke test. All six harness unit tests pass.
- CUDA13 compilation at N24/sm89 passes. Ranked fast prepare uses128 registers,
  fast finish80, both without spills; unchanged from candidate4. The generic
  finish retains the prior 8-byte spill. Ranked CUDA12.8 execution remains
  necessary; this machine has no GPU. OpenSSL3 deprecation warnings are present.

## Submission decision

Both tracks are open with a 100-bip (1%) minimum. At the checked frontier of
644,546,620, the nominal next integer pinning target is **650,992,087**.
Recheck the live frontier and PR24/PR53 results before selecting a follow-up.
Pending pinning PR29 `bb4d7f81-8e9b-4c24-afd9-5d913e067b5a` is preserved;
no duplicate, cancellation, artifact replacement, or admission bypass is made.
Subset PR28 subsequently finished at 433,564,114 (+0.05015%), below the 1%
gate. Its independent follow-up 3b1bba4b is now validating and is preserved.
Candidate5's 1GiB
wide-window experiment remains separate because its cache tradeoff is unknown.
