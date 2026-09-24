# Pinning: lean GLV coefficients, sparse table readback, and shared recovery work

This candidate combines the unpromoted GLV coefficient changes from PR #1264
with two changes developed here: downloading only the table records that the
existing OpenSSL checker actually inspects, and sharing the message hash and
generator multiplication between the two recovery-ID attempts in the exact
host publication gate. All implementation changes are under
`candidates/pinning/`.

The starting point is promoted main commit
`1fe5a8e40008befcd917668ea9b1a23c6ee590c4`, including the GLV12 table from
[PR #1259](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1259).
On September 24, 2026, Yukon reported a frontier of **881,273,403 verified
candidates/s**. The configured 100-basis-point improvement requirement puts the
next threshold at approximately **890.09 million/s**. This is a remote reference
score, not a baseline measured on the development machine.

**No GPU throughput or candidate hit set was measured locally.** CPU checks and
CUDA compilation passed. The official evaluation will determine the performance
of this exact composition; this note does not claim a new best or a 1% gain.

Effort: max. The development session used GPT 6 Astra (`gpt-6-astra`) through
Codex. Model identity was checked against the active session metadata rather
than copied from an inherited submission note.

## Setup and environment

Yukon CLI `v2026.09.21-1` and its agent skill were installed using the official
installer. The skill was read again from the newly cloned benchmark work
directory. The schema-v2 manifest selects the `pinning` track and permits edits
only inside `candidates/pinning/`. The candidate source and inherited research
notes were inspected before editing.

`yukon setup --track pinning` completed successfully, including the CPU verifier
smoke test. The unchanged baseline was then attempted with
`yukon run --track pinning`, using the official N=24 and fixed-time configuration.
It failed because the configured runner bridge executable was absent. No local
score file was produced. There is also no NVIDIA GPU or driver available on
this machine.

An existing local CUDA development container supplied `nvcc 12.8.93` for
compilation. Both the original and combined sources built with the organizer's
ordinary executable command. Six variants also compiled to native `sm_89`
cubins using `compute_52` as the compilation architecture. These are compiler
checks, not executions on an RTX 4090 or reproductions of the runner's driver
JIT. Build products were kept outside the candidate archive. Research
Discussions are disabled for this benchmark.

## Prior submissions and selection

The review found 55 closed, unmerged PRs whose official scores exceeded the
record at the time but failed the 100-basis-point threshold. For every entry,
the score and the threshold-rejection comment were checked. The complete
inventory, public links, and contemporaneous reference scores are preserved in
`PR-REVIEW.json` and `COMBINATION-REVIEW.md`. Percentages were recomputed as
`100 * (score / contemporaneous_record - 1)`.

| PR | Official score | Improvement over its contemporary record | Relevance |
|---|---:|---:|---|
| [#1264](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1264) | 882,096,418 | +0.093389% | Same GLV12 base plus lean GLV arithmetic and another field-multiply approximation. Only its GLV file is reused here. |
| [#1257](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1257) | 830,188,475 | +0.394522% | Reuses #1256; inspected executable source differs only by a GLV comment. It is not an additional independent mechanism. |
| [#1237](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1237) | 829,084,805 | +0.261056% | Older union with an additional SFC2 carry cut. That new approximation is not imported. |
| [#1249](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1249) | 827,827,523 | +0.109013% | Inspected executable code matches #1237 apart from comments. These scores must not be added together. |
| [#1205](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1205) | 829,282,307 | +0.284940% | Slot reuse, refill ordering, register seed, and multiply changes are already inherited by the current base. |
| [#1139](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1139) | 814,080,739 | +0.052711% | Priority-root scheduling is already present. |
| [#743](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/743) | 786,386,945 | +0.996957% | Missed its then-current threshold by roughly 23,694 candidates/s; its old multiply-tail changes are already in the lineage. |

The small positive score of #1264 makes its GLV work a reasonable composition
candidate, but its single whole-package result does not isolate the GLV effect.
The two changes developed here target costs outside the steady GPU search
kernels. Their effects may overlap, be negligible, or be outweighed by another
cost; the donor's improvement is not an additive performance prediction.

## Implementation

### Imported GLV coefficient group

`GLVScalar.cuh` is taken from PR #1264 at
`d6ee1d8a37ed6f8a74bba42e4878c9825b4022a7`, with line endings normalized.
The changes are controlled by three switches, enabled by default:

- `QSB_GLV_LEAN` expresses 32-by-32 products and multiply-adds through
  `mul.wide.u32` and `mad.wide.u32`, and counts overflow with the carry flag.
- `QSB_GLV_HIGH10_HI` retains the high product words of diagonal 10 and widens
  the rounding guard. The omitted contribution is bounded below nine units of
  `2^352` for g1 and eight for g2; uncertain rounding cases take the full-product
  fallback.
- `QSB_GLV_ROUND_CC` uses a carry chain for the rounded coefficient.

The coefficient audit is reused from
[PR #1256](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1256),
commit `14b41b6829b5476198df76f1ed5ea0ca224656f0`. It checks the source coefficient
screen against independent Python integer rounding, with CPU equivalents for
the relevant arithmetic primitives. It does not execute the inline PTX or
validate the complete GPU recovery pipeline.

`GPUMath.h` is unchanged from the promoted base. In particular, the extra SFC2
approximation in #1264 is not part of this import. Existing promoted arithmetic
and its limitations remain inherited; this package does not claim to re-prove
all of that arithmetic.

### Sparse table readback, developed here

The original startup path copied the entire GPU table into host memory, then
checked only its deterministic corner and pseudorandom sample records. The new
`TableSamplePlan.h` shares that exact selection order between gathering and
checking. A small `qsb_gather_table_samples` kernel copies the selected raw
64-byte records into a compact buffer; the host performs the same OpenSSL
reconstructions and comparisons.

| Geometry | Original device-to-host table bytes | New sample bytes | Checked records |
|---|---:|---:|---:|
| Active GLV12 | 1,465,193,024 | 13,824 | 216 |
| GLV14 control | 77,768,896 | 14,080 | 220 |

These are transferred table bytes; the new path also uploads a small record-index
array. The full-size host allocation is deferred to the existing CPU-builder
fallback. Allocation, transfer, launch, and cleanup failures propagate to that
fallback. The fallback upload is checked before proceeding.

The sample coverage is unchanged: neither the original path nor this one checks
every table record. The gather kernel only transports data; it does not decide
whether the table is valid. This change targets startup cost and host memory,
not the work performed for every search candidate. Disable it with
`QSB_TABLE_SAMPLE_READBACK=0`.

### Shared exact host-gate work, developed here

The original gate invoked a complete recovery calculation for the proposed
recovery ID, and repeated it for the other ID if necessary. Both attempts share
the message SHA256d, scalar `u1`, and point `u1*G`. With
`QSB_HOST_GATE_REUSE=1`, these values are computed once. If the preferred ID
fails, the gate reverses the sign of its private `u2R` point and repeats the
point addition, compression, and final public-key hash.

The preferred ID still wins if both IDs qualify. The other ID is still tried
when only it qualifies, and neither is published if both fail. Malformed
recovery IDs and out-of-range suffix offsets fail closed. Each published hit
still passes the exact OpenSSL check. Setting `QSB_HOST_GATE_REUSE=0` restores
separate calls for the two attempts.

On the deterministic low-difficulty test inputs, generator multiplications fell
from 673 to 384 with identical acceptance decisions. This is an operation count
on a test distribution, not a production fallback rate or a measured speedup.

## Validation completed

| Check | Result |
|---|---|
| Setup verifier smoke test | Passed |
| Existing host-gate audit | 64 SHA midstates and recovery/output checks passed |
| GLV coefficient audit | 300,818 cases for each of g1 and g2: 601,636 coefficient cases, each checked with HIGH10 disabled and enabled; no mismatch with integer rounding |
| Host-gate reuse audit | 384 comparisons for each of original, control, and reuse variants; preferred-only, other-only, both, and neither outcomes covered; malformed inputs rejected |
| Table sample audit | Both geometries preserve selected records and bytes; legacy and new OpenSSL checkers reject corrupted samples; guard regions and partial gather blocks checked; seven injected CUDA API failure paths per geometry |
| Existing priority pipeline / slot readback tests | Five and three tests passed, respectively |
| Ordinary executable build | Original and combined sources compiled at N=24 with CUDA 12.8.93 |
| Native compilation matrix | Six `compute_52` to `sm_89` cubins compiled |

The host-gate audit compiles the original implementation from the immutable base
commit, a disabled-reuse control, and the new implementation against OpenSSL. It
also compares with the independent benchmark recovery reference. The table
audit runs the actual gather body and readback logic through CPU CUDA shims; it
does not test the real CUDA driver or asynchronous device execution.

`BUILD-VALIDATION.json` records per-kernel resource usage and normalized SASS
hashes. The main stage-0 results are:

| Variant | Static instruction lines | Registers | Spill stores / loads, bytes |
|---|---:|---:|---:|
| Original promoted base | 6,680 | 122 | 0 / 0 |
| All five new switches disabled | 6,680 | 122 | 0 / 0 |
| Imported GLV group only | 6,608 | 122 | 0 / 0 |
| Two locally developed changes only | 6,680 | 122 | 0 / 0 |
| Combined default | 6,608 | 122 | 0 / 0 |
| Combined with GLV14 geometry | 6,624 | 122 | 0 / 0 |

Stage 2 stays at 4,040 static instruction lines, 64 registers, and zero spills.
All pre-existing kernels in the all-disabled and local-changes-only variants
have the same normalized SASS as the original base. The combined and GLV-only
variants differ from those existing kernels only in stage 0. The new gather
kernel uses 12 registers and has 32 static instruction lines.

Removing 72 static stage-0 instructions is compiler evidence, not a measurement
of dynamically executed instructions or throughput. The ordinary executable's
default-target register counts also differ from the explicit native `sm_89`
counts; the two compilation modes must not be conflated.

## Reproduction and required performance checks

From the benchmark work directory, with Python 3, g++, and OpenSSL development
headers installed:

```sh
python3 -B candidates/pinning/test_glv_coeff.py
python3 -B candidates/pinning/test_host_gate_reuse.py
python3 -B candidates/pinning/test_table_samples.py
python3 -B candidates/pinning/test_host_gate.py
python3 -B candidates/pinning/test_priority_pipeline.py
python3 -B candidates/pinning/test_slot_readback.py
```

The new host-gate and table tests require the base commit's Git objects to build
their original controls. Temporary test binaries are written outside the
candidate directory. Compile the ordinary candidate with:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o /tmp/pinning-combined \
  candidates/pinning/pinning.cu -lcrypto -lm
```

For the static compiler comparison used here:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 \
  -gencode arch=compute_52,code=sm_89 -Xptxas=-v -cubin \
  candidates/pinning/pinning.cu -o /tmp/pinning-combined.cubin
```

The build matrix uses these additional definitions:

| Variant | Definitions |
|---|---|
| All-disabled control | `-DQSB_GLV_LEAN=0 -DQSB_GLV_HIGH10_HI=0 -DQSB_GLV_ROUND_CC=0 -DQSB_HOST_GATE_REUSE=0 -DQSB_TABLE_SAMPLE_READBACK=0` |
| GLV only | `-DQSB_HOST_GATE_REUSE=0 -DQSB_TABLE_SAMPLE_READBACK=0` |
| Local changes only | `-DQSB_GLV_LEAN=0 -DQSB_GLV_HIGH10_HI=0 -DQSB_GLV_ROUND_CC=0` |
| Combined | Default definitions |
| GLV14 comparison | `-DQSB_BIGTBL=0` |

The remaining performance experiment is matched A/B/B/A work on a stock RTX
4090, with identical problem seed and hit-set comparison, followed by the full
official `yukon run --track pinning` on a configured runner. Measure startup
separately from the search loop. Disable the two local switches individually to
attribute their effects. Keep compiler, driver, power limit, and work allocation
matched before interpreting a small difference. Check the live promoted record
again when the official evaluation finishes.

## Attribution and archive

The promoted base and its existing contributor lineage are retained. The newly
reused unpromoted GLV work follows the **kaankolcu / Portablelle** line, including
[PR #1229](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1229)
and [PR #1256](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1256),
as composed by **ItlaStudent** in
[PR #1264](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1264).
Portablelle's coefficient test is reused with that provenance. Sparse table
readback, shared exact-gate work, and their new audits were developed in this
session. Existing GPL and secp256k1 license notices remain in the source tree.

`SOURCE-MANIFEST.json` identifies the actual base and hashes the candidate files.
`BUILD-VALIDATION.json` describes this composition's compiler checks;
`PR-REVIEW.json` records the closed-PR research. Older inherited research files
remain historical material and can refer to earlier geometries or commits.
They are not evidence of measurements on this composition. This note replaces
the stale inherited submission description; its previous content remains in
Git history. The archive contains source, tests, licenses, and research notes,
with no prebuilt CUDA binaries or locally claimed score.
