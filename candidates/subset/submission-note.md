# Subset successor from the PR1137 source

Model: GPT 6 Sol, xhigh reasoning. Harness: Codex.

This subset candidate starts from the public
[PR1137](https://github.com/Layr-Labs/quantum-safe-bitcoin-challenge/pull/1137)
source (`d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`), whose official score was
624,752,385 candidates/s. The current leader is 623,518,629 candidates/s, so
the 1% promotion floor is 629,753,816 candidates/s. The PR1137 source fell
short of that floor. The original benchmark source base for this worktree is
`b59484345df5208f5caffc82c25a4a3b50cbe523`.

## Changes

- `QSB_DROP_Z2_EARLY=1` removes the early carry propagation into limb 2 at
  nine lean field-square fold sites. The later carry propagation remains. This
  belongs to the speculative filter and can very rarely lose a hit; an exact
  host gate rechecks every tentative hit before publication. Setting the switch
  to 0 restores the inherited fold code.
- `QSB_RECODE_BASE_A=1` uses the base point A directly rather than A/2 in the
  fixed-base table. The scalar recoding, GPU table construction, host spot
  check, and CPU fallback were changed together. Setting the switch to 0
  restores the inherited A/2 path.
- `QSB_SHA_FOLD=1` uses the packaged `_SHA256TransformDigest32Q` for the second
  SHA-256 transform in the paired-epoch path. The routine already exists in
  the inherited package; this change connects it to the active call site.
  Setting the switch to 0 restores `_SHA256Transform`.

All other executable changes in this directory are inherited from PR1137,
including its exact host verification gate, field chain, and SHA routine.
Source licenses and contributor notices are retained. The PR1137 source
inherits the PR1088 composite credited to terrapinelf and its earlier
contributors. The active SHA idea follows the public pinning SHA work; the
base-A and field carry changes were prepared here.

## Verification and limits

- The organizer-style CUDA 12.6 build completed on arm64, including native
  `sm_89`. `kernel_digest` uses 128 registers and 49,152 B shared memory per
  CTA, with zero stack frame and zero spills in that static build.
- The base-A scalar recoding passed 20,008 random and boundary identity/bounds
  checks. The default preprocessed PTX point-update path matched a reference
  over 1,000 random inputs.
- `tests/sha_digest32_host.cpp` compiles the actual optimized SHA digest
  function as host C++ and compares 10,000 random 32-byte messages with
  OpenSSL SHA-256; all eight output words matched in every case. This tests
  the algebra and call contract, not CUDA scheduling or target-GPU throughput.
- A native sm_89 static ablation at CUDA 12.6 found 14,896 `kernel_digest`
  SASS instructions and 4 B spill stores/loads with all three new switches
  disabled. Enabling only the carry cut gave 14,880 instructions and zero
  spills; only base-A gave 14,840 and 4 B spills; only SHA fold gave 14,888
  and 4 B spills. All three enabled gave 14,808 instructions and zero spills.
  These are compile artifacts, not an end-to-end throughput measurement.
- No NVIDIA GPU is available in this environment. No candidate-vs-base
  throughput or full hit-set measurement exists for this combination. Static
  resource counts do not establish a ranked speedup or correctness on the
  official CUDA 12.8 runner. The three changes may interact, especially in
  register scheduling and the SHA path. This submission is the target-GPU
  measurement for the strongest locally verified subset candidate available
  in this handoff; no score claim is made from the static measurements.

The package contains only `candidates/subset/` changes relative to the
promoted benchmark source. `SOURCE-MANIFEST.json` records every source file and
hash, excluding this note and the manifest itself to avoid self-reference.

## Why this package is a separate candidate

The promoted subset source and PR1137 differ substantially. PR1137 already
bundles the optimized paired-epoch and field paths, which makes it the
relevant starting point for these three cuts. The source tree here was copied
from that public submitted package before the new switches were applied. The
package itself stays within the editable subset directory; the benchmark
schema, harness, problem generation, scoring, and verification files remain at
the promoted source commit. `SOURCE-MANIFEST.json` identifies the precise
implementation commit and records byte hashes, so the candidate can be
reconstructed without relying on a worktree name or an untracked build output.

The direct base-A recode removes the need to multiply each scalar by two modulo
the group order before digit extraction. Its table uses A in both the GPU build
and the host fallback, and the host sample checker expects the same point. A
one-sided change at any of these sites would yield a valid-looking table with
incorrect scalar products. The scalar identity/bounds property test covers
zero, boundary, order-adjacent, and random inputs. It does not substitute for
an end-to-end hit-set comparison on an NVIDIA GPU.

The paired SHA transform is called only from `qsb_pair_second_sha_z` when the
paired-epoch path is enabled. The inherited `sha_gate_fma.cuh` implementation
contains the second-block schedule and digest extraction. The switch is local
to that call; the first SHA transform, candidate enumeration, and host
publication check follow the inherited PR1137 package. The file's CRLF line
endings were normalized for a clean source diff; the code tokens are the same
as the public donor file.

The field cut removes one `addc.u32` at each selected fold. PTX carry flags are
sensitive to instruction order, so the verification model used the candidate's
preprocessed PTX rather than only a handwritten algebraic rewrite. There were
no differences in the 1,000 tested point updates. That sample cannot bound
extremely rare carries. The exact host gate can reject false tentative hits,
while any lost tentative hit would lower verified throughput; that risk is part
of the reason this source has not been uploaded.

## Promotion gate

A rank-worthy upload needs a target-GPU comparison against the same source
with these switches disabled, with identical problem seeds and verified hit
sets, plus a comparison against the current frontier. The official score uses
verified hits over elapsed time. A favorable SASS count or a donor's earlier
rate is not an official throughput measurement for this combination. A future
submission should refresh the leader and 1% floor immediately before upload,
then replace this note's status and measurements with the observed data.
