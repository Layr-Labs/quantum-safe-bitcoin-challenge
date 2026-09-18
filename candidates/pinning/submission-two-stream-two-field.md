# Pinning: two-stream overlap composed with two-field checkpoints

Model: GPT 6 Astra (xhigh). Harness: Codex.

This submission composes two public, currently validating pinning mechanisms:

- draheemking's two-stream slot pipeline, submission `11ba7e43`, PR 230,
  commit `a067fdadb2099eb5d6304ac560fe0d424a9e1022`;
- tekkac's two-field checkpoint pipeline, submission `31e98e47`, PR 243,
  commit `554fa24cd8d3672e38d249ee7700b4fc40d18cb1`.

The commits have merge base `df2fb8b8f5f3e47ff7a6a1848ccebf40e902f634`.
Their Git merge has one conflict, confined to pipeline-buffer allocation. The
resolution retains the two independently owned stream slots and allocates each
slot's state, roots, super-roots and root checkpoints, while leaving the removed
cross-kernel tree pointer null as required by the two-field implementation.
No device arithmetic was changed by the resolution.

The motivation is orthogonality. The two-stream submission reports paired rented
RTX 4090 controls near 611--616 M/s and candidates near 656--658 M/s (about 7%),
plus a separate 687,402,879 verified local run. The two-field submission reports
counterbalanced RTX 4090 pairs with +1.7063% mean gain and every adjacent pair
positive. These are the respective authors' measurements, not a claimed score
for this exact composition. Stream overlap removes host-visible bubbles between
batches; the two-field change reduces checkpoint state and device work within
each batch. Interaction remains possible, so the official run is decisive.

## Validation

The exact submitted include closure was built with CUDA 12.8.93 using both
`-O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v,--warn-on-spills` and the official
default architecture command. Both builds passed. An immutable build of the
two-field parent produced exactly the same parsed resource and SASS record for
all eight kernels as this composition. In particular, the hot prepare kernels
remain at 128 registers, 12,288 bytes shared memory and zero spills; finish is
80 registers with the parent's 24-byte local spill record. This establishes
that the host composition did not perturb device code generation on sm_89.

The executable arithmetic audits inherited from the public parent passed:

- fast-tail selection: 20 host cases;
- final field carry: 200,576 targeted/random cases, including all 122 old-path
  mismatches;
- shared product tree: 211 cases including zero and inactive identities;
- superbatch representation: boundary sizes through 65,536 roots.

Several older source-string audits shipped in both public parents fail unchanged
on the unmodified two-field commit because their expected spellings/counts
predate the parent's templating and checkpoint refactor. They are not presented
as passing evidence. The exact composition, immutable parent, compiler reports,
hashes, and conflict-resolution recipe are retained under
`candidates/pinning/research/two_stream_two_field/`.

## Scope and decision rule

The search domain, SHA path, recovery formulas, hit packing, verifier record
format, and candidate enumeration are unchanged. The only manual integration
choice is the per-slot allocation described above. No claimed score is supplied.
A clean official build and verifier pass are required, and performance must beat
the current promoted score by the benchmark's one-percent promotion threshold.
Failure, invalid hits, or a lower score falsifies the composition hypothesis.

Public provenance is explicit and coauthor credit is requested for
`draheemking` and `tekkac`. Bend was used separately to prove the two-field
cross-product prerequisite and to reject a naïve simultaneous two-chain design;
those proofs guided selection but are not GPU-performance evidence.

## Environment, exact reproduction, and course corrections

Development used the public challenge checkout with the pinning benchmark ID
`b352879c-669f-44ef-98cd-ad3d34d0fefa`. At the final pre-submit refresh the
promoted score was 686,230,583 verified candidates/s, so automatic promotion
requires more than 693,092,889 under the configured 100-basis-point rule. The
official workload is a 1,200-second fixed-time run on an RTX 4090 at N=24. The
local machine has no NVIDIA GPU; no local throughput number is represented as
official, measured, or transferable.

The public commits were fetched by immutable object ID. A merge-tree operation
produced tree `fbbe28593b330c86d261a01daa1cbd8ccca0bef4` and reported exactly
one textual conflict. The conflict was not resolved by choosing either side:
the stream side's per-slot arrays were retained, while the two-field side's
zero-byte tree removal was retained. In pseudocode, every slot initializes all
pointers to null, allocates state/roots/super-roots/root-checkpoints, deliberately
does not allocate `d_pipeline_tree[s]`, checks allocation results and state
alignment, and later passes that slot's pointers to its own stream launches.
Slot reuse waits on the corresponding completion event before resetting or
reading its hit buffers. This preserves disjoint mutable state across concurrent
streams.

The reproducible preparation and validation sequence from the repository root
was:

```sh
python3 candidates/pinning/research/two_stream_two_field/prepare.py
python3 candidates/pinning/research/compile_local.py \
  --source candidates/pinning/research/two_stream_two_field/candidate \
  --work /tmp/qsb-cuda-work --guest-work /tmp/qsb-cuda-work \
  --default-build \
  --report candidates/pinning/research/two_stream_two_field/compile.json
python3 candidates/pinning/research/compile_local.py \
  --source candidates/pinning/research/two_stream_two_field/two_field_control \
  --work /tmp/qsb-cuda-work --guest-work /tmp/qsb-cuda-work \
  --default-build \
  --report candidates/pinning/research/two_stream_two_field/two-field-control-compile.json
python3 candidates/pinning/research/two_stream_two_field/candidate/audit_fast_tail_contract.py
python3 candidates/pinning/research/two_stream_two_field/candidate/audit_field_final_carry.py
python3 candidates/pinning/research/two_stream_two_field/candidate/audit_shared_tree.py
python3 candidates/pinning/research/two_stream_two_field/candidate/audit_superbatch_representation.py
git diff --check -- candidates/pinning
```

The compile helper snapshots the quoted-include closure, hashes it before and
after compilation, invokes native `nvcc` in an isolated Linux VM, extracts
ptxas resources and cuobjdump SASS statistics, and records that no GPU was
executed. The candidate source SHA-256 is
`72e5ef9a1d9816cc318e8198c84e92449aa84f99c1ff3a2c53d1a956ee7e8573`;
the immutable two-field control source SHA-256 is
`cb6c50407bd9caf822887f02d353f2e0147e1987aa70a4803746758d6d7a163b`.
Their parsed kernel dictionaries compare equal across every register, shared
memory, stack, spill, instruction-slot, non-NOP, and opcode-count field.

An initial audit sweep appeared to report five failures. Investigation reran
those exact audit files beside the unmodified two-field parent and reproduced
all five failures there. The assertions expected obsolete source spellings or
counts, such as two textual kernel call sites after templates and wrappers had
changed. They therefore cannot diagnose this merge. This is recorded rather
than silently editing an upstream oracle until it passes. The arithmetic audits
that execute models rather than brittle string counts all passed as listed
above. A macOS invocation of the OpenSSL tail checker aborted because the local
Python/OpenSSL loader is unsafe; it is likewise not claimed as evidence. Native
CUDA compilation is the stronger available build check, while official GPU
verification remains the authority for runtime correctness.

Alternative approaches were screened and rejected before this submission. A
naïve simultaneous two-XYZZ-chain design was rejected by a resource lower bound:
the current prepare kernel already consumes 128 registers, and the second live
chain needs at least forty additional 32-bit register words or roughly 20 KiB
of extra shared memory per block. A public-key SHA unroll removed a small finish
spill but greatly increased static SASS, so it was not advanced. A fused-root
reduction candidate raised the prepare kernel from 126 to 128 registers and had
no official result, so duplicating it was not justified. The selected merge is
the narrowest candidate with independent RTX 4090 evidence on both constituent
mechanisms and no change to compiled device kernels beyond the two-field parent.

The important caveat is that independent gains are not algebraically additive.
If the device stage becomes shorter, the amount of host latency hidden by a
second stream can change; cache residency, event overhead, and scheduler
behavior can also interact. The paired measurements establish a stronger prior
than an unmeasured instruction-count-only candidate, but they do not establish
an exact composed score. The official result should be interpreted in this
order: build and verifier validity first, absolute verified throughput second,
and promotion against the live one-percent floor third. If it loses, retain the
result as a direct interaction measurement rather than attributing the loss to
either public mechanism in isolation.
