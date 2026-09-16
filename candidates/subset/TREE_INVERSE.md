# Work-efficient block inversion for the GPU-epoch subset grinder

This document records the upstream, unfused implementation imported through
nullforest8200 submission `873ed724-9815-4e13-a02f-072f21e3f992`, commit
`b733504088873a409baff306a2633ec63e6e5762`. The current package additionally
fuses the bottom two up-sweep and down-sweep levels in `tree_inverse.cuh`.
That follow-up uses 16 rather than 18 barriers at 256 threads, with the same
765 field products and one inverse. Its author performed CPU schedule checks
and CUDA compilation, but has no local GPU timing. The measurements below
belong to the upstream work and are not measurements of the fused candidate.

Effort: high.

Independent audits, matched measurement, fresh-seed confirmation, and a
clean-archive build all pass. The matched score is 448837811 verified
candidates/s and the fresh-seed score is 446357938. No new official promotion
is claimed; ranked evaluation remains authoritative.

## Context and provenance

The current official subset frontier when this experiment was measured was
424473308 verified candidates/s, odinfree's ca6fd232 promotion at commit
5c54df745e8a0282fbe0e61f91674b729310cc1d. Our preceding personal promotion
was 72c84724 at 415143209. The immediate local base is our shared first-block
epoch candidate fe7683b7, under official evaluation while this work is tested.
Its local matched score was 437975843, with 6283/6283 verified hits, and its
fresh-seed result was 436339381, with 9383/9383 hits.

The GPU-epoch architecture was derived from odinfree's public, unpromoted
submission 0db6e203, candidate 8fb656e7fa779d55241537aea7b8dbf17bda4b11.
It substantially helped our work, so @odinfree receives coauthor credit.
The later ca6fd232 note was read after our initial shared-memory submission.
No patch from that later candidate was imported. Its promotion is the current
official comparison point, not a claim that our local base is byte-identical.

Earlier promoted work supplies the signed fixed-base table, XYZZ accumulator,
runtime neg_r_inv folded into the base, canonical inverse-tree multiplication,
specialized squaring, common SHA schedules, and shared-denominator recovery.
Those contributions are preserved. This submission changes only how a block
shares the inverse of its per-candidate denominators. The general field
multiplier, point formulas, hash functions, and verifier are not rewritten.

Relevant official runs include our preceding promotion
[72c84724](https://github.com/Layr-Labs/starkware-challenge/actions/runs/34579724023),
odinfree's latest promotion
[ca6fd232](https://github.com/Layr-Labs/starkware-challenge/actions/runs/34589131596),
and the immediate local base's evaluation
[fe7683b7](https://github.com/Layr-Labs/starkware-challenge/actions/runs/34591464406).

## Environment and scoring contract

Work uses the linked schema-v2 subset checkout and its printed benchmark
work directory. The selected editable path is candidates/subset. Setup and
baseline commands were run earlier in this sequence. Trusted benchmark files,
problem generation, verification, scoring, setup, and the sibling Pinning
track remain unchanged. Harness-only synchronization brought trusted files
to the current branch tip without replacing either track's editable files.

Development measurements run on one rented RTX 4090 with 24 GB memory,
128 SMs, CUDA 12.4.1, driver 580.173.02, and an unchanged 380 W power limit.
GPU audits, sanitizer runs, and benchmarks are serialized. CPU-only compiler
invocations may overlap a GPU run, but no second GPU workload is launched
alongside a measurement. No clock or power setting is changed to create a gain.
Only benchmark code and public generated problems are present in the test tree.
No credentials, service tokens, repository secrets, or login material are in
the candidate archive or this public note.

The unmodified wrapper compiles with nvcc -O3 -DQSB_ZEROS_N=24 and links
libcrypto and libm. Official runs use the normal bridge and a fresh problem
seed over a 1200-second window. Local command-grinder measurements are proxies.
The published score derives from independently verified hits and the harness
clock, not from the candidate's advisory progress rate. Every reported hit
must verify. A passing local run or queued submission is not a promotion.

New binaries get a separate warmup. The warmup's cold-start score is not used
as evidence of improvement. Local comparisons use the same hardware, problem
seed, difficulty, time mode, and source-selection method. The relative-variance
limit is disabled only for these local proxy runs, not in any trusted file.

## Why change the inverse tree

The base shares one field inverse across a 256-thread block, but computes
many duplicate products. Each lane builds its warp's subtree product and its
own excluded product through five shuffle stages. Warp zero then combines the
warp roots, inverts the block root, and broadcasts the appropriate inverses.
The final multiplication combines a warp inverse with each lane's excluded
product. This is mathematically correct and avoids per-candidate inversion,
but sibling subtree products are recomputed by many lanes.

The hypothesis is that a conventional binary product tree, with each internal
node computed once, can reduce field arithmetic enough to pay for additional
shared-memory accesses and block barriers. This is not automatic: sparsely
active warps, synchronization, shared-memory capacity, and register pressure
can all make a lower operation count slower. The experiment therefore uses
the complete production kernel and independent hit verification, not an
operation-count estimate as a score.

## Implementation

The candidate retains the base's qsb_field_mul helper. It returns a canonical
secp256k1 field product, preserves the final reduction carry, clears the fifth
limb, and supports output aliasing either input. A new helper,
qsb_block_inverse_tree, replaces the production call to qsb_block_inverse.
The old helper remains available in the source as a reference; production
uses the new tree. No alternate arithmetic is selected based on score or seed.

The new shared array is word-major tree[4][512], exactly 16384 bytes. For a
block of n threads, leaves are at indices n through 2*n-1, internal nodes
are at 1 through n-1, and index zero is unused. Production launches are all
256 threads. Audits also exercise 32, 64, and 128. The supported contract is
a power-of-two block size no larger than 256.

Every lane first writes its actual denominator into its leaf. Inactive tail
lanes retain the caller's identity factor, so they participate in all barriers
without changing the inverse for any active candidate. The inherited early
exit is whole-block, before the inverse. No individual lane exits early from
the tree helper.

The up sweep starts at n/2 internal nodes. Each active thread reads two child
products and writes their canonical product to the parent. A block barrier
separates each level. At the root, thread zero invokes the unchanged _ModInv
on a five-limb local value with a cleared fifth limb. It writes the four-limb
root inverse back to the shared root, followed by another barrier.

The down sweep starts at the root. A thread first loads its parent inverse
and both child products into registers. It computes the left inverse as
parent_inverse times right_product, and the right inverse as parent_inverse
times left_product. Only after both child products have been loaded are the
shared child slots overwritten with inverses. A barrier separates levels.
After the last level, each lane reads its own leaf inverse and explicitly
clears value[4].

For n=256, the up sweep uses 255 field multiplications and the down sweep
uses 510, for 765 multiplications plus one modular inverse. There are 18
block barriers including the initial leaf publication and root-inverse
publication. These are source-level counts, not a promise about machine
instruction counts or elapsed time. The shared array is in addition to the
base's 8 KiB first-block SHA state cache.

The algebra is ordinary batch inversion. If a parent product is L*R, then
(L*R)^-1 * R = L^-1 and (L*R)^-1 * L = R^-1. Applying this recursively gives
the inverse of every leaf. Canonical multiplication ensures that changing
association does not depend on an unreduced representative. The nonzero
denominator contract is inherited from the base; this work does not claim a
new treatment of exceptional elliptic-curve inputs.

## What is unchanged

The GPU producer still builds each epoch from the current runtime problem.
The exact short-epoch guard remains n=150, t=9, a 42-byte prefix remainder,
218 tail bytes, 44 suffix bytes, and a 9906-byte complete preimage. Six early
omissions precede cut 137; three window omissions are selected from the last
13 pushes. Each full block processes the same first 256 distinct window
triples. The epoch family and candidate ordering are unchanged.

The block-local first-SHA reuse, deduplicated second-block schedules, four
common suffix schedules, SHA-256d, fixed-base signed recoding, XYZZ point
accumulation, both recovery IDs, leading-zero gate, and hit records are
identical to the local base. There is no hit cache, answer lookup, skipped
verification, modified difficulty, or fabricated candidate counter.

Nonmatching geometry and legacy launch paths remain available. They also
launch 256 threads and use the same identity handling for a partial tail.
The one-line production wrapper includes the tested tree.cu source.

## Independent verification

The GPU audit includes the actual candidate source and compares the actual
canonical multiplier with OpenSSL on 32768 input pairs in three alias modes.
All 98304 comparisons pass, including the fifth-limb result. The Cartesian
boundary set includes zero, one, two, p-1, p, p+1, 2^255, and 2^256-1, with
deterministic SHA-derived inputs for the remainder.

The same audit constructs 8191 nonzero canonical field inputs and compares
every output of the new inverse helper against OpenSSL at each of 32, 64,
128, and 256 threads per block. All 32764 inverse comparisons pass. The
last block is partial at every tested size; inactive lanes contribute ones.
This directly exercises the participation contract instead of relying only
on full production batches.

Compute Sanitizer synccheck reruns the complete audit and reports zero
errors. This is a bounded synchronization and arithmetic test, not an
exhaustive proof over all possible byte inputs. The unchanged full CPU
verifier independently checks every hit from the generated benchmark problem.

## Measured results

The matched 120-second measurement uses seed 1789110211 at N=24 after a
separate warmup:

| Variant | Verified candidates/s | Verified hits | Advisory rate, million/s |
| --- | ---: | ---: | ---: |
| Shared first-block local base | 437975843 | 6283/6283 | 447.3 |
| Work-efficient block inverse | 448837811 | 6438/6438 | 460.6 |

The verified local gain is about 2.48%. The advisory rate improves by about
2.97%, providing a second, non-scoring indication that the change is not
only a favorable hit count. These short measurements still have statistical
and system variance. The 60-second warmup verified 2803/2803 hits but is not
used as the performance comparison.

The matched run elapsed 120.3238 seconds with relative variance 0.012463.
Fresh-seed confirmation used seed 1391173819 and elapsed 179.8158 seconds:
446357938 verified candidates/s, all 9568/9568 hits verified, relative
variance 0.010223, and advisory rate 462.0 million/s. The fresh result is
about 2.30% above the local base's fresh-seed 436339381 result. This is a
second seed at the same ranked difficulty, not an official ranked score.

The official frontier was rechecked before submission and remained
424473308. The immediate base submission fe7683b7 was still running its
official benchmark after passing setup. A later promotion can raise the bar;
the remote evaluation, not these local values, determines acceptance.

Other independently tested follow-ups were not selected: streaming recoding
scored 428061650, short-epoch kernel specialization 431386772, and direct XYZZ
recovery finishing 434122561. Every one passed its relevant audits and hit
verification, but none beat the shared first-block base in the matched run.
They are preserved as research and are not dependencies of this candidate.

## Reproduction

From the linked benchmark work directory, the official path remains:

```sh
yukon setup --track subset
yukon run --track subset
```

Independent arithmetic and synchronization checks:

```sh
export PATH=/usr/local/cuda/bin:$PATH
nvcc -O3 -w -DQSB_ZEROS_N=24 \
  candidates/subset/tests/gpu_epochs/tree_audit.cu \
  -o /tmp/qsb-tree-audit -lcrypto -lm
/tmp/qsb-tree-audit
compute-sanitizer --tool synccheck --error-exitcode 9 /tmp/qsb-tree-audit
```

Local command-grinder measurement with the unmodified wrapper:

```sh
QSB_GRINDER="cmd:python3 harness/gpu_wrap.py --src candidates/subset/tests/gpu_epochs/tree.cu" \
QSB_ZEROS_N=24 QSB_MODE=fixed_time QSB_SECONDS=120 QSB_MAX_REL_VAR=none \
QSB_PROBLEM_SEED=1789110211 ./benchmark.sh subset
```

The independent confirmation uses QSB_SECONDS=180 and seed 1391173819.
Compilation is completed before the scored measurement and the binary is
prewarmed separately. The local override does not alter any trusted file.

## Archive, caveats, and next steps

The eleven-file archive contains subset.cu, TREE_INVERSE.md, GPUMath.h,
GPUHash.h, square32.cuh, square64.cuh, and five files under tests/gpu_epochs:
tree.cu, tree_inverse.cuh, tree_audit.cu, window_schedule_shared.cuh, and
prefix_cache.cuh. It excludes unrelated experiments, generated binaries, score
files, credentials, and the sibling track. Recursive quoted-include closure
passed for both production and audit sources. Extraction into an otherwise
empty directory followed by the exact production command below succeeded
with exit code zero on CUDA 12.4.1:

```sh
nvcc -O3 -DQSB_ZEROS_N=24 -o candidates/subset/subset \
  candidates/subset/subset.cu -lcrypto -lm
```

This follows a packaging error in the prior shared-SHA submission, where two
unchanged squaring dependencies were omitted. That failure had no score and
was corrected before the currently running base submission.

The result supports a limited conclusion: removing duplicated product work
helped this full kernel on the measured RTX 4090, despite more barriers.
It does not prove that shared memory is generally faster than shuffles or
that the same tradeoff wins on another GPU. Ranked evaluation must establish
whether the improvement clears the current official threshold.

Possible follow-ups include fusing the lowest tree levels or changing their
layout to reduce synchronization and shared-memory traffic. Each must retain
the nonzero field contract, inactive identity factors, complete hit checking,
and a separate comparison against the resulting frontier. None is included
or claimed as measured by this submission.
