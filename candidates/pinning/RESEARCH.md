# Pinning: 32-lane super-root inverse groups

## Candidate delta

This candidate is based on accepted commit
`92f27aef95e0df4496537b576a30be793fddfcd0`, submission
`f7412e96-b0aa-4804-a108-54f49ede6a96`, whose recorded official score is
741,053,306. It changes only the final super-root inversion grouping in
`pinning.cu`. The other seven production or license files are byte-identical to
the accepted commit.

The existing block inverse helper is generalized from a fixed width of 256 to
a power-of-two template. Only its `N=32` specialization is used by the
super-root inverse kernel. That kernel changes from one 256-lane CTA per 256
group roots to one 32-lane CTA per 32 group roots. The host launch changes from
`ceil(root_groups/256) x 256` to `ceil(root_groups/32) x 32`. Identity padding,
the active-lane predicate, all full-block barriers, and the global write guard
remain in place.

For the ranked full batch, the accepted geometry produces 256 group roots.
The accepted kernel processes them with one CTA and one field inversion. This
candidate processes them with eight CTAs and eight field inversions. The intent
is to expose more independent work at the final inverse stage. This is a
measurement candidate, not a claim that the extra parallelism outweighs seven
additional inversions.

## Source and arithmetic scope

The packed product tree uses leaf indices `0..N-1`, then contiguous levels,
ending at root index `2*N-2`. Its inverse array stores the `N-1` internal-node
inverses. For `N=32`, the leaf, internal, and root ranges are therefore bounded
by the allocated `2*N` product slots and `N` inverse slots. The downward pass
retains the accepted identity
`I=1/(L*R) => I*R=1/L` and `I*L=1/R`; only constants are expressed in terms of
`N`. Every lane reaches every barrier. Inactive tail lanes contribute one and
do not write global output.

The candidate retains the accepted `QSB_REMEASURE_TAG_09190538` no-op marker
and the accepted generic host path. The stage was restored against the accepted
source by replacing only the reviewed helper, nearby explanatory comments,
the super-root inverse kernel, and its host launch expression. Replacing those
candidate spans with the accepted spans reproduces the complete accepted
`pinning.cu` byte-for-byte.

## Verification evidence

The exact organizer build command succeeded with CUDA 12.8.93:

`nvcc -O3 -DQSB_ZEROS_N=24 -o pinning candidates/pinning/pinning.cu -lcrypto -lm`

PTX generation with the same flags and offline `ptxas -arch=sm_89 -v` also
succeeded. The retained analyzer compares this build with a separately built,
hash-bound accepted-92f baseline. It requires exactly eight PTX entries, the
super-root inverse entry as the sole changed body, and byte identity for the
other seven entry bodies. It also checks the complete parsed sm_89 resource
map rather than assuming that the requested geometry is the only compiler
effect.

The source-bound CPU traversal fixture imports the previously reviewed test
runner and verifies its hash before use. It extracts the actual accepted and
candidate helper/kernel bodies, compiles them with ASan and UBSan, and exercises
12 counts: 0, 1, 2, 31, 32, 33, 127, 128, 255, 256, 257, and 513. Accepted and
candidate traversals produce the same transcript. The candidate trace has ten
full-block barrier phases, 93 modeled field multiplications, and one modeled
inversion per CTA; the accepted trace has sixteen phases, 765 multiplications,
and one inversion. Sentinels, inactive tails, identity padding, canonical
outputs, multi-CTA indexing, and multiplicative invariants are checked.

Negative controls alter sibling selection, padding, global stride, launch
width, helper width, barrier placement, and an unrelated batch constant. The
fixture or source-scope guards reject them. All seven non-`pinning.cu`
production/license files are independently compared with the accepted commit.

## Limits

The traversal fixture models field multiplication and inversion with a
canonical big-integer secp256k1 prime-field implementation. It does not execute
the inherited raw CUDA field assembly. The count-zero fixture case is a
zero-iteration host model; it is not native proof of a zero-grid CUDA launch.
The PTX and offline sm_89 resource checks are static compiler evidence. No
native GPU kernel, driver JIT, SASS, latency, throughput, ranked score, or
promotion is measured by this package. The package therefore makes no speedup
or optimality claim.

The eight production/license files total 280,703 bytes. Their hashes and the
exact baseline provenance are recorded in `SOURCE-MANIFEST.json`. The source
archive contains only those eight files plus this note and the manifest.
