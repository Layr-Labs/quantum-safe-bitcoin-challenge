# Subset: one candidate per thread with a spill-free 96-register point chain

Prepared with GPT 6 Astra, xhigh effort, using Codex. This is a new execution
layout on hybridnoise's public PR #1137, commit
`d5d2283e47fdb0108ff36b3e6acc5d16c115d8ad`. That source scored 624,752,385
verified candidates/s. The promoted subset frontier at preparation is
623,518,629, with a 100-bip promotion threshold of 629,753,816. No measured
speedup or claimed score is supplied for this candidate. The authoring machine
has no NVIDIA GPU; the official RTX 4090 evaluation is its first device run.

## Mechanism and selection

The parent processes two candidates per thread in 256-thread blocks. Its digest
uses 128 registers and 49,152 bytes of shared memory, permitting two blocks and
16 warps per Ada SM. This candidate processes one candidate per thread in
192-thread blocks. The final digest uses 96 registers, 32,256 shared bytes,
zero stack, and zero spill stores or loads. CUDA 12.8's standalone occupancy
calculator, supplied with Ada SM89 resource limits and the compiler's resource
counts, reports three blocks and 18 warps per SM. This is a resource calculation,
not an observed scheduling result, and 18 versus 16 warps does not establish a
12.5 percent speedup.

The layout is made feasible by controlling register lifetimes. A plain
single-candidate port spills heavily at 96 registers. The implemented changes
are all parts of that register reduction:

1. The existing scalar setup produces the same signed odd recoding. All fifteen
   digits are calculated before the point chain and packed into shared columns.
   The chain reads one packed digit per iteration. Seven rolling scalar words
   and the sign no longer remain live throughout the chain. The packing keeps
   an 18-bit index field and a separate sign bit, including the larger first
   table segment and the positive-remainder final digit.
2. The digit columns reuse the 16 KiB product-tree arena. The sixteenth plane
   holds each thread's active flag. Before any thread initializes products,
   a block barrier ensures all threads have finished reading their digits and
   active flags. The digit and product layouts differ, so this extra barrier
   is required even though each digit column has one owner.
3. Inside the existing point-add PTX, four ZZZ limbs are stored after their
   first use and reloaded immediately before their next use. One ZZ limb is
   likewise parked between its first use and the multiply that updates it.
   The scratch array has five planes of 192 uint64 values, or 7,680 bytes.
   Each thread owns one column. Ten added load/store instructions preserve
   every arithmetic instruction and every bit of the saved values.
4. The final point addition parks its affine Y in the same thread's already
   consumed digit columns, then reloads it for the final Y resolution. It uses
   separate 32-bit planes so it cannot overwrite another thread's digits while
   that thread is still running its loop. The active-flag plane is disjoint.
5. Front and tail wrappers are inlined for the single-candidate route. The
   published hit index is recomputed from the block and thread IDs when needed.
   There is no per-thread local-memory storage for a second candidate.

The final body retains the parent's point formulas, field operations, table,
hash functions, and exact host publication gate. It introduces no further
carry truncation. The parent already uses approximate speculative field
arithmetic before exact host verification; a different inversion grouping can
therefore change which tentative hits survive that approximate filter. CPU
algebra checks do not prove identical CUDA hit sets. Any missed hit reduces the
official verified score, and the unchanged host gate must validate every
published result.

## Mapping, inversion and publication

The linear index is `block * blockDim + thread`. Epoch and window are
`index / 128` and `index % 128`. A 192-thread block can cross an epoch boundary;
each thread independently selects the correct epoch descriptor and first-state
cache. The first-state builder and the underlying omission family are inherited.
Inactive tail threads use epoch zero for safe loads and the identity denominator
for the collective inverse; they cannot publish a tentative hit.

The level-packed inverse tree retains 256 leaves. Threads 0 through 63 initialize
the 64 virtual leaves with multiplicative identities before the inherited first
tree barrier. All internal levels have at most 128 writers, so the 192 physical
threads cover every operation. Every physical thread participates in the block
barriers. The existing isomorphic root scale and recovery tail are retained.

The host computes the number of blocks by rounding `epochs * 128` up to 192
threads. Epoch buffers are allocated for the integer capacity selected by the
host, and the final partial block is masked. Accounting advances by the actual
number of epochs times 128, not by the padded thread count. The tag remains
`epoch * 128 + lane`, with the recovery ID in the inherited high bits. Six early
omissions and three window omissions retain their original record positions.
The atomic count, 1,024-record capacity, exact OpenSSL gate and final pending-hit
drain are inherited. The candidate family is not reduced or repeated.

## Compiler evidence and controls

All final builds use CUDA 12.8.93. The organizer-style full executable command is:

```
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v -o subset subset.cu -lcrypto -lm
```

The default-target executable compiles with a 96-register, 32,256-byte-shared
digest and zero stack/spills. Default-target PTX reassembled with
`ptxas -arch=sm_89 -v` and a native `-arch=sm_89 -cubin` build have the same
resource counts. Every emitted function's spill report was inspected, including
called functions; inspecting only the entry kernel had concealed spills in an
earlier discarded prototype. The final builds have no reported spills in any
function. Build artifacts are excluded from this source archive.

`-DQSB_SINGLE_EPOCH=0` restores the PR #1137 execution path. Its complete parsed
SM89 instruction listings are identical to the measured-source control under
the same default-PTX-to-SM89 build route. The enabled configuration selects
`QSB_SINGLE_THREADS=192`, `QSB_SINGLE_MAXREG=96`, `QSB_SINGLE_INLINE=2`,
`QSB_SINGLE_DIGITS=1`, and `QSB_SINGLE_POINT_PARK=2`. Disabling the root switch
also disables all new point storage and final-Y storage. Experimental component
controls exist for diagnosis; unsupported geometry combinations fail compilation.

The new shape costs more inversions per candidate and adds shared-memory traffic.
It also replaces the parent's paired SHA scheduling with its existing single
SHA helper. These costs may outweigh extra residency. Resource counts and
instruction listings cannot resolve that tradeoff, GPU clock behavior, or
full-duration verified throughput. This package is one selected measurement,
not an unchanged replay or an upload queue of register settings.

## Focused CPU validation

Run `python3 -B test_single_layout.py` from the candidate directory. It places
temporary compiler files outside the archive and performs these checks:

- The literal source recoder and new digit-staging loop are compiled with
  undefined-behavior checks. 10,772 boundary/random scalars and 161,580 packed
  digits match an independent unbounded-integer signed recurrence. The weighted
  digits reconstruct twice the input scalar modulo the secp256k1 group order.
- 12,730,752 candidate mappings cover 1 through 257 epochs with 128-, 192-, and
  256-thread blocks. Every active candidate appears once, padded threads are
  excluded, and both recovery-ID tags round-trip through the host decoder.
- A level-packed product-tree model checks 4,608 lane inverses against Python's
  independent modular inverse, with identity tails, virtual lanes and a random
  nontrivial root scale. This checks the padding and index algebra, not CUDA
  barrier execution or the inherited approximate PTX multiplication.
- The literal preprocessed point PTX is compared with parking disabled. For
  both parking modes, removing only the balanced shared stores and loads leaves
  all arithmetic instructions identical. Each saved register is unused between
  its store and reload, and every scratch address belongs to one thread.

These checks do not execute CUDA, measure performance, or replace an end-to-end
fresh-problem hit comparison. The official verifier and measured score remain
the authority. The package preflight inventories all on-disk files, validates
source hashes and note size, and checks the track scope and unchanged host gate.

## Prior work and attribution

The substantial baseline is hybridnoise's PR #1137 and its inherited lineage,
including terrapinelf, Akashneelesh, dun999, Meganpark980320, ercumentyildirim,
EvanYan1024, DPZZxlz, fkiene, jacklightChen, owizdom, DrCleverHans, mitchuski,
Babbaragga, and Saviour1001 as credited in the retained source and notes. All
existing license notices remain.

Akashneelesh's public `6b24afc3` / `828448651875689bb70eaacebfb98c10346b19f3`
tested 192-thread blocks, shared rolling digits and local storage for the parked
candidate. It scored 540,264,097 and was rejected. Its note's simple register
total suggested three blocks at 112 registers, but the CUDA 12.8 occupancy
calculator reports only two for SM89. This candidate reaches 96 registers,
removes the second candidate, precomputes digits instead of repeatedly moving
seven rolling words, and has no local spills. The prior rejected result remains
evidence of risk; it is not being counted as a speedup or copied unchanged.
The new storage layout, register-lifetime changes, CPU checks and resource
analysis were implemented here. No donor's measured result is attributed to
this new source.
