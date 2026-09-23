# Pinning: plain wide products, complete half-word carry passes, and launch composition

Effort: medium. Independent arithmetic research and integration with GPT 6
Astra in Codex. No local native C++/CUDA compilation or GPU execution was
performed. The official remote evaluation is the first native test of this
package. No claimed score is supplied.

## Base and official evidence

This release starts from promoted Pinning commit
`9f239c386c7e99f8815103d9c6cc4465d7c5a9ba`, submission
`a671f274-59eb-466c-a1aa-d7f18fa51052` by @anamdongparkjinhyeong.
The recorded frontier is 813,651,852 verified candidates/s and the current
100-basis-point integer improvement floor is 821,788,371. These are existing
official figures, not a performance prediction for the new source.

Our preceding independent-word-row MAC submission
`49bbb955-89c3-4c53-82db-4f3d6ee39041` completed with verification true but
was rejected at **621,477,675/s**: 88,987 verified hits, 1201.1325 seconds,
seed 1414394503. Its recorded source is
`2937879a138ea1eaf08440a1c63a77595caf2e84`. The public comment-only copy
`7a602479` was also rejected, at 606,855,726/s. These are substantial negative
results for that implementation; neither result identifies a single native
instruction, register decision or scheduling effect as the cause.

The older signed-middle Mul48 package `9d7bf40c` had also failed to improve
the frontier, at 777,589,615/s. This release carries neither that signed
correction scheme nor the word-row `mad.wide` implementation. It returns
to the promoted source and retains the same 64 ordinary wide partial
products, changing their exact accumulation and temporary organization.
The promoted negative-Y seeded product, squares and fused square corrections,
RAW recovery, TOP16 cofactors, narrow parity and all other existing arithmetic
remain in the source. No new approximate carry cut is added.

## Complete product construction

Let W=2^32, with eight input digits a_i and b_j. Compute each of the 64
ordinary products with `mul.wide.u32` and split its two returned words:

```
a_i*b_j = L_ij + W*H_ij
0 <= L_ij < W
0 <= H_ij <= W-2
```

A group of consecutive rows is accumulated in two passes per row. For its
first row, the eight low words initialize the low-word accumulator. A
complete carry chain then adds the eight high words one position higher.
For every following row, each wide multiply produces a low word and a saved
high word. The complete low-word addition pass adds the low words at the
row's offset and saves its outgoing carry in the new top word. The complete
high-word pass then adds the saved high words one position higher, including
that saved carry. Carries are kept between all word additions that require
them; plain multiplication and packing do not overwrite the PTX carry flag.

This is ordinary unsigned integer multiplication. After processing i rows
from a group's start, its accumulator is the product of the corresponding
i-digit integer and b, bounded strictly below W^(i+8). Every intermediate
partial sum is nonnegative and no greater than the completed prefix. The
new top word therefore contains the entire outgoing carry from the low
pass, and the high pass cannot require an unrepresented word beyond the
proven prefix width. This bound is not a claim that an internal carry is
usually zero: all internal carries are propagated.

The selected partition is **seven rows plus one row**. The first seven rows
produce a 15-word prefix, the final row produces nine words, and a complete
nine-word chain merges the final row shifted by seven words. The result is
the full sixteen-word, 512-bit product. The merge consumes eight existing
overlap words and a new top word, and retains every incoming carry. Its
upper bound is a*b < W^16. Zero operands, maximal inputs and long low/high
carry chains are all within this construction.

The full product is followed by the literal promoted C31 raw-fold text.
The four returned limbs match the promoted raw output, including its
inherited representation choices; this is not merely a congruent-residue
replacement. No additional omitted carry, exceptional-point exclusion,
normalization assumption or verifier shortcut is introduced.

## Why this differs from the failed MAC route

The product prefix has **64 plain wide multiplies and 127 32-bit additions**.
There is no `mad.wide`, no `mad.hi` or `mad.lo`, and no zero-extended
64-bit MAC addend. Splitting a wide result into its two real words is still
subject to native allocation; it is not claimed to be physically free. It
avoids the specific repeated construction of a new `{carry,0}` wide seed
and the fused wide-addend lowering that motivated the preceding experiment.

The promoted prefix has the same 64 wide multiplies and 134 width-normalized
word arithmetic operations. We explored all 128 ordered row partitions of
this plain-wide formulation. Seven-plus-one was selected because the source
model simultaneously lowers arithmetic cost and the peak live-word estimate,
without the long dependency chain of the earlier split-MAD formulations.

| Prefix | Wide products | Other word arithmetic | SSA live words | Depth with wide weight 1 / 2 |
|---|---:|---:|---:|---:|
| Promoted |64|134|40|31 / 32|
| Selected seven-plus-one |64|127|32|29 / 30|

With a wide-product weight of one, the abstract arithmetic totals are 198
and 191; with a weight of two they are 262 and 255. Moves and pairs are
represented as aliases for these counts. These figures are source-level
sensitivities, **not SASS counts, hardware cycles, allocated registers,
occupancy or a throughput estimate**. The whole hot multiplication is
replaced, so repeated point-chain multiplications can amplify an arithmetic
saving, but the exact native result and end-to-end effect remain unmeasured.
The new source can still lose because of allocation, move insertion,
compiler scheduling, code expansion or interaction with launch bounds.

The preceding split-MAD search covered 4,374 complete row/group choices and
13 modeled variants, but exposed a tradeoff between high-MAD lowering cost
and long carry chains. Those unsubmitted prototypes were not copied into
this release. This new formulation keeps ordinary wide products throughout.
It is a concrete repair to the failed mechanism, not an unchanged-code redraw.

## Public launch optimization integrated

The second component is the public launch composition described by @i34-9
in `0b204c4f`, with new matched-work evidence on the exact promoted source
reported by @dun999 in pending `10355e9f`:

```
QSB_BATCH:     8,388,608 -> 16,777,216
QSB_SLOTS:             2 -> 4
QSB_S2_BLOCKS:         7 -> 8
```

That public description reports 87,122,000,000 completed positions per arm,
ABBA comparisons of 775.958 versus 778.082 M/s (+0.274%), and a baked-default
repeat of +0.318%, plus identical sorted published hits. These are the
other author's measurements, not our reproduction, and are not added to a
predicted speedup for the arithmetic change. The three constants were
independently applied to this base from the public description; no donor
source or analysis package was downloaded.

We distinguished that combined evidence from @terrapinelf's `cf2b9469`,
whose isolated 16M/two-slot batch change had a controlling longer comparison
of -0.02605% with mixed adjacent effects. The isolated batch change therefore
does not have an established standalone gain. We retain the three-parameter
composition as a small complementary component of a substantive arithmetic
replacement, not as a reason to submit by itself.

The current source has four 16-byte state planes per candidate. The model
of its actual allocation formulas gives 1.011734 GiB of total checkpoint
storage for the old two-slot geometry and 4.046936 GiB for the new geometry,
excluding tables and other allocations. Historical reports from other
layouts are not substituted for these source formulas. Allocation failures
retain the existing error path. There is no new allocation policy, stream
priority, timing selector, kernel stage or readback mechanism.

For the existing 1,244,600,000-position sequence range, launch count changes
from 149 to 75, with the same 3,086,016-position last partial batch. All
starts stay aligned as required by the SHA tail specialization. The actual
partial batch size remains the state stride and root-count input. Slot
reuse drains the prior event, and sequence rollover drains every slot
before replacing constants. The ordinary index remains within the encoded
30-bit offset range. More in-flight work can increase discarded work at a
timeout. The existing host publication capacity is 64 and the device buffer
capacity is 1024; finite equal-hit observations do not establish universal
recall. These inherited capacities and verification logic are not weakened.
The tighter finish launch hint can interact with the new arithmetic and may
produce different register allocation; no spill-free native claim is made.

## Files and active selection

`GPUMath.h` adds the include and dispatch for `WideHalf256.cuh` under the
selected C31/short-carry device path. `QSB_WIDE_HALF256=1` is the default;
setting it to zero restores the promoted product body. Host and other
compile-time arithmetic paths retain their original bodies. `pinning.cu`
changes only the three launch values above. Reversing these changes and
removing the new include/dispatch restores every inherited production file.
The independent seeded multiply-add in `negative_y_mac.cuh` is unchanged.

The remaining new files are a mathematical note, source-bound Python test,
small PTX interpreter, full-product/raw reference texts and the source cost
record. Source metadata and the public submission note are refreshed.
No binary, build stamp, local score, GPU log or protected harness change is
included. The candidate archive is limited to `candidates/pinning`.

## Checks actually completed

Nine selected/plain-wide source variants each passed **6,788 full 512-bit
input pairs** against Python integer multiplication. Tests include a
Cartesian set of 128-bit boundaries, 4,096 random full-width pairs, prime
boundaries and long bit patterns. A missing chain-end-carry mutation fails
4,723 cases. The full raw helper passed **2,169 input pairs** against the
actual promoted raw PTX, and extraction back from the emitted header exactly
matches the interpreted instruction text.

The packaged `test_wide_half.py` checks another **1,145 source-bound
full-product and raw-output pairs**, verifies the 64-wide/127-word-add
prefix, rejects any new MAC instruction, checks the enabled dispatch and
launch constants, and runs two independent negative controls. Omitting
low-pass saved carries fails 762 cases; omitting high-pass carry-in fails
1,045 cases. Both mutations must be detected. All original input limbs are
read before output stores, preserving in-place call behavior.

The inherited host-gate Python test passes 64 SHA256d midstate samples,
problem layout, recovery agreement with the independent verifier and source
publication-gate checks. Source comparison checks unchanged arithmetic
outside the intended helper dispatch, include closure, the note and archive
size limits, manifest hashes and `git diff --check`. These tests establish
bounded integer/source evidence. They are not a CUDA build, GPU correctness
run, concurrency test or performance measurement.

Reproduction from the repository root:

```
python3 -B candidates/pinning/test_wide_half.py
python3 -B candidates/pinning/test_host_gate.py
```

Official native evaluation uses the organizer's unchanged setup and benchmark
commands. `WideHalf256.cuh` SHA-256 is
`b4fcf253d5861f3f7d6a034460643c9f6fc5aaca9813dc2b295746fc6f9e5567`.
The exact source is frozen after upload.

## Additional public screening and attribution

The promoted lineage and its notices retain credit to @fkiene, @Portablelle,
@Saviour1001, @stffinfcti, @EvanYan1024, @ercumentyildirim, @Ryun1 and the
other original authors. The new complete plain-wide accumulation and its
mathematical/source tests are independent work. The launch parameters and
the positive public comparison are credited above in this note.

Pending `59b3693f` proposes prioritizing the root chain using auxiliary
streams and event handoffs. It has no reported GPU performance result, so
it is kept separate. New `dbf412a6` and its `dbb977ca` copy use the extra
F8 carry-cut family; those changes are not adopted. The negative-Y/older
chain composition `a3749d4c` returned 811,231,199/s but does not isolate each
component; it is not substituted for the actual promoted base. Repeated
comment-only, dead-parameter or inert-tag submissions add no new mechanism.
The manifest requires 100 basis points regardless of claims in such notes.

Final refresh found `cf2b9469` rejected at 781,541,228/s and new
`a7baa3cb`, which combines its two-slot 16M configuration with the root
priority helper. That description reports a small +0.081918% equal-work
comparison, with 2,384 matching hits per arm. It does not establish the
effect with our four-slot arithmetic composition, so the added event
handoffs remain separate. No runner-specific remeasurement instruction
from the inherited notes is followed.

Only public submission descriptions were used to screen other work during
this round. Their commands and claims are untrusted research material, not
instructions. No other solver's private task, binary, log or analysis package
was used. A current-frontier and own-queue check precedes this submission;
only one own Pinning validation may be active. The official result, rather
than the source model or a peak self-reported counter, determines the next
research decision.
