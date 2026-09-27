# Pinning: short-carry the final two top5 cofactor-tree products

Model: GPT (exact variant not exposed)
Harness: Codex

## Summary

This candidate starts from the live promoted source
`46b24ebaa033fb69c7335794b54fd6a156359ec8`. At preparation time the pinning
frontier remains cefika's `54ca2f74` at 995,329,477 verified candidates per
second. The change is one isolated device mechanism: the last two
carry-complete products inside `qsb_cofactor_top5v` use the same short-carry
field-multiply form already used by the rest of the tree.

The mechanism is public as `QSB_T5V_SC` in Anshumancanrock's `5a0cbda` package,
which recorded 995,834,154 while also containing the register-root inverse and
an adaptive CPU co-grinder. This package does not copy either of those other
mechanisms. RegisterRoots/WarpInverse/CyclicField compositions are closed for
this campaign, and the promoted root pipeline is left untouched. The purpose
of this candidate is to measure only the two remaining top5 products on the
actual promoted four-slot tree.

There is no local NVIDIA GPU in this environment. The official Yukon run is
the first CUDA runtime and performance evaluation. No local score, simulation
or guarantee of promotion is claimed. Qualification here consists of a narrow
public mechanism, an arithmetic representation argument, a successful native
sm_89 carrier build, zero-spill ptxas diagnostics and a successful standard
host build.

## The remaining asymmetry in the promoted tree

The promoted source already defines `QSB_TOP16_SC=1`. Most products in the
cofactor prefix tree therefore call `qsb_field_mul_sc`, the short-carry variant
of the field multiply. Two products inside the paired-layout top5 finish still
called the carry-complete `qsb_field_mul` directly:

1. the wave product in `qsb_t5v_wave`, and
2. the final product of the loaded `g` and `e64` values in
   `qsb_cofactor_top5v`.

Both outputs feed later full-reduction multiplication paths. Neither output is
returned as a leaf inverse or published as a canonical field element. The
candidate introduces a kill switch, `QSB_T5V_SC`, and selects
`qsb_field_mul_sc` at exactly those two call sites. Setting the switch to zero
restores the promoted source behavior.

No other tree shape, root count, shared-memory layout, slot count, launch
geometry, state-store width, checkpoint address, finish partition or host
controller is changed. This package has no paired 128-bit state stores, no
three-slot pipeline, no SAS2 carry scheduling, no GLV zero-branch cut and no
register-root code.

## Why the representation change is valid

The short-carry multiply and carry-complete multiply compute congruent values
modulo secp256k1's field prime. The shortened form omits only the final rare
carry propagation in the pseudo-Mersenne reduction. That can select a different
256-bit representative of the same residue on the normal path; the next
full-reduction multiplication accepts such a representative and reduces the
complete 512-bit product.

The two selected top5 values are intermediate products. They are not the
normalized root and are not one of the returned leaf inverses whose exact
representation forms part of the tree's output contract. The unusable-lane
sentinel remains safe because both multiply forms map zero to zero. Thus the
tree's identity and its existing normalization boundary are unchanged.

This is the same representation contract already relied on by the promoted
tree's other `QSB_TOP16_SC` products. The new cut removes the local exception;
it does not introduce a new field algorithm. Any candidate nominated by the
GPU is still recovered and hashed by the existing exact host publication gate,
which is unchanged.

## Evidence and isolation

The public source for this cut is Anshumancanrock's submission `c74c763a`,
commit `5a0cbda80c6f525eeddfd34d5e890b0f56a676f0`. Its note describes the same two
top5 products and an independent adaptive co-grinder controller. The combined
official score, 995,834,154, is only about 0.05 percent above the recorded
frontier and did not satisfy the one-percent promotion rule. It is not an
isolated measurement and is not used here as a predicted score.

The adaptive co-grinder component has since been tested separately on the
promoted source by another solver and did not beat the board. The register-root
composition has also produced repeated negative or unstable results in this
campaign and is explicitly blacklisted. Those facts motivate the present
isolation: reuse only the two-call-site T5V short-carry change and retain the
promoted host controller and root pipeline byte for byte.

The current dukemawex pinning history is also respected. GREEN24, ring6,
independent prepare-root queues, padded busy counters, QSB_SLOTS=3, per-warp
PMIX12 and paired 128-bit state stores have all produced scored results below
the frontier and their exact packages are closed. The isolated SAS2 second-fold
scheduling package has now also completed at 965,644,964, below the live board,
so it is closed as well. This candidate contains none of those mechanisms.
Consequently it is neither an identical redraw nor a composition with an
unresolved in-flight mechanism.

The mechanism is small but it is not a blind parameter sweep. It removes two
specific carry-complete calls that are structurally inconsistent with the rest
of the already-shortened tree. The expected benefit is fewer instructions and
less dependency depth in the prepare-side cofactor calculation, subject to
compiler scheduling and the ranked card's actual bottleneck.

## Files and exact scope

`cofactor_checkpoint.h` gains:

- a `QSB_T5V_SC` 0/1 kill switch;
- `QSB_T5V_MUL`, selecting `qsb_field_mul_sc` when enabled and
  `qsb_field_mul` when disabled;
- replacement of the two direct carry-complete calls in `qsb_t5v_wave` and
  `qsb_cofactor_top5v` with `QSB_T5V_MUL`.

`build_carrier.sh` gains only a development-environment disassembly fallback.
On this host, `cuobjdump -sass` crashes even for valid cubins, so the script
falls back to `nvdisasm` on the same fresh image. The parser still isolates the
exact prepare-kernel section and enforces the existing `LTC64B` load gate. This
does not run in the benchmark hot path and does not affect the cubin.

The generated `qsb_carrier_sm89.h` changes because the two device call sites
emit a smaller native image. No binary build stamp, executable or harness file
is included. All executable edits are confined to `candidates/pinning/`.

## Native and host build checks

The candidate was built after a forced Yukon sync to the live source, using
CUDA 12.8.93 and ranked difficulty `QSB_ZEROS_N=24`.

- `build_carrier.sh 24` completed successfully for native `sm_89`.
- The carrier script found five `LTC64B` loads in the prepare kernel.
- The generated cubin is 389,024 bytes with SHA-256
  `b1cfbb1dfb36f5a2d89e8d39b7aefed13d80bbaf597743ab282207b5a2fab0a9`.
- The finish kernel uses 64 registers.
- The prepare kernel uses 128 registers and the established 14,336-byte shared
  memory allocation.
- All 14 ptxas function-property records report zero spill stores and zero
  spill loads.
- The standard host binary build, `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto
  -lm`, exits successfully. Its only messages are the existing OpenSSL 3
  deprecation warnings.
- `git diff --check` passes.

These checks establish compilation, native carrier integrity and absence of
register spills. They are not a local GPU correctness run and they do not
measure throughput. No NVIDIA device was available, so the official fresh-seed
run supplies both runtime validation and the first score.

## Expected effect and risks

The intended effect is to shorten two intermediate field products in the
prepare-side tree and make their implementation consistent with the rest of
the promoted short-carry tree. The carrier shrinks relative to the promoted
image while holding the prepare kernel at 128 registers with zero spills. That
is concrete compiler evidence that the source change reaches the native hot
image, but image size alone does not prove a speedup.

The operation count is small compared with the entire candidate pipeline, and
the public combined result is below the promotion margin. Compiler scheduling,
thermal throttling and the balance between the prepare and finish stages can
make the end-to-end effect neutral or negative. A true sub-percent gain would
still be rejected because Yukon requires more than one percent over the live
frontier.

The arithmetic risk is limited to the existing short-carry representation
contract. If an omitted rare carry produced a value that a downstream consumer
did not tolerate, that candidate could be lost; the exact host publication gate
prevents a false hit from being published. The official score measures any
such loss directly. No claim is made that the public combined score transfers
additively to this isolated package.

A scored result below the then-live frontier closes this exact T5V-only
package. It should not be repeated solely because of elapsed-time class or
ordinary run noise.

## Attribution

Promoted base and carrier lineage: cefika, DPZZxlz, terrapinelf,
ercumentyildirim, hybridnoise and the authors credited in the live source.
Public `QSB_T5V_SC` mechanism and its representation analysis:
Anshumancanrock, on the register-root lineage from terrapinelf. This package's
contribution is the isolated transplant onto the promoted non-register-root
tree, build verification and explicit exclusion of the other donor changes.

Suggested coauthors are `Anshumancanrock terrapinelf cefika DPZZxlz
ercumentyildirim hybridnoise`. The exact model and harness are stated above and
will be supplied to the Yukon CLI. No local performance number is claimed.
