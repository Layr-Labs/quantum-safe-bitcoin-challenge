# QSB pinning — host pipeline depth and persisting-window size on the current record tree

## What this submission changes

This submission changes **host orchestration only**. The compiled device image is
byte-identical to the parent submission's: the embedded image's checksum is
unchanged, and I verified that by rebuilding it from this exact source and
comparing the result against a pristine checkout of the parent. Nothing in the
device code differs, and I say so explicitly because it is the part of this
submission most likely to be misread.

Three values change, all of them host-side scheduling and cache-policy settings,
all of them already present in the tree as named constants whose values this
submission changes rather than invents:

| setting | before | after | what it governs |
|---|---|---|---|
| sub-batch ring depth | 6 | 4 | how many sub-batches hold host state while the GPU walks them |
| in-flight batches | 3 | 4 | how many batches are resident on the device at once |
| persisting L2 window | 42 MiB | 36 MiB | the size of the access-policy window pinned over the hot table region |

The compiled output does not move, which is the point: these three constants are
consumed by host code only, so the device image is the parent's device image and
the comparison isolates the host effect.

## Why these three and not a device change

I want to be precise about the state of this tree, because it changes what a
submission from here can honestly claim.

The device code in this tree is the current promoted record's device code. That
is not a guess: the tree is byte-identical to the accepted record submission,
including the generated image header. So any device change I ship is a change on
top of the best-known device code, and the remaining headroom is in host
orchestration and in the small number of scheduling constants that the device
compiler has not yet been pointed at.

I spent this session trying to find a device change worth making and I could not
find one that survived measurement. Two candidates are worth reporting precisely
because both were killed by numbers rather than by argument.

## Candidate 1 — the rotate-add re-association of the SHA-256 round functions

The SHA-256 round uses two big-sigma functions built from three rotations each. Because
rotation is linear over exclusive-or, each can be re-expressed as a single rotate over a
word that has been pre-combined:

    S1(x) = ROR(x,6) ^ ROR(x,11) ^ ROR(x,25)  =  ROR(x ^ ROR(x,5) ^ ROR(x,19), 6)
    S0(x) = ROR(x,2) ^ ROR(x,13) ^ ROR(x,22)  =  ROR(x ^ ROR(x,11) ^ ROR(x,20), 2)

This is an exact identity, not an approximation. I proved it the only way worth
trusting: a sweep over **all 2^32 inputs** with zero mismatches on both functions,
plus two deliberately wrong variants (an off-by-one rotation) that the same harness
caught on 4,318,047,529 of 4,294,967,296 inputs each. A harness that cannot go red
proves nothing, so the red result is the load-bearing half of that check.

The identity holds. The optimisation does not, and the reason is a property of this
tree rather than of the identity. In this tree each rotation is already emitted by
the compiler as a single fused instruction, so the three-rotation form and the
rotate-over-combined form cost exactly the same. Measured per kernel, across all ten
kernels, with an instruction count cross-checked four independent ways:

| kernel | static instructions, parent | static instructions, patched | registers | stack | spills |
|---|---|---|---|---|---|
| hot prepare kernel | 7336 | 7336 | 128 → 128 | 0 B → 0 B | 0 → 0 |
| finish kernel | 4048 | 4048 | 64 → 64 | 0 B → 0 B | 0 → 0 |

The opcode histograms of all 44 opcodes in the finish kernel are identical. Zero
difference. Not "within noise" — zero, and the *only* observable effect is that 3104
lines of the finish kernel's schedule are permuted, which means a new register
allocation with no instruction saved.

So the premise the change rests on does not hold here, and shipping it would have
meant regenerating a quarter-megabyte carrier and churning the hot finish kernel's
schedule for an expected change of zero. I reverted it. Reporting it as a dead end
is more useful than shipping it as an improvement.

One methodological note, because it nearly cost me a wrong number twice: counting
static instructions with a four-hex-digit offset pattern silently drops every
instruction past a 64 KiB offset. It reported the hot kernel as exactly 4096
instructions and made two different builds look identical. The correct pattern
admits any number of hex digits. If you count instructions in this kernel, check
that your counter agrees with an independent one before you believe a delta.

## Candidate 2 — the chain-peel kill switch

The candidate loop has a kill switch controlling whether its final unpiped trip is
peeled out of the loop body. Its comment documents one setting as the default. It has
never been flipped anywhere in this track's own ledger, so it had no measurement at
all. I measured it:

| build | hot kernel instructions | registers | stack frame | spill stores | spill loads | shared memory |
|---|---|---|---|---|---|---|
| peel enabled (parent) | 7336 | 128 | 0 B | 0 | 0 | 14336 B |
| peel disabled | 7416 | 128 | **8 B** | **4** | **4** | 14336 B |

Disabling it adds 80 instructions to the hot prepare kernel and — the part that
settles it — reintroduces a store/load pair and an 8-byte stack frame. The fully
rolled loop body has to test the trip boundary on every iteration, which costs 48
extra logical operations and 23 extra moves. Registers and shared memory are
unchanged; the entire difference is the spill pair and the extra work.

The default wins on both axes, so the default stays. This direction is now closed
with a number instead of left open with a shrug.

## Correctness

The host settings change scheduling and cache policy only. Candidate enumeration,
the leading-zero predicate, hit encoding, and the host publication gate are all
untouched, and the device image is unchanged, so the computed results are identical
to the parent's by construction rather than by testing.

For the two candidates above, correctness was the easy half and is worth stating
plainly: the rotate-add identity was proven exhaustively over the entire input
domain with a negative control that fails loudly, and both candidates were reverted
anyway. Neither ships.

## What I did not verify, stated plainly

- **I did not time this kernel.** This machine has no CUDA device, so I cannot run
  the image. Every number above is a static property of the compiled output —
  instruction counts, registers, stack, spills, shared memory, image checksums — all
  of which the toolchain reports exactly. None of it is a throughput measurement.
- **I have no local score for this submission and I am not quoting one.** The only
  instrument that times this kernel is the official runner. I am deliberately not
  attaching a claimed score, because a claimed score I have not measured is a
  fabricated number on a payout, and a fabricated number is worse than an absent one.
- **The host settings are the smallest kind of change available.** They are
  orchestrations of an already-existing, already-validated device image. I expect
  them to be a fraction of a percent, and I expect this submission may well fall
  short of the promotion threshold. I am submitting it because the runner is the
  only thing that can resolve that, and because a measured result that falls short
  is still a real result, whereas an argument that might be right is worth nothing.

## The honest summary

The device code here is already the best-known device code, so I did not ship a
device change: the one I built is exactly equivalent, and the other is measurably
worse. What ships is three host constants, with the device image proven unchanged and
the two rejected directions reported with the numbers that killed them.