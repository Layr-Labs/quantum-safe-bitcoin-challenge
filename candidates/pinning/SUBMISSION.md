# Pinning: six exact-or-host-gated refinements on the 32bc0c54 record

Author: [kaankolcu](https://github.com/kaankolcu).

## Starting point

This submission builds directly on the promoted record **32bc0c54** (official score
826,926,066; upstream commit `b5948434`, `candidates/pinning` tree `937c0b0e`). Everything in that
record is kept: the GLV recovery chain, the packed recovery tables, the slot readback, the priority
pipeline and the host-side exact re-verification of every hit. Nothing outside `candidates/pinning/`
is touched and the judge is unchanged.

The changes below were each built as a compile-time switch whose "off" position reproduces the
previous tree's code, gated statically (register count, spills, instruction mix of the hot loops),
checked for correctness, and then timed against the tree it landed on in the same rental before
being adopted. They are stacked in the order listed; each step was timed on top of the previous one.

## Changes

### 1. SHA-256 pipe balance (`QSB_SHA_ALU_ADD`, `QSB_SHA_FMA_EARLY`)

Stage 0's SHA-256d additions are forced onto the integer ALU pipe, while the pubkey hash in the
finish kernel keeps its IMAD-pipe adds (the finish kernel is already ALU-bound; moving those adds
cost time). Rounds 2-15 and the first message-schedule block of the pubkey hash use the same
FMA-add form as rounds 16-63. Only the association of mod-2^32 additions changes, so every value is
identical: exact.

### 2. Leaner GLV split and window decode (`QSB_GLV_LEAN`)

The scalar split's 32x32 products are written as explicit `mul.wide.u32` / `mad.wide.u32`, and the
high-part overflow count is taken from the add's carry flag. The C form `(uint64_t)a*b` computes the
same product, but ptxas lowered it through a 64x64 multiply and kept the zero high halves in a
uniform register. The signed-window decode produces the same fourteen codes as the reference
decode in fewer instructions (one `lop3` against the sign spread). Every value is identical: exact.

### 3. 64-byte L2 fills for table records (`QSB_TBL_L2_64B`)

Each 64-byte table record spans two 32-byte sectors, so an L2 miss used to fetch it as two sector
fills. Loading with the `.L2::64B` hint fills the whole record on the first miss. Same loads, same
answers; the saved DRAM traffic shows up as clock at the 4090's power cap. The hint needs
`sm_75` or newer, so it is compiled only when the target supports it and falls back to `__ldg`
otherwise. The ranked build passes no `-arch` (compute_52 PTX, JIT-compiled on the 4090), so in the
scored binary this change is inert; it only helps an `-arch=sm_89` build.

### 4. Paired codes and phi out of the loop (`QSB_CODE_PAIR`, `QSB_BETA_OUT`)

Each GLV term's code plane stores the record index and its decoded 32-bit sign mask as one
`uint64_t`, so the chain loop reads both with a single `LDS.64` instead of recovering the mask from
bit 31. The endomorphism phi (one multiply of X by beta) used to sit inside the rolled loop behind a
per-trip test and branch; now the loop runs to the component boundary, phi is applied once, and the
same loop code continues. Same operations in the same order: exact.

### 5. Host refill before the gate, overlapped sequences (`QSB_REFILL_BEFORE_GATE`, `QSB_OVERLAP_SEQUENCES`)

The host refills the next batch before waiting on the gate and overlaps consecutive sequences, so
the GPU does not idle between launches. Hit collection reads the same slots. Host-only; the kernel
is unchanged. Before this change the host waited for the gate, then prepared and launched the next batch,
so the GPU sat idle for the host's share of every batch boundary; with both switches on, that host
work runs while the GPU is still busy. Because the candidates, their order and the slots are the
same, the set of hits for a given amount of completed work is the same; only a batch still in flight
when the process is stopped is lost, exactly as with any timed stop.

### 6. Register-resident seed codes, lean offset correction, LEA in recovery (`QSB_GLV_SEED_Q=2`, `QSB_OFF_LEAN`, `QSB_REC_LEA`)

- `QSB_GLV_SEED_Q=2` keeps the GLV seed codes in registers. A candidate whose Q component is zero
  (probability about 2^-128) is skipped.
- `QSB_OFF_LEAN` computes the offset-anchor sum's correction in four SASS ops instead of nine by
  folding the `+kk` into the add chain's carry and stopping the correction at limb 0. Limb 1 would
  change with probability under 2^-32 per call (twelve calls per candidate, about 2^-29.4 per
  candidate). It is inexact in that rare case and is covered by the existing exact host
  re-verification of every hit, in the same class as the record's own first-fold carry drops.
- `QSB_REC_LEA` uses `LEA` for the paired code's record address in recovery. Exact.

## Correctness

- Every hit the kernel reports is re-verified exactly on the host before it is counted, as in the
  record; the harness then verifies them again independently on the CPU.
- Each change was checked on a GPU with CPU-verified hits: identical hit sets between the "off" and
  "on" arms over equal work, and on the 4090 timing boxes the same 13,354 hits in every block.
- The SHA changes were also checked against a CPU oracle (3M messages under four switch settings
  against a reference SHA-256, with a deliberate mutant caught).
- The inexact shortcut (`QSB_OFF_LEAN`) cannot produce a false hit: a wrong candidate fails the
  host's exact check and is dropped; its only cost is a lost hit at about 2^-29.4 per candidate.
- `QSB_HOST_GATE` stays on, as in the record: the host recomputes every reported candidate with
  exact arithmetic, so no GPU shortcut, carry drop or scheduling change can turn into a reported
  false hit. The only failure mode any of these changes can have is a missed hit.
- Every switch's "off" position compiles to SASS byte-identical to the previous tree, so each step
  is an isolated, reviewable delta: turning one switch off measures and checks exactly that step.
- The tree compiles with the grader's own line, which passes no `-arch`, so the scored binary is
  compute_52 PTX that the driver JIT-compiles for the 4090. No step relies on an instruction that
  compute_52 lacks; the one that did (step 3) is guarded and falls back to the plain load.

## Measurements

How the steps were chosen: each switch was timed against the tree before it on one RTX 4090, P C C P
quartets in a single rental, at least 20 minutes of timed blocks, identical hits in every block, and
kept only when its mean gain was positive in every round. Those step timings used `-arch=sm_89`
builds, so they are not quoted here; the ranked build targets compute_52 and is JIT-compiled on the
4090, where step 3 is inert.

## Reproducing

```sh
./setup.sh pinning      # builds candidates/pinning with the grader's line:
                        # nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm
./benchmark.sh pinning
```

Every switch above defaults to its adopted value; building with `-D<SWITCH>=0` restores the previous
code for that step.
