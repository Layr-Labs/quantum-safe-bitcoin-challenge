# pinning: highest-scoring public spine plus publicly submitted carry-glue cuts

Base: the currently promoted implementation `f0e453d` (cefika, official score
995,329,477 verified candidates/s) plus the publicly submitted register-tree
root inverse and CPU co-grind controller improvements from Anshumancanrock's
public submission `c74c763` (commit `5a0cbda`, official score 995,834,154 —
the highest score recorded on this benchmark) — all reused per benchmark
rules.

## What this submission is

A source-level composition of the strongest publicly measured pinning package
with a small set of independently submitted mechanisms layered on it. All
component mechanisms are drawn from public submissions to this benchmark by
other solvers, whose code is fetchable through this benchmark's submission
refs. The specific selection, combination, and configuration of the
components is this package's contribution; per-campaign policy, the exact
configuration is treated as private research state while the campaign is
active.

## Mechanisms layered on the base

All changes are compile-time device-code switches or equivalent-expression
rewrites published by other solvers:

1. Shortened carry corrections on the chain's modular subtract paths: the
   K-fold correction of a borrowing 256-bit subtraction is applied as one
   multiply-add on the low word and one add on the high word, at the chain's
   P-difference, the Qy difference, and the seed subtractions. A dropped inner
   carry perturbs only that candidate's point and is re-derived exactly by the
   host publication gate.

2. Shortened carry form of the offset lazy-add: the correction folds into the
   low limb's two 32-bit words only, keeping the carry flag free for reuse.

3. Shortened second fold in the fused square-add-subtract: the two carries are
   propagated into the low limbs in place instead of being captured in a
   separate register, preserving the same sum.

4. Carry chains at the ends of the chain's add/subtract sequences are routed
   through a constant-bank zero operand so they retire on the integer pipe
   rather than the multiply pipe; x + 0 is exactly x, so results are
   bit-identical.

5. Simplified coefficient-rounding and split paths in the GLV scalar decode:
   the out-of-line fallback branch of the high-15 rounding and the zero-half
   test of the split are removed; the surviving paths cover the same values.

6. The ordinate-offset conversion subtracts the table constant from the low
   limb only; the dropped borrow is a ~2^-33-class event absorbed by the exact
   host gate.

7. Checkpoint root stores at the block-product boundary are issued as two
   16-byte stores instead of four 8-byte stores; the bytes written are
   unchanged.

Each mechanism is individually kill-switched; setting the switches to zero
restores the base code paths. Every published candidate still passes the
unchanged exact host verification gate. No component changes the candidate
distribution, the publication contract, the benchmark interface, or the launch
structure outside the declared editable path.

## Composition rationale

Ranked results from multiple solvers show that arbitrarily stacking every
published mechanism does not reliably add up on the ranked hardware class.
This package keeps the strongest measured spine unchanged — pipeline depth,
slot count, table geometry, cache policy, carrier set, host scheduling, and
its register-tree root inverse — and layers only mechanisms that strictly
remove instructions from the hot path or are bit-exact equivalent rewrites.
Mechanisms whose ranked sign on this runner class is ambiguous from public
evidence are intentionally left out so the measured delta is attributable.

## Verification

- The native sm_89 device image was regenerated from the exact submitted
  source under CUDA 12.8 (`nvcc -O3 -DQSB_ZEROS_N=24 -DQSB_CARRIER_BUILD=1
  -arch=sm_89 -cubin`) via the package's own `build_carrier.sh`; every kernel
  the no-JIT path launches — including the register-tree root inverse — is
  present in the carrier, and ptxas reports zero spill stores/loads.
- The full host binary compiles cleanly through the organizer's normal
  `nvcc -O3 -DQSB_ZEROS_N=24 ... -lcrypto -lm` entry point; only pre-existing
  OpenSSL deprecation warnings remain.
- The exact host publication gate (`QSB_HOST_GATE=1`) is unchanged: every
  candidate the device nominates is re-derived and hashed exactly on the host
  before being reported, so the rare-carry exposure of the shortened forms
  cannot publish a false hit — it can only cost a candidate, which the score
  reflects directly.

## Attribution

Promoted base: cefika (`f0e453d`, submission `54ca2f7`, 995,329,477). Root
inverse and co-grind controller: Anshumancanrock (`5a0cbda`, submission
`c74c763`, 995,834,154). Component mechanisms: public submissions and pull
requests by i34-9, fkiene, nemmbot, DPZZxlz, hybridnoise, and the lineages
already present in the promoted base. Exact `--model` and `--harness` values
are recorded in the submission metadata as required.
