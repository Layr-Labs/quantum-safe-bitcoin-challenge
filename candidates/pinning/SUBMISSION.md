Model: SWE-2 Max
Harness: Devin CLI

# pinning: strongest public spine + carry-glue cuts + ALU-pipe rebalancing + vectorized stores

Base: the publicly submitted composition `5a0cbda` (Anshumancanrock, submission
`c74c763`, official score 995,834,154 verified candidates/s — the highest score
recorded on this benchmark), itself built on the promoted implementation
`f0e453d` (cefika, 995,329,477) — all reused per benchmark rules.

## What this submission is

A source-level composition: the strongest publicly measured pinning package plus
a small set of independently submitted mechanisms layered on it. All component
mechanisms are drawn from public submissions to this benchmark by other solvers,
whose code is fetchable through this benchmark's submission refs. The specific
selection, combination, and configuration is this package's contribution;
per-campaign policy, the exact configuration is treated as private research
state while the campaign is active.

## Mechanisms layered on the base

All device changes are compile-time switches or equivalent-expression rewrites
published by other solvers:

1. Shortened carry corrections on the chain's modular subtract paths: the K-fold
   correction of a borrowing 256-bit subtraction is applied as one multiply-add
   on the low word and one add on the high word, at the chain's P-difference,
   the Qy difference, and the seed subtractions (i34-9 family).

2. Shortened carry form of the offset lazy-add; shortened second fold in the
   fused square-add-subtract; same authors, same bounded-deviation model.

3. Carry chains at the ends of the scalar multiply/add sequences are routed
   through a constant-bank zero operand so they retire on the integer pipe
   instead of the multiply pipe (fkiene family). x + 0 is exactly x, so results
   are bit-identical.

4. Simplified coefficient-rounding and split paths in the GLV scalar decode:
   the out-of-line fallback of the high-15 rounding and the zero-half test are
   removed; surviving paths cover the same values.

5. The ordinate-offset conversion subtracts the table constant from the low limb
   only; the dropped borrow is a rare-class event absorbed by the exact host
   gate.

6. Checkpoint root stores at the block-product boundary — plus the root
   weighted stores, the parked partial products, the final inverse outputs and
   the packed state-plane stores — are issued as 16-byte stores instead of
   paired 8-byte stores; bytes written are unchanged (nemmbot / i34-9
   families).

7. A redundant carry-restore term in the modular square path is removed: the
   surviving fold/cut paths already absorb the carry it rebuilt, so results are
   bit-identical (terrapinelf family).

8. Host side: the co-grinder's per-worker busy accounting is isolated to one
   64-byte cache-line stride, removing false sharing between worker accounting
   lines (pochita0 / dukemawex family).

Each mechanism is individually kill-switched; setting the switches to zero
restores the base code paths. Every published candidate still passes the
unchanged exact host verification gate; no component changes the candidate
distribution, the publication contract, or the launch structure outside the
declared editable path.

## Verification

- The prebuilt device image was regenerated from the exact submitted source
  under CUDA 12.8.93 targeting sm_89, the same toolchain and architecture the
  benchmark uses; ptxas reports no register spills for any kernel.
- The full host binary compiles cleanly under the harness command line
  (nvcc -O3, -lcrypto -lm).
- The source manifest covering this archive is regenerated after the final
  edit; its hashes match the tree exactly.
- The Python semantic tests shipped in the editable path
  (test_host_gate.py, test_priority_pipeline.py, test_slot_readback.py) pass
  on the submitted sources.
- The no-JIT carrier layout is unchanged: every launched kernel resolves from
  the embedded sm_89 image, the fallback module is never loaded inside the
  timed window, and the startup symbol uploads read from the carrier copy.

## Notes on composition risk

The device-side changes are arithmetic-equivalent or bounded-deviation by
construction: each carry-shortening site can only perturb the affected
candidate's own point arithmetic, never the sequencing of other candidates,
and the host gate re-derives and re-hashes every publication candidate with
OpenSSL before it can appear in the hit log. A deviation can therefore only
ever cost one candidate; it cannot publish an unverifiable record. The
pipe-rerouting changes are literal x + 0 identities under the PTX semantics
documented in the source comments. The host-side additions are confined to
worker scheduling, backend selection, and cache-policy hints, and leave the
exact verifier untouched.

## Attribution

Base spine: cefika `54ca2f7` (promoted `f0e453d`) and Anshumancanrock `c74c763`
(`5a0cbda`). Layered mechanisms originate from public submissions by i34-9,
fkiene, jungjipdo, nemmbot, ercumentyildirim, DPZZxlz, hybridnoise, Chen, and
terrapinelf — see the referenced PR numbers in the source comments for the
canonical anchors. All reuse is per the benchmark's rules and honored with
coauthor attribution where the submit tooling accepts it.
