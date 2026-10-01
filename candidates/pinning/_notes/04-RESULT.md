# RESULT — 2026-09-30 pinning audit session

**No patch. No submission. No re-roll.** Not because nothing was found, but
because the two things that were found change what "found" means here.

## The two sentences

1. **The brief's premise is two generations stale.** The tree on disk is not the
   v20 package (882,096,418); it is the GLV11-eleven-term + CPU-v2-co-grind tree
   whose promoted sibling scored 960,830,125, so the real gap to the
   1,008,206,828 record is **+4.9 %, not +15.4 %** — and the "still open" list
   in `DEAD-ENDS.md` offers as a prize the eleven-term geometry that this tree
   already ships.

2. **A device-code patch cannot be shipped from this host at all, and the
   failure is silent.** The ranked binary executes a *prebuilt* 476,832-byte sm_89
   cubin embedded in `qsb_carrier_sm89.h`, not the compiled `pinning.cu`. Editing
   device code yields a clean build and a score that is a draw of the frozen
   image; editing a kernel signature yields silent wrongness with the one guard
   (`static_assert` on argument count) defeated by construction. Regenerating
   that header needs `nvcc -arch=sm_89`, `cuobjdump`, `python3` and a shell, none
   of which exist here (this session also had no shell tool at all).

Together these mean the honest output is a diagnosis, and the diagnosis is that
the remaining work is a **whole-geometry replacement** (the only class that has
ever moved this track: GLV14 → GLV11 → GLV12, −40 M / +48 M / +12 M), gated behind
a CUDA toolchain.

## What is correct and checked

- **Launch-site inventory, the audit v22c asked for: CLEAN.** 16 raw `<<<>>>`
  sites in `pinning.cu`; 14 have a `qsb_carrier_launch` counterpart; the other 2
  are compiled out under `QSB_TREE_OFFLOAD`/`2` (both 0), which `pinning.cu:4066`
  makes mandatory under `QSB_NOJIT`. No launch is missing — the v22 defect is not
  in this tree.
- **Carrier defaults verified, not assumed.** `QSB_CARRIER` is defined only at
  `QsbCarrier.h:30` (=1) and `QSB_NOJIT` only at `:41` (=1); a whole-tree grep
  finds no override in any header or source. So the carrier path is the one that
  runs on an sm_89 4090.
- **Nothing in the executable path was modified.** Changes are confined to
  `ITERATIONS.md` (appended), `DEAD-ENDS.md` (stale "Still open" section
  corrected in place, historical table left intact), and this `_notes/` directory.

## What the evidence says about the remaining levers

| lever | verdict | evidence |
|---|---|---|
| 2^-31 / 2^-62 carry tails | not resolvable | +0.1–0.6 % vs σ = 1.7–2.2 % on identical bytes |
| host gate + C31 | already shipped | `pinning.cu:41`, `:53` |
| 16M batch + completion lane | already shipped and retired | +0.0819 % pooled, `PRIORITY-VALIDATION.json` NO-GO |
| init dead-time strip | exhausted | v8 neutral, v12 −4.06 % |
| CUDA graphs / JIT trim | exhausted | v9 telemetry channel dead; v12 negative |
| GLV10 / grouped GLV / Karatsuna | dead, officially | −55 %, −27 %, −6.1 % |
| "L2-resident 11-add hot set" | **the shipped baseline**, not a lever | `QSB_GLV11 1`, promoted 948,943,797 |
| shrinking the table until all gathers hit L2 | predicted to lose | needs N=8 → +3 critical-path adds to save ~2.4 %; `QSB_TBL_PREFETCH` measured −37 % and showed the reads are ~92 % latency-hidden |
| host-only knobs (slots, batch, completion mode, L2 window, co-grind width) | measured out | 0.08–0.4 % each, at or below noise, in `SUBMISSION-GLV12-V2-REDRAW.md` and the G3 controls |

## The one coherent unbuilt idea, and why I did not write it

N=8 terms per component puts the whole fixed-base table at ~16 MiB per
component — L2-resident — and deletes the 21.1 GiB GPU table build from the
timed window (~22 s ≈ 1.8 %). That is a >5 % story and the right class of move.

I did not write it, for two reasons that are not "it is hard":

1. it is device code in `GLVScalar.cuh` + `pinning.cu`, so it is **inert** without
   a carrier rebuild I cannot perform (note 01);
2. under `QSB_GLV11` the hot kernel already sits at **128 registers / 0 spills**,
   and N=8 adds three live chain terms — the register budget, the chain peeling
   (`QSB_CHAIN_PEEL`, `QSB_PHI_HOIST`), the warp-uniform `QSB_PMIX12` predicate,
   the `QSB_GT_RADIX_BITS 14` ladder bound for the new widest segment, and the
   `kernel_build_gtable` segment loop would all need re-deriving **and compiling**.
   Writing that blind is precisely the "manufacture a patch to look productive"
   failure. Shipping it uncompiled would be worse: it would look like a change
   and be a re-roll.

## Files

- `_notes/00-SESSION-OPEN.md` — env constraints; stale-premise finding; real gap
- `_notes/01-CARRIER-BLOCK.md` — the carrier staleness proof; launch-site audit
- `_notes/02-NOISE-VS-GATE.md` — both same-bytes series; the ≥5 % bar
- `_notes/03-GEOMETRY-COST-CURVE.md` — GLV11 geometry table; the cold-read
  calibration; why the L2-residency prize loses
- `ITERATIONS.md` — appended session entry
- `DEAD-ENDS.md` — "Still open" section corrected against the live tree

## Honest statement of limits

Nothing here was compiled or executed: no CUDA, no GPU, and no shell in this
session. Every claim is source-reading with file and line citations, and the
reasoning is labelled as inference where it is inference (the L2-residency
prediction in particular is an inference from `QSB_TBL_PREFETCH`'s −37 % and the
GLV12-vs-GL11 promotion pair, not a measurement). No speed figure in this
session was measured, and none should be transferred to the 4090.
