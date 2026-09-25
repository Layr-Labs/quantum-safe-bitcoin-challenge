# BABYDOV subset redraw: GLV11 + native sm_89 carrier + FMA gate + lean GLV split

## Submission type

This is a fully disclosed independent ranked redraw of the public Yukon source from submission
`7ee5c52a-b3d6-4873-9856-8943399d1b8d` (PR #1557, source commit `a8c9a7d`), prepared by
BABYDOV / williamleewilliam1-star. I do **not** claim authorship of the underlying optimization.
The only BABYDOV source edit is an unused provenance macro,
`QSB_BABYDOV_REDRAW_20260925`, in `subset.cu`; it is not referenced by candidate code and is
intended only to give this redraw an auditable source identity.

The executable candidate is therefore the public GLV11 package authored and assembled by the
contributors credited below. This submission exists because Yukon scores a fresh official RTX 4090
run and public exact-source/redraw submissions are already part of this track's history.

## Current target

At preparation time Yukon reports:

- track: `eigenlabs/quantum-safe-bitcoin-challenge/subset`
- current promoted record: **623,518,629 verified candidates/s**
- required promotion improvement: **100 bips**
- promotion threshold: **629,753,815 candidates/s**
- official source ref: `d59a969777f4223a330bab759e38e1dd16dec810`

The selected public package is materially different from the promoted record and was chosen instead
of our earlier GLV12 + SHA-unroll experiment. Public same-lineage measurements show that enabling
`QSB_PAIR_SHA_UNROLL_CONST=1` is a regression (about -0.24% locally), so that experiment was
discarded before submission rather than spending an official runner draw on a self-cancelling
combination.

## What the selected package contains

The source package from PR #1557 combines:

1. **GLV11 P18 geometry** from i34-9's public work: eleven table gathers / ten point additions per
   candidate rather than the GLV12 twelve/eleven shape.
2. **Native sm_89 carrier** derived from newjordan/Ryun1 lineage so cold 64-byte records can use the
   one-access L2 prefetch-size hint on the benchmark RTX 4090.
3. **QSB_SHA_FMA_ADD=1**, shifting exact pubkey-hash additions onto the FMA-heavy pipe.
4. **Lean GLV split** from i34-9 (`QSB_GLV_LEAN`, `QSB_GLV_ROUND_CC`,
   `QSB_GLV_HIGH15_HI`).
5. **Exact group-capacity sizing**, required because the GLV11 table occupies most of a 24 GB card.
6. Existing exact host/OpenSSL publication verification and the inherited speculative/exact split.

The original PR #1557 author reports paired local RTX 4090 measurements of roughly +2.743% for GLV11
over its GLV12 base and another +0.213% for the lean GLV split, with 9,240 / 9,240 hits verified in a
90-second unmodified-harness run. These are **source-author measurements**, not BABYDOV measurements.
I make no claimed score; the official Yukon runner is authoritative.

## BABYDOV verification performed

Before this submission:

- Yukon CLI was updated to v2026.09.23-3; telemetry and trace capture were disabled.
- A fresh Yukon-linked checkout was created from the official source ref.
- The source was fetched directly from the public Yukon submission ref, not reconstructed from prose.
- `git diff --check` passes.
- Every changed path is under the benchmark's declared editable path `candidates/subset/`.
- `yukon setup --track subset` completed successfully on macOS:
  - synthetic problems generated;
  - verifier smoke test passed;
  - as expected, CUDA compilation/device execution was skipped because this Mac has no `nvcc` or
    NVIDIA GPU.
- The source package retains the generated sm_89 carrier and the original license/attribution files.

No local GPU throughput is claimed. No claimed-score field is supplied.

## Why redraw this source

The promoted subset record is 623.519M. The earlier public GLV12 native-carrier candidate
(`d1ddefca`) scored 626.795M: faster than the record but below the +1% promotion threshold.
The selected GLV11 package attacks the remaining gap by removing an additional point addition while
retaining the native cold-record fetch path and FMA gate improvements. Its public local measurement
is large enough to justify one independent official draw, unlike the discarded SHA-unroll branch.

The outcome is intentionally falsifiable:

- If verified score >= the live promotion threshold, Yukon can promote it.
- If it verifies but remains below threshold, this redraw is recorded as a non-promotion; it must not
  be repeatedly resubmitted unchanged merely to chase noise.
- If validation fails, the public PR/source gives an exact reproducer for investigation.

## Attribution

Primary source package:
- terrapinelf — public PR #1557 / submission `7ee5c52a-b3d6-4873-9856-8943399d1b8d`.

Key directly reused mechanisms:
- i34-9 — GLV11 P18 geometry, exact group-capacity work, lean GLV split.
- newjordan — GLV12/native-carrier base and warp-root-inverse lineage.
- Ryun1 — native carrier design lineage.
- ercumentyildirim — GLV12 subset port lineage.
- fkiene — fk-lean / FMA-gate lineage.
- Akashneelesh and the contributors credited by the promoted crown — inherited subset frontier work.

All inherited GPL/MIT notices and source attribution in the package are retained.

## Scope and safety

Only `candidates/subset/` is submitted. No benchmark harness, verifier, scorer, problem generator,
workflow, sibling track, credential, token, wallet data, or private material is included.

The official Yukon benchmark is the first BABYDOV GPU execution of this exact redraw.
