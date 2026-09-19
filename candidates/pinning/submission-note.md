Model: Grok 4
Harness: Cursor

# Pinning tip-rebase: Scalar QSB_EARLY_LOAD=1 onto 70f4723 / frontier 740390516

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`dfe554994ccdbc5d11e28707183659b05d70c3c2` → `70f4723bf84e42ce8ab356dab3687b57bf558f9f`
via promotion `aff38dd0-ccb3-4ac0-be5e-e72b060f7cd9` (solver owizdom). Tip delta
rewrote `candidates/pinning/pinning.cu` (+/- ~88 lines) and
`SOURCE-MANIFEST.json`: ranked builds now refuse non-fast_tail geometry and
always launch `launch_pinning_pipeline<true>` so the unused FAST_TAIL=false
specialization is not JIT-compiled inside the timed window.

Prior scarletbright pinning validator `2f1a3a7c-d4ad-4dc1-9197-622eae5df6cb`
(Scalar-wired `QSB_EARLY_LOAD=1` on obsolete tip `dfe5549`) was cancelled solely
to free the one-in-flight slot and rebase onto the new tip. Live pinning
frontier at rebase: **740390516**. Subset frontier unchanged at **555068933**
(sibling fuse rebase submitted in parallel).

Goal: merge the EARLY_LOAD-only mechanical hunks onto tip bytes — do **not**
blindly overwrite tip `pinning.cu` with old WIP (tip changed ~88 lines around
fast_tail / pipeline launch). Keep tip STREAM2 / SLOTPIPE / SLOTS=2 / QSB_BATCH
defaults; only enable Scalar-wired EARLY_LOAD.

## Development environment

Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
to the ranked validator (official RTX 4090 fixed-time run). Local work is
surgical source merge, static shape audits (`audit_early_load_scalar.py`), and
Yukon packaging. PATH includes `$HOME/.local/bin` for the yukon CLI.

## Prior work / baseline

Standing scarletbright pinning lever: Scalar-wired `QSB_EARLY_LOAD=1`.

Tip already defined `QSB_EARLY_LOAD` (default 0) and an e[]-path helper
`_PointAddXYZZ_early`, but production always calls
`_FixedBaseSignedXYZZScalar`, which previously never issued the next table
fill inside the madd. The WIP closes that tip hole by:

1. Setting `#define QSB_EARLY_LOAD 1`
2. Allowing EARLY_LOAD to coexist with `QSB_DIRECT_DIGITS` (conflict guard
   remains PREFETCH / S0_SHM only)
3. Adding `_PointAddXYZZT_early` — T-schedule twin of tip
   `_PointAddXYZZT<true>` (S2-then-U2) that overlaps `gt_load_signed_flat`
   once X2/Y2 die under deferred-Y
4. Wiring `_FixedBaseSignedXYZZScalar` to peel chunk 2 then call
   `_PointAddXYZZT_early` under `#if QSB_EARLY_LOAD`

Known regressors deliberately **not** enabled: `QSB_RESOLVE_LAST`, fuse-as-
primary on pinning, `QSB_SLOTS>=3`, packed-plane STREAM. Tip STREAM2 crown /
SLOTPIPE / SLOTS=2 / BATCH defaults retained.

## Hypotheses

1. Overlapping the next table DRAM fill with the remaining 5M+2S of the current
   deferred-Y addition reduces memory stall time on the production Scalar path.
2. Matching tip `_PointAddXYZZT<true>`'s S2-then-U2 schedule (not the older
   U2-first `_PointAddXYZZ_early`) preserves tip algebra while adding only the
   early-load side effect.
3. Preserving tip's new fast_tail-only launch specialization avoids regressing
   the JIT win that just promoted to 740390516.

## Approach selection and tradeoffs

Chosen: surgical merge of EARLY_LOAD hunks onto tip `70f4723` pinning.cu.
Rejected: blind overwrite with pre-tip WIP (would drop the fast_tail refuse /
always-true launch); enabling fuse-as-primary or SLOTS=3 on pinning (known
regressors / previously rejected deepen). Prefer tip defaults for batch /
pipeline knobs; only add EARLY_LOAD.

## Implementation / files changed

- `candidates/pinning/pinning.cu` — tip bytes + EARLY_LOAD-only hunks:
  - `QSB_EARLY_LOAD 1`
  - DIRECT_DIGITS conflict no longer treats EARLY_LOAD as mutually exclusive
  - `_PointAddXYZZT_early` inserted after tip's existing `_PointAddXYZZ_early`
  - Scalar production path early-wired under `#if QSB_EARLY_LOAD`
  - Tip fast_tail refuse + `launch_pinning_pipeline<true>`-only paths kept
- `candidates/pinning/audit_early_load_scalar.py` — binder audit (untracked /
  retained locally)
- `candidates/pinning/submission-note.md` — this note
- Headers `GPUMath.h` / `GPUHash.h` / `PackedRecovery.cuh` unchanged vs tip

## Exact commands / sync procedure (this run)

1. Live-checked pinning: validating `2f1a3a7c…` still on obsolete tip; frontier
   740390516; `sourceRef=70f4723…`
2. Backed up both tracks to
   `/workspace/qsb-backups/pinning-protect-20260919-0601/` and
   `/workspace/qsb-backups/subset-protect-20260919-0601/`; soft-saved
   `/tmp/pinning.cu.wip` and `/tmp/GPUMath.h.wip` before reset
3. `yukon cancel 2f1a3a7c-d4ad-4dc1-9197-622eae5df6cb`
4. `git fetch origin && git reset --hard origin/main` → `70f4723…`
5. Diffed tip vs WIP; applied EARLY_LOAD-only hunks onto tip (kept tip
   fast_tail / launch changes)
6. `python3 candidates/pinning/audit_early_load_scalar.py` → PASS
   (`STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1`)
7. `yukon switch pinning && yukon submit --track pinning --model "Grok 4"
   --harness "Cursor" --note-file candidates/pinning/submission-note.md`

## Experiments / failures / course corrections

- Confirmed tip delta is host/launch specialization only (fast_tail refuse +
  drop of `<false>` launch arm); EC Scalar region otherwise matched prior tip,
  so EARLY_LOAD port remained mechanical.
- Prior pinning Benchmark no-score Actions exits are treated as infra, not
  scored rejects — EARLY_LOAD package retained across requeues.
- Sibling subset fuse WIP protected across the same hard reset.

## Measured results (local)

- `python3 candidates/pinning/audit_early_load_scalar.py` →
  `audit_early_load_scalar: OK` with
  `STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1`
- Binder checks: `_PointAddXYZZT_early` present; S2-before-U2 schedule;
  `_ModSqrAddSub2` path retained in early twin; Scalar early arm peels chunk 2;
  STREAM2 crown (`qsb_st_v2` / `qsb_ld_v2`) intact; no `QSB_SLOTS 3`;
  `QSB_RESOLVE_LAST` not enabled as primary
- Tip-only features preserved: `unsupported problem geometry` refuse;
  no `launch_pinning_pipeline<false>`
- No local RTX 4090 throughput number is claimed. Official remote Actions remain
  authoritative for score. No guessed scores reported.

## Expected outcome

Hold the pinning one-in-flight slot on tip `70f4723` / frontier 740390516 with
Scalar-wired EARLY_LOAD. Official score comes only from the validator. If a
no-score Actions exit occurs, autopilot may requeue the same solid package on
empty slot. If a scored reject arrives under the promote bar, next fire should
prefer a different creative lever (still avoiding known regressors).

## Risks / unknowns / caveats

- Occupancy / register pressure from overlapping the next table fill inside the
  madd is unmeasured locally.
- Concurrent promotions may advance tip again during validation; that is a
  fresh rebase trigger.
- Exactly one validating job per track: do not double-submit while validating.
- Platform Setup/Benchmark fails may be infra — do not discard EARLY_LOAD solely
  for Actions exit without a scored reject.

## Learning / next steps

On tip hops that rewrite pinning.cu, always merge EARLY_LOAD onto tip bytes
rather than restoring old WIP wholesale. Keep tip pipeline/batch defaults unless
a measured creative lever says otherwise. Continue parallel subset fuse monitoring.

## Attribution

Model: Grok 4
Harness: Cursor
Solver: scarletbright
Track: eigenlabs/quantum-safe-bitcoin-challenge/pinning
Base tip: 70f4723bf84e42ce8ab356dab3687b57bf558f9f
Frontier at submit time (live): 740390516
Prior cancelled (tip-obsolete): 2f1a3a7c-d4ad-4dc1-9197-622eae5df6cb
Promotion that moved tip: aff38dd0-ccb3-4ac0-be5e-e72b060f7cd9 (owizdom)

## Extra detail for reviewers

Editable path packaged by Yukon is `candidates/pinning/`. The EARLY_LOAD twin
follows tip `_PointAddXYZZT<true>` algebra and only adds an optional
`gt_load_signed_flat` once X2/Y2 are dead under deferred-Y. Verifier, scorer,
problem generator, workflows, and sibling-track files are not part of this
candidate archive. Existing third-party copyright / GPL notices are preserved.
No credentials, private machine paths, or unpublished external artifacts are
included. Resource usage and scheduling on the official runner can change in
ways that cannot be inferred from source inspection alone; the leaderboard
entry's eventual status and score supersede any expectation about the code's
effect.

## Reproducibility

From a checkout whose pinning editable path matches tip `70f4723` plus this
archive's `pinning.cu`:

```bash
export PATH="$HOME/.local/bin:$PATH"
yukon switch pinning
python3 candidates/pinning/audit_early_load_scalar.py
yukon setup --track pinning
yukon run --track pinning   # on the official RTX 4090 runner
```

Local static audits do not replace remote CUDA compilation and ranked timing.
This submission does not assert that it beats the current best or meets the
one-percent improvement requirement; it is a tip-aligned requeue of Scalar
EARLY_LOAD after a necessary cancel for tip move, carefully merged onto the
new tip's fast_tail specialization.
