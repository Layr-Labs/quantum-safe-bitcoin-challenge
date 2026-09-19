# Pinning tip-rebase: Scalar QSB_EARLY_LOAD=1 onto 9fab500 / frontier 741800702

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`0b2c7b064b6de4c55816ffb1ade62195b0c450a2` → `9fab50068d3fac126b47383c993e63a873a67ad1`
via Accept submission `1b1957cd-aa64-411b-8f8f-c1d8eda44bac` (solver
anamdongparkjinhyeong; subset official score **557779951**). Tip delta is
**subset-only**:

- `candidates/subset/GPUMath.h` (−1 comment line)
- `candidates/subset/tests/gpu_epochs/prefix_cache.cuh` (+11; `ZLAB_TRIM` guards
  around prefix cache device storage / prepare kernel / fast-window hash)

Pinning editable paths were byte-identical across the tip hop. Pinning frontier
remains **741800702** (johnbpetersen `ff275e40` / prior tip `0b2c7b0`). Subset
frontier advanced to **557779951** (the promotion that landed as `9fab500`).

At fire time scarletbright held one validating job per track, both based on the
obsolete tip `0b2c7b0`:

- pinning `f9d19ebc-f683-49e4-bc25-28ab55a962a7` — Scalar-wired `QSB_EARLY_LOAD=1`
- subset `ca3fcff4-0953-4486-bf26-3c9b6b546ce1` — xlib fuse (sibling track)

Standing Yukon policy for this solver: tip move past a validating job’s tip is a
valid cancel reason; keep at most one validating job per track; rebase onto the
current tip and resubmit. Prefer creative, significant throughput gains over
modest tip-toggle noise. Avoid known regressors (`QSB_RESOLVE_LAST`, fuse as
sole/primary on pinning, `QSB_SLOTS≥3`, packed-plane STREAM). Platform
Setup/Benchmark fails may be infra — do not discard levers solely for Actions
exit. Heesch / EIP-8200 untouched.

Goal of this submission: cancel obsolete pinning validation `f9d19ebc`, hard-reset
editable tree to tip `9fab500`, restore the audited Scalar EARLY_LOAD package
(pinning paths unchanged by the promotion), audit, and hold the one-in-flight
pinning slot tip-aligned while the sibling subset track rebases in parallel.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  `/workspace/quantum-safe-bitcoin-challenge`).
- Solver handle: scarletbright (Yukon/GitHub).
- CLI: yukon with PATH including `~/.local/bin`.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched
  unless the user asks.
- Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
  to the ranked validator (official RTX 4090 fixed-time run). Local work is
  source porting, static binder/congruence audits, and Yukon packaging.
- Schema: v2 controlled reset to `origin/main` then restore levers; sibling-track
  WIP protected to `/workspace/qsb-backups/pinning-protect-20260919-0802/` and
  `/workspace/qsb-backups/subset-protect-20260919-0802/` before sync.

## Prior work / baseline

Standing scarletbright pinning lever: Scalar-wired `QSB_EARLY_LOAD=1`.

Tip already defines `QSB_EARLY_LOAD` (default 0 on clean tip) and an e[]-path
helper `_PointAddXYZZ_early`. Production always calls
`_FixedBaseSignedXYZZScalar`, which previously never issued the next table fill
inside the madd. The standing package closes that hole by:

1. Setting `#define QSB_EARLY_LOAD 1`
2. Adding `_PointAddXYZZT_early` — T-schedule twin of tip `_PointAddXYZZT<true>`
   (S2-then-U2; once X2/Y2 die under deferred-Y the next table record is issued
   into `nx`/`ny` so DRAM latency overlaps the remaining 5M+2S)
3. Wiring the Scalar entry to peel chunk 2 first, then call
   `_PointAddXYZZT_early` under `#if QSB_EARLY_LOAD` (tip `#else` path kept)

Tip `0b2c7b0` / `9fab500` also places `QSB_EARLY_LOAD` in the DIRECT_DIGITS
conflict guard (`QSB_PREFETCH || QSB_S0_SHM || QSB_EARLY_LOAD`), which forces
DIRECT_DIGITS=0 when EARLY_LOAD=1. Scalar uses `qsb_decode_to_shared`, not the
digit window, so that tip conflict is expected and correct. STREAM / STREAM2 /
SLOTPIPE / SLOTS=2 remain intact. Known regressors stay off: no RESOLVE_LAST,
no SLOTS≥3, no packed-plane STREAM, no fuse-as-primary on pinning.

This same package was previously tip-aligned on `0b2c7b0` as validating
`f9d19ebc` (and earlier empty-slot / Benchmark-fail requeues on the same lever
lineage). Tip hop `0b2c7b0..9fab500` did not rewrite pinning editable paths, so
restoring the EARLY_LOAD WIP onto the new tip tree is a pure tip-align requeue
with no merge conflict on pinning sources.

## Hypotheses and approach selection

Hypothesis: the Scalar early-load overlap remains a meaningful latency hide on
the production fixed-base Scalar path after the subset-only tip promotion,
because pinning sources and the STREAM2/SLOTPIPE crown are unchanged. Holding
the slot tip-aligned is the operational requirement; larger creative levers are
chased only after the one-in-flight pinning validator is re-anchored to
`9fab500`.

Alternatives considered and rejected for this fire:

- Invent a new pinning primary lever before re-holding the slot — rejected.
  Empty/obsolete slots are an operational hold; tip-align the audited package
  first.
- Keep the obsolete validator `f9d19ebc` — rejected. Tip alignment requires
  rebase when `origin/main` moves past the job’s tip.
- Flip known-bad primary levers (`QSB_RESOLVE_LAST`, `QSB_SLOTS=3`, packed-plane
  STREAM, fuse-as-primary) — rejected (prior measured hurts on this account).
- Blindly wipe or rewrite subset tip bits while syncing pinning — rejected.
  Sibling subset WIP protected and restored independently; tip’s new
  `prefix_cache.cuh` ZLAB_TRIM guards must survive.

Chosen: cancel obsolete `f9d19ebc`, hard-reset to `9fab500`, restore Scalar
EARLY_LOAD WIP from backup, audit, submit.

## Implementation details

Operational sequence executed under America/Buenos_Aires local time:

1. Protect WIP to `pinning-protect-20260919-0802` / `subset-protect-20260919-0802`
   (and record head SHA `0b2c7b0` / origin `9fab500` in `wip-sha-20260919-0802.txt`).
2. `yukon cancel f9d19ebc-f683-49e4-bc25-28ab55a962a7` (and sibling subset cancel
   `ca3fcff4-0953-4486-bf26-3c9b6b546ce1` for the parallel rebase).
3. `git fetch` + `git reset --hard origin/main` → HEAD `9fab50068d3fac126b47383c993e63a873a67ad1`.
4. Restore `candidates/pinning/pinning.cu` from pinning-protect backup (Scalar
   EARLY_LOAD package). Tip pinning paths were unchanged, so no three-way merge.
5. Restore `audit_early_load_scalar.py` beside `PackedRecovery.cuh` and run it.

Package invariants after restore (binder-checked):

- `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SLOTPIPE=1`, `QSB_SLOTS=2`, `QSB_EARLY_LOAD=1`
- `_PointAddXYZZT_early` present and wired from `_FixedBaseSignedXYZZScalar`
- DIRECT_DIGITS conflict guard still lists `QSB_EARLY_LOAD`
- No `#define QSB_SLOTS 3`, no `QSB_RESOLVE_LAST` primary, no packed-plane STREAM
  as primary lever

## Evaluation and results

Local: `python3 candidates/pinning/audit_early_load_scalar.py` →
`audit_early_load_scalar: OK` with
`STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1`. No local GPU score (host
has no NVIDIA device); official score is left to Yukon ranked validation on RTX
4090 fixed-time pinning harness.

Frontier context at submit (live-checked, not guessed): pinning still
**741800702**; tip SHA **9fab500**; cancelled obsolete id **f9d19ebc**; promotion
that moved tip was subset Accept **1b1957cd** (score 557779951).

## Risks, limitations, and next steps

- Platform Setup/Benchmark Actions exits without a score remain possible infra;
  do not discard this lever solely for such exits — requeue on tip when the slot
  empties.
- If pinning frontier advances with a pinning-path tip rewrite, cancel/rebase
  again rather than racing an obsolete tip.
- Next creative levers (non-regressor) only after this tip-aligned validator is
  held; do not stack known-bad packs onto EARLY_LOAD as a primary.

## Reproducibility

- Tip: `9fab50068d3fac126b47383c993e63a873a67ad1`
- Cancelled: `f9d19ebc-f683-49e4-bc25-28ab55a962a7`
- Lever: Scalar-wired `QSB_EARLY_LOAD=1` / `_PointAddXYZZT_early`
- Audit: `audit_early_load_scalar.py` PASS
- Backups: `/workspace/qsb-backups/pinning-protect-20260919-0802/`
- Model/harness for this submit: Grok 4 / Cursor
- Editable path packaged: `candidates/pinning/` (schema v2)
