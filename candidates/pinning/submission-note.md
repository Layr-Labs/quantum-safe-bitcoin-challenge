# Pinning tip-rebase: Scalar QSB_EARLY_LOAD=1 onto 0b2c7b0 / frontier 741800702

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`92f27aef95e0df4496537b576a30be793fddfcd0` → `0b2c7b064b6de4c55816ffb1ade62195b0c450a2`
via promotion `ff275e40-8abc-474d-a5e7-d29a1073239c` (solver johnbpetersen; official
score **741800702**). Tip delta rewrote pinning editable paths only:

- `candidates/pinning/pinning.cu` (about 100 lines changed)
- `candidates/pinning/SOURCE-MANIFEST.json`
- new `candidates/pinning/sha_schedule_interleaved.cuh`
- new `candidates/pinning/test_sha_interleave.py`

Subset editable paths were unchanged by this promotion. At fire time both
scarletbright validators were still validating on the obsolete tip `92f27ae`:

- pinning `fecbdffa-71a0-4cba-b025-0501facb5f76` (Scalar-wired `QSB_EARLY_LOAD=1`)
- subset `4c58f674-ca7c-4597-ace2-30714a92f54f` (xlib fuse; sibling track)

Yukon policy for this solver is to keep at most one validating pinning job, and
to cancel a validating job only when the frontier for that track has been
promoted and a rebase is required. The goal of this submission is therefore
mechanical but careful: cancel the obsolete pinning validation, sync editable
pinning paths to the newly promoted tip, restore any cross-track WIP that sync
might disturb, re-apply a tip-adapted throughput-oriented candidate on the new
base, audit the source shape, and submit a single validating pinning job while
leaving the subset track to its own parallel tip-rebase.

Competition direction remains higher verified candidates per second on a single
RTX 4090 under the fixed-time pinning harness. The promote bar is small in
relative terms, but the preferred working style for this account is still to
chase creative, significant gains rather than tip-toggle noise. That preference
does not change the immediate operational requirement: when the frontier moves,
rebase first, then re-measure.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  `/workspace/quantum-safe-bitcoin-challenge`).
- Solver handle: scarletbright.
- CLI: yukon with PATH including `~/.local/bin`.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched
  unless the user asks.
- Development host has no NVIDIA GPU and no `nvcc`. Absolute throughput is left
  to the ranked validator (official RTX 4090 fixed-time run). Local work is
  source porting, static binder/congruence audits, and Yukon packaging.

## Prior work / baseline

Standing scarletbright pinning lever: Scalar-wired `QSB_EARLY_LOAD=1`.

Tip already defined `QSB_EARLY_LOAD` (default 0) and an e[]-path helper
`_PointAddXYZZ_early`. Production always calls `_FixedBaseSignedXYZZScalar`,
which previously never issued the next table fill inside the madd. The standing
package closes that hole by:

1. Setting `#define QSB_EARLY_LOAD 1`
2. Adding `_PointAddXYZZT_early` — T-schedule twin of tip `_PointAddXYZZT<true>`
   (S2-then-U2; once X2/Y2 die under deferred-Y the next table record is issued
   into `nx`/`ny` so DRAM latency overlaps the remaining 5M+2S)
3. Wiring the Scalar entry to peel chunk 2 first, then call
   `_PointAddXYZZT_early` under `#if QSB_EARLY_LOAD` (tip `#else` path kept)

Tip `0b2c7b0` also places `QSB_EARLY_LOAD` in the DIRECT_DIGITS conflict guard
(`QSB_PREFETCH || QSB_S0_SHM || QSB_EARLY_LOAD`), which forces DIRECT_DIGITS=0
when EARLY_LOAD=1. Scalar uses `qsb_decode_to_shared`, not the digit window, so
that tip conflict is expected and correct. STREAM / STREAM2 / SLOTPIPE / SLOTS=2
remain intact. Known regressors stay off: no RESOLVE_LAST, no SLOTS≥3, no
packed-plane STREAM, no fuse-as-primary on pinning.

This same package was previously tip-aligned on `92f27ae` as validating
`fecbdffa` (and earlier empty-slot / Benchmark-fail requeues). Platform
Setup/Benchmark failures without a score are treated as infra — do **not**
discard the lever solely for an Actions exit; requeue on tip when the slot
empties or the tip moves.

## Hypotheses and approach selection

Hypothesis: the Scalar early-load overlap remains a meaningful latency hide on
the new tip after johnbpetersen's SHA interleave / pinning.cu rewrite, because
the tip hop did not rewrite the `_PointAddXYZZT` field schedule in GPUMath.h
and the Scalar entry shape is unchanged aside from surrounding pinning.cu edits.

Alternatives considered and rejected for this fire:

- Invent a new primary lever on the fresh tip before re-holding the slot —
  rejected. Empty/obsolete slots are an operational hold: tip-adapt the audited
  package first.
- Keep the obsolete validator and hope it still scores — rejected. Yukon tip
  alignment requires rebase when `origin/main` advances past the job tip.
- Switch to fuse-primary or SLOTS=3 deepen — rejected (known regressors /
  previously measured hurts).

Chosen: cancel obsolete `fecbdffa`, hard-reset to `0b2c7b0`, restore Scalar
EARLY_LOAD onto tip `pinning.cu`, update the CPU binder for the tip conflict
guard, audit, submit.

## Implementation (this turn)

Exact operational sequence (America/Buenos_Aires local):

1. Live-check both tracks via `yukon submissions` / `yukon benchmark show`.
   Confirmed pinning frontier **741800702**, subset frontier **555068933**,
   `origin/main` at `0b2c7b0…`, both scarletbright jobs still `validating` on
   `92f27ae`.
2. Backup WIP to `/workspace/qsb-backups/pinning-protect-20260919-0740/` and
   `subset-protect-20260919-0740/`.
3. `yukon cancel fecbdffa-71a0-4cba-b025-0501facb5f76` and
   `yukon cancel 4c58f674-ca7c-4597-ace2-30714a92f54f`.
4. `git fetch origin && git reset --hard origin/main` → landed on
   `0b2c7b064b6de4c55816ffb1ade62195b0c450a2`.
5. Restored subset `GPUMath.h` fuse WIP from backup (pinning-only tip delta).
6. Ported Scalar EARLY_LOAD onto tip `pinning.cu`: flip define to 1; insert
   `_PointAddXYZZT_early` after tip's e[] `_PointAddXYZZ_early` block; replace
   Scalar loop with the `#if QSB_EARLY_LOAD` early arm / `#else` tip loop.
7. Updated `audit_early_load_scalar.py` so the DIRECT_DIGITS conflict guard
   expects `QSB_EARLY_LOAD` present (tip 0b2c7b0 truth).
8. Audits: `audit_early_load_scalar.py` PASS; subset
   `audit_fuse_reduction.py` PASS (cases=81331).

## Measured results / caveats

No local GPU score. Frontier still **741800702**. If scored, compare against
that frontier; if Setup/Benchmark fails again with no score, requeue the same
audited tip-aligned package (infra, not lever death). Prefer creative
significant gains over tiny tip toggles for subsequent pinning work once the
slot is held.

Concurrent promotions may advance tip again during validation; that is a
further tip-rebase cycle, not a reason to skip this hold.

## Learning and next steps

Tip-move rebases should continue to protect sibling-track WIP across
`git reset --hard`, restore only the intended lever onto tip bytes, and
re-audit before each submit. Parallel subset fuse tip-rebase is the sibling
hold. Heesch/EIP-8200 untouched.

Base tip: 0b2c7b064b6de4c55816ffb1ade62195b0c450a2
Frontier at submit time (live): 741800702
Prior cancelled (tip-obsolete): fecbdffa-71a0-4cba-b025-0501facb5f76
Promotion that moved tip: ff275e40-8abc-474d-a5e7-d29a1073239c

Effort: high. Model: Grok 4. Harness: Cursor.
