# Pinning resubmit: Scalar EARLY_LOAD on tip b62eb79 after third Benchmark fail

## Initial context and goal

Scarletbright pinning submission 7bf31e0e (after prior acfda355 / 4b17e2c4)
FAILED at the GitHub Actions Benchmark step with no official score. Platform
Benchmark failures without a score are treated as infra / runner noise rather
than evidence that the tip-adapted Scalar EARLY_LOAD lever is bad. Yukon
competition policy prefers holding the pinning EARLY_LOAD package over waiting
for another window.

Live check at about 2026-09-19 04:25 ART (America/Buenos_Aires, UTC-3) before
this resubmit:

- Pinning frontier still **739180224** on tip **b62eb79** (unchanged vs all
  three failed EARLY_LOAD jobs). Confirmed via `yukon benchmark show` and
  `git rev-parse` HEAD == origin/main == b62eb79d21ac6d1db6bff3732448f20ac84ce30b.
- Latest EARLY_LOAD row 7bf31e0e failed — no EARLY_LOAD package currently
  validating. Other unrelated scarletbright pinning experiments may still be
  validating; this cycle does not cancel them. Requeue the identical audited
  Scalar EARLY_LOAD package to hold that lever.
- Subset submission **f6bb671c** still **validating** on frontier **548846182**
  — LEAVE ALONE. Do not cancel. Protect subset WIP across pinning-only ops.
- Do not invent a new primary lever. Resubmit the same tip-adapted Scalar
  EARLY_LOAD (or tip-aligned solid lever). Avoid RESOLVE_LAST / SLOTS≥3 /
  packed-plane STREAM / fuse-primary.

Competition direction remains higher verified candidates per second on a
single RTX 4090 under the fixed-time pinning harness. An empty EARLY_LOAD
hold after infra fails is an operational requeue: identical tip-aligned
candidate immediately.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  /workspace/quantum-safe-bitcoin-challenge).
- Solver handle: scarletbright.
- CLI: yukon with PATH including ~/.local/bin.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched.
- Pinning editable path: candidates/pinning.
- Subset editable path: candidates/subset (protected; backed up under
  /workspace/subset-dirty-preserve-20260919-042527-pre-pinning-earlyload-resubmit).
- Hardware / runner: GitHub Actions GPU workflow for benchmark-pinning.yml.
- Local box: Linux workspace for source edits, audits, backups, and yukon
  orchestration. No claim of a local full-score GPU run for this cycle;
  official score comes from the Actions benchmark.

Baseline commands:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon tracks
yukon switch pinning
yukon submissions --json
yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa
git fetch origin
git log --oneline -5 origin/main
python3 candidates/pinning/audit_early_load_scalar.py
```

Confirmed live: currentBestScore=739180224,
sourceRef=b62eb79d21ac6d1db6bff3732448f20ac84ce30b, local HEAD matches
origin/main at b62eb79.

## Prior work and baseline

The tip-adapted Scalar EARLY_LOAD was rebased onto tip b62eb79 after the
frontier advanced from 37922c7 / 739010506 to b62eb79 / 739180224. That
produced 4b17e2c4, which reached Benchmark and failed with no score. Immediate
resubmits of the same tip-aligned tree were acfda355 and then 7bf31e0e; both
likewise failed at Benchmark with no score. This cycle is the next slot-hold
resubmit of that same audited candidate while the frontier remains unchanged.

Recent pinning history on this account includes both measured rejects and a
string of Actions-level Benchmark failures without scores (including e16f991,
3c7d088, 1940726, 4b17e2c4, acfda355, 7bf31e0e). Measured rejects are real
lever feedback. Benchmark-step failures are not. This resubmit therefore does
not change the lever shape.

Subset independently sits at frontier 548846182 with f6bb671c validating.
Subset was intentionally left alone: no cancel, no sync, WIP protected.

## Hypotheses for this cycle

1. Tip b62eb79 / score 739180224 is unchanged since 4b17e2c4 / acfda355 /
   7bf31e0e. The tip-adapted Scalar EARLY_LOAD wiring remains tip-aligned; no
   ambitious rebase is required.
2. A Benchmark-step failure with no score is not a regression signal. Holding
   the EARLY_LOAD package with the same audited lever is higher EV than waiting
   or switching to a known-regressor stack. Platform fails may be infra —
   still resubmit.
3. Syncing pinning when the tip has not moved would only risk disturbing
   subset WIP for no tip benefit. Therefore this cycle skips cancel/sync and
   submits the existing tip-adapted working tree.
4. Keeping STREAM / STREAM2 / SLOTPIPE / SLOTS=2 intact on tip is safer than
   stacking known regressors. Known-bad primary levers for this account remain
   out of scope (RESOLVE_LAST, SLOTS≥3, packed-plane STREAM experiments,
   fuse-as-sole/primary pinning gambits).
5. Source-shape audits without nvcc remain a useful gate before submit: they
   catch accidental disablement of crown switches and accidental wiring
   mistakes in the production scalar entry.

## Approach selection and tradeoffs

Selected approach: resubmit same tip-adapted Scalar EARLY_LOAD on
unchanged tip b62eb79, with subset left validating.

Tradeoffs considered and rejected for this cycle:

- Waiting for more Actions signal before requeue: empties the EARLY_LOAD hold
  and loses queue position; policy prefers holding the package.
- Switching to RESOLVE_LAST / SLOTS≥3 / packed-plane STREAM / fuse-primary:
  known regressors or high-risk primary gambits for this account.
- Force-syncing both tracks or cancelling subset f6bb671c: forbidden;
  subset frontier unchanged and job still validating.
- Inventing a brand-new unrelated lever during the empty EARLY_LOAD window:
  higher risk of shipping an un-audited interaction while the prior tip-adapted
  candidate never received a score.
- Cancelling unrelated in-flight pinning experiments on this account: not
  required for this hold; leave them alone.

## Implementation and files changed

Only `candidates/pinning/` is in scope. Relative to tip b62eb79 production
pinning sources, the tip-adapted Scalar EARLY_LOAD changes are confined to
`pinning.cu`:

- `#define QSB_EARLY_LOAD 1` (tip default was 0).
- DIRECT_DIGITS conflict guard no longer treats EARLY_LOAD as exclusive with
  DIRECT_DIGITS (PREFETCH / S0_SHM still force the shared digit-plane path).
- New `_PointAddXYZZT_early` twin of tip `_PointAddXYZZT<true>`: same
  S2-then-U2 T-schedule and fuse path; once X2/Y2 die under deferred-Y it
  issues `gt_load_signed_flat` for the next table record into nx/ny.
- `_FixedBaseSignedXYZZScalar` production entry peels chunk 2 then overlaps
  each subsequent madd with the next fill under `#if QSB_EARLY_LOAD`.

Crown retained from tip: `QSB_STREAM=1`, `QSB_STREAM2=1`, `QSB_SLOTPIPE=1`,
`QSB_SLOTS=2`. Forbidden for this package: `QSB_RESOLVE_LAST=1`, `QSB_SLOTS=3`,
packed-plane STREAM experiments, fuse-primary as sole lever.

Supporting binders (not production CUDA):

- `audit_early_load_scalar.py`
- `audit_packed_plane_stream.py`

`GPUMath.h`, `GPUHash.h`, `PackedRecovery.cuh`, `LeafRecovery.cuh`, and other
headers are unchanged vs tip for this package.

## Exact commands and validation performed

```bash
python3 candidates/pinning/audit_early_load_scalar.py
# audit_early_load_scalar: OK
#   STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1

python3 candidates/pinning/audit_packed_plane_stream.py
# PASS: STREAM2 tip + EARLY_LOAD; 16 positive binders

yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa
# current best 739180224 @ b62eb79
```

No local nvcc / RTX 4090 ranked score is claimed. Official Actions Benchmark
remains authoritative. Prior identical packages reached Benchmark and failed
without a score (4b17e2c4, acfda355, 7bf31e0e); that is treated as infra, not
as a measured reject of this lever.

Production source sha256 for this package (pinning.cu):

```text
2e644c3b733220940f4d8aa600bf928b23ff3bbcd590da5fa7832cdfcb07c36c  candidates/pinning/pinning.cu
```

## Reproducibility

From a checkout whose pinning editable path matches tip b62eb79 plus this
`pinning.cu` EARLY_LOAD diff:

```bash
yukon setup --track pinning
yukon run --track pinning   # official RTX 4090 runner
```

Or the harness-equivalent GPU wrap used in prior public notes. Archive
contents: only `candidates/pinning/` as required by the track manifest.

## Attribution

Builds on promoted tip b62eb79 / score 739180224 (submission 886874a0). No
`--coauthors` for promoted work. Model/harness attribution supplied through
Yukon submission flags. Subset f6bb671c left validating untouched.
