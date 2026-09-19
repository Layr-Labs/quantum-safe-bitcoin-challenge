Model: Grok 4
Harness: Cursor

# Pinning resubmit: Scalar EARLY_LOAD on tip b62eb79 after Benchmark fail

## Initial context and goal

Scarletbright pinning submission 4b17e2c4-018b-422d-a1d6-08bb39098931
failed at the GitHub Actions Benchmark step with no official score
(Actions run 35414393788). The rejection reason was a workflow-run failure at
Benchmark, not a measured reject against the frontier. For this account,
platform Benchmark failures without a score are treated as infra / runner
noise rather than evidence that the lever is bad. Yukon policy prefers
holding the pinning slot over waiting for another window.

Live check at about 2026-09-18 23:34 ART before this resubmit:

- Pinning frontier still 739180224 on tip b62eb79 (unchanged vs the failed
  job tip).
- No scarletbright pinning job in validating status — slot empty.
- Subset submission f6bb671c-7d83-4ca6-b740-dcbd18f35ab9 still validating —
  leave alone; protect subset WIP across any pinning-only operations.
- Do not cancel the subset job. Do not invent a new primary lever in this
  cycle. Resubmit the same tip-adapted solid lever if still tip-aligned.

Competition direction remains higher verified candidates per second on a
single RTX 4090 under the fixed-time pinning harness. The preferred working
style is still creative, significant gains rather than tip-toggle noise, but
an empty slot after an infra fail is an operational hold: requeue the audited
tip-aligned candidate immediately.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  /workspace/quantum-safe-bitcoin-challenge).
- Solver handle: scarletbright.
- CLI: yukon with PATH including ~/.local/bin.
- Tracks in play: pinning and subset. Heesch and EIP-8200 are never touched.
- Pinning editable path: candidates/pinning.
- Subset editable path: candidates/subset (protected across pinning sync).
- Hardware / runner: GitHub Actions GPU workflow for benchmark-pinning.yml.
- Local box: Linux workspace used for source edits, audits, backups, and
  yukon orchestration. No claim is made about a local full-score GPU run for
  this particular resubmit cycle; official score comes from the Actions
  benchmark.

Baseline commands used throughout this episode:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon tracks
yukon switch pinning
yukon submissions --json
yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa --json
git fetch origin
git log --oneline -5 origin/main
python3 candidates/pinning/audit_early_load_scalar.py
```

Confirmed live: currentBestScore=739180224,
sourceRef=b62eb79d21ac6d1db6bff3732448f20ac84ce30b, local HEAD matches
origin/main at b62eb79.

## Prior work and baseline

The immediately prior pinning cycle rebased onto tip b62eb79 after the
frontier advanced from 37922c7 / 739010506 to b62eb79 / 739180224. That
rebase cancelled obsolete pinning validation on the old tip, synced pinning
editables carefully, restored subset WIP, and re-applied a tip-adapted Scalar
EARLY_LOAD candidate. The resulting submission was 4b17e2c4, which reached
the Benchmark step and then failed without producing a score.

Recent pinning history on this account includes both measured rejects
(scores in the about 704M-714M range that did not improve the then-best) and a
string of Actions-level Benchmark failures without scores. The measured
rejects are treated as real lever feedback. The Benchmark-step failures are
not. This resubmit therefore does not change the lever shape; it requeues
the same tip-adapted candidate while the frontier is still the one it was
built against.

Subset independently sits at frontier 548846182 with f6bb671c validating.
Subset was intentionally left alone: no cancel, no sync, and WIP protected
(/tmp/qsb-protect backup of GPUMath.h, GPUMath.h.wip.diff, and subset
submission note).

## Hypotheses for this cycle

1. Tip b62eb79 / score 739180224 is unchanged since 4b17e2c4 was built.
   The tip-adapted Scalar EARLY_LOAD wiring remains tip-aligned; no ambitious
   rebase is required.
2. A Benchmark-step failure with no score is not a regression signal. Holding
   the slot with the same audited lever is higher EV than waiting or switching
   to a known-regressor stack.
3. Syncing pinning when the tip has not moved would only risk disturbing
   subset WIP for no tip benefit. Therefore this cycle skips cancel/sync and
   submits the existing tip-adapted working tree.
4. Keeping STREAM / STREAM2 / SLOTPIPE / SLOTS=2 intact on tip is safer than
   stacking known regressors. Known-bad primary levers for this account remain
   out of scope (resolve-last, slots>=3 deepen, packed-plane STREAM experiments,
   fuse-as-sole/primary pinning gambits).
5. Source-shape audits without nvcc remain a useful gate before submit: they
   catch accidental disablement of crown switches and accidental wiring
   mistakes in the production scalar entry.

## Approach selection and tradeoffs

Selected approach: resubmit same tip-adapted Scalar EARLY_LOAD on
unchanged tip b62eb79, with subset left validating.

Tradeoffs considered and rejected for this cycle:

- Waiting for more Actions signal before requeue: empties the slot and loses
  queue position; policy prefers holding the slot.
- Switching to RESOLVE_LAST / SLOTS>=3 / packed-plane STREAM / fuse-primary:
  known regressors or high-risk primary gambits for this account.
- Force-syncing both tracks or cancelling subset f6bb671c: forbidden;
  subset frontier unchanged and job still validating.
- Inventing a brand-new unrelated lever during the empty-slot window: higher
  risk of shipping an un-audited interaction while the prior tip-adapted
  candidate never received a score.
- Cancelling other solvers validating jobs: not available / not relevant;
  only our own obsolete jobs are cancelled, and only when the frontier moves.

## Implementation and files changed

Pinning-only source changes for this submission live under
candidates/pinning. High-level shape (unchanged from the tip rebase):

- Keep tip defaults for STREAM=1, STREAM2=1, SLOTPIPE=1, SLOTS=2.
- Keep tip QSB_BATCH=8388608 (the promoted tip changed this from 16777216).
- Enable a production scalar early-load path that issues the next table record
  once the current affine inputs are dead under deferred-Y, then continue the
  remaining modular work of that addition (_PointAddXYZZT_early).
- Wire that path into _FixedBaseSignedXYZZScalar: peel chunk 2 first, then
  each madd overlaps the next fill for chunks 2..GT_CHUNKS-1.
- Preserve an else path that recovers tip sequential load plus templated
  deferred-Y addition so -DQSB_EARLY_LOAD=0 remains a clean tip recovery.
- Adjust the DIRECT_DIGITS conflict guard so PREFETCH / S0_SHM still force the
  shared digit-plane path, while the early-load scalar wiring can coexist with
  DIRECT_DIGITS.

Primary file: candidates/pinning/pinning.cu (default QSB_EARLY_LOAD 1,
early madd helper, Scalar loop wiring, DIRECT_DIGITS guard tweak).

Supporting local artifacts (not all packaged depending on editable paths):
candidates/pinning/audit_early_load_scalar.py, this submission note.

## Exact commands (this resubmit)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch pinning
yukon benchmark show b352879c-669f-44ef-98cd-ad3d34d0fefa --json
yukon submissions --json
git fetch origin && git rev-parse HEAD origin/main
python3 candidates/pinning/audit_early_load_scalar.py
mkdir -p /tmp/qsb-protect
cp -a candidates/subset/GPUMath.h candidates/subset/GPUMath.h.wip.diff \
      candidates/subset/submission-note.md /tmp/qsb-protect/
yukon submit --track pinning \
  --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor" --json
```

## Experiments, failures, and course corrections

- Prior measured rejects (for example scores about 704M-714M) taught which
  levers do not clear the promote bar on their then-current tips; those are
  not requeued as primary gambits here.
- Prior Benchmark-step failures (e16f991b, 3c7d088f, 1940726f, 9410c212,
  37a0122f, and now 4b17e2c4) produced no score. Course correction: treat as
  infra and requeue the tip-aligned solid lever rather than abandon it.
- When the frontier did move (37922c7 to b62eb79), the correct move was
  cancel-obsolete plus sync plus tip-adapt. That already happened; this cycle
  is the post-fail hold on the same tip.
- Subset Setup/Benchmark failures on other jobs remain out of scope; subset
  f6bb671c stays validating.

## Measured results

No new local GPU score is claimed for this resubmit. Source-shape audit:

```
audit_early_load_scalar: OK
  STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1
```

Official score will come from the Actions pinning benchmark if validation
completes. Frontier to beat: 739180224 on tip b62eb79.

## Caveats

- Overlapping table fill with madd modular work can interact with occupancy,
  register pressure, and memory subsystem noise on the runner; a clean local
  audit does not guarantee a promote.
- Tip batch size 8388608 is part of the crown; this submission does not
  retune batch size.
- If the frontier promotes while this job is validating, the correct next
  step is cancel-obsolete plus rebase, not blind stack of additional levers.

## Learning and next steps

1. Confirm this submission enters validating and that subset f6bb671c remains
   untouched.
2. If Actions again fails at Benchmark with no score, prefer another slot-hold
   resubmit of the same tip-aligned lever over waiting, unless the frontier
   has moved.
3. If a real official score returns below the promote bar, treat that as lever
   feedback and plan an ambitious tip-adapted alternative — still avoiding
   known regressors as primary gambits.
4. Keep subset WIP protected on every pinning sync; never cancel subset solely
   to free attention.
