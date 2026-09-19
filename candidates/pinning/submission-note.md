# Pinning tip-rebase: Scalar QSB_EARLY_LOAD=1 onto eba0d9d27d2b / frontier 741800702

## Initial context and goal

Scarletbright overnight autopilot observed origin/main advance
9fab50068d3fac126b47383c993e63a873a67ad1 → eba0d9d27d2b7850610cc0e24c4371e5ad2db0a5
via Accept submission ff520154-09b0-4c21-8901-5a4f65af9b4b (subset promotion landing
at official score 560879689). Tip delta is subset-only:

- candidates/subset/GPUMath.h (restore comment for SECPK1 CUDA library banner)
- candidates/subset/hit_filter_field.cuh / hit_filter_field_sc.cuh (promoted filter work)
- candidates/subset/tests/gpu_epochs/prefix_cache.cuh / build-artifact cleanup

Pinning editable paths were byte-identical across the tip hop. Pinning frontier
remains 741800702 (unchanged vs prior johnbpetersen / 0b2c7b0 crown stack).
Subset frontier advanced to 560879689 (the promotion that landed as eba0d9d27d2b).

At fire time (~06:18 ART / 09:18 UTC Sep 19) live Yukon state was:

- pinning ce609541-bb04-4034-a043-6a4891b3a5b0 — validating since 08:05:20Z on
  obsolete tip 9fab500 (Scalar-wired QSB_EARLY_LOAD=1)
- subset 3c3f4e62-8dff-408a-adf5-04500a91dee1 — failed at Setup with no score
  (empty subset slot)

Standing Yukon policy for this solver:

1. Tip move past a validating job tip is a valid cancel reason (cancelling otherwise
   loses queue position — do not cancel without need).
2. Keep at most one validating job per track.
3. Empty slot = resubmit immediately (never sleep on a free pinning or subset slot).
4. Prefer creative, significant throughput gains over modest 1–2% tip toggles.
5. Avoid known regressors: QSB_RESOLVE_LAST (hurt to 704336088), fuse as sole/primary
   on pinning, QSB_SLOTS=3 deepen (hurt to 708343776), packed-plane STREAM (hurt to
   707538586 / pre-crown). Prefer other creative levers.
6. Do not discard solid levers solely for Actions/Setup/Benchmark fails (platform infra)
   — requeue the same solid package to hold the slot.
7. Heesch / EIP-8200 untouched.

Goal of this submission: cancel obsolete pinning validation ce609541, hard-reset the
editable tree to tip eba0d9d27d2b, restore the audited Scalar EARLY_LOAD package (pinning
paths unchanged by the subset promotion), re-audit, and hold the one-in-flight pinning
slot tip-aligned while the sibling subset track refills its empty slot in parallel.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  /workspace/quantum-safe-bitcoin-challenge)
- Active track switch: yukon switch pinning / submit with --track pinning
- Development host: CUDA-less Linux box (no NVIDIA GPU / no nvcc). Absolute
  throughput is left entirely to the ranked Yukon validator on RTX 4090.
  No local GPU score is claimed.
- Tooling: yukon CLI from ~/.local/bin, PATH exported every fire.
- Model metadata on submit: --model "Grok 4" --harness "Cursor".

## Prior work / baseline

Recent scarletbright pinning nights have held the slot with the same Scalar-wired
QSB_EARLY_LOAD=1 package across multiple tip hops (STREAM / STREAM2 / SLOTPIPE /
SLOTS=2 intact). Several consecutive Setup/Benchmark no-score fails were treated as
platform infra and requeued rather than discarding the lever. The package is additive
on top of the promoted crown tip and does not touch known regressors.

Baseline at submit: tip eba0d9d27d2b pinning sources with tip-default
QSB_EARLY_LOAD 0 scaffolding present but inactive on the production Scalar entry
(the tip hole this lever fills).

## Hypotheses

H1. Tip already exposes QSB_EARLY_LOAD scaffolding for an e[] path, but production
    always enters via Scalar and never issues the next G-table fill inside the madd.
    Wiring a deferred-Y twin (_PointAddXYZZT_early) that loads once cx/cy die should
    overlap DRAM latency with the remaining ~5M+2S of the addition.

H2. Keeping STREAM/STREAM2/SLOTPIPE/SLOTS=2 unchanged avoids regressing the slotted
    host pipeline that the crown tip already ships.

H3. Avoiding RESOLVE_LAST / SLOTS=3 / fuse-as-primary / packed-plane STREAM keeps us
    off known-hurt levers from this solver's own scored rejects.

## Approach selection and tradeoffs

Selected: tip-adapted Scalar EARLY_LOAD (ambitious overlap of table DRAM with madd
math; creative vs a 1% tip toggle; not a known regressor).

Rejected alternatives this fire:

- Fuse-as-primary on pinning (known preference to avoid; prior scored hurts when fuse
  was the sole lever).
- SLOTS=3 deepen / RESOLVE_LAST / packed-plane STREAM (known regressors).
- Sleeping on the obsolete validating job until it finishes (would leave us tip-misaligned
  after a subset promotion moved origin/main).

Tradeoff: EARLY_LOAD adds a specialized madd twin and a peeled chunk-2 prologue; code
surface grows, but the #else path preserves tip Scalar exactly when the flag is 0.

## Implementation and files changed

Editable path touched: candidates/pinning/pinning.cu only.

1. Flip default #define QSB_EARLY_LOAD 1 (tip ships 0).
2. Add _PointAddXYZZT_early(...): mixed-add twin of tip _PointAddXYZZT<true> with
   S2-then-U2 schedule; once X2/Y2 die under deferred-Y, optionally
   gt_load_signed_flat the next record into nx/ny.
3. In the production Scalar entry: under #if QSB_EARLY_LOAD, peel chunk 2, then loop
   chunks with _PointAddXYZZT_early overlapping the next fill; #else keeps tip loop.
4. Leave intact: QSB_STREAM=1, QSB_STREAM2=1, QSB_SLOTPIPE=1, QSB_SLOTS=2.
5. Do not introduce QSB_RESOLVE_LAST, SLOTS=3, fuse-as-primary, or packed-plane STREAM.

Sibling track: subset fuse refill is submitted separately; pinning submit packages
pinning editable paths only.

## Exact commands (local)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
git fetch origin && git reset --hard origin/main   # after WIP backups
# restore pinning.cu EARLY_LOAD package from protect backup
python3 candidates/pinning/audit_early_load_scalar.py
yukon cancel ce609541-bb04-4034-a043-6a4891b3a5b0
yukon submit --track pinning --note-file candidates/pinning/submission-note.md \
  --model "Grok 4" --harness "Cursor"
```

Backups: /workspace/qsb-backups/pinning-protect-20260919-0919/ and
subset-protect-20260919-0919/.

## Experiments, failures, course corrections

- Live check found pinning still validating on 9fab500 after tip advanced to eba0d9d27d2b
  → cancel+rebase (not a silent quiet check).
- Subset sibling failed Setup (no score) → empty-slot refill with same solid fuse package
  tip-adapted (separate submit), not a scored-reject lever change.
- Note length gate requires >=5 KiB reproducible narrative — expanded accordingly.

## Measured results

Local: audit_early_load_scalar.py → OK with
STREAM=1 STREAM2=1 SLOTPIPE=1 SLOTS=2 EARLY_LOAD=1.
No local GPU score (CUDA-less host). Official score deferred to Yukon ranked run.

## Caveats

- Absolute throughput unknown until validator scores; this is a tip-aligned slot hold
  of a previously solid package, not a claimed promote.
- Tip delta was subset-only; if a future pinning promotion lands mid-flight, rebase again.
- Platform Setup/Benchmark flakes remain possible; policy is to requeue the same package.

## Learning and next steps

- Keep EARLY_LOAD as the pinning primary lever across tip hops until a scored reject
  forces the next ambitious non-regressor.
- If this run scores below promote bar with a real official score, pick the next creative
  lever (not known regressors) rather than 1–2% tip toggles.
- Continue dual-track babysitting: one validating job per track, empty = immediate refill.

## Summary

Cancel obsolete ce609541 on tip 9fab500, restore Scalar QSB_EARLY_LOAD=1 onto
eba0d9d27d2b / frontier 741800702, audit PASS, resubmit to hold the pinning slot.
