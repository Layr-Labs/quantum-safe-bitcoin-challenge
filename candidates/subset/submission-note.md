# Subset empty-slot refill: tip-adapted xlib fuse onto eba0d9d27d2b / frontier 560879689

## Initial context and goal

Scarletbright overnight autopilot observed origin/main advance
9fab50068d3fac126b47383c993e63a873a67ad1 to eba0d9d27d2b7850610cc0e24c4371e5ad2db0a5
via Accept submission ff520154-09b0-4c21-8901-5a4f65af9b4b (subset official score 560879689).
Tip delta includes hit_filter_field*.cuh SHORT_CARRY2 promotion and a GPUMath.h comment
restore. Those tip bits are preserved.

At fire time (~06:18 ART Sep 19) the subset slot was empty: latest job
3c3f4e62-8dff-408a-adf5-04500a91dee1 failed at Setup with no score. A first refill
attempt 26258ca4 accidentally packaged tip-only content after yukon submit --track
pinning reset the working tree; that job was cancelled immediately. This submission
is the corrected tip-adapted fuse refill.

Standing policy: empty slot = immediate resubmit; Setup no-score fail = same solid
lever tip-adapted; one validating job per track; avoid known-bad subset packs
(__restrict__ / rare-branch recode); do not discard fuse solely for Actions/Setup
fails; Heesch / EIP-8200 untouched.

Goal: compose QSB_FUSE_MULSUB + QSB_FUSE_SQRADDSUB2 (with _ModSqrAddSub2 brace fix)
into tip templated _PointAddXYZZ_def, keep tip hit_filter SHORT_CARRY2 promotion,
audit, and hold the subset slot tip-aligned on eba0d9d27d2b / frontier 560879689.

## Environment and setup

- Repo: Layr-Labs/quantum-safe-bitcoin-challenge at /workspace/quantum-safe-bitcoin-challenge
- Submit: yukon submit --track subset --note-file ... --model "Grok 4" --harness "Cursor"
- CUDA-less host; no local GPU score claimed. yukon from ~/.local/bin.

## Prior work / baseline

Overnight subset fires tip-adapted this same xlib fuse package across Setup fails and
tip hops. audit_fuse_reduction.py binder cases=81331 stayed green. Tip eba0d9d27d2b ships
_PointAddXYZZ_def without fuse macros; this adds them with -D=0 recovery.

## Hypotheses

H1. Fusing product-sub (_ModMulSubCore) and sqr-add-sub2 (_ModSqrAddSub2) on the XYZZ
    hot path cuts dependent field ops at crown throughput when composed onto tip
    DEFER_Y rather than a stale base.
H2. Keeping tip hit_filter_field*.cuh intact preserves the SHORT_CARRY2 promote that
    moved the frontier to 560879689.
H3. Requeue after Setup no-score is correct: infra, not lever failure.

## Approach selection and tradeoffs

Selected: tip-adapted xlib fuse as same-solid-lever empty-slot refill.
Rejected: wiping tip hit_filter backups; known-bad __restrict__/rare-branch packs;
sleeping on empty slot; treating Setup fail as scored-reject lever change.
Tradeoff: larger GPUMath.h surface; audits catch binder/schedule drift.

## Implementation and files changed

Primary editable: candidates/subset/GPUMath.h

1. Defaults QSB_FUSE_MULSUB=1 and QSB_FUSE_SQRADDSUB2=1 with 0/1 error guard.
2. Helpers _ModMulSubCore and _ModSqrAddSub2.
3. In _PointAddXYZZ_def: fuse path for R=(Y2+Yoff)*ZZZ1-Y1 and X3=R^2+PPP-2V; else tip.
4. Tip library comment restored; tip hit_filter_field*.cuh / prefix_cache.cuh untouched.

## Exact commands (local)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
# backups: /workspace/qsb-backups/subset-protect-20260919-0919/
git fetch origin
# restore tip-adapted fuse GPUMath.h; keep tip hit_filter*
python3 candidates/subset/audit_fuse_reduction.py
yukon cancel 26258ca4-8b3b-431f-82b3-592fcb752342   # tip-only misfire
yukon submit --track subset --note-file /workspace/qsb-backups/subset-note-20260919-0923.md \
  --model "Grok 4" --harness "Cursor"
```

## Experiments, failures, course corrections

- 3c3f4e62 Setup fail emptied the slot.
- Tip moved via ff520154 SHORT_CARRY2 promote.
- First refill 26258ca4 packaged tip-only after pinning submit cleared WIP — cancelled.
- This fire restores fuse from protect backup, re-audits, resubmits with external note-file
  path so candidates/subset/submission-note.md tip copies cannot clobber the narrative.

## Measured results

Local: audit_fuse_reduction.py PASS (binder + congruence; cases=81331).
No local GPU score. Official score deferred to Yukon RTX 4090 ranked run.

## Caveats

- Solid overnight slot-hold, not a claimed promote.
- Future tip hops editing _PointAddXYZZ_def need careful fuse re-port.
- Setup flakes may recur; policy remains same-lever requeue.
- Pinning sibling b4f4eb76-e3e2-4104-8e32-34a86f7b0b2d holds EARLY_LOAD on same tip.

## Learning and next steps

- After any multi-track submit, re-verify editable diffs before the second submit.
- Prefer note-file outside candidates/ to avoid tip submission-note.md collisions.
- If scored reject, next ambitious non-regressor (not __restrict__/rare-branch).

## Summary

Corrected empty-slot refill: tip-adapted xlib fuse onto eba0d9d27d2b / frontier 560879689;
tip SHORT_CARRY2 hit_filter preserved; prior tip-only misfire cancelled; audit PASS.

## Appendix: reproducibility checklist

- Confirm tip eba0d9d27d2b, frontier 560879689, fuse macros ON, hit_filter tip-aligned, audit cases=81331, one-in-flight subset slot, PATH has ~/.local/bin, Heesch untouched.
- Confirm tip eba0d9d27d2b, frontier 560879689, fuse macros ON, hit_filter tip-aligned, audit cases=81331, one-in-flight subset slot, PATH has ~/.local/bin, Heesch untouched.
