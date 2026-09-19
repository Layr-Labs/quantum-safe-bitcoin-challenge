# Subset tip-rebase: xlib fuse onto 9fab500 / frontier 557779951

## Initial context and goal

Scarletbright overnight autopilot observed `origin/main` advance
`0b2c7b064b6de4c55816ffb1ade62195b0c450a2` → `9fab50068d3fac126b47383c993e63a873a67ad1`
via Accept submission `1b1957cd-aa64-411b-8f8f-c1d8eda44bac` (solver
anamdongparkjinhyeong; subset official score **557779951**). Tip delta is
**subset-only**:

- `candidates/subset/GPUMath.h` (−1 line: remove comment
  `// 256(+64) bits integer CUDA libray for SECPK1`)
- `candidates/subset/tests/gpu_epochs/prefix_cache.cuh` (+11): wrap
  `QSB_PREFIX_CACHE`, `qsb_prepare_prefix_cache`, and `qsb_fast_window_hash` in
  `#if !ZLAB_TRIM` so the GPU-enum-only prefix path is not emitted into PTX /
  device globals when `ZLAB_TRIM=1` (shipped default)

Pinning editable paths were unchanged. Pinning frontier remains **741800702**.
Subset frontier is now **557779951** (this promotion).

At fire time scarletbright held one validating job per track on obsolete tip
`0b2c7b0`:

- pinning `f9d19ebc-f683-49e4-bc25-28ab55a962a7` — Scalar `QSB_EARLY_LOAD=1`
- subset `ca3fcff4-0953-4486-bf26-3c9b6b546ce1` — xlib fuse

Standing policy: tip move is a valid cancel/rebase reason; one validating job
per track max; compose fuse onto the new tip subset sources — do **not** blindly
wipe tip’s new subset bits (`prefix_cache.cuh` ZLAB_TRIM guards must remain).
Prefer significant gains; avoid known-bad subset packs (`__restrict__` /
rare-branch recode). Platform Setup/Benchmark fails may be infra — do not
discard the fuse lever solely for Actions exit. Heesch / EIP-8200 untouched.

Goal: cancel obsolete subset validation `ca3fcff4`, hard-reset to tip `9fab500`,
restore tip-adapted xlib fuse (`QSB_FUSE_MULSUB` + `QSB_FUSE_SQRADDSUB2` with
`_ModSqrAddSub2` brace fix) into tip’s templated `_PointAddXYZZ_def<DEFER_Y>`,
preserve tip `prefix_cache.cuh`, audit, and resubmit so the one-in-flight subset
slot stays filled and tip-aligned.

## Environment and setup

- Challenge repo: Layr-Labs/quantum-safe-bitcoin-challenge (local clone under
  `/workspace/quantum-safe-bitcoin-challenge`).
- Solver handle: scarletbright (Yukon/GitHub).
- CLI: yukon with PATH including `~/.local/bin`.
- Tracks: pinning and subset in parallel one-in-flight slots. Heesch / EIP-8200
  never touched unless asked.
- No local NVIDIA GPU / `nvcc`. Throughput is left to Yukon RTX 4090 fixed-time
  subset harness. Local work: compose, static binder + modular-identity audit,
  Yukon package.
- Schema v2: controlled `git reset --hard origin/main` then restore levers;
  WIP protected to `/workspace/qsb-backups/subset-protect-20260919-0802/` (and
  sibling pinning-protect) before reset.

## Prior work / baseline

Standing scarletbright subset package is xlib fused modular reductions in
`candidates/subset/GPUMath.h`:

- `QSB_FUSE_MULSUB=1` (default ON; `-D=0` recovers tip field path)
- `QSB_FUSE_SQRADDSUB2=1` (default ON; includes `_ModSqrAddSub2` closing-brace fix)

Wiring targets tip’s single templated deferred madd
`template<bool DEFER_Y> _PointAddXYZZ_def(...)`:

- Under `QSB_FUSE_MULSUB`: `_ModMulSubCore(R, S2, ZZZ1, Y1)` replaces separate
  S2 multiply + R subtract for `(Y2+Yoff)*ZZZ1 - Y1`
- Under `QSB_FUSE_SQRADDSUB2`: `_ModSqrAddSub2(T, R, PPP, Q)` builds
  `X3 = R^2 + PPP - 2V` in one fused path
- `-D=0` tip paths retained (`_ModMult(S2, ZZZ1)`, `_ModAdd256(T, T, PPP)`, …)

This is the same package previously tip-aligned on `0b2c7b0` as validating
`ca3fcff4` (and earlier empty-slot / Setup-Benchmark no-score requeues). Tip hop
`0b2c7b0..9fab500` rewrote subset sources only at the comment line and
`prefix_cache.cuh`; field-math edit sites for fuse remain the compose surface.

## Hypotheses and approach selection

Hypothesis: fuse composition remains tip-valid on `9fab500` after applying the
one-line tip comment removal to the fuse WIP and keeping tip’s new
`prefix_cache.cuh` ZLAB_TRIM guards intact. The promotion’s subset score jump
does not invalidate the fuse binder; it does require tip-alignment so the
validator is not racing an obsolete tip.

Alternatives considered and rejected for this fire:

- Blind restore of pre-tip `prefix_cache.cuh` from backup — rejected. Would wipe
  tip’s ZLAB_TRIM PTX/global fix that landed with `1b1957cd`.
- Invent a new subset primary lever before re-holding the slot — rejected.
  Obsolete tip under a validating job is an operational rebase first.
- Keep obsolete validator `ca3fcff4` — rejected.
- Reintroduce known-bad subset packs (`__restrict__` / rare-branch recode) —
  rejected (prior measured hurts).
- Use fuse as sole/primary on the pinning track — rejected (pinning policy;
  pinning keeps Scalar EARLY_LOAD).

Chosen: cancel `ca3fcff4`, hard-reset to `9fab500`, restore fuse WIP GPUMath,
delete the tip-removed comment line, leave tip `prefix_cache.cuh` untouched,
audit, submit.

## Implementation details

Operational sequence (America/Buenos_Aires):

1. Protect WIP → `subset-protect-20260919-0802` / `pinning-protect-20260919-0802`.
2. `yukon cancel ca3fcff4-0953-4486-bf26-3c9b6b546ce1` (and sibling pinning
   `f9d19ebc-…`).
3. `git reset --hard origin/main` → `9fab50068d3fac126b47383c993e63a873a67ad1`
   (tip `prefix_cache.cuh` with ZLAB_TRIM now in tree).
4. Restore fused `GPUMath.h` from subset-protect backup; remove the single tip
   comment line so the file matches tip’s non-fuse delta plus fuse additions.
5. Confirm `prefix_cache.cuh` still contains `#if !ZLAB_TRIM` guards (not
   overwritten).
6. Restore `audit_fuse_reduction.py`; run binder + congruence.

Package invariants after compose (binder-checked):

- `#define QSB_FUSE_MULSUB 1` and `#define QSB_FUSE_SQRADDSUB2 1`
- exactly one `_ModMulSubCore` and one `_ModSqrAddSub2` definition
- exactly one call site each inside templated `_PointAddXYZZ_def`
- tip `-D=0` paths retained; no `gt_recode_setup`; single `_BinarySearch`
- tip comment `256(+64) bits integer CUDA libray…` absent
- tip `prefix_cache.cuh` ZLAB_TRIM intact

## Evaluation and results

Local: `python3 candidates/subset/audit_fuse_reduction.py` →
`PASS: subset fuse reduction binder + congruence; cases=81331`. No local GPU
score. Official score left to Yukon ranked validation.

Frontier context at submit (live-checked): subset **557779951**; tip SHA
**9fab500**; cancelled obsolete id **ca3fcff4**; promotion **1b1957cd**.

## Risks, limitations, and next steps

- Setup/Benchmark Actions exits without score: treat as infra; requeue on tip
  rather than discarding fuse.
- If a future tip rewrites `_PointAddXYZZ_def` shape, re-port fuse call sites
  rather than blind file restore.
- Do not regress tip’s ZLAB_TRIM prefix_cache change when rebasing.
- Next creative subset levers only after tip-aligned slot is held; avoid
  known-bad packs.

## Reproducibility

- Tip: `9fab50068d3fac126b47383c993e63a873a67ad1`
- Cancelled: `ca3fcff4-0953-4486-bf26-3c9b6b546ce1`
- Lever: `QSB_FUSE_MULSUB=1` + `QSB_FUSE_SQRADDSUB2=1` (`_ModSqrAddSub2` brace fix)
  into `_PointAddXYZZ_def<DEFER_Y>`; tip `prefix_cache.cuh` preserved
- Audit: `audit_fuse_reduction.py` PASS (81331 cases)
- Backups: `/workspace/qsb-backups/subset-protect-20260919-0802/`
- Model/harness: Grok 4 / Cursor
- Editable path packaged: `candidates/subset/` (schema v2)
