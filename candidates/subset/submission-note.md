# Subset: __ldg + DEFER_Y + host-drain on fkiene outlined-last tip (safe rebase)

Effort: high. Agent: Cursor. Host has no NVIDIA GPU / no `nvcc`; absolute
throughput is left to the ranked validator. Local checks are CPU verifier smoke
on a CUDA-less box.

## Initial context and goal

Subset track of `eigenlabs/quantum-safe-bitcoin-challenge`. Working directory
`/workspace/quantum-safe-bitcoin-challenge`. PATH includes `$HOME/.local/bin`.
Account: scarletbright.

Live promoted subset record at preparation:

- Submission `2c71a386` / **fkiene** / tip `ba418f2` / official score
  **539,150,559** verified candidates/s on the ranked RTX 4090.
- Prior crown was `591a2239` / i34-9 / **536,484,898**. The new crown outlines
  the last-window exact-Y XYZZ mixed-add as `_PointAddXYZZ_def_last` so the
  rolled madd never carries the Y2*ZZZ3 multiply.
- Shared branch tip after this subset promote is `ba418f2` (pinning score
  frontier unchanged at **702,050,398**; pinning dirty WIP restored after sync).

Obsolete scarletbright subset validation `45210302` (queued on the prior
536.5M tip) was cancelled solely for this frontier rebase. Pinning validation
`702e3b6f` remains validating on frontier 702050398 and is **not** cancelled.
Heesch / EIP-8200 untouched.

Promote bar for a ≥1% lift over 539150559 is approximately **544,542,065**.
Schema currently reports `minScoreImprovementBips = 0`; this archive still
targets a meaningful ≥1% lever family.

## Environment and setup

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/quantum-safe-bitcoin-challenge
yukon switch subset
# cancelled 45210302 only; pinning 702e3b6f left validating
yukon sync --force
# restored pinning QSB_L2_SKIP=1 dirty from backup after sync
./setup.sh subset
QSB_GRINDER=cpu QSB_ZEROS_N=10 QSB_SECONDS=3 QSB_MODE=fixed_time ./benchmark.sh subset
```

Sync restored editable `candidates/subset` from promoted `2c71a386` @ tip
`ba418f2` / score 539150559.

## Prior work / baseline read

Tip already owns: squaring-free finish lineage, register-carried digit window
(`gt_direct_digit_p` + `gt_window_advance`), `_ModAddLazy` / `_ModX3Fused`
inside deferred madd, **outlined last-window** `_PointAddXYZZ_def` /
`_PointAddXYZZ_def_last`, `ZLAB_DIRDIG=1`, `ZLAB_HITPATH=1`, `ZLAB_TRIM=1`,
`ZLAB_T14=0`.

Prior scarletbright subset `5f75d186` failed ranked Benchmark in ~16s after
adding rare-branch `gt_recode_setup` and `__restrict__` on madd pointers.
Follow-up `45210302` was the safe stack on the *old* 536.5M tip and was
cancelled only to rebase here.

Missing vs pinning sibling XYZZ stacks that previously validated for hours on
older subset crowns:

| mechanism | tip before this patch | this archive |
| --- | --- | --- |
| `__ldg` table loads in `gt_load_signed_flat` | plain loads | `__ldg` |
| arithmetic-shift sign mask in `gt_digit_idx` | compare/ternary | arithmetic shift |
| `DEFER_Y` template specialization | tip twin functions (outlined last) | `template<bool DEFER_Y>` shared body + tip-named wrappers (`_def` / `_def_last`) |
| host-drain (remove redundant `cudaDeviceSynchronize` before blocking hit D2H) | present on three host loops | drained; gtable sync kept |
| rare-branch `k≥n` reduce | tip branchless | **unchanged tip branchless** |
| `__restrict__` on madd limbs | no | **not added** |

## Hypothesis and approach

**Selected:** XYZZ hot-path codegen + host-drain, adapted to fkiene's outlined
last-window tip. Tip already split interior vs last add; this archive keeps
those call sites and specializes the shared arithmetic via `DEFER_Y`, adds
`__ldg` + arithmetic-shift digit index, and drains redundant host syncs.

**Excluded:** rare-branch recode; restrict on madd; compact residual schedules;
Heesch; pinning edits; cancelling `702e3b6f`.

## Implementation

Editable surface only: `candidates/subset/`.

### `GPUMath.h`

- Replace tip twin bodies with `template<bool DEFER_Y> _PointAddXYZZ_def_body`
  plus `__forceinline__`.
- Keep tip `_ModAddLazy` / `_ModX3Fused` body byte-identical (verified op-for-op
  against `ba418f2` tip `_def` / `_def_last`).
- Thin wrappers `_PointAddXYZZ_def` → `<true>` and `_PointAddXYZZ_def_last` →
  `<false>` so tip FixedBase call sites stay unchanged.
- No `__restrict__` on madd limb pointers.

### `tests/gpu_epochs/tree.cu`

- `gt_load_signed_flat`: `__ldg` on the four `ulonglong2` loads.
- `gt_digit_idx`: arithmetic-shift sign mask.
- Host-drain on short-epoch HITPATH and sibling host loops: remove redundant
  `cudaDeviceSynchronize` before blocking hit D2H; error-check the D2H itself.
  One-time `kernel_build_gtable` sync retained.
- `gt_recode_setup`: **left as tip branchless reduce** (byte-identical to tip).
- FixedBase stream: tip's `_PointAddXYZZ_def` / `_def_last` call sites preserved.

## Local results

| Check | Result |
| --- | --- |
| Frontier | 539,150,559 (`2c71a386` / `ba418f2`) |
| ≥1% bar | ≈ 544,542,065 |
| `./setup.sh subset` verifier smoke | PASS |
| CPU grind N=10 / 3s | ran; 1 verified hit / 40 candidates in 3.1s on CPU (not a throughput claim; variance gate rejects as expected) |
| Diff scope | only `GPUMath.h` + `tests/gpu_epochs/tree.cu` under `candidates/subset/` |
| Pinning `702e3b6f` | still validating; not cancelled |
| Prior failure `5f75d186` | CI Benchmark exit 1 ~16s; this archive drops restrict + rare-recode |
| Prior `45210302` | cancelled only for rebase onto 539.2M tip |

No local GPU throughput claimed. Ranked RTX 4090 is authoritative.

## Scope and safety

- Track: subset only. Editable paths packaged: `candidates/subset`.
- Pinning `702e3b6f` left validating; pinning dirty `QSB_L2_SKIP=1` restored
  after sync and not part of this archive.
- Heesch / EIP-8200: not touched.
- Fail-mode avoidance: no `__restrict__` on madd; no rare-branch
  `gt_recode_setup`; tip lazy/X3 and outlined-last call shape preserved.

## Dual watch

After submit, watch the new subset SHA together with pinning `702e3b6f`
(validating on 702050398). Do not cancel pinning.

## Reproducible reasoning narrative

1. Confirmed frontier `yukon benchmark show …/subset` → current best
   539150559 @ source `ba418f2` (fkiene `2c71a386`).
2. Confirmed pinning best still 702050398; shared tip moved due to subset
   promote only.
3. Cancelled only `45210302-b1b0-4143-925e-caa0f9306bfe` (validating on old tip).
4. Backed up WIP to `wip-backup-frontier-rebase-*` and pinning dirty aside.
5. `yukon sync --force` on subset → editable from `2c71a386` @ `ba418f2`.
6. Restored pinning `QSB_L2_SKIP=1` dirty after sync.
7. Studied tip: already has `_PointAddXYZZ_def` / `_def_last` outline; still
   missing `__ldg`, arithmetic digit idx, host-drain, and DEFER_Y template
   forceinline specialization of the shared body.
8. Applied safer lever stack; verified tip arithmetic ops and
   `gt_recode_setup` unchanged; smoked; submitted.

## Learning and next steps

- Outlined-last on tip already captures most of the old loop-split DEFER_Y
  benefit; template+forceinline still removes residual twin-body divergence
  risk and matches pinning sibling codegen style.
- If ranked gain is thin, next levers should stay off the 5f75d186 fail pair
  (restrict / rare recode) and prefer measured host/path tweaks or table-load
  hierarchy experiments with local GPU evidence.

## Caveats

- No local RTX 4090; claimed improvement is hypothesized from prior validating
  evidence of this lever family on older crowns, not measured here.
- Host-drain changes completion signaling; D2H error checks replace the
  removed synchronize+GetLastError pair on the three drained loops.

