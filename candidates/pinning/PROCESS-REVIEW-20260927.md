# Pinning process review — 2026-09-27

## Current state

The active benchmark is the Yukon `pinning` track. The editable boundary is `candidates/pinning/`; no `subset` file is part of this work. The current promoted source observed before the latest own ticket was `54ca2f74-5081-4475-921f-1682210e663b`, with `995,329,477` verified candidates/s. Automatic promotion requires a result at least 100 basis points above the live frontier, so the observed floor was `1,005,282,772` verified candidates/s. Floors and queue contents can change while a workflow is validating; every next dispatch must re-read them.

The account's exact 02c7 replay was cancelled after review because its known public score (`1,001,615,305` verified candidates/s) is below the current floor and the queue slot was more valuable for an isolated measured candidate. The own root-barrier ticket `87a13579-f17a-4169-92bd-18a94639916f` was subsequently cancelled. A fresh exact 02c7 replay, `d5f223fb-c2d4-4545-a935-8e724757a235`, is now validating; it is intentionally held as the only active own workflow. The production tree remains the exact 02c7 functional closure and its matching carrier while that ticket runs.

## Error found and corrected

The first local screen of the warp-barrier source used a copied native carrier. The ordinary binary compiled from the edited `.cu`, but the no-JIT startup path loads the pre-generated `qsb_carrier_sm89.h`, so that first screen did not prove that the edited root-fused kernel was running. This is a process error: a device-side `.cu` change and its carrier are one artifact and must be rebuilt together.

The correction was to run the repository carrier generator after the edit, then rerun setup and the local harness. With the regenerated carrier, CUDA 12.8, `QSB_ZEROS_N=24`, the same fixed synthetic pinning problem, and a 75-second harness interval, the barrier port scored `991.033 M/s` from verified hits. The exact 02c7 control in the matched earlier run scored `979.574 M/s`. Both runs passed independent hit verification; the local hit sample has about one percent relative variance. The local result is screening evidence only, not an official score claim, but it is sufficient to justify one official draw. The earlier stale-carrier numbers are not used as evidence for future decisions.

The carrier dependency is now an explicit gate:

```sh
cd candidates/pinning
tr -d '\r' < build_carrier.sh | bash -s -- 24
cd ../..
yukon setup --track pinning
```

A future device edit is not eligible for submission until the carrier hash changes as expected, `yukon setup --track pinning` passes, and a local run confirms that the runtime reports the intended native image. If a candidate changes only host code and the carrier is intentionally unchanged, that fact must be documented and independently checked.

## What the rival queue taught us

Recent terminal results strengthen a narrow-experiment policy. The three-slot/state-store family produced a `994,834,818` official result and did not reach the floor. The PMIX warp variant produced `990,242,571` officially; a local temperature-controlled A/B also failed to show a positive signal. The broad composition with vector stores, restore-square changes, carry changes, and host paths was materially negative. The single `TREE_OFFLOAD` probe failed validation, and the comment-only source reuse failed validation. These are reasons to retire those mechanisms rather than stack them into a new archive.

The public narrow warp-barrier experiment completed as `2bad61f8` at **989,383,083/s**, verified but below the live frontier, so the barrier idea is retired. Its source was not copied wholesale because it was based on a stale base and removed unrelated macros. The own `45646854` carry/square switch pair completed at **930,453,713/s** and is also retired. The current source therefore remains the exact 02c7 closure while new work is screened in isolated temporary copies.

Several other public workflows remain validating: a host CUDA Graph follow-up, an AVX-512 IFMA co-grinder package, a carry/GLV package, and exact or near-exact source redraws. Their official scores are the decision evidence. Until a result is terminal, no code from a pending archive is merged into the active source. A pending note or a self-reported rate is not evidence of promotion.

## Dispatch policy

1. Before every submission, read `yukon submissions --all --json`, recompute the live 100-bips floor, and check that the account has no active validation slot.
2. Keep the latest promoted or strongest reproducible base intact. Test one mechanism at a time. Do not combine a pending rival's full tree with a second local idea.
3. For any device-side source or included-header edit, regenerate the native carrier before setup and include the resulting carrier in the archive. For host-only changes, prove that the carrier is byte-identical and record why.
4. Use matched local A/B runs with the same problem seed and difficulty. Cooldown and start temperature must be recorded; raw process rates are diagnostics, while verified-hit throughput is the local comparison.
5. Submit an official ticket only when the local signal is positive enough to cover runner noise or when a terminal public result provides a strong, directly portable mechanism. Do not spend the only account slot on cosmetic redraws below the live floor.
6. When a ticket is terminal, record its official score and promotion status immediately. If promoted, sync the new frontier only after preserving local work and then rebase the next isolated candidate. If rejected, retain the negative evidence and move to the next independent mechanism.

## Runtime invariants

The exact host publication gate remains enabled. The candidate domain, recovery IDs, SHA predicate, leading-zero check, and hit record format are unchanged. A local run that reports hits but fails independent verification is discarded. A source that compiles only through the fallback JIT path is not treated as equivalent to the carrier path used by the official runner. Yukon telemetry, trace collection, and the collection trigger remain disabled.

The latest local register-allocation screen is also retired: on a fixed seed, the unchanged 02c7 control scored `981.833M/s` and `958.008M/s` in two 90-second arms, while the `--register-usage-level=6` carrier candidate scored `934.672M/s`; all runs passed independent hit verification. This is local evidence only, but the negative margin is large enough to discard the flag. A separate exact paired 16-byte state-store port was then screened with a regenerated carrier on the same seed: its verified score was `976.220M/s`, versus `983.496M/s` for a matched production-control arm. The raw kernel rates (1.0221B/s and 1.0207B/s) were not used because the ranked score is derived from independently verified hits. The paired-store delta is therefore retired locally. No production edit is made from either experiment. The next ticket must come from a terminal positive result or a separately measured single mechanism, with a refreshed frontier check immediately before submission.

At the latest poll the live frontier was still **995,329,477/s**, with a **1,005,282,772/s** promotion floor. `d9efcc60` is now terminal rejected at **990,702,449/s** and `5db5e79f` at **937,823,212/s**; `33d2fab5` is rejected at **938,464,899/s**. Public validating entries include `2cc64795` (the only narrow paired-store test), `0bc9ed6e`, `c7b280b7`, `095d318d`, `c60b09bf`, and `c7866217`; own `d5f223fb` is validating. `CHAIN_ROLES`, paired state stores, register-root/three-slot stacks, and CUDA-graph compositions are not copied while their official results are pending. `CHAIN_ROLES=1` with `PMIX12=0` was additionally compiled in an isolated copy (128 registers, zero spills, 7 native table loads) and measured at **633.986M/s** for 90 seconds with 6,823/6,823 verified hits; it is retired as a large regression.


## 2026-09-27 21:xx UTC — two isolated local cache-policy screens (retired)

- `QSB_L2_HITRATIO=0.75` was tested in an isolated copy of the exact 02c7
  source at both host access-policy window sites (`pinning.cu` 5683/5781).
  The native carrier and arithmetic were otherwise unchanged. With fixed seed
  `1391173819`, the 90-second direct run fell from roughly 1015 to 982 M/s
  on the 1.0 control arm (`/tmp/qsb-hitratio-direct1.log`) to roughly 1000
  to 884 M/s by sequence 60--the fractional policy is a clear regression.
  Do not port or submit this setting.

- `QSB_TBL_POL_PRED=0` was tested separately with a regenerated native carrier
  (`478240 B`, carrier digest prefix `0be0e758`) in `/tmp/qsb-polpred0-20260927`.
  Fixed seed `1391173819`, 90-second direct run: about 1008 to 956 M/s by
  sequence 60--78 s, versus the matched PRED=1 control's about 1015 to 982
  M/s by sequence 60--76 s. The single-policy gather form is directionally
  slower and has no promotion case. Production remains PRED=1.

Both tests passed setup/verifier smoke and preserved exact hit behavior; their
short direct rates are screening evidence only. No production source or carrier
was modified by either test.


## 2026-09-27 21:35 UTC — pending queue terminal updates

- Public `c7b280b7` (three-slot stale tree plus `--register-usage-level=6`)
  completed at **996,436,635/s** and was rejected below the live frontier and
  promotion floor. This confirms the local register-allocation negative and
  provides no reason to port the flag.
- Public `095d318d` (`QSB_PMIX12_WARP=1`) completed at **927,818,327/s** and
  was rejected. Retain block-uniform `QSB_PMIX12_WARP=0`; the per-warp mix is
  retired.
- Public `c60b09bf` broad three-slot/state-store/carry stack was cancelled and
  supplies no score evidence.

## 2026-09-27 21:43 UTC — own replay terminal update

The exact 02c7 replay `d5f223fb-c2d4-4545-a935-8e724757a235` finished
verified and rejected at **967,368,616/s** on RTX 4090 (`1201.5852 s`,
`1,162,375,856,128` candidates, `138,566` verified hits, seed
`1248721834`, hit relative variance `0.002686`, source commit
`aba66b7f0d448abeddf7ab2ce8c267f7bc3cf441`). This is a valid but slow-runner
near-miss draw, about 3.4% under the 995,329,477 frontier and far below the
1,005,282,772 promotion floor; it is not evidence for a new optimization.
The prior 02c7 official draw at 1,001,615,305/s was also below the floor, so
further byte-identical redraws are retired. The production source and carrier
remain unchanged.

The broad c60 redraw `9f97b039` was cancelled without a score at 21:43 UTC;
no mechanism is imported from it. Pending queue review continues with the
isolated SAS2-glue ticket `1a89f12a` and unscored identity/reuse entries only.
