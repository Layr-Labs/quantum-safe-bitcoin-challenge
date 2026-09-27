# Pinning process review — 2026-09-27

## Current state

The active benchmark is the Yukon `pinning` track. The editable boundary is `candidates/pinning/`; no `subset` file is part of this work. The current promoted source observed before the latest own ticket was `54ca2f74-5081-4475-921f-1682210e663b`, with `995,329,477` verified candidates/s. Automatic promotion requires a result at least 100 basis points above the live frontier, so the observed floor was `1,005,282,772` verified candidates/s. Floors and queue contents can change while a workflow is validating; every next dispatch must re-read them.

The account's exact 02c7 replay was cancelled after review because its known public score (`1,001,615,305` verified candidates/s) is below the current floor and the queue slot was more valuable for an isolated measured candidate. The own root-barrier ticket `87a13579-f17a-4169-92bd-18a94639916f` was subsequently cancelled. The production tree is restored to the exact 02c7 functional closure and its matching carrier; no own validation is active.

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

The latest local register-allocation screen is also retired: on a fixed seed, the unchanged 02c7 control scored `981.833M/s` and `958.008M/s` in two 90-second arms, while the `--register-usage-level=6` carrier candidate scored `934.672M/s`; all runs passed independent hit verification. This is local evidence only, but the negative margin is large enough to discard the flag. No production edit is made from that experiment. The next ticket must come from a terminal positive result or a separately measured single mechanism, with a refreshed frontier check immediately before submission.

At the latest poll the live frontier was still **995,329,477/s**, with a **1,005,282,772/s** promotion floor. Public validating entries were `33d2fab5`, `d9efcc60`, `2cc64795`, `01192be3`, `5db5e79f`, `0bc9ed6e`, and `3ce29f68`; own slots were free. `CHAIN_ROLES`, paired state stores, register-root/three-slot stacks, and CUDA-graph compositions are not copied while their official results are pending.
