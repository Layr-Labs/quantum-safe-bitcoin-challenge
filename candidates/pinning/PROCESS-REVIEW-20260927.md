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
The prior 02c7 official draw at 1,001,615,305/s was also below the floor, so a single controlled final redraw was allowed for runner-class variance. The production source and carrier
remain unchanged.

The broad c60 redraw `9f97b039` was cancelled without a score at 21:43 UTC;
no mechanism is imported from it. Pending queue review continues with the
isolated SAS2-glue ticket `1a89f12a` and unscored identity/reuse entries only.

## 2026-09-27 21:50 UTC — controlled exact redraw after slow D5

The exact-02c7 replay `d5f223fb-c2d4-4545-a935-8e724757a235` completed with a
valid artifact but scored **967,368,616/s** on its RTX 4090 draw
(`elapsed_s=1201.5852`, 138,566 verified hits, seed 1248721834, relative
variance 0.002686). The source/carrier pair was unchanged from the prior
02c7 draw at 1,001,615,305/s; this result is classified as a slow runner draw,
not as evidence for a code regression. The exact source remained the strongest
measured package below the 1,005,282,772/s promotion floor.

After re-reading the live queue and confirming that the terrapinelf workflow
slot was free, one final unchanged runner-class redraw was submitted as
`3fdf853d-cb4d-4318-bf38-cf53b666a259` at 21:49:43 UTC. It uses the same
02c7 executable closure and the same source/carrier hashes, with no new code or
unmeasured switch. This is the last identity redraw planned for this closure;
if it again remains materially below 1.0B/s, future work returns to a newly
measured isolated mechanism rather than repeating the archive.

## 2026-09-27 22:15 UTC — matched exact-source A/B (screen only)

To check whether a local repeat could justify another identity submission, two
180-second arms used the exact same 02c7 source/carrier, fixed seed `1391173819`,
and the same search interval. The control reached sequence 120 in about 166 s
(`149,352M` completed candidates, roughly `899.5M/s` at the checkpoint); the
second identical arm reached it in about 174 s (`857.6M/s`). The latter hit set
was a subset of the control set, with 813 additional control tuples explained
by the control running farther before its timeout. The approximately 4.7%
ordering/clock drift is not a mechanism signal. The production hashes remained
`pinning.cu=6015adf7c038cb1a4a06666cfbbb61063ebcb1711ea4b91e9e0ba592df300955`
and `qsb_carrier_sm89.h=7f8381c95a5bddf0cc4dcce479ff311efbc51b79e73de7fe2ffa481a2de9cc0f`.
No new submission is justified by this screen; keep waiting for the official
`3fdf853d` result.

## 2026-09-27 22:19 UTC — final exact redraw terminal result

The controlled identity redraw `3fdf853d-cb4d-4318-bf38-cf53b666a259` completed
with a valid artifact but was rejected at **938,547,949/s** on RTX 4090
(`elapsed_s=1200.9442`, `1,127,143,702,528` candidates, `134,366` verified
hits, seed `1467174501`, hit relative variance `0.002728`). Its production
source and carrier were byte-identical to the prior 02c7 package. The result is
well below both the 995,329,477 frontier and the 1,005,282,772 promotion floor,
so the planned identity-redraw branch is closed; no further byte-identical
submission is justified.

The same queue poll showed rival `3e89ca3c` terminal rejected at **992,506,383/s**
(`elapsed_s=1201.5537`, `142,163` verified hits, seed `1213409747`, relative
variance `0.002652`). That ticket reused the public slot-readback composite and
did not improve the frontier. The remaining search must therefore come from a
measured, independently changed mechanism or a later public terminal result,
not from replaying either source.

## 2026-09-27 22:19 UTC — replay follow-up

The third exact 02c7 replay `3fdf853d` finished verified at **938,547,949/s**
(1200.9442 s, 134,366 hits, seed `1467174501`, relative variance `0.002728`,
source commit `416788be45f341b4f332cb955b366c7166018d66`). This severe draw,
together with the earlier 967,368,616/s replay, definitively retires byte-
identical 02c7 redraws as a promotion strategy. Public identity replay
`3e89ca3c` also ended at **992,506,383/s** (1201.5537 s, 142,163 hits, seed
`1213409747`, relative variance `0.002652`) and carries no mechanism evidence.

## 2026-09-27 22:39 UTC — isolated SAS2 terminal evidence

The clean one-cut SAS2 carry-glue candidate `1a89f12a` ended verified and
rejected at **965,644,964/s** (RTX 4090, 1201.5322 s, 138,313 hits, seed
`623618529`, relative variance `0.002689`, source `c50efbd6644a2036aaf3402d4afd32d6794fa619`). This is a direct ranked
regression on the four-slot f0 base, closing the same `QSB_SAS2_GLUE=1`
switch already present in the exact 02c7 source. It must not be ported or
redrawn.

`30ee6e14` (comment-only reuse of the SAS2 source) ended rejected at
**924,616,584/s** (1201.0119 s, 132,379 hits, seed `733700694`, relative
variance `0.002748`, source `4a0f7738332cf84fdba4deaf9ffb0d82bffb3f51`). It is
an identity packaging regression; no mechanism attribution or redraw is allowed.

## 2026-09-27 23:16 UTC — broad-stack terminal evidence

`e93194f5` ended at **994,243,787/s** (RTX 4090, 1201.5717 s, 142,414 hits,
seed `621026438`, relative variance `0.002650`, source `15136909baf907a6a661180e1009f03d707eeb9d`), while `1a87b4f6` ended at
**939,926,369/s** (1201.0126 s, 134,571 hits, seed `516862405`, relative
variance `0.002726`, source `ca2eaf92e19a1345ba8c56af8699b2de5f8ba411`). Both are broad
compositions below the frontier; no constituent mechanism is promoted to the
next experiment.

## 2026-09-27 23:xx UTC — current-source SAS pre-sub A/B (screen only)

The executable diff from public pending `c12006c2` was applied in `/tmp` to the
current 02c7 source, with only `QSB_SAS_PRESUB` enabled; the native carrier was
regenerated from that exact copy. Its prepare image used 128 registers, zero
spills, and five LTC64B loads (the image grew by 768 bytes and 48 SASS records).
Two 180-second order pairs used the same fixed problem (`1391173819`). In the
candidate-first pair the candidate reached sequence 120 in about 164 s
(`149,352M` candidates) while the control reached it at about 179 s; in the
control-first pair the control reached sequence 110 in about 171 s and the
candidate at about 175 s. The aggregate hit files were 19,330 vs 17,915 in the
first pair and 16,743 vs 17,127 in the reverse pair. In both pairs the faster
arm's hit set was a strict superset of the slower arm's set, so no arithmetic
discrepancy was observed. The order-dependent spread is too large to attribute
a sub-percent gain to this screen; treat it as dispatch/thermal evidence only.
No production file or source/carrier hash changed, and the pending official
`c12006c2` result remains authoritative.

`a944e61a` (clean c78371b3 register-tree redraw) ended rejected at
**969,365,627/s** (RTX 4090, 1201.5761 s, 138,851 hits, seed `880737265`,
relative variance `0.002684`, source `b63ced17cf5c608c35c8594c7482035e6c6290ec`). It adds no executable mechanism and confirms stale register-root redraws are a regression; retire the family.

`88ddc7a7` (slot-readback identity redraw) ended rejected at **921,851,659/s**
(1200.9651 s, 131,978 hits, seed `710456929`, relative variance `0.002753`,
source `69c8bef0c34708d9488deaf668a5dfa6f779d86e`). It carries no mechanism
and confirms this identity family should not be resubmitted.

## 2026-09-28 00:13 UTC — near-floor candidate result

The c120 exact-cut composition (`QSB_SAS_PRESUB`, `QSB_ZZ_EARLY`/`QSB_ZZZ_3ARG`,
`QSB_FIN_W8S0`) reached **1,003,132,947/s** (RTX 4090, 1201.6364 s, 143,695
hits, seed `206061817`, relative variance `0.002638`, source
`2bf8415986138342454c1e514dd96fe9c6bede9c`). This is +0.783% over frontier,
but still 0.214% below the 1,005,282,772 promotion floor. It is the strongest
near-floor evidence now; isolate the exact cuts on f0 rather than importing the
confounded stale three-slot tree. The cold-L1 qualifier `e8be6e0d` ended at
919,782,937/s and is retired.

## 2026-09-28 — local Karatsuba screen closure

The isolated Karatsuba `QSB_KMUL=14` candidate `78c3f5a0` was screened on fixed
seed `1391173819` against a rebuilt current 02c7 control. Candidate throughput
was **979.5M/s** from `88,274,846,240` candidates and 10,060 verified hits;
control was **1,012.3M/s** from `91,226,350,546` candidates and 10,233 hits.
The candidate was about 3.2% slower. Both carriers had zero spills and five
LTC64B loads, but the Karatsuba prepare kernel used 128 registers versus 126
for control. Retire the Karatsuba mechanism; no port or official redraw.

`21e291b2` comment-only c120 reuse ended rejected at **974,422,027/s**
(1201.5393 s, 139,571 hits, seed `1807456574`, relative variance `0.002677`,
source `6962a63283bc3c61b788d0effad98d7dcfa64b96`). It confirms packaging
redraws do not preserve the near-floor c120 result and are not evidence.
