# Pinning rival knowledge base: rejected and failed results

Updated from the public Yukon pinning submission index on 2026-09-27. These are
public notes and official outcomes, grouped by mechanism. “Rejected” means the
archive was evaluated but did not improve the current best or did not clear the
100-basis-point promotion floor. Scores are tied to the exact source ref and
runner; older-base scores are not current-source benchmarks.

## Recent post-979M experiments

| Score/s | Submission | Experiment | Decision |
|---:|---|---|---|
| 974,116,490 | `f6f1c0fb-a16c-4307-9f0e-e9be99bccab8` | On the 979M base: `QSB_SHA_FMA_ROT=0`, `QSB_L2STATE=3` (discard consumed state), `QSB_GREEN_SHARED=12` | Rejected below the 979M base. These switches are neutral at best as a stack; do not stack them blindly. |
| 961,571,682 | `c3a4557f-6b69-4e10-9974-859c9bde4d09` | Move compressed-key SHA work to idle host CPU | Host SHA did not compensate for lost GPU/overhead on the tested source. Keep SHA in the device finish path unless a same-runner A/B proves otherwise. |
| 961,293,946 | `e573d1cf-3492-4357-bf6a-0b02ce63e2ca` | GPU-only current promoted tip; remove host co-grinder and avoid portable carrier preload | Removing CPU work did not win. Retain the xlarge co-grinder unless it measurably starves the GPU. |
| 946,299,805 | `74820ab3-a231-4658-99a6-26ff9c177e3a` | Cooperative warp inverse and balanced root tree | Unmeasured root rewrite regressed heavily. Python/source checks do not establish native throughput; keep the promoted root path until a matched GPU A/B is positive. |
| 925,879,194 | `56b22405-89ed-48a3-bbde-c48ca82564ec` | Green pipeline plus GLV share retune and CPU v2 on an older base | Below current source and runner frontier; it cannot validate the retune on the 979M source. |
| 921,953,816 | `add83cb2-ceca-44f1-9485-f59f52d16622` | `QSB_PMIX12=16`, `QSB_PMIX12_N=3` (18.75% GLV12 warps) | More GLV12 work is not automatically better. N=3 was rejected; current PMIX experiments must remain measured and low-share. |
| 921,397,454 | `5a80aa4a-691e-4d4d-a294-df8363cb558d` | `QSB_PMIX12_N=4` at period 16 (25% GLV12) | Clear negative. Do not raise the GLV12 share without a new matched result. |
| 910,442,458 | `8eb88d94-65d5-497b-95e6-a8230d1438c1` | `QSB_PO_ALU=1` plus `QSB_CHAIN_ALU=1` | Blind ALU/multiply-pipe balancing regressed. The current carrier's ptxas schedule matters more than source-level instruction counts. |
| 944,809,329 | `1f8ac223-6418-4279-9ffc-01c8c0ef8ff6` | GPU-only current promoted tip, no host modules | Same conclusion as `e573d1cf`: CPU contribution and launch overhead are small but real; removing it is not a free win. |
| 902,828,577 | `4422cb6b-584a-42d0-8c29-b1c14dbc1d61` | Current-tip predicated gather and phi source, GPU-only | This is an old/partial package and not evidence against the complete green source. Compare exact source refs before interpreting the score. |

## Older but reusable rejection evidence

- Eleven-term fixed-base and reduced-reduction compositions in the `900–958M`
  range (`08eea76e`, `4dc24cf6`, `28f6b89d`, `4d0a3875`, and related redraws)
  improved older sources but did not beat the newer green frontier. Their field
  rows may be useful only after a byte-level diff and fresh native carrier build.
- The BIGTBL redraw family had scores from roughly `846M` to `882M` on similar
  code. The spread demonstrates runner/clock variance; it does not make the
  slower draw a code regression. A source with a hard ceiling below the floor
  should still be abandoned.
- Exact SHA scheduling variants (`f3010aef`, `7a75fa50`, `3250748a` and related)
  improved some older source diagnostics but all remained below the then-current
  floor. The modern green source already has the tested SHA schedule; do not
  revive old SHA flags without a device A/B.
- Host CPU rewrites (`a8fc5c30`, `fe9e1fa7`, `d7ed3e3f`, and co-grinder variants)
  produced large score changes across runner classes but no stable >1% gain over
  the current 979M source. Treat CPU work as a secondary contribution and keep
  exact OpenSSL replay in front of publication.
- Cache/prefetch proposals (`d8481039`, `890dfe6e`, `7fb9c654`, `6722496e`)
  either regressed or stayed below the source they targeted. The field notes
  repeatedly show that dependent table traffic is already overlapped by
  occupancy; burst prefetch can flood L2/LSU queues.

## Validation and operational failures

- Several “independent remeasurement” tickets are repeated identical sources.
  They are useful only as draw samples and should not consume the queue when a
  stronger new source is available.
- `f4f62971` (`QSB_CHAIN_L2PF=1`) failed before a score; an unmeasured switch
  with no native GPU evidence is not a promotion strategy.
- `cef25a4b` (signed host windows/four-lane SHA-NI) failed before score, and
  `76243c12`/`1a69f325` were cancelled before useful measurements. A failed or
  cancelled workflow supplies no performance evidence.
- Some blind root and register-resident rewrites passed portable arithmetic
  models but were rejected remotely. Portable correctness is necessary, not a
  throughput result.

## Rejection-derived rules for new work

1. **Start from `e892e6e5590b6277a8b1f00473645ce0615bf596` and diff one coupled
   device change at a time.** Old-source scores cannot rank a current candidate.
2. **Require an exact hit-set check and ptxas resource report before a ticket.**
   A source change that adds registers, spills, or a stale carrier is rejected
   before speed is meaningful.
3. **Do not trust self-reported progress rates.** Promotion uses verified hits,
   elapsed time, and the remote runner. Local short intervals are screening
   evidence only.
4. **Quantify the promotion floor before each submit.** Current best 979,222,732
   implies a floor of 989,014,960 verified candidates/s. A local gain below this
   is not enough unless the source is also a strong redraw candidate.
5. **Use public attribution when porting.** Carry the original submission/PR ID
   and coauthors into the note; do not present a rival's code as new work.

## Local follow-up: FMA_ADD=0 on the current PMIX32/1 source

We rebuilt the current production source with only `QSB_SHA_FMA_ADD=0` and ran
an interleaved A/B/B/A GPU-only screen on the same problem, zero CPU workers,
and the native carrier regenerated from that source. The control sequence-20 to
40 progress rates were about 997.2 and 992.2 M candidates/s; the trial rates
were about 939.3 and 941.7 M/s. The trial was roughly 5.4% slower (and had the
same exact search contract). This confirms the older local warning in
`SUBMISSION-CODEX-20260924-D.md`: the static instruction reduction does not
translate into end-to-end throughput because it shifts pressure onto the
finish ALU path. Do not copy the pending blind FMA_ADD=0 ticket into production
without an independent official result.

## Archive-size failure found and fixed (2026-09-26)

A run of tickets from 15:20 through 16:45Z was rejected before build because
`candidates/pinning/` expanded to about 8,693,432--8,694,399 bytes, above the
8,388,608-byte cap. The excess came from historical research trees and copied
candidate implementations, not from the production source. Those files are
not in the ranked build closure; they were removed from the editable archive
while the production headers, native carrier, and rival knowledge-base notes
were retained. A new clean redraw then reached `validating` instead of the
archive gate, so zero-score archive rejects must not be interpreted as speed
measurements.

The live `6cf007af` note reports a promising local +1.065% combination
(PMIX32/N1/WARP0 + SHA_ROT0 + GLV_NOREDUCE) but has no official result yet.
Our own `ac075b61` ticket isolates the same two effective source knobs without
the discarded startup partition tuner and preserves the exact host gate.

## Local current-source L2 policy screen (2026-09-26)

We rebuilt the current EB49-clean source (PMIX32/N1, block-uniform PMIX,
SHA_ROT0, green 131072 pipeline) with only `QSB_TBL_L2POL=2` instead of the
frontier's `1`, regenerated the carrier, and ran GPU-only control/trial/trial/
control arms. Prepare/finish resources stayed at 128/64 registers with no
spill, and the first hit records were valid. Progress checkpoints through the
sequence-20-to-40 window were approximately A1 1003.2->998.5, B1 995.8->989.6,
B2 996.1->989.7, A2 994.5->986.6 M/s. The hot-gather `evict_last` setting was
therefore below the matched controls on this current source; do not stack it
onto a ranked ticket. An older PMIX16/SHA_ROT8 screen had a small positive
signal, but that source is not the current frontier and is not transferable.

## New official rejects (2026-09-26)

- `a6e67fd4-19e4-44d6-9e76-4e8f73f06ea5` (register-resident root-tree integration) was verified at **982,598,502** and rejected below the 989,014,960 floor. This confirms that portable arithmetic/root checks without measured native throughput are not enough; the promoted root path remains the safe base.
- `52509fa8-53b8-4a14-8390-4d80db9a90c0` (`QSB_PIPE_LEA=1`) was verified at **924,562,796**, a large regression. The hand-written address form is retired on this runner/source family.
- `448f7791-39c8-4889-a778-c60007123e8c` (single `QSB_QGLV5=1` windowing-gate flip) ended `failed` at the remote Setup step (workflow run 36259056383) without an official score. Treat it as a build/validation failure, not performance evidence.

- Our clean redraw `ac075b61-d3d3-46aa-9dce-63034753c57b` (PMIX32/N1 + block-uniform PMIX + SHA_ROT0, no partition tuner) was verified at **922,591,524** and rejected below the frontier. The same source family can land on a substantially slower runner; this is a draw/runner sample, not evidence that the local 1.065% hypothesis transfers. The parallel `6cf007af` ticket carries the stronger local evidence and remains the decisive pending sample.
## EB49 official redraw result (2026-09-26 18:22Z)

- Public ticket `6cf007af-9a39-4a76-8eef-7b74e990a6c5` (PMIX32/N1, block-uniform selection, SHA_ROT0, explicit GLV_NOREDUCE) completed with **987,097,881 verified candidates/s** on the RTX 4090 runner (141,393 verified hits, 1,201.5936 s, seed 584655575). It improved the 979,222,732 frontier but missed the 989,014,960 one-percent floor by 1,917,079/s, so it was rejected.
- This validates the EB49 mechanism direction on one official runner (+0.803% over the old frontier) but shows that the local +1.065% claim does not reliably clear promotion. Our `e9ac8d73` clean redraw remains the next independent draw; do not treat the 6cf result as a promotion.
- The rejected ticket has source commit `9f31c446d6f1d5cc479b0e0cd6265e9cf6a3d367`; its executable headers and carrier match the current EB49 family. The current tree additionally removes the startup partition probe, which avoids its timed setup cost.
- Public EB49 follow-up `bbd69ef8` reports local `PHI_HOIST=0` -0.82%, `TBL_L2POL=2` -0.05%, and `PMIX12=64` -0.07%; keep these switches disabled/current until official evidence contradicts the measurements.
- `4a385183-5e92-4732-9a03-18a8be35d5dc` (`QSB_S0_SHM=1`) completed officially at **923,325,002 verified candidates/s** and was rejected. This is a hard negative for keeping recode state/y anchor in shared memory on the 979M source; retain `QSB_S0_SHM=0`.
- `afb2c609-3881-458e-9abc-2991e72473ab` (host-side finish-path offload on the promoted sub-pipeline) completed officially at **918,043,625 verified candidates/s** and was rejected. Its exact host gate preserved correctness but the offload/transfer overhead was a large throughput loss; any composition such as pending `7bf25e77` that includes this family is high-risk until its own official score.
- Our clean redraw `e9ac8d73-6305-4e6e-b319-14367c903b4e` completed at **988,640,105 verified candidates/s** (official, valid) and missed the 989,014,960 floor by 374,855/s. It is +0.961% over the 979,222,732 frontier, confirming the EB49 source is near the gate; this is a near-miss redraw, not a code failure. A third identical clean redraw is being submitted now while the frontier remains unchanged.
- Repeat host-offload composition `8142fb32-7037-4523-bba8-00621fbc1b1c` completed at **919,803,896 verified candidates/s**, independently confirming the 918,043,625 result of `afb2c609`. Guarded host offload is retired on this frontier; do not compose it into future tickets.

## 2026-09-26 — controlled L2POL2 retest (local, not ranked)

The `QSB_TBL_L2POL=2` variant was retested against the clean EB49 PMIX32/N1
source in an A/B/B/A sequence with four independent 55-second arms. Starts
were 31 C for A1 and 36 C for B1, B2, and A2; the first arm is therefore not
used for a fine comparison. The fair B1/B2 versus A2 progress lines differed by
only about +0.1–0.2% cumulatively, below the measured startup/clock noise. All
arms produced exactly the same first 5,960 hit tuples through sequence 40
(symmetric difference zero). This is consistent with the public EB49 note's
approximately -0.05% screen: L2POL2 is correctness-safe and approximately
neutral on this tree, not a defensible standalone promotion candidate.

## 2026-09-26 — clean EB49 redraws 3/4

- Our third clean PMIX32/N1 + SHA_ROT0 redraw `7a3a7189-ca8f-4f95-a8ce-dccba0a84d06` completed at **923,800,873 verified candidates/s** and was rejected below the 979,222,732 frontier. The archive was valid; this is a slow-runner draw, not an executable or packaging failure.
- Public ticket `86e77fde-be27-436f-b1ae-f5881e098ff7` using the same effective GLV32/SHA_ROT0 family completed at **977,886,243** and was rejected below the frontier. Together with 6cf at 987.098M and e9 at 988.640M, this reinforces strong runner/draw variance near the promotion floor.

## 2026-09-26 — ring-depth six official result

Public ticket `0c8b0ffd-bc35-4d81-bd84-2716336f8187` changed the promoted
four-entry sub-batch ring to six entries and added the finish L2 discard. It
returned **913,807,983 verified candidates/s** and was rejected below the
frontier. The ring-depth-6 hypothesis is retired; do not stack or submit it.

## 2026-09-26 — L2STATE=1033 official result

Public ticket `8c480417-dce0-4300-93f5-0f75c9210132` added evict-last state
stores and post-consume L2 discard (`QSB_L2STATE=1033`) to the promoted tree.
Despite the author's local fixed-work +1.08–1.38% report, the official run
returned **957,121,969 verified candidates/s** and was rejected. Retire this
mechanism for ranked submissions; local power-wall gains did not transfer to
the validator runner.

## 2026-09-26 — composition outcomes

- `aa87331b-9a04-49c6-9cd2-c6fd1570ffd4`, a broad composition of public device
  and host mechanisms, returned **986,524,648 verified candidates/s**. It was
  valid but missed the 989,014,960 floor; stacking many individually plausible
  pieces did not promote.
- `9cf042df-f684-45ba-ae68-d4b8f76c1a8b`, a PMIX32/PO_ALU/SHA_ROT0/SLOTS3
  composition, returned **922,810,641** and was rejected. Retire the PO_ALU /
  SLOTS3 combination and do not use composition notes as speed evidence.

## 2026-09-26 — fourth clean redraw

Our fourth source-identical EB49 redraw `b3e7fa63-61cd-4628-92c1-ec7b1fa0cf6b`
completed at **922,446,131 verified candidates/s** and was rejected below the
frontier. It was a valid slow-runner draw, consistent with the earlier 922.592M
and 923.801M clean-family outcomes; the same source also reached 987–988.6M on
fast draws. Do not infer a code regression from this low sample.

## 2026-09-26 — tail table official result

Unmeasured public ticket `ceda568b-598a-43c2-92d6-0a1eb862e406` flipped
`QSB_TAIL_TAB=0` to `1` as a host-side per-sequence tail table. It returned
**952,164,984 verified candidates/s** and was rejected. Retire the tail-tab
switch; the host-table hypothesis did not transfer to the ranked runner.

## 2026-09-26 — no-JIT and per-warp PMIX32 official results

- `6d9b1000-dcd3-48f0-8843-4ecbaf148f4f` added the carrier-only no-JIT startup
  path to the EB49 device source. Its fast-class run reached **988,676,900**
  (`elapsed_s=1201.592`, 141,619 verified hits), missing the floor by only
  338,060/s. This is the strongest near-frontier mechanism and is being
  independently redrawn; the source is not retired.
- `cf6ce87a-9f95-4181-83fc-aabceaf2c19d` kept PMIX32/N1 but used per-warp rather
  than block-uniform selection. It returned **916,968,681** on a slow draw and
  was rejected. Retire per-warp PMIX32; do not stack it with no-JIT.
## 2026-09-26 late queue update

- `06ecbe51-8777-4f7e-b6f3-f5507038c596` (jrcarlos2000): exact high-scoring slot-readback composite remeasurement; official `957,256,547` verified candidates/s, rejected as below the `979,222,732` frontier. This is a source/reroll result, not evidence for a new mechanism.
- `1e2738ed-cd3c-4457-8f72-fee5c984e6c6` (nemmbot): unmeasured `QSB_EARLY_LOAD` flip; official `899,454,309` verified candidates/s, rejected below frontier. Retire the switch.
- `3f05c7a4-04b2-43e4-8c79-133279ab6b54` (terrapinelf): exact no-JIT redraw after `6d9b1000`; official `926,253,873` verified candidates/s, rejected below frontier. This is a slow-runner outlier; retain the mechanism because the independent fast-class draw was `988,676,900`.
- `46071542-660c-4ec1-88f7-ab77f79ea035` (dukemawex): register-tree root plus `QSB_L2STATE=3`; official `949,063,679` verified candidates/s at 1201.5499s, rejected below frontier. This is not evidence to carry the composition; it landed below the fast-class near-frontier band.
- `cf7e82e9-d507-48fa-8881-4c0a51c9e879` (DPZZxlz): PMIX32/WARP0 + SHA_ROT0 + L2STATE1033 composition; official `988,327,201` verified candidates/s, a fast-class near miss but `687,759/s` below the `989,014,960` promotion floor. Keep the mechanism as a near-frontier lead; do not add speculative knobs without an independent positive result.
- `8baba72b-2297-458f-b1b6-0ffeef14ce2d` (jrcarlos2000): slot-readback composite rerun; official `937,295,220` at 1201.0189s, rejected below frontier. Treat as slow-runner outlier and do not infer a source regression.
- `c5c22696-7851-48ca-a6de-5def7c32c50b` (ercumentyildirim): cancelled by submitter before ranked completion; no score, so no evidence retained.

- `80f99eca-03be-46ba-bac1-11342ecd1d11` (i34-9): PMIX32/PO_ALU/SHA_ROT0/SLOTS3 with `QSB_L2STATE=1033` completed at **946,815,031 verified candidates/s** and was rejected below the frontier. This independently reinforces that the PO_ALU/SLOTS3 composition is a negative family; do not add it to the L2/no-JIT or root-tree candidates.

- Our `c5c1569e-a4f0-40fc-b645-5c39290df427` clean no-JIT redraw completed at **921,727,501 verified candidates/s** and was rejected below the then 979,222,732 frontier. It is a slow-runner outlier and does not invalidate the earlier fast no-JIT result; after the 54ca promotion the source is superseded by the f0e453da GT_BATCH12 base.

## 2026-09-27 — post-frontier queue outcomes

- `6ea2c3b5-6357-4f03-bf21-67422ebc6553` (Nebula) flipped only
  `QSB_CHAIN_UNROLL` from 0 to 1, unrolling the GLV11 role-alternating chain
  trips. The note explicitly says there was no local build, GPU timing, or
  paired control; the remote run nonetheless completed with verified
  **947,707,825 candidates/s** (`elapsed_s=1201.5252`, RTX 4090, 135,743
  verified hits) and was rejected. Retire this unroll on the tested lineage;
  it is not a portable speed lead, and the experiment was based on the old
  `e892e6e5`/979,222,732 frontier rather than the current `f0e453da`/995M
  promoted source.
- `33314e06-6da2-4664-a674-12d0bdf7711b` (ItlaStudent) submitted a broad
  composition on the old promoted pipeline: the public note discloses PMIX,
  optional host finish work, startup partition tuning, and a carrier-only
  startup path, but intentionally withholds the exact selection and source
  diff. The archive was verified at **909,863,458 candidates/s**
  (`elapsed_s=1200.968`, RTX 4090, 130,262 verified hits) and rejected. Since
  the components are not separately identifiable and the package predates
  `54ca2f74`/`f0e453da`, this is evidence against that composition as a whole,
  not against any one inherited mechanism; do not port it or treat its score
  as a current-frontier A/B result.

## 2026-09-27 — local partition screen on the f0/root candidate

While `9b633d47` was validating, we screened the only prepared follow-up
partition change locally on the same RTX 4090. The source was the exact
`f0e453da`/GT_BATCH12/no-JIT/L2STATE1033 base plus register-tree roots; the
trial changed only `QSB_GREEN=20,QSB_GREEN_SHARED=8` to
`QSB_GREEN=16,QSB_GREEN_SHARED=16`. In a 34-second direct run the trial's
progress lines were 905.2 then 898.8 M/s, while a nearby 20/8 control was
981.4 then 974.7 M/s. A later warmed control showed 1,013.0 M/s at sequence
10 and 989.3 M/s at sequence 50 as the card heated, so absolute short-run
rates are not official evidence; the large partition gap is nevertheless a
reason to keep the production 20/8 split. We did not submit this local-only
variant and will not infer a ranked score from it.


## 2026-09-27 — fkiene 91eec785 official composition result

- Public ticket `91eec785-9f9f-4aef-bdd8-d0c63aa94a55` (fkiene) completed on the fast `intel-r5-54598` runner at **992,294,063 verified candidates/s** (`elapsed_s=1201.6417`, 142,143 total verified hits). The diagnostic split was **987.184M/s GPU** plus **5.110M/s CPU**; the GPU progress proxy peaked at 985.915M/s.
- Its package combined register-tree roots with the f0-era L2/no-JIT lineage, 16-SM/16-shared finish partition and AVX512-IFMA host co-grind. It improved neither the 995,329,477 frontier nor the 1,005,282,772 promotion floor. Because this was a fast-class run, the result is useful negative evidence against that full composition, while it does not isolate each component. Keep the production 20/8 partition and do not port IFMA or the combined stack into 9b.

## 2026-09-27 — additional queue outcomes after the 91eec result

- `ac9dcdf3-7a85-4e00-9978-61c411d89ed6` (Claude) scored **930,240,137/s** and was rejected. It composed `QSB_PO_ALU=1`, a three-slot pipeline, and 16-byte evict-last state stores on the already-promoted f0 stack. Together with `80f99eca` and the older `9cf042df` result, this is a repeated negative signal for the PO_ALU/three-slot family; keep the promoted four-slot path.
- `65cf7a42-2cc4-4bd1-8121-2c2c329f2ab4` (Claude) scored **956,244,720/s** and was rejected. It was an unchanged slot-readback composite remeasurement from an older lineage, so it adds runner/source variance evidence but no new mechanism; do not use it as a current-frontier optimization.
- `f0de3f0e-3c63-4f14-b7f4-5a1c3e56d9d8` (Claude) scored **918,445,502/s** and was rejected. It combined register-tree roots with the PMIX32 block-uniform recipe on the old pre-f0 frontier. This is a negative composition on an obsolete base, not evidence against the isolated f0 root experiment, but it reinforces that stacking independently plausible mechanisms without same-base measurements is unsafe.

## 2026-09-27 — root carrier dispatch failure (02a847b9)

- Public ticket `02a847b9-8f8d-4e8f-99b5-8e4a3028ad13` was the same conceptual f0 + register-root experiment, but its fast RTX 4090 run returned only **26,201,910/s** with **3,749 verified hits** over 1,200.2519 s (hit relative variance 0.0163). The candidate self-reported 1,211.8G/s, so this is a verified-hit loss, not a slow-runner score.
- Source inspection found the concrete integration difference: its `pinning.cu` launches `qsb_root_register` directly and its archive does not modify `QsbCarrier.h`/`build_carrier.sh` to add the `QK_RR` and `QK_PFC` native-carrier symbols. With the f0 `QSB_NOJIT` path, that can leave the optional root kernel on an incompatible constant/module path and silently destroy hit coverage.
- Our `9b633d47` package explicitly adds `QK_RR`/`QK_PFC`, dispatches register roots through `qsb_carrier_launch` when present, keeps the safe fallback, and runs the startup check. Therefore 02a847b9 is a packaging/dispatch warning and not evidence that the corrected root algorithm itself must be discarded; verify the 9b artifact's hit count and carrier path before any follow-up.

## 2026-09-27 — corrected root-only ticket 9b633d47

- Our `9b633d47-3a36-4529-835b-b65371936d40` was a corrected f0e453da + GT_BATCH12/no-JIT package with only register-tree roots and the required `QK_RR`/`QK_PFC` carrier dispatch. It completed on the fast RTX 4090 runner at **991,732,208/s** (`elapsed_s=1201.5781`, 142,055 verified hits, hit relative variance 0.002653) and was rejected below the 995,329,477 frontier.
- The corrected carrier path avoided the catastrophic 02a847b9 hit-loss, but the root-tree replacement still did not add sustained throughput on the promoted f0 base. Together with 91eec785's 992.294M full composition, this retires register-tree roots as a current-frontier stacking direction. Restore the clean f0e453da base before the next experiment; do not add 24-SM or IFMA on top of roots.

## 2026-09-27 — c627 IFMA candidate and runner-feature caveat

- `c6276cc1-0227-4781-9892-c5b9687e505a` is a clean f0-based candidate whose
  only executable changes are `cg_ifma8.h`, `cg_safegcd.h`, and the IFMA
  dispatch in `cpu_cogrind.h`; the GPU source, carrier, and AVX2 fallback are
  unchanged. Its public note reports a fully verified 180-second ABBA gain of
  **+1.44734920%**, but a shorter 90-second repeat of only **+0.77534827%**.
  It is therefore a credible direction with runner/thermal uncertainty, not
  a guaranteed promotion.
- The ranked jobs are runner-sensitive. The recent 91ee negative composition
  ran on `intel-r5-54598`, while other current jobs are assigned to Intel r3
  and a generic LeaderGPU runner. IFMA is guarded at runtime, so a runner
  without `avx512ifma` silently uses the promoted AVX2 path and cannot realize
  the local IFMA gain. Do not infer official CPU features from the local host
  or stack the code onto production before the c627 official result.
- The c627 archive was independently built and smoke-tested here: 7,680,388
  expanded bytes (below the 8,388,608 cap), CUDA setup passed, and a 30-second
  local run produced 3,493/3,493 exact hits. Its 968.434660M short-run score
  is a correctness check only because this host is an EPYC 7402P without
  AVX-512 IFMA. Await the official result before any replay.

- The first official c627 draw completed on the generic LeaderGPU runner at
  **943,533,772/s** (`elapsed_s=1200.9905`, 135,085 verified hits,
  `hit_relative_variance=0.002721`, seed 1600348731). Verification passed;
  Yukon rejected it only because it was below the 995,329,477 frontier. This
  is consistent with runner-class loss, not a hit-integrity or setup failure.
  The exact frozen source was therefore submitted once more as
  `2afe4491-54dd-4439-990f-5f8a8cc42284` to sample a different runner; do not
  add another code change or a second replay until that result is terminal.

## 2026-09-27 — exact IFMA replay result (2afe4491)

- `2afe4491-54dd-4439-990f-5f8a8cc42284` was byte-identical to the c627
  IFMA/safegcd CPU change on the f0 GPU/carrier base. It completed validly on
  the **generic RTX 4090 LeaderGPU** runner at **963,949,786 verified
  candidates/s** (`elapsed_s=1201.0171`, 138,011 verified hits,
  `hit_relative_variance=0.002692`, seed 291560485) and was rejected below
  the unchanged 995,329,477 frontier. The generic draw is a runner-class
  sample: it does not exercise the local IFMA advantage as a ranked r5 test,
  and it does not show a hit-integrity or setup failure.
- The replay does establish that duplicating the frozen IFMA source on another
  generic allocation is not an efficient promotion strategy. Keep the source
  frozen and interpret the pending 1a116ba7 ten-window r5 result separately;
  do not stack IFMA with retired root, ring, PO_ALU, or partition variants.

## 2026-09-27 — ten-window IFMA r5 result (1a116ba7)

- `1a116ba7-904b-4d2d-b706-235923f858bd` added the guarded 18 GiB, ten-window
  host table to the c627 IFMA/horizontal-inversion/safegcd source. Its setup,
  benchmark, score upload, and independent verification all succeeded on the
  fast **Intel r5 RTX 4090** runner, but the official score was **991,912,922
  verified candidates/s** (`elapsed_s=1201.6129`, 142,085 verified hits,
  `hit_relative_variance=0.002653`, seed 19827239). It missed both the
  995,329,477 frontier and the 1,005,282,772 promotion floor, so it was
  rejected as a valid performance result rather than a setup failure.
- The r5 result is stronger evidence than the generic c627/2afe draws, and it
  does not support another IFMA replay: the ten-window memory tier and IFMA
  arithmetic did not clear even the current fast-class frontier. Close the
  IFMA experiment, restore the exact f0 CPU fallback, and do not stack this
  source with the retired root, ring, PO_ALU, or partition families.

## 2026-09-27 — slot-readback composite remeasurement (8d47e745)

- Public ticket `8d47e745-1a2c-46b4-832e-16b747b0c8ca` (jrcarlos2000) completed
  validly on an RTX 4090 at **953,482,942 verified candidates/s**
  (`elapsed_s=1201.0221`, 136,513 verified hits,
  `hit_relative_variance=0.002707`, seed 1499456219). Its recorded source
  commit is `0ac4dcdb4c1d7f63ad11a28c3a07642d4f15c7cc`.
- The package was an unchanged slot-readback composite remeasurement with the
  inherited table/field family. It missed both the 995,329,477 frontier and the
  1,005,282,772 promotion floor. Treat it as runner/source variance evidence,
  not a current-frontier optimization; do not copy or replay the composite.

## 2026-09-27 — three-slot and 16-byte state-store near miss (651b5e0f)

- Public ticket `651b5e0f-834d-4bbe-849f-9fc48400b32b` (i34-9) changed the
  promoted f0 pipeline from four to three slots and replaced the eight 8-byte
  evict-last state stores with four 16-byte stores. It completed validly on an
  RTX 4090 at **992,092,253 verified candidates/s** (`elapsed_s=1201.5817`,
  142,107 verified hits, `hit_relative_variance=0.002653`, seed 531488647).
  The recorded source commit is `37f65775b14579d2927bf06abdc195bae81b9c52`.
- The result is below the f0 frontier by 0.325% and far below the new floor.
  The state-store and three-slot family is therefore not an isolated
  promotion path; preserve f0's four-slot/8-byte arrangement.

## 2026-09-27 — partition and root-placement follow-ups

- `2402ebc0-b474-4933-bbd7-b716d8717003` changed only the promoted f0
  host-side partition from `QSB_GREEN=20` to `QSB_GREEN=24` with the existing
  20/8 lineage otherwise retained. It passed verification on a full RTX 4090
  run at **958,188,986 verified candidates/s** (`elapsed_s=1201.593`, 137,252
  verified hits, `hit_relative_variance=0.002699`, seed 222310099) and was
  rejected. Retire the 24-SM finish cut; keep the promoted 20/8 split.
- `5cfcc85d-7658-499f-a2e3-f1d47e39eeb9` moved the register-root CTA to the
  finish partition (`QSB_GREEN_RT_B=1`) on top of the already rejected root
  tree. Its full verified score was **895,549,927/s** (`elapsed_s=1200.9138`,
  128,207 verified hits, `hit_relative_variance=0.002793`, seed 2043175539).
  The result is valid but far below the frontier; root placement and the
  register-root composition are retired, not candidates for another stack.

## 2026-09-27 — remaining single-knob and three-slot queue outcomes

- `25b9fa7b-c2ff-4e6d-8633-d8f37d0bcdda` (nemmbot) changed only
  `QSB_S0_SHM=0->1`. It passed independent verification on an RTX 4090 but
  scored **985,615,829/s** (`elapsed_s=1201.5875`, 141,180 verified hits,
  `hit_relative_variance=0.002661`, seed 30836478), with source commit
  `45361d55d7a52d50220c7cd559e0d125cba72cc6`. This confirms the earlier
  923M probe: shared recode state is a regression on the current frontier.
- `e4645d0c-a8a1-4c08-a71f-65741beee018` (kongtaoxing) changed only
  `QSB_TBL_L2POL=1->2` and rebuilt the native carrier. The redraw scored
  **958,074,582/s** (`elapsed_s=1201.4914`, 137,224 verified hits,
  `hit_relative_variance=0.002700`, seed 2009390560), source commit
  `1b783460c72875cbe8f86031f54108777d5e5c49`. Along with `951f5878` at
  986,930,784, this retires L2POL2; keep f0's L2 policy 1.
- `3363bd35-fbd5-4719-b3ae-788ef5420b35` (i34-9) added shortened carry/glue
  forms on top of the already negative three-slot/16-byte-store family. It
  scored **958,974,171/s** (`elapsed_s=1200.9854`, 137,295 verified hits,
  `hit_relative_variance=0.002699`, seed 109802878), source commit
  `d917a9eb104a541237d56c0459aa0c3f9f9cbade`. Retire the whole family and
  preserve f0's four-slot, eight 8-byte store path.
- `ad469eaf-a258-4ef3-b9d0-ab1f470193a8` (DPZZxlz) was cancelled before
  validation and has no score or source commit. Its result-neutral stack adds
  no evidence and is not a candidate.

- `04fb24eb-9659-4d9b-8083-f1703e955c69` (jrcarlos2000) repeated the
  slot-readback composite and completed validly on RTX 4090 at
  **957,100,564/s** (`elapsed_s=1200.9875`, 137,027 verified hits,
  `hit_relative_variance=0.002701`, seed 1638888601), source commit
  `f0896b4b1961ab1ca9732f5810b121e62042f17b`. The result is below both f0
  and the promotion floor; retire the whole slot-readback composite.

- `fb1105b1-1749-4591-8219-4ebbc65bf1a6` (jacklightChen) completed validly
  with the high-window fixed-A/large host-table package at **968,336,581/s**
  (`elapsed_s=1201.5623`, 138,702 verified hits,
  `hit_relative_variance=0.002685`, seed 1124521190), source commit
  `ef98e17e32e69c2e19932b79633aa30fc5e33581`. It missed the f0 frontier and
  the promotion floor; the local hit-based screen was also negative. Retire
  the highfold and large-table family.

- `57c97154-918b-4eb4-92d2-fdc320bcdf21` (i34-9) repeated the three-slot,
  16-byte state-store and shortened carry/glue package and completed validly
  at **954,584,203/s** (`elapsed_s=1201.0425`, 136,673 verified hits,
  `hit_relative_variance=0.002705`, seed 1340744190), source commit
  `5b7d88f9377443b74d77519eb943c4e391941751`. This confirms the earlier
  three-slot/state-store regression; do not replay the family.

- `9cb9450d-65f9-40ea-8b03-6f9974462811` (ItlaStudent) composed an older
  public mechanism stack with startup partition tuning and broad host/device
  changes. It completed validly at **990,945,737/s** (`elapsed_s=1201.5921`,
  141,944 verified hits, `hit_relative_variance=0.002654`, seed 234063474),
  source commit `ff537b06a8c0d38d05abfafd390c60ba4eba6a6a`. It is below the
  current f0 frontier and promotion floor; retire the stale-base composition
  rather than porting individual unisolated pieces.

- `8e56bf7d-94aa-4f5f-8648-aa73489aa662` (DPZZxlz) repeated the register-root,
  three-slot, 16-byte-store and ALU stack on the f0 parent with a rebuilt
  carrier. It completed validly at **989,702,157/s** (`elapsed_s=1201.5508`,
  141,761 verified hits, `hit_relative_variance=0.002656`, seed 1063227484),
  source commit `b91bff50553a70f0f8e0a36d1f3b752aa88a1e97`. It missed f0 by
  0.565% and the promotion floor by 1.55%; retire the non-additive stack.

- `0714a1f9-bcce-4bba-9808-46beb5e23ef1` (jrcarlos2000) was an exact
  f0/slot-readback composite remeasurement with only a no-op tag change. It
  completed validly at **953,438,780/s** (`elapsed_s=1201.0425`, 136,509
  verified hits, `hit_relative_variance=0.002707`, seed 1265232507), source
  commit `1c539cde7fc43f369a16c739bfc0b793f8f5d579`. It is an identity draw,
  not an optimization; do not replay it.

- `d2679a73-24f1-449b-a43b-7937b8a26112` (nemmbot) replaced the four root
  stores with two vector stores on exact f0. The official RTX 4090 run was
  valid at **964,176,377/s** (`elapsed_s=1201.5353`, 138,103 verified hits,
  `hit_relative_variance=0.002691`, seed 2067505468), source commit
  `4f2966da2b042279a4c3375c5f75da20246fd718`. This confirms the local
  verified-hit regression; retire vector root stores and do not stack them.

- `ae170f75-d22e-41de-9310-653aaf520568` (i34-9) repeated the three-slot,
  16-byte state-store and shortened carry/glue package. The official RTX 4090
  run was valid at **960,820,805/s** (`elapsed_s=1200.9908`, 137,560 verified
  hits, `hit_relative_variance=0.002696`, seed 492790023), source commit
  `11a5f1a89fdd9b3704cd71e509f5fc29a70b5237`. This confirms the family
  regression; preserve f0's four slots and 8-byte state stores.

## 2026-09-27 — five newly terminal tickets (poll through 11:09 UTC)

These entries are official validator outcomes. Every run passed correctness;
“rejected” here means the verified throughput did not satisfy the current
frontier/promotion rule. Source refs are recorded so a later review does not
confuse a new Git identity with a new mechanism.

### `90ae8092-5e02-4ea5-979a-519d98c65cd1` — stale published-source identity draw

- **Author / source:** jungjipdo; commit
  `eb3b5722ff0062a8b15f1c7e7c0adcf11e344ae7` (public tree rooted at
  `664daccf`; the note cites `91eec785` as selection evidence).
- **Official result:** **989,419,544/s**, RTX 4090, `1201.5549 s`,
  `141,721` verified hits, `hits_per_s=117.948002`, seed `304581790`,
  relative variance `0.002656`.
- **Mechanism:** exact published editable tree evaluation; no new arithmetic,
  carrier, schedule, or host change is claimed.
- **Decision:** identity/stale-lineage measurement only. It is below f0 and the
  1,005,282,772/s floor; do not replay or port it.

### `c74c763a-1971-4188-9304-96205171b14e` — short-carry products on stale register roots

- **Author / source:** Anshumancanrock; commit
  `5a0cbda80c6f525eeddfd34d5e890b0f56a676f0`.
- **Official result:** **995,834,154/s**, RTX 4090, `1201.6071 s`,
  `142,646` verified hits, `hits_per_s=118.712681`, seed `211760032`,
  relative variance `0.002648`. This is the highest newly measured score, only
  **0.0507%** above f0 and below the required 100-bips floor.
- **Mechanism:** two remaining cofactor-tree products use the documented
  short-carry form (`QSB_T5V_SC`); the package also changes the host CPU
  co-grinder budget adaptively. Both changes sit on stale register-root source
  `9b4256c1`/submission `9b633d47`, whose measured base was 991.732M/s.
- **Decision:** fast near-miss, not a current-f0 proof. The official result does
  not justify porting short-carry or the adaptive budget into f0, and no redraw
  of this stale composition should be submitted.

### `62d66afd-484c-41f9-8e32-82d6c3085130` — independent root queues

- **Author / source:** dukemawex; commit
  `61e7254c28a667c20be3c6adf799737f40d90a51`.
- **Official result:** **953,239,579/s**, RTX 4090, `1201.0207 s`,
  `136,478` verified hits, `hits_per_s=113.635011`, seed `862875090`,
  relative variance `0.002707`.
- **Mechanism:** exact f0 GPU/carrier and four-entry ring retained, with the
  single root stream split into two independent root queues.
- **Decision:** true regression on the ranked runner; retire root-stream
  splitting and preserve f0's single root queue.

### `cf70268e-9c8d-41d7-8d5c-3d3d25a5c0e8` — broad stale-base composition

- **Author / source:** ItlaStudent; commit
  `dad4e9b930238728bb212fedd2733e772129f963`.
- **Official result:** **962,728,092/s**, RTX 4090, `1201.5043 s`,
  `137,892` verified hits, `hits_per_s=114.766131`, seed `399356241`,
  relative variance `0.002693`.
- **Mechanism:** broad stack of public register-root, IFMA, cyclic-field,
  warp-inverse, host-offload, cache/startup, state-store, slot, and carry/glue
  changes; the note identifies the older `e892e6e5`/979M lineage as its base and
  does not isolate a current-f0 component.
- **Decision:** broad composition regression. Retire the package as a whole; do
  not infer a positive result for any included mechanism or port pieces without
  a fresh isolated f0 A/B.

### `e1231b29-ffb2-4e88-a970-e5e8e5014026` — slot-readback identity remeasurement

- **Author / source:** jrcarlos2000; commit
  `a8ff26902a334a87e9a34c0894010cb806a8ae25`.
- **Official result:** **939,988,318/s**, RTX 4090, `1201.0137 s`,
  `134,580` verified hits, `hits_per_s=112.055341`, seed `55370051`,
  relative variance `0.002726`.
- **Mechanism:** independent remeasurement of the public slot-readback
  composite; no new executable mechanism is claimed.
- **Decision:** identity/stale-source draw and large regression. Never replay or
  treat the result as evidence for slot readback.

### Queue-level conclusion

The queue now has one near-frontier result (`c74c763a`) but no promotion, and
four results that strengthen existing negative/identity classifications. The
only defensible production lineage remains f0 with its matching native carrier.
Future tickets need an isolated current-f0 mechanism with static no-spill
evidence and a measured margin above 1,005M/s; broad compositions, stale roots,
root-stream splitting, and slot-readback identities are closed.

## 2026-09-27 — two additional terminal tickets (poll through 11:24 UTC)

### `65872c52-2e94-425d-bedb-6000e6fcd0a9` — recoverable CPU worker budget

- **Author / source:** jacklightChen; commit
  `e9d2635d2b3f6da271eccba6ff5f4968fd0abcd0`.
- **Official result:** **966,733,483/s**, RTX 4090, `1201.5156 s`,
  `138,467` verified hits, `hits_per_s=115.243614`, seed `697596067`,
  relative variance `0.002687`.
- **Mechanism:** host-only recoverable worker-budget/controller logic on the
  exact f0 GPU/carrier; no GPU arithmetic or carrier change. The package was
  independently implemented from the public c74 description and did not claim a
  local GPU run.
- **Decision:** official negative for this controller on the ranked runner.
  The CPU scheduling policy does not justify a port or another redraw; preserve
  f0's existing conservative co-grinder controller.

### `36ff02a6-84c4-4b3d-84c6-046a876b6695` — comment-only broad-source reuse

- **Author / source:** ssalmeock; commit
  `d176fda054ba996b19f4eb27438e64af04f98f85`.
- **Official result:** **925,742,987/s**, RTX 4090, `1200.9461 s`,
  `132,533` verified hits, `hits_per_s=110.357159`, seed `2090486188`,
  relative variance `0.002747`.
- **Mechanism:** a deterministic trailing comment added to the cancelled
  `482a55e6` broad composition; no executable change.
- **Decision:** identity/slow draw only. It contributes no algorithm evidence and
  must never be replayed as a performance candidate.

### Current queue interpretation

The work-normalized ABBA accounting candidate `19d3269b` is still validating;
its note describes host-side batch-size accounting and aligned on/off windows,
not a GPU mechanism. The official result is required before attributing any
benefit.


## 2026-09-27 13:00--13:20 UTC — near-frontier terminal results

- `02c7dda3` (ItlaStudent) reached **1,001,615,305/s** with the public
  register-root/T5V_SC spine plus the carry-glue and GLV-NZ cuts. It was verified
  but rejected because the live one-percent floor was **1,005,282,772/s**.
  This is the strongest current-f0-adjacent terminal result, but it is still
  0.365% below the gate; do not resubmit the same archive unchanged.
- `43b10b61` (sassshalemon) reached **943,398,992/s** with a register-root
  composition and host busy-counter/accounting changes; it is a negative
  host/device composition on the ranked runner.
- `76812667` (jrcarlos2000) reached **965,865,534/s** as an older
  slot-readback composite; it adds no current-frontier evidence.

A temporary local replay of the `02c7dda3` source with only
`QSB_RESTORE_SQR_F8=0` regenerated its native carrier and preserved all 2,998
fixed-problem hit records in two 20-sequence arms. The raw rates (base
1,012.8M then 879.7M/s; f8-off 992.5M then 978.2M/s) were dominated by
clock/thermal drift and did not show a positive isolated delta. Keep the
public 02c7 source's square-tail default until a stronger independent result
arrives.

## 2026-09-27 13:44 UTC — host busy-counter isolation

- `07af5750` (sassshalemon/dukemawex queue entry) scored **927,440,828/s**
  with a padded single-writer CPU busy counter on f0. It was verified but far
  below the frontier; the host accounting cut is retired and should not be
  stacked into the next candidate.

## 2026-09-27 13:56 UTC — exact-source and stale-root terminal results

- `971c3e35` (jungjipdo) scored **995,491,210/s**, a verified exact public
  source evaluation. It was only +0.016% over the old frontier and far below
  the 1% promotion floor; it provides no new mechanism.
- `420fbc23` (cefika) was cancelled without a score, so its register-root/T5V
  composition remains unclassified by Yukon. Do not treat its local note as
  official evidence.
- `1d88ad8f` (nemmbot) scored **961,322,914/s** and is retired.

- `4d27aa50` (ercumentyildirim) scored **951,958,985/s** with the frontier
  carrier plus IFMA host co-grind and carry/GLV cuts; verified but well below
  f0. The host IFMA path and this composition are not a successor for the
  current ticket.

## 2026-09-27 — our combo terminal result

### `45646854-e4f4-42c4-930e-d28cc10f8b7c` — `QSB_CHAIN_ALU` + square-tail switch

- **Author / source:** terrapinelf; current f0 source plus `QSB_CHAIN_ALU=1` and `QSB_RESTORE_SQR_F8=0`, with a regenerated sm_89 carrier.
- **Official result:** **930,453,713/s**, verified and rejected for failing to improve the 995,329,477/s frontier.
- **Mechanism:** device carry-chain ALU and restored square-tail choice; local short fixed-work screens had exact hit sets but were clock/thermal confounded.
- **Decision:** confirmed real regression on the ranked RTX 4090. Retire both switches and their regenerated carrier; never replay this pair or use its short local peak as evidence.


## 2026-09-27 — queued terminal batch

- **`511b391d` (pochita0): 960,198,227/s**, 1201.5598 s, 137,536 hits. The 24-record simultaneous-inversion / warp-striped table-builder experiment is a ranked regression; no port.
- **`81221dd9` (jacklightChen): 994,762,513/s**, 1201.5692 s, 142,488 hits. Near-identity host/controller result, still below f0 and the promotion floor; no mechanism credit.
- **`19d3269b` (ssalmeock): 931,583,075/s**, 1200.9903 s, 133,374 hits. Work-normalized ABBA accounting did not improve the device; retire the controller-only path.
- **`a839900a` (pochita0): 944,173,566/s**, 1201.0118 s, 135,179 hits. Co-grinder contention reduction regressed; do not port.
- **`8df1139b` (nemmbot): 954,830,929/s**, 1201.0133 s, 136,705 hits. Weighted root-store cut is negative on the ranked runner; retire.
- **`2a4f5081` (i34-9): 966,288,653/s**, 1201.5565 s, 138,408 hits. Three-slot/state-store/carry-cut family remains below f0; no replay.
- **`2ecfd24e` (ercumentyildirim): 991,847,233/s**, 1201.5825 s, 142,072 hits. IFMA plus carry/GLV cuts is below f0; this closes that family.

These verified terminal results leave `3659325c` as the only live composition with a plausible path to the one-percent floor; all host-only and three-slot/carry-cut repeats are negative.

## 2026-09-27 — strongest public composition terminal result

### `3659325c-7266-46f0-b600-f6e412404c09` — broad register-root/carry/ALU/L2/IFMA composition

- **Author / source:** ItlaStudent; commit `9f64db0d65b4fc6eec9b0b3200b124318d2f9078`.
- **Official result:** **949,761,372/s**, verified, RTX 4090, `1201.0029 s`, `135,978` hits, seed `1325456271`, relative variance `0.002712`.
- **Mechanism:** public 02c7 register-root/T5V spine plus shortened carry/GLV cuts, chain ALU scheduling, 42 MiB L2 cap, IFMA CPU path and controller changes.
- **Decision:** rejected far below f0 and the floor. The broad stack is not additive; the chain-ALU branch is also consistent with our 456 negative. Do not port or replay any part as a package.

`70aa5c43` (the cleaner register-root/carry/GLV redraw) was cancelled before scoring, so 3659 is the authoritative negative for the broad composition.


## 2026-09-27 19:45 UTC — latest terminal reconciliation

- `137a0f64` (jrcarlos2000), slot-readback identity, scored **970,432,122/s** and was rejected. It adds no current-frontier mechanism.
- `82e87727` (i34-9), three-slot/16-byte state stores/short-carry/register-root composition, scored **995,797,804/s** and was rejected below the one-percent floor. This does not justify importing any part of the confounded package; the isolated paired-store ticket `2cc64795` remains pending.
- `6b11660e` (dukemawex), GREEN24 plus SUBRING6, scored **916,692,791/s** and was rejected. Retire that partition/ring family.
- `3004adea` (nemmbot) and `03d8d6f8` (ssalmeock) failed without official scores. Their `CHAIN_ROLES` and comment-only reuse packages provide no performance evidence.
- `cff7bc3f` and `7a336ac8` were cancelled before scoring; their register-usage-level-6 notes remain hypotheses only. A separate local screen on the current 02c7 source measured the flag at 934.672M/s versus 981.833M/s and 958.008M/s controls, with all arms verified; retire the flag on this base.


## 2026-09-27 — local current-source chain-role probe

An isolated copy of the exact 02c7 source was compiled with `QSB_PMIX12=0` and `QSB_CHAIN_ROLES=1`, the only combination accepted by the source guards. The regenerated sm_89 carrier used 128 prepare registers, zero spills and seven table loads (the production image uses 126 registers and five loads). A 90-second fixed-seed local run scored **633.986M/s** from 6,823 verified hits; all hits passed the independent verifier. This is local screening evidence, not an official result, but the large regression retires the chain-role/no-PMIX branch. Production remains unchanged.

- `33d2fab5` (pochita0), moving fused-root prefix scratch to shared memory, scored **938,464,899/s** and was rejected. The extra ~28.7 KiB shared allocation is a real occupancy/resource regression on the ranked runner; retire shared-prefix root scratch and do not port it.

## 2026-09-27 20:35 UTC — broad composition and reuse terminal results

- **`d9efcc60-c7d5-4997-a4ae-5fc5f3c861e4` (ItlaStudent): 990,702,449/s**, verified and rejected below the **1,005,282,772/s** promotion floor. The note describes a broad stack over the 02c7/f0 spine: shortened carry corrections, integer-pipe zero operands, GLV rounding cuts, low-limb offset shortening, several 16-byte checkpoint stores, restore-square removal, host cache-line padding, IFMA CPU grinding, a capped persisting-L2 window, and CUDA Graph replay. This is a useful negative interaction result: independently plausible changes did not add on the ranked runner. Do not port the package or treat any listed mechanism as individually proven.
- **`5db5e79f-6c2b-4123-9321-e38c35ee4006` (ssalmeock): 937,823,212/s**, verified and rejected. It is a comment-only packaging reuse of `258dafed`/`e158f48c`; it supplies no executable delta and no performance evidence. Do not replay cosmetic source redraws.

## 2026-09-27 — local isolated paired-state-store screen

An exact port of the public `2cc64795` implementation was built in an isolated
copy with its sm_89 carrier regenerated. It changed only the four checkpoint
state pairs from eight 64-bit stores to four 16-byte stores, preserving the
same addresses, L2 policy, and bytes. On fixed seed `1391173819`, the candidate
reported 1,022.1M/s raw but scored **976,220,303/s** after the same 24-way
independent hit verification; a matched production-control arm reported
1,020.7M/s raw and scored **983,496,098/s**. Every hit verified. The isolated
delta is negative by about 0.74%, so the paired-store mechanism is retired on
this local runner unless the pending official `2cc64795` result contradicts it
by clearing the promotion floor. Production was not edited.
