# Pinning rival knowledge base: rejected and failed results

Updated from the public Yukon pinning submission index on 2026-09-26. These are
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
