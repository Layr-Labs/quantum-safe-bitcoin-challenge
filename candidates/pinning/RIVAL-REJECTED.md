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
