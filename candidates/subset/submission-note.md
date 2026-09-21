<!-- Do NOT put Model:/Harness: lines here: yukon submit --model/--harness prepends them. -->
Effort: high
This work is produced by the **agentprivacy dual-agent harness**, running as the `qpcbtc_mage`
instance: a seated loop in which a proposing seat plans exactly one lever through a named lens, a
hold-apart seat draws verification witnesses by hashing the proposal so the prover cannot choose its
own test set, an adversarial prover seat measures the lever and returns a verdict, and a critic seat
classifies what closed and names the next lead. Levers that fail a gate are recorded as killed rather
than retried, and every claim below is a measurement made under that discipline rather than an
expectation.

# Subset: exact host publication gate replaces the GPU verify kernel, and the unreachable direct producer leaves the fatbin (on 9ac2515)

## What changed (candidates/subset/ only)

`kernel_verify_pair_hits` — 21,728 sm_89 instructions, 136 registers, launched `<<<1,64>>>` once per
batch to recompute at most 1024 tentative hits — leaves the fatbin. Its job moves to the host
(`tests/gpu_epochs/qsb_host_verify.h`, ~120 lines of OpenSSL): each tentative record's `(epoch rank,
lane)` is rebuilt into its nine skip indices with the same `qsb_host_unrank` the batch loop already
uses (never from the tentative's combo bytes), the preimage is reassembled exactly as
`harness/verify.py` reads it, `SHA-256d` runs from the committed midstate, `Q = u1·G ± u2R` is
recovered with OpenSSL, `SHA-256(compress(Q))` is checked for `N` leading zeros, and the GPU's recid
is tried first, the other recid second. Only records that pass are written, in the same
`indices=… recid=…` line format. The verification of batch *i* runs on the host while batch *i+1*
grinds, so it is off the GPU's critical path; the last batch in flight at SIGTERM is dropped exactly
as before, and a pending batch is drained on natural loop exit. Two cosmetic consequences, stated so the
logs are not misread: the kernel's own hit counter (progress line, STATUS= line) now lags the hit file by
one batch, and the 1024-record tentative cap is unchanged from the frontier (≈16–32 expected per launch at
N=24; low-N diagnostic runs must shrink ZLAB_LAUNCH_BLOCKS as HIT-CHECK.md already prescribes). `-DQSB_HOST_VERIFY=0` restores the e876032 device code byte for byte. Nothing else changes:
speculative filter, enumeration, counter, table, launch geometry, hit format. A second, smaller change in the same
mechanism: `kernel_build_epochs` (3,120 instructions) is reachable only from a guard the code marks "Cannot happen for the
pinned 6-of-137 shape"; behind `QSB_TRIM_DIRECT_PRODUCER` that guard becomes a hard error and the kernel is compiled out
(5 kernels, 2.11 MB PTX, ptxas 2.1 s). Every reachable input behaves identically.

## Why: the timed window contains work that is not grinding

The organizer's `benchmark.sh` output for every ranked subset run carries the warning "verified hits
outside expected Poisson band … candidate count may be misreported". Mapping the hits of ranked run
35587149800 (public diagnostics artifact) back to candidate indices shows the kernel enumerated
718.7×10⁹ candidates while its counter extrapolated 840.2×10⁹; on the enumerated work the hit yield
is 0.9924 ± 0.0034, so no hits are lost — the kernel simply did not grind for ~174 s of the 1201 s
window (14.5%). The promoted 9ac2515 run (35619323301) shows 160 s. Seven promoted runs in the
lineage show 131–195 s; pinning shows 29–34 s on Intel hosts and 48–52 s on the LeaderGPU-class host at the same peak rate.

Locally (RTX 4060, driver 610, nvcc 12.4) that time is the driver's JIT of the organizer's
compute_52 PTX for sm_89: with the JIT cache disabled the binary's search clock starts 4.8–11 s after
launch (0.76 s with a warm cache), and `ptxas -arch=sm_89` on the PTX takes 5.1 s, of which the
verify kernel is 35–37%. On the promoted 9ac2515 tree this change cuts the PTX from 3.52 MB (7 entries) to 2.33 MB
(6 entries) and `ptxas -arch=sm_89` from 3.7 s to 2.2 s (on the previous crown e876032: 3.37 → 2.18 MB, 5.1 → 3.2 s). The ranked-box cause is not established from public data alone (a
CPU-limited sandbox, thermal throttling, or a failed table spot-check are all consistent with the
lineage); if the JIT share on the runner is what it is locally, the expected gain is about
+5% of score, if not, this change is neutral. It is submitted as a measurement of that question.

## Correctness evidence (local, RTX 4060)

- Identity on the promoted 9ac2515 tree, harness-drawn problem seed 1842180 at N=24, 60 s each: the frontier binary and this
  binary publish identical hit sets over the common enumerated prefix (471 = 471 over 4,291,735,936 candidates for this bundle;
  495 = 495 for the host-gate change alone), every hit re-derived by the unchanged `harness/verify.py` (496/496 and 472/472).
- Cold start on that tree (JIT cache disabled, interleaved): frontier 5.07 / 4.28 s → 2.55 s to the kernel's clock start.
- Kernel counter rate, four 60 s arms frontier/bundle/bundle/frontier: −0.13% (484 verified hits in every arm), i.e. unchanged
  within the 4060's ±1% drift; the host gate handles ≈15 tentatives per batch.
- A first re-based build published zero hits because it decoded tentative tags with the previous 256-window layout; the gate
  caught it locally and the decode now reads the tree's own per-epoch constant. Recorded here so the mechanism is auditable.

- On the previous crown e876032 the same change was gated the same way: identical hit sets on two
  harness-drawn seeds (664 = 664 and 633 = 633, all re-derived by `harness/verify.py`), cold start
  −42% / −46%, 15.4 tentatives per batch, counter throughput −0.12 ± 0.77% pooled over 16 arms.

## Provenance

Promoted runtime 9ac2515 (Akashneelesh: three exact chain-loop deletions on the dun999 negfold + windows-128 +
EvanYan1024 parity-window composite carried by terrapinelf), on e876032 (jacklightChen) and the credited subset lineage: owizdom, DPZZxlz, Meganpark980320, odinfree, AbdelStark, i34-9, scarletbright, fkiene,
anamdongparkjinhyeong, jrcarlos2000, terrapinelf, Saviour1001; VanitySearch GPL primitives, COPYING
retained). The host publication gate follows the pattern of the pinning frontier's `QSB_HOST_GATE`
(Saviour1001, dcd0147). The measurement of the counter/score gap, the census tooling and the
host-verification port are this submission's work.

## Reproduction

```bash
./setup.sh subset && QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu --no-build' \
QSB_SECONDS=60 QSB_PROBLEM_SEED=1842180 ./benchmark.sh subset
nvcc -O3 -DQSB_ZEROS_N=24 -ptx -o k.ptx candidates/subset/subset.cu && grep -c '^.visible .entry' k.ptx   # 6
```
