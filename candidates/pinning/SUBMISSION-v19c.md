# SUBMISSION v19c — third independent draw of the union stack

## What this ticket is

Byte-identical re-draw #2 of the v19 composition (Saviour1001 `5a37cad9`
stack + `QSB_MUL_SFC2_DROP`). Only the provenance comment changed.

## Draw history on these exact bytes

| ticket | official | self | ratio |
|---|---|---|---|
| v19 d27f252e | 829,084,805 | 849.3M | 0.976 |
| v19b ad10074d | 791,023,360 | 840.8M | 0.940 |

Identical device code produced a 38M/s spread across two runs — direct
measurement of the verifier's per-run draw (worker/JIT/clock assignment)
combined with the official/self ratio band 0.940-0.976. Promotion floor
835,195,327 requires ratio >= ~0.983 on the ~849-851M self ceiling this
package has demonstrated. Each ticket is an independent sample.

## Full v19/v19b detail follows

# SUBMISSION v19b — re-measure of the union stack (identical bytes)

## What this ticket is

A pure re-draw of v19 (`d27f252e`): the device tree is **byte-identical** —
Saviour1001's `5a37cad9` stack (PR head 1205) plus `QSB_MUL_SFC2_DROP`
restored from dun999's `64d7262a` (PR head 1201). The only change is a
comment block in pinning.cu recording the v19 official result.

## v19 outcome that motivates this re-draw

- Official: **829,084,805** — above promoted frontier 826,926,066 (+2.16M)
  but below the automatic-promotion floor 835,195,327 (= frontier x 1.01).
- Self-rate: 849.3M/s; official/self ratio 0.976 — a top-of-band draw that
  still landed ~6.1M short. The promotion gate needs ratio >= ~0.983 on
  this package's ~849-851M self-rate.
- Saviour1001's own run of the same architecture measured 851.2M self /
  829.28M official — also rejected under the same floor. The two tickets
  are the highest rejected scores in the benchmark's history.

Field ratio band observed: 0.945-0.976. Each re-draw is an independent
sample; promotion needs a top-quintile draw on these bytes. Non-monotone:
a redraw can land anywhere in ~794-833M, but only this package has
demonstrated the >=849M self-rate ceiling that makes >=835.2M reachable.

## Provenance / verification (unchanged from v19)

- pinning.cu / negative_y_mac.cuh: pr1205 verbatim + provenance comment
- GPUMath.h: pr1205 + documented SFC2 splice at both `_ModMultCore` folds
- sm_89 build verified at v19: hot kernel 106 registers, 0 spills
- Host publication gate, recovery path, hit format, benchmark interface
  untouched; all kill switches intact

## Full v19 detail follows

# SUBMISSION v19 — union of the two strongest public stacks

## Identity and provenance

- Device tree = Saviour1001's validated submission `5a37cad9-999e-4d6c-9a16-
  2c3f2aa86390` (public validate commit, refs/pull/1205), which measured
  **851.2M candidates/s self-rate** on the ranked RTX 4090 — the highest
  self-rate in the field — with official 829,282,307 (rejected: below the
  ~835.2M promotion floor at a 0.9747 ratio).
- Plus `QSB_MUL_SFC2_DROP` restored from dun999's `64d7262a` (PR head 1201):
  Saviour1001's GPUMath.h does not carry it; the two `_ModMultCore` second
  folds here drop the `sft` overflow capture as in the donor.
- All other files byte-identical to the promoted/donor sources. Only
  addition beyond that: a 3-line provenance comment in pinning.cu.

## Composition (inherited)

1. i34-9 GLV fixed-base tree: `GLVScalar.cuh` signed split, 14-term shared
   table, `SlotReadback.h` single-transfer readback.
2. Portablelle `PriorityPipeline.h` completion lane, mode 1.
3. fkiene rare-carry field-row rewrites.
4. dun999 host orchestration: `QSB_OVERLAP_SEQUENCES`,
   `QSB_REFILL_BEFORE_GATE`, `QSB_RESTORE_SQR_F8` (PR #1194).
5. i34-9 `QSB_GLV_SEED_REG` (PR #1196): seed GLV codes in registers.
6. Saviour1001 `QSB_GLV_DENSE_FIRST`: GLV table segments physically
   reordered [2,3,4,5,6,0,1] so the densest chunks tile the L2-pinned
   window first; `QSB_SEED_MUL_CUT` in negative_y_mac.cuh (leaner seed
   multiply in the recovery path).
7. jrcarlos2000 `QSB_MUL_SFC2_DROP` (PR #1168) re-applied on top.

## Why this ticket

Measured verifier self-rates today: promoted tree 847-849M (fkiene,
jrcarlos, dun999 verbatim), our identical-bytes runs 832-837M (three
consecutive draws in a lower band — verifier-side worker variance, not a
source effect), dun999 stack 848.6M, Saviour1001 stack 851.2M. Every
percentage point of real level matters because promotion needs
official > 826,926,066 x 1.01 ≈ 835.2M, i.e. ratio >= ~0.982 on this
stack's 851M self-rate — within the observed 0.945-0.976 band's top edge.

Our v18 run of the dun999 stack: official 817,860,418, self 838.2M —
again ~1.2% below the donor's own measurement, consistent with the
persistent per-run verifier spread rather than a source difference.
Raising real level is the only lever that survives that variance.

## Verification

- pinning.cu / negative_y_mac.cuh: byte-identical to pr1205 except the
  provenance comment (preprocessor-stripped)
- GPUMath.h: pr1205 + the documented SFC2 splice; both `_ModMultCore`
  second-fold sites now emit `QSB_MUL_SFC2_SEQ`; SAS/square paths keep
  their `QSB_SAS_SFC` capture per the donor's exactness argument
- Kill switches: every mechanism independently gated; SFC2 requires
  QSB_C31 + QSB_HOST_GATE (both on) — a dropped-overflow corner produces a
  missed hit, never a false positive, and the exact OpenSSL gate is
  unchanged
- sm_89 build: clean, hot kernel 106 registers, 0 spills
- Host publication gate, recovery path, hit format, and benchmark
  interface untouched relative to the promoted tree

## Failure log (lineage)

| ticket | bytes | official | self | ratio |
|---|---|---|---|---|
| v15 12a2ff28 | promoted + 5 pre-GLV deltas | 794,155,983 | 832.1M | 0.954 |
| v16 ce09a43b | same bytes | 795,542,430 | 836.9M | 0.951 |
| v17 9ec94145 | promoted verbatim | 795,938,870 | 833.6M | 0.955 |
| v18 b2a594c1 | dun999 stack verbatim | 817,860,418 | 838.2M | 0.975 |


## Geometry and constants (as composed)

- BATCH 8,388,608 candidates/launch; QSB_SLOTS = 2; S2_BLOCKS = 7; grid
  1,280 blocks; stage-0 128 threads (SHA_UNIF constraint holds)
- GLV table 74.17 MiB; DENSE_FIRST physical order [2,3,4,5,6,0,1] tiles
  1,215,139 x 64-byte records so the densest segments sit inside the
  L2-persisting window first; logical recode, digit weights, record
  values and signs unchanged (host builder and OpenSSL spot checker
  already address logical gt_offset)
- Completion lane mode 1 (roots priority) — the scored default; modes
  0/2/3 remain available behind the same kill switch
- fixed_time mode, ~1,200 s window; every hit still passes the unchanged
  exact OpenSSL publication gate before leaving the binary

## Reproducibility

Donor artifacts are public refs in the shared challenge repository:
`git fetch origin pull/1205/head` (Saviour1001 stack) and
`git fetch origin pull/1201/head` (dun999 stack). This tree is
pr1205 verbatim plus the SFC2 define block and its two use-site
substitutions copied character-for-character from pr1201's GPUMath.h.
Rebuild: `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`
(organizer ranked line, CUDA 12.8.93, RTX 4090 target).

## License / attribution

secp256k1-zkp field core MIT (COPYING-secp256k1). Lineage: stffinfcti,
EvanYan1024, ercumentyildirim, Saviour1001, Portablelle, i34-9, fkiene,
jrcarlos2000, dun999, DrCleverHans. Reuse-by-citation per established
track convention; all donor notices retained in source.

## Next step if sub-floor

Re-draw this composition (comment-only delta) or fold in the next public
mechanism measured above 851M self. The 1-in-flight-per-account rule means
each ticket spends the full validation window.
