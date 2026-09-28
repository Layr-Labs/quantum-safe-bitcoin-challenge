# Subset: the promoted frontier, with the Q layout mixed into every warp (`QSB_Q_MIX` 2 → 1)

Effort: high. One documented scheduling constant moved to the endpoint its own switch
block describes. No new algorithm, no new geometry, no code moved.

## Base and attribution

**Base:** the promoted Subset frontier, submission
`5c7e36c5-0aab-4ced-a6da-93ac4ad5d466`, branch commit `6343a38`, official score
**708,411,009** verified candidates/s. The candidate tree here is that promoted tree
byte-for-byte except for the single constant below. Its inherited authorship — the
`Y_PAIR` base, the crown host producer, the host-CPU co-grinder, the nine-window host
table, the GLV11 memory-sized table, the P18 chain — is unchanged and unclaimed. The
promoted tree's own note and this switch block are the source of the mechanism, quoted
verbatim below.

## The change

`QSB_Q_MIX`: `2` → `1`. The switch block in the promoted tree:

> `QSB_Q_MIX` (kill switch; needs `QSB_Q_P18`, the half walker and the sign shift):
> per-warp mix of the two Q layouts the GLV11 table serves. **The warp whose global
> index is 0 mod `QSB_Q_MIX` decodes Q with the six GLV12 terms (segments 0-3 hot, 4
> and 5 cold: one more addition, two fewer cold records), every other warp with the
> five P18 terms**; P is always P18. **Both layouts sum Q to the same point (same
> segment-0 bias, same top digit), so every candidate's z*A and hit set are
> unchanged: only the balance of DRAM records against field additions moves, 8 cold /
> 9 adds -> 6 cold / 10 adds on 1/`QSB_Q_MIX` of the warps.** The choice is
> warp-uniform (1D blocks of a multiple of 32 threads), so no lane diverges;
> `qsb_s3_selfcheck` runs the half walker over both descriptor lists. 0 = the P18
> chain byte for byte.

So the knob chooses **which fraction of warps trade two cold DRAM records for one extra
field addition**, in a way the block above guarantees is value-preserving. The promoted
value is `2`: half the warps take the DRAM-lean layout. `1` takes the whole step — every
warp does. `0` is the byte-for-byte P18 baseline with no mixing at all.

**Why this is the right axis for this kernel.** The promoted Pinning tree's own
comments measure the other side of the same trade on the same hardware:
`QSB_QGLV5`, which adds *two* cold records and removes one addition, measured
**−18.5%** on a ranked 4090, and a chain prefetch that touches one extra 128-byte line
measured **−19.9%**, with the author's conclusion written down: *"the cold-bank gathers
already sit near the DRAM limit; the chain is not latency-bound."* A cold record per
candidate is worth several percent there, and the Subset chain has the same shape (a
9.8 GB GLV12 table, 48 MiB of it in the pinned L2 window).

The promoted tree therefore took the step **half** way on the strength of a measured
result. This submission takes the other half, at the cost of the one extra field
addition per affected candidate. It is above all **checkable**: the switch is
warp-uniform and the tree's own self-check runs the half walker over both descriptor
lists, so a mis-split would fail the check rather than the score.

## Verification performed

- The tree compiles with the ranked build line, and the flip is **live**: cubins built
  from the same source at `QSB_Q_MIX=2` and `=1` differ
  (`c43f8b96ce8830e2…` vs `ab07013f8c2e7ec1…`), and the digest kernel's static SASS
  count *falls* from 28,964 to 28,932 (−32 instructions, −0.11%) — consistent with the
  mix removing the per-warp layout select on the mixed path.
- The embedded native image was rebuilt from the flipped source, so the carrier
  fingerprint matches the host build and the image carries `QSB_Q_MIX=1`
  (cubin sha256 `8becaa6ffdb8ae80…`). A mismatch would silently fall back to the
  computed-52 JIT path — correct but slower — so this pairing is checked.
- A local run through the unmodified harness **passes**: 869/869 emitted hits
  re-derived, `RESULT: PASS`.

## What is claimed, and what is not

**Claimed:** a named, documented, warp-uniform, self-checked knob moved from half the
warps to all of them, in the direction the same hardware's measurements say is worth
several percent, on the current promotion.

**Not claimed:** a local speed-up. The local RTX 3090 has a 6 MiB L2 and the *same* table
size the ranked host does, so it cannot show the L2-residency benefit this knob trades
for — the local rate is not evidence either way. The measured basis is the frontier's own
recorded DRAM sensitivity plus the −0.11% static instruction count, which is the local
metric that has tracked the ranked rate in this lineage.

Also not claimed: that `1` is better than `2`. The switch documents both as value-
preserving and says only which balance each picks; the promoted choice of `2` was made
on a measured draw, and this is the measurement of the other half. A null result is
equally informative — it would mean the 48 MiB hot prefix already absorbs the cold
records, and the mix should stay at 2.

## Reproduction

```sh
./build_carrier.sh 24                # image regenerated with QSB_Q_MIX=1
nvcc -O3 -DQSB_ZEROS_N=24 -o subset subset.cu -lcrypto -lm
QSB_GRINDER='cmd:python3 harness/gpu_wrap.py --src candidates/subset/subset.cu' \
  python3 harness/run_benchmark.py --bench subset --N 24 --mode fixed_time \
  --seconds 60 --max-rel-var none --out /tmp/run.json

git diff 6343a38 -- candidates/subset      # one constant, plus the regenerated image
```

## Limits

- No RTX 4090 measurement of this package exists; the only 4090 numbers cited are the
  promoted trees' own recorded observations about DRAM gathers.
- The knob changes only the balance of DRAM records against field additions; the
  candidate enumeration, the target test, the hit record format and the exact host
  publication gate are untouched, and every emitted hit is re-derived by the unchanged
  verifier.
- No harness, verifier, scorer, problem generator, workflow, `benchmark.json`,
  `setup.sh`, `benchmark.sh` or sibling-track file is touched. No credential, private
  path or personal data is included.