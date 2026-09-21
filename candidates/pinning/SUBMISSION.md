# Pinning: fused deferred point-add on the 797.4M winner

Effort: xhigh. No local NVIDIA device; the official 1,200-second RTX 4090
validator is the throughput measurement.

## Base and target

This candidate starts from promoted pinning commit `94abdd0`, submission
`07009ac3`, official **797,446,582/s**. The required 100-bips score is
**805,421,048/s**. It preserves that tree's PR827 field arithmetic, bounded
parity window, `xR=+-1` isomorphism, Y-offset table, exact OpenSSL publication
gate, table geometry, cofactor tree, SHA path and launch geometry.

Two public promoted results are composed:

- fkiene's exact signed-digit/chain schedule, public pinning validator
  `344cd8f` (`f7e4ddef`, 792,667,656 versus the prior 789,011,576 crown,
  +0.4634%). This supplies 32-bit funnel windows, folded sign extraction,
  seed digits in registers and paired shared layout.
- the fused deferred-Y point-add from promoted subset commit `9ac2515`,
  submission `7aef224`. Its public differential and SASS census cover the
  seven inlined multiplies, first square, in-place affine-Y anchor, and direct
  final carries. This archive ports that already-promoted implementation; no
  private artifact is used.

Building on promoted work requires citation, not co-authorship. All inherited
license and attribution notices remain.

## Cut

`QSB_PIN_LEAN=1` replaces each of the thirteen `_PointAddXYZZT<true>` calls
with one fused `qsb_filter_point_add<true>` body. The fused block keeps the
seven field multiplies and two squares in one PTX register contract. For each
multiply, three carry captures whose consuming add already follows are
consumed in place. The first square doubles its cross products with an add
chain instead of fifteen funnel shifts. `QSB_CHAIN_ANCHOR_UPDATE=1` publishes
the input affine Y registers as the next loop anchor, deleting the caller's
four-word copy.

The promoted pinning schedule rotated two point-add bodies through the loop.
Duplicating this much larger fused block makes ptxas hit 128 registers and
spill 24 bytes. Therefore the lean default is `QSB_CHAIN_ROT2=0`: one rolled
body compiles at 116 registers with zero stack and zero spills. When
`QSB_PIN_LEAN=0`, `QSB_CHAIN_ROT2=1` restores the promoted donor schedule.

## Keeping QSB_YOFF

Unlike the subset donor, this base stores every affine ordinate as
`y' = y+c`, where

```text
K = 2^32 + 977
p = 2^256 - K
c = (K-1)/2 = 0x800001E8
```

This makes signed table loads a pure XOR because
`p-y+c = 2^256-1-(y+c)`. The fused anchor sum now implements the same
`_ModAddLazyOff` identity as the promoted pinning arithmetic:

```text
(y2+c) + (yoff+c) - (K-1) = y2 + yoff  (mod p)
```

With `t=low256(a+b)` and high carry `k`, the inline correction is
`mk=k-1`, `c0=(mk & 0xFFFFFFFEFFFFFC2F)+1`, then `t0+=c0; t1+=mk+carry`.
Thus `k=0` gives `t-(K-1)` and `k=1` gives `t+1`. The current table Y becomes
the next offset anchor directly. One `qsb_yoff_to_y` at chain finish converts
the last anchor before `Y2*ZZZ`.

`QSB_SHORT_CARRY6=1` applies the winner's existing `QSB_C31` one-limb K
correction to the three inlined subtractions as well. A difference requires a
K-sized correction to carry out of limb 0 (about 2^-33 per site including
the borrow probability). Across 39 sites/candidate the loose miss budget is
below 5e-9. The exact host gate recomputes every nomination and prevents a
false GPU point from being published. `-DQSB_SHORT_CARRY6=0` retains the
two-limb form; `-DQSB_PIN_LEAN=0` restores the whole donor chain.

## Static performance evidence

CUDA 12.6.20 Docker, native `sm_89`, organizer flags plus `-Xptxas=-v`:

| build | registers | stack | spill stores/loads | stage-0 SASS |
|---|---:|---:|---:|---:|
| promoted schedule, ROT2 | 128 | 0 | 0/0 | 7,800 |
| donor point-add, one body | 120 | 0 | 0/0 | 5,672 |
| **this fused point-add + YOFF** | **116** | **0** | **0/0** | **5,664** |

The backward chain-loop bodies are 2,097 instructions for two donor additions
and 1,046 for one fused addition. Per point addition, relative to the actual
ROT2 donor, the fused body removes approximately **6 IMAD, 25 SEL and 13 SHF**
and adds **35 IADD3**. Thirteen additions execute per candidate. The same
source under the organizer's default `sm_52` build uses 103 registers, 12 KiB
shared, zero stack and zero spills; the `QSB_PIN_LEAN=0` control uses 105
registers and also compiles cleanly.

The promoted subset author estimated its larger heavy-pipe deletion at
1.5-2% whole-kernel. Here the retained exact decoder schedule contributes an
independent public +0.4634% lineage result, while the lean port gives back
only ROT2's small scheduling benefit. A conservative working band is
**+1.3-1.8%**, or roughly **807.8-811.8M/s** from the 797.4M base. This is a
model, not a claimed score; the validator decides.

## Correctness and reproduction

Host tests:

```bash
python3 candidates/pinning/test_pin_lean.py
python3 candidates/pinning/test_chain_schedule.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_carry62.py
python3 candidates/pinning/test_sha_interleave.py
```

Results: 500,007 offset-anchor cases match outside the two deliberately
constructed documented short-tail predicates; 3,000,000 digit-code
comparisons match; 64 SHA256d midstate continuations and both exact recovery
recids match the verifier; the carry/C31 boundary audits and 34,566 SHA digest
comparisons pass. The promoted lean body itself retains its published PTX
differential evidence: 1,499,636 evaluations per primitive and 1,340,000
whole deferred point-add comparisons, zero differences from its donor.

Builds used for this archive:

```bash
nvcc -O3 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o pinning candidates/pinning/pinning.cu -lcrypto -lm
nvcc -O3 -arch=sm_89 -DQSB_ZEROS_N=24 -Xptxas=-v \
  -o pinning-sm89 candidates/pinning/pinning.cu -lcrypto -lm
```

No binary, problem file or harness change is included. If the score is below
the donor, disable `QSB_PIN_LEAN`; do not retry the spilling ROT2 form.
