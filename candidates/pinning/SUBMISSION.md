# Pinning: QSB_RAW_X (finish products skip normalize)

Base: `dcd0147c` **promoted 789,011,576** (`66fede0`, RTX 4090). Land bar **796,901,692**.

`4ce3607` (`QSB_FKIENE`) **rejected 762,268,179** (−3.39%, 109,172 hits, 90.87 hits/s, RTX 4090). Promoted 789.0 M is ~112.9k hits / ~94.1 hits/s over 1200 s. FKIENE lost ~3.7k hits of throughput, not of packing: 109k verified hits means idx/neg matched the ranked seed. Register `codes[15]` did not beat volatile shared. **`QSB_FKIENE=0`. Do not retry.** `b0fbfb1a` (`QSB_RP_SQR`) rejected +0.26% with `f8`. **`QSB_RP_SQR=0`. Do not retry `f8`.** One switch on the promoted arithmetic (743 mul tail + carry62 + C31 + host gate + promoted decoder). SHA flags 0.

## Why RAW_X

`qsb_recovery_mul` is `qsb_field_mul_sc` plus `qsb_field_normalize` to `[0,p)`. The packed finish already accepts `[0,2^256)` representatives on every multiply that is not hashed and on `qsb_sum_parity`. Two remaining canonical muls sit on the hashed x products:

```
s = sum * (l - c)     // r1 = x1 - a
s = sum * (m - c)     // r2 = x2 - a
x_i = s + a
```

`QSB_RAW_X=1` uses `qsb_packed_raw_mul` for those two, then `qsb_add_boundary` (normalize only if `a[3]==2^64-1`) and the existing `_ModAdd256(x,s,a)`. Let `B=2^256`, `p=B-K`. If `a[3] != 2^64-1` then `a ≤ B-2^192-1`, so `raw+a < 2p` and one conditional subtract of `p` is canonical. `a` is a coordinate of `u2·R` (`< p`). `qsb_sum_parity` already documents raw `w ∈ [0,B)`.

The previous `#if QSB_RAW_X` block computed `x1,x2` and was then overwritten by `QSB_PARITY_SUM=1`. That made RAW_X a no-op on the production finish. This archive wires the raw products **inside** the parity path. `-DQSB_RAW_X=0` restores `qsb_recovery_mul` on those two sites.

`PackedRecovery.cuh` `qsb_packed_finish` with `QSB_PARITY_SUM=1` (production):

```
_ModSub256(t, l, c)
s = packed_raw_mul(sum, t)     // was recovery_mul = mul + normalize
qsb_add_boundary(s, a)         // no-op unless a[3] == 2^64-1
x1 = s + a                     // _ModAdd256, one p-subtract
u  = packed_raw_mul(l, s)      // unchanged; feeds qsb_sum_parity
```

Same for `(m, x2, v)`. Two `qsb_field_normalize` calls deleted per candidate. The 15-add comb, the signed-odd decoder (`qsb_decode_to_shared`), and `GPUMath.h` are unchanged. Host gate stays on because C31 is still in the promoted ancestor.

`pinning.cu`: `QSB_RAW_X 1`, `QSB_FKIENE 0`, banner `RAW_X: on`. `LeafRecovery.cuh` is the unused per-leaf path and is not this switch.

## Not this switch

FKIENE, RP_SQR/`f8`/`sfc`, F16 extra `z8`, SAS `g8` only, `cp.async`, C6-14 orbit comb (host recoder first), joint last-window+`±R` (product count first), GLV, unroll, top-16, Karatsuba, SHA ST, `_ModAddLazyOff` t1.

## Evidence

`GPUMath.h` unchanged (`713ca68723cce6f4e168e77523d01bdf5f9e62c3440b56bd70ab4ed45dce9343`). Ten production files, 374938 bytes, SHA-256:

```
3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986  COPYING
8cf9b303b6f5a09051e433a8bc21e2f3a66631e3bd22b2b1a10e71fe0b21b2bc  GPUHash.h
713ca68723cce6f4e168e77523d01bdf5f9e62c3440b56bd70ab4ed45dce9343  GPUMath.h
b09d39fe723810215c916bffcb9ba01f57375d8d9ed059da66900a45ad00f1c3  LeafRecovery.cuh
2206b1d27fda5b06cf5b84b31ebeac71c434b929e8dbc7ce82c9a508be2643aa  PackedRecovery.cuh
6f6c0347ab0bb4abca13b2cdbb9294a997c9b3c6a076fd7ed7493e531ac0e369  RecoveryConstant.h
2c551a75fce84d0e10a780b2689f48eb19b609cf6bab31450c165a8806223ed6  cofactor_checkpoint.h
a8043dcb6b939c86ad453e0550ce44ddb2eaf6b25a06f6bb78a355f3c61a24c1  pinning.cu
bd811f32f3560fe5fd694f4afd4990c3da451819dd486579d74695cb85ed5462  sha_pinsha.cuh
629417bb86ff908078b8f1358a2b2773b44f02f1c88c3bce9f61a622e24b403f  sha_schedule_interleaved.cuh
```

Host, no GPU:

```
python3 candidates/pinning/test_raw_x.py
python3 candidates/pinning/test_host_gate.py
python3 candidates/pinning/test_carry62.py
python3 candidates/pinning/test_fkiene.py
```

`test_raw_x.py`: 4000 random `raw ∈ [0,B)`, `a < p`: `_ModAdd256(add_boundary(raw,a), a) == (raw+a) mod p`; `a[3]==MAX` forces normalize. Source: `QSB_RAW_X 1`, `QSB_FKIENE 0`, `QSB_RP_SQR 0`, parity path calls `qsb_packed_raw_mul` + `qsb_add_boundary`. `test_host_gate.py`: 64 SHA-256d midstates, both recids match `harness/problem.py:candidate_hash`. `test_carry62.py`: C31 predicates; RP_SQR macros present, default off. `test_fkiene.py`: funnel still equals the old extract (code remains behind `#if QSB_FKIENE`).

Banner prints `RAW_X: on` and `FKIENE: off`. Ranked: `nvcc -O3 -DQSB_ZEROS_N=24 -o pinning pinning.cu -lcrypto -lm`, CUDA 12.8.93, 1200 s, RTX 4090. No local device run.

Expected band: **0–1%** finish-only. May miss 796.9 M. Honest leftover after FKIENE, not a 1.0 G jump.

## After the official score

Keep `QSB_HOST_GATE` (C31 is still on). If this promotes, the new record is the floor; next is SAS `g8` only or `cp.async` of table chunk `c+1`, not a bundle. If a near-miss above 789 M, same next list. If ~0 hits, `qsb_add_boundary` vs `_ModAdd256(s,a)` on the ranked `a=u2R`. If ~−3.7%, Issue #505 host draw. Do not revive FKIENE or RP_SQR. Subset `c29af055` (YOFF) is validating separately; this archive does not touch `candidates/subset/`.
