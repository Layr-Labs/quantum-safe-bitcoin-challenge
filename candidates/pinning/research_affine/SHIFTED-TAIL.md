# Affine window walk with the recovery pair folded into its last table

This is an isolated research path. The mathematical fast path is implemented
and audited in `shifted_tail.cuh`; the production candidate is unchanged.
It targets a reduction in field operations, but no GPU speed or recall result
has been measured for the resulting solver architecture.

## Construction and signs

Write the original-curve fixed scalar point as `A = neg_r_inv * G`. The active
signed odd-digit decomposition uses 15 windows: width 18, then width 17. It
expresses the target scalar point as the sum of signed odd multiples of
`2^offset * (A/2)`. The final offset is 239. Its positive table entry is

```
L_j = (2j+1) * 2^238 * A,       0 <= j < 65536.
T_0(j) = L_j + R,              T_1(j) = L_j - R.
```

The first 14 windows leave an affine prefix `P`. Instead of adding `L_j`,
normalizing a projective result and recovering both signs of `R`, finish with
the two affine additions `P + T_0(j)` and `P + T_1(j)`. This is ordinary group
associativity. For a negative final digit, the sign rule is essential:

```
T_recid(-L) = -T_(recid XOR 1)(L).
```

Thus load the opposite stored arm and negate its ordinate. Negating both
stored arms without swapping them would reverse the recovery identifiers.
`qsb_shifted_tail_load` performs this swap and negation.

Each entry stores two canonical affine points, 128 bytes. Replacing the final
4 MiB table with this 8 MiB table increases all fixed-base tables from 64 MiB
to 68 MiB. Build these points on the original curve. The production recovery
isomorphism and its weighted inverse do not belong in this path.

## One denominator per candidate for both final outputs

Let `P=(x,y)` and `T_r=(a_r,b_r)`, and define

```
delta_r  = a_r - x
n_r      = b_r - y
D        = delta_0 * delta_1
lambda_r = n_r * delta_(r XOR 1) * inverse(D)
x_r      = lambda_r^2 - x - a_r
y_r      = lambda_r * (a_r - x_r) - b_r.
```

When both deltas are nonzero, cancellation gives the usual affine addition
slope for each arm. Consequently the produced x coordinate and y parity are
identical to the full two-step group calculation. A compressed key only
needs the parity, so its last line can call the inherited parity window as
`parity_product(lambda_r, x_r-a_r, b_r, 1)`. **Both arms use negation flag 1**
because each stored point already includes the correct sign of `R`.

The tail costs `5M + 2S + 2W` before the batch inverse: one denominator
product and two multiplications per slope. `W` means the existing
parity-product window, not a full field multiplication. Prefix infinity,
tail infinity, doubling and cancellation with a zero delta explicitly decline
this fast path. A complete solver must send those cases to an exact fallback;
the research primitive does not claim complete exceptional-case handling.

## Complete operation comparison

Let `M`, `S`, `W`, `I` denote a field multiplication, dedicated square,
parity-product window and inversion. These symbols do not imply equal GPU
latencies. Counts exclude field additions, normalization, loads, barriers,
SHA256, setup and exceptional-case work.

The active baseline scalar chain is `3M+2S` for its affine-pair seed,
`13*(7M+2S)` for mixed additions and `1M` to resolve its final deferred
ordinate: **95M+28S**. Recovery adds `8M+2W` outside the collectives: one
denominator product, three packed-prepare products, two slope products and
two x-coordinate products. The recovery isomorphism makes its fixed x factor
a sign and adds no per-candidate multiplication there.

For the active `QSB_TOP16` traversal with `N=128`, the local cofactor work is
`112+63+96+128 = 399 = 3N+15` multiplications. This is more than the minimal
inverse-tree work because the merged schedule duplicates arithmetic to remove
synchronization waves. The 256-root hierarchy adds its product/downward
passes, one ordinate weight per local root and one isomorphism weight at the
outer inverse. With the default full batch `B=2^23`, all recovery collectives
therefore total `3B + 19*(B/N) - 2` multiplications and one inversion.

The new worker instead uses a complete local inverse tree costing `3N-3`
multiplications and **one inversion per N candidates in each wave**. Thirteen
ordinary affine additions cost `13*((5-3/N)M+S+I/N)`. The paired tail costs
`(8-3/N)M+2S+2W+I/N`. Its inverse service preserves this local `N=128` boundary;
there is no extra global inversion hierarchy in this prototype.

| Path, per candidate | M | S | W | I |
| --- | ---: | ---: | ---: | ---: |
| Active baseline, B=2^23, N=128 | 106.14843726 | 28 | 2 | 1 / 8388608 |
| Affine target, N=128 | 72.671875 | 15 | 2 | 14 / 128 |
| Initial exact reference probe | 89.671875 | 0 | 0 | 14 / 128 |

The target formula is `(73-42/N)M + 15S + 2W + 14I/N`. The current reference
probe deliberately implements each square and each parity product with a
full multiplication, hence `(90-42/N)M + 14I/N`. A later specialized square
and parity window would be required to attain the target operation ledger.
The much larger inversion demand, service scheduling, exact arithmetic cost,
register pressure and table traffic can erase the arithmetic advantage. A
separate service can overlap inversion latency; it cannot remove that work.
`count_operations.py` reproduces this ledger in `operation_counts.json`.

## Audit and prior work

`audit_shifted_tail.py` compiles the actual C++ builder, signed loader,
denominator and finish bodies with exact big-integer field shims. Independent
Python curve arithmetic checks table points, full scalar reconstruction,
both recovered keys, compressed-key SHA256, recid priority and explicit
singularity/infinity declines. It includes a table slice crossing the
builder's 1024-entry batching boundary. Counts and the exact header hash are
recorded in `shifted_tail_result.json`. This is a host correctness audit;
it does not execute the CUDA arithmetic or mailbox protocol.

Affine addition and Montgomery batch inversion are established techniques;
the latter costs `1I+3(n-1)M` for n inputs. The GPU use of batched affine
arithmetic, inversion bottlenecks and intermediate-state traffic are also
explicitly studied in [gECC](https://arxiv.org/html/2501.03245v1).
[The EFD](https://www.hyperelliptic.org/EFD/g1p/auto-shortw.html) gives the
standard curve formulas. There is no claim that these methods, table shifts
or their algebra are unique discoveries.

The bounded contribution here is their adaptation to this candidate's
particular 15-window walk: paired final-table recovery, the exact signed-digit
arm mapping, parity-only finishing, and integration with resident affine
workers and a local inversion service. Existing field arithmetic, parity
windows and tree lineage retain their credits in the production files.

Reproduce the host audit and ledger from the benchmark work directory:

```sh
rtk proxy python3 candidates/pinning/research_affine/audit_shifted_tail.py
rtk proxy python3 candidates/pinning/research_affine/count_operations.py
```

The audit needs a C++ compiler, OpenSSL headers/library, and Boost headers.
Its include override uses the session's cached `/tmp/qsb-boost-audit/usr/include`;
system-installed Boost is found normally when that directory is absent.
