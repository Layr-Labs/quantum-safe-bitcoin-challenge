# Results

## Identity

- Date: 2026-09-17 (America/Indiana/Indianapolis)
- Promoted base: `1248235b7bab9471e6cc4c2f301180837e039fc8`
- Promoted `zinv32.cuh` SHA-256:
  `86b52456b9f7f5f7ec20018b38ad75279ba97a37e762915c8bca31eec30436db`
- Bend: `bend 2.0.5` from `https://github.com/bendlang/bend`
- `LAWS.bend` SHA-256:
  `cd5d01d0c8642ce9998013a4e1e8fa65b2a938bf62701f1a4fd690fd5e7a7596`
- `PROOF.bend` SHA-256:
  `df95f26a240bcf2d917d9a8c6479fda2a46cba14b4ba14af964421f1a9d9b826`
- `MODEL.bend` SHA-256:
  `97b59186638baa54e94a2e8d80f534f937e492c42de9e0019764d1b640d1757f`

Correction: an earlier pass mistakenly installed the unrelated historical
HigherOrderCO Bend 1 crate. The user supplied the authoritative Bend 2 project;
the official Bend 2.0.5 installer was then used with telemetry disabled,
`bend --version` and `bend guide` were run, and these files were rewritten in
Bend 2 syntax. The obsolete Bend 1 outputs are superseded by the results below.

## Commands and results

```text
BEND_NO_TELEMETRY=1 ~/.bend/bin/bend PROOF.bend
All terms check.

BEND_NO_TELEMETRY=1 ~/.bend/bin/bend MODEL.bend -o model
./model --threads 8
MkStats{0, 2994, 16320}
```

The `MkStats` fields mean:

- split/reconstruction failures: `0`;
- truncating-mutation mismatches: `2994`;
- first encoded witness: `16320` (index plus one).

Decoded first witness: `a=3, b=15, x=11, y=15`, so `acc=258`.
The full/split shift returns `16`; the truncating mutation returns `0`.

## Evidence boundary and decision

The two concrete reduced-radix laws are mechanically proven by Bend 2. The
separate model is an exhaustive finite search over `a,b,x,y in [0,15]`, not a
universal proof. Together they establish that the boundary split is preserved
and that storing only the low word before the delayed shift is not an algebraic
identity. Native execution used Apple clang 21 and eight CPU threads; it does
not validate the CUDA transcription.

The result independently supports keeping the signed accumulator through the
final production shift. It does not prove production `int64_t` bounds, signed
right-shift portability, the exact 30-bit coefficient bounds, warp exchange
correctness, or GPU throughput. Because a public pending submission already
contains this repair, do not submit a duplicate; use its official result and
source-bound 288-bit oracle before deciding whether to inherit it.
