# Lane-local zero-product inversion guard

Base: conservative Terrapin `8a952d94b87f6e89eb26d722761d32db540eec6b`.
Prepared by GPT-6 Astra / Codex. The only production-code change is the zero-product
fallback in `fe8_inv_lanes`, plus its corrected comment:

```
if (fe_is_zero(pr)) { fe8_inv1(x); return; }
```

`pr` is already reduced to a canonical field element before this check. The
original eight input lanes in `x` are still intact. Nonzero products use the
existing scalar-inversion path unchanged; a zero product instead uses the
existing eight independent Fermat chains. Zero inputs return zero by this code's
convention, and nonzero inputs retain their own inverses. OpenSSL inversion is
used only as an oracle for nonzero inputs; zero has no field inverse.

This fixes contamination across the eight SIMD lane indices. It does **not**
repair equal-x affine addition, point doubling, infinity handling, or the existing
`fe8_inv4` combination of four chain products within each SIMD lane. If any of
those four products is zero, all four outputs at that same lane still become zero,
as they do on the original Fermat path. Other lane indices remain independent.
The downstream exact gate validates published hits; it cannot restore work lost
through exceptional affine states.

## Focused regression

```
g++ -std=c++17 -O2 -Wno-deprecated-declarations -pthread \
 candidates/subset/tests/cpu_inverse/zero_guard_test.cpp -lcrypto -o /tmp/qsb-zero-guard-test
/tmp/qsb-zero-guard-test
```

Requires AVX-512 IFMA; exits 77 when unavailable. No tables, worker threads,
benchmark timing or remote jobs are launched. Fixtures use one, small integers,
values immediately below the field modulus, and deterministic wide nonzero values.
All 256 zero masks cover no zeros, each single zero, all multiple-zero combinations
and all zero lanes. Four-chain checks additionally put a zero in one chain for each
selected lane and verify the documented remaining lane-local contamination.

Saved `result.json`: 1,024 lane cases and 257 four-chain cases;
8,192 canonical Fermat comparisons, 8,192 OpenSSL nonzero comparisons,
8,224 zero-output comparisons; **0 failures**. OpenSSL inverses for the 32 unique
nonzero fixtures are reused across masks.

The identical executable source compiled with HEADER pointing to the unmodified
base header reproduces **12,192 failed assertions** (`before-result.json`), all
from the existing cross-lane contamination. This is a regression demonstration,
not a count of independent incorrect candidates. No full arithmetic audit was repeated.

`CpuWorkerPlacement.h`, worker counts, geometry selection, row prefetch, epoch
stride, benchmark entry (DIAG_EPOCH=0), producer, GPU and carrier are unchanged
from the base. The 8-lane Terrapin pipeline is retained; this candidate does not
contain the separate 16-lane stream optimization. Standalone wrapper syntax with
QSB_HP_ON=1 passes. Native CUDA compilation and matched all-hit validation remain
the parent's final gate.
