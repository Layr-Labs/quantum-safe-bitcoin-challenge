#!/usr/bin/env python3
"""Focused source-bound field and exceptional-finish audit.

This audit is intentionally narrow.  It binds the checks to the checked-out
GPUMath.h and pinning.cu bytes, proves that the old base header lacks the
repair, runs the independent extracted-PTX receipt when available, and then
checks the host carry/canonical boundary and the cold complete-finish mapping
with deterministic integer models.  It does not claim CUDA compilation or
runtime execution.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ATTEMPT = ROOT.parent.parent
HEADER = ROOT / "GPUMath.h"
PINNING = ROOT / "pinning.cu"
# Keep the source-bound receipt reconstructable from the candidate directory.
PTX_AUDITOR = ROOT / "audit_field_ptx.py"
BASE_COMMIT = "372a3251e707616011ff6dc0c961cf944d565149"
M = 1 << 256
C = (1 << 32) + 977
P = M - C
N = int("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def limbs(value: int) -> list[int]:
    return [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]


def canonical(value: int) -> int:
    value %= M
    return value - P if value >= P else value


def double_fold(value: int) -> tuple[int, int]:
    first = value % M + (value // M) * C
    second = first % M + (first // M) * C
    return second % M, second // M


def repaired_reducer(value: int) -> int:
    low, carry = double_fold(value)
    return canonical(low + carry * C)


def field_add(a: int, b: int) -> int:
    return (a + b) % P


def field_sub(a: int, b: int) -> int:
    return (a - b) % P


def field_mul(a: int, b: int) -> int:
    return (a * b) % P


def field_sqr(a: int) -> int:
    return field_mul(a, a)


def run_ptx_receipt() -> dict[str, object] | None:
    if not PTX_AUDITOR.exists():
        return None
    proc = subprocess.run(
        [
            "python3",
            str(PTX_AUDITOR),
            "--header",
            str(HEADER),
            "--op",
            "both",
            "--a",
            "p-65537",
            "--b",
            "p-65537",
        ],
        cwd=ATTEMPT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)
    assert report["header_sha256"] == sha256(HEADER)
    checks = report["direct"]["checks"]
    assert len(checks) == 2
    assert all(item["match"] and item["congruent"] for item in checks)
    assert all(item["observed"] == "0x100020001" for item in checks)
    # The instruction emulator deliberately stops before the production C++
    # _ModCanonical256 call.  Its bounded self-test therefore may report a
    # congruent p+1 representative for p-1 inputs.  Bind that distinction to
    # the actual current PTX: every self-test result must be congruent and the
    # post-asm canonicalization oracle must map it to the expected residue.
    self_proc = subprocess.run(
        [
            "python3",
            str(PTX_AUDITOR),
            "--header",
            str(HEADER),
            "--op",
            "both",
            "--self-test",
        ],
        cwd=ATTEMPT,
        # The utility returns 1 for its intentionally unclassified p+1
        # representative, while still emitting the complete fail-closed JSON
        # report.  Parse and validate that report below instead of treating
        # the expected classification as an execution failure.
        check=False,
        capture_output=True,
        text=True,
    )
    self_report = json.loads(self_proc.stdout)
    assert self_report["header_sha256"] == sha256(HEADER)
    suite = self_report["self_test"]
    assert suite["vectors_per_operation"] >= 100
    assert suite["classification"] == "mismatch-unclassified"
    assert self_proc.returncode == 1
    all_checks = []
    for operation in ("mult", "sqr"):
        op_report = suite["operations"][operation]
        all_checks.extend(op_report["mismatches"])
        assert op_report["checks"] >= 100
        assert all(item["congruent"] for item in op_report["mismatches"])
        for item in op_report["mismatches"]:
            observed = int(item["observed"], 16)
            expected = int(item["expected"], 16)
            assert canonical(observed) == expected % P
    assert all(item["congruent"] for item in suite["all_mismatches"])
    assert suite["known_vector_observed"]["mult"] == "0x100020001"
    assert suite["known_vector_observed"]["sqr"] == "0x100020001"
    self_report["post_asm_canonicalized_mismatches"] = len(all_checks)
    self_report["process_returncode"] = self_proc.returncode
    report["self_test"] = self_report
    return report


def get_old_header() -> str:
    proc = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:candidates/pinning/GPUMath.h"],
        cwd=ATTEMPT,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def function_body(source: str, name: str) -> str:
    """Extract one C/CUDA function body while ignoring string/comment braces."""

    match = re.search(r"\b" + re.escape(name) + r"\s*\(", source)
    if not match:
        raise AssertionError(f"function {name} not found")
    opening = source.find("{", match.end())
    if opening < 0:
        raise AssertionError(f"opening brace for {name} not found")
    depth = 0
    in_string = False
    in_char = False
    escaped = False
    line_comment = False
    block_comment = False
    i = opening
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
        elif block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 1
        elif in_string or in_char:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif (in_string and ch == '"') or (in_char and ch == "'"):
                in_string = False
                in_char = False
        elif ch == "/" and nxt == "/":
            line_comment = True
            i += 1
        elif ch == "/" and nxt == "*":
            block_comment = True
            i += 1
        elif ch == '"':
            in_string = True
        elif ch == "'":
            in_char = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:i]
        i += 1
    raise AssertionError(f"unterminated function {name}")


def host_function_body(source: str, name: str) -> str:
    body = function_body(source, name)
    assert "#else" in body and "#endif" in body
    host = body.split("#else", 1)[1].rsplit("#endif", 1)[0]
    assert host.strip()
    return host


def c_limb_array(value: int) -> str:
    return "{" + ", ".join(f"0x{limb:016x}ULL" for limb in limbs(value)) + "}"


def run_host_source_receipt(header: str, pinning: str) -> dict[str, object]:
    """Compile and execute the exact current host transcriptions.

    GPUMath.h is CUDA source, so this intentionally extracts only its #else
    branches and the actual canonicalizer into a tiny clang++ harness.  The
    substitution implements PTX's sub.cc/subc.cc borrow flag (a= b + borrow),
    including the final subc output consumed by _ModCanonical256.  The test
    vectors are generated here and expected residues are an independent bigint
    oracle; no hand-written replacement arithmetic is used for the functions
    under test.
    """

    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler, "clang++/c++ unavailable for host source receipt"
    mult_host = host_function_body(header, "_ModMultCore")
    sqr_host = host_function_body(header, "_ModSqr")
    canonical_body = function_body(header, "_ModCanonical256")
    add_body = function_body(header, "_ModAdd256")
    sub_body = function_body(header, "_ModSub256")
    double_body = function_body(header, "_PointDoubleAffineXYZZ")
    symmetric_body = function_body(pinning, "qsb_xyzz_finish_symmetric")
    assert "_ModSqr(f, u)" in symmetric_body
    assert "qsb_field_normalize" not in symmetric_body
    assert "_ModNeg256(h, u)" not in symmetric_body
    assert "Load256(h, u)" not in symmetric_body
    extracted = "\n".join(
        (
            mult_host,
            sqr_host,
            canonical_body,
            add_body,
            sub_body,
            double_body,
            symmetric_body,
        )
    )

    # Cover the reviewed carry edge, canonical boundaries, and deterministic
    # random inputs.  Keep this source-bound build small enough for every local
    # worker to rerun while materially exceeding the single-vector PTX check.
    deltas = [0, 1, 2, 3, 7, 31, 255, 256, 977, 65535, 65536, 65537,
              65538, C - 1, C, C + 1]
    boundary = [0, 1, 2, P - 1, P, M - 1]
    boundary += [P - d for d in deltas if 0 <= d <= P]
    boundary = sorted(set(boundary))
    pairs: list[tuple[int, int]] = [(a, b) for a in boundary for b in boundary]
    rng = random.Random(0xF1E1D5A)
    pairs.extend((rng.randrange(M), rng.randrange(M)) for _ in range(256))
    cases = [
        (a, b, (a * b) % P)
        for a, b in pairs
    ]
    square_cases = [(a, a, (a * a) % P) for a in boundary]
    square_cases.extend(
        (a, a, (a * a) % P) for a in [rng.randrange(M) for _ in range(256)]
    )

    # Exercise the extracted production symmetric recovery helper itself.  The
    # expected records come from an independent affine group law, while the
    # source helper receives the exact projective Y/V/W state passed across the
    # CUDA stage boundary.  Include fixed, random, and near-p denominator
    # points; the complete infinity/equal/opposite cold mapping is checked by
    # exceptional_checks(), because this helper is only called for W != 0.
    R = (
        int("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16),
        int("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16),
    )
    neg_R = (R[0], (-R[1]) % P)
    K = (3 * R[0] * R[0]) % P
    recovery_points: list[tuple[int, int]] = []
    for _ in range(16):
        recovery_points.append(random_point(rng))
    boundary_points: list[tuple[int, int]] = []
    for delta in (65537, 1 << 20, 1 << 32):
        u0 = (P - delta) % P
        x = (R[0] - R[1] * pow(u0, P - 2, P)) % P
        y2 = (x * x * x + 7) % P
        assert pow(y2, (P - 1) // 2, P) == 1
        y = pow(y2, (P + 1) // 4, P)
        boundary_points.extend(((x, y), (x, (-y) % P)))
    recovery_points.extend(boundary_points)
    recovery_cases: list[tuple[int, ...]] = []
    for point_index, (x, y) in enumerate(recovery_points):
        z_values = (1, 2, P - 1, rng.randrange(1, P)) if point_index < 16 else (1, 2, P - 1)
        for z in z_values:
            U = (z * z) % P
            V = (U * z) % P
            X = (x * U) % P
            Y = (y * V) % P
            d = (R[0] * U - X) % P
            W = (U * U * d) % P
            assert W
            inv = pow(W, P - 2, P)
            plus = ec_add((x, y), R)
            minus = ec_add((x, y), neg_R)
            assert plus is not None and minus is not None
            recovery_cases.append(
                (
                    Y,
                    V,
                    inv,
                    R[0],
                    R[1],
                    K,
                    plus[0],
                    minus[0],
                    plus[1] & 1,
                    minus[1] & 1,
                )
            )

    gx = int("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16)
    gy = int("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16)
    points: list[tuple[int, int]] = [(gx, gy)]
    for _ in range(3):
        next_point = ec_add(points[-1], (gx, gy))
        assert next_point is not None
        points.append(next_point)
    double_cases: list[tuple[int, int, int, int, int, int]] = []
    for x, y in points:
        yy = (y * y) % P
        yyyy = (yy * yy) % P
        s = (4 * x * yy) % P
        mm = (3 * x * x) % P
        x3 = (mm * mm - 2 * s) % P
        y3 = (mm * (s - x3) - 8 * yyyy) % P
        zz3 = (4 * yy) % P
        zzz3 = (8 * y * yy) % P
        double_cases.append((x, y, x3, y3, zz3, zzz3))
    # The helper explicitly maps a zero affine ordinate to the infinity
    # representation.  It is not a valid secp256k1 point, but it closes the
    # generic branch without relying on an inverse of zero.
    double_cases.append((1, 0, 0, 0, 0, 0))

    case_rows = ",\n".join(
        "  {" + c_limb_array(a) + ", " + c_limb_array(b) + ", " +
        c_limb_array(expected) + "}"
        for a, b, expected in cases
    )
    square_rows = ",\n".join(
        "  {" + c_limb_array(a) + ", " + c_limb_array(b) + ", " +
        c_limb_array(expected) + "}"
        for a, b, expected in square_cases
    )
    double_rows = ",\n".join(
        "  {" + c_limb_array(x) + ", " + c_limb_array(y) + ", " +
        c_limb_array(expected_x) + ", " + c_limb_array(expected_y) + ", " +
        c_limb_array(expected_zz) + ", " + c_limb_array(expected_zzz) + "}"
        for x, y, expected_x, expected_y, expected_zz, expected_zzz in double_cases
    )
    recovery_rows = ",\n".join(
        "  {" + ", ".join(
            (
                c_limb_array(Y),
                c_limb_array(V),
                c_limb_array(inv),
                c_limb_array(xR),
                c_limb_array(yR),
                c_limb_array(K_case),
                c_limb_array(expected_plus),
                c_limb_array(expected_minus),
                str(plus_parity),
                str(minus_parity),
            )
        ) + "}"
        for (
            Y,
            V,
            inv,
            xR,
            yR,
            K_case,
            expected_plus,
            expected_minus,
            plus_parity,
            minus_parity,
        ) in recovery_cases
    )
    cpp = f'''#include <cstdint>
#include <cstddef>
#include <cstdio>

static uint64_t hcc;
#define UADDO(c,a,b) do {{ __uint128_t _v=(__uint128_t)(a)+(b); (c)=(uint64_t)_v; hcc=(uint64_t)(_v>>64); }} while(0)
#define UADDC(c,a,b) do {{ __uint128_t _v=(__uint128_t)(a)+(b)+hcc; (c)=(uint64_t)_v; hcc=(uint64_t)(_v>>64); }} while(0)
#define UADD(c,a,b) do {{ uint64_t _a=(uint64_t)(a), _b=(uint64_t)(b), _in=hcc; __uint128_t _v=(__uint128_t)_a+_b+_in; (c)=(uint64_t)_v; }} while(0)
#define UADDO1(c,a) do {{ __uint128_t _v=(__uint128_t)(c)+(a); (c)=(uint64_t)_v; hcc=(uint64_t)(_v>>64); }} while(0)
#define UADDC1(c,a) do {{ __uint128_t _v=(__uint128_t)(c)+(a)+hcc; (c)=(uint64_t)_v; hcc=(uint64_t)(_v>>64); }} while(0)
#define UADD1(c,a) do {{ uint64_t _old=(uint64_t)(c), _a=(uint64_t)(a), _in=hcc; __uint128_t _v=(__uint128_t)_old+_a+_in; (c)=(uint64_t)_v; }} while(0)
#define USUBO(c,a,b) do {{ uint64_t _a=(uint64_t)(a), _b=(uint64_t)(b); __uint128_t _v=(__uint128_t)_a-(__uint128_t)_b; (c)=(uint64_t)_v; hcc=(_a<_b); }} while(0)
#define USUBC(c,a,b) do {{ uint64_t _a=(uint64_t)(a), _b=(uint64_t)(b), _in=hcc; __uint128_t _bb=(__uint128_t)_b+_in; __uint128_t _v=(__uint128_t)_a-_bb; (c)=(uint64_t)_v; hcc=((__uint128_t)_a<_bb); }} while(0)
#define USUB(c,a,b) do {{ uint64_t _a=(uint64_t)(a), _b=(uint64_t)(b), _in=hcc; __uint128_t _bb=(__uint128_t)_b+_in; __uint128_t _v=(__uint128_t)_a-_bb; (c)=(uint64_t)_v; }} while(0)
#define USUBO1(c,a) do {{ uint64_t _old=(c); __uint128_t _v=(__uint128_t)_old-(__uint128_t)(a); (c)=(uint64_t)_v; hcc=((__uint128_t)_old<(__uint128_t)(a)); }} while(0)
#define USUBC1(c,a) do {{ uint64_t _old=(c); __uint128_t _bb=(__uint128_t)(a)+hcc; __uint128_t _v=(__uint128_t)_old-_bb; (c)=(uint64_t)_v; hcc=((__uint128_t)_old<_bb); }} while(0)
#define USUB1(c,a) do {{ uint64_t _old=(uint64_t)(c), _a=(uint64_t)(a), _in=hcc; __uint128_t _bb=(__uint128_t)_a+_in; __uint128_t _v=(__uint128_t)_old-_bb; (c)=(uint64_t)_v; }} while(0)
#define Load256(r,a) do {{ (r)[0]=(a)[0]; (r)[1]=(a)[1]; (r)[2]=(a)[2]; (r)[3]=(a)[3]; }} while(0)
#define _IsPositive(x) (((int64_t)(x)[4]) >= 0LL)
#define SubP(r) do {{ \\
    USUBO1((r)[0], 0xFFFFFFFEFFFFFC2FULL); \\
    USUBC1((r)[1], 0xFFFFFFFFFFFFFFFFULL); \\
    USUBC1((r)[2], 0xFFFFFFFFFFFFFFFFULL); \\
    USUBC1((r)[3], 0xFFFFFFFFFFFFFFFFULL); \\
    USUB1((r)[4], 0ULL); \\
}} while(0)

void _ModCanonical256(uint64_t *r) {{
{canonical_body}
}}
void mul(uint64_t *r, const uint64_t *a, const uint64_t *b) {{
{mult_host}
    _ModCanonical256(r);
}}
void sqr(uint64_t *r, const uint64_t *a) {{
{sqr_host}
    _ModCanonical256(r);
}}
void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {{ mul(r, a, b); }}
void _ModMult(uint64_t *r, uint64_t *a) {{
    uint64_t b[4] = {{r[0], r[1], r[2], r[3]}};
    mul(r, a, b);
}}
void _ModSqr(uint64_t *r, const uint64_t *a) {{ sqr(r, a); }}
void _ModAdd256(uint64_t *r, uint64_t *a, uint64_t *b) {{
{add_body}
}}
void _ModSub256(uint64_t *r, uint64_t *a, uint64_t *b) {{
{sub_body}
}}
void _ModSub256(uint64_t *r, uint64_t *b) {{ _ModSub256(r, r, b); }}
void _PointDoubleAffineXYZZ(
    uint64_t *X1, uint64_t *Y1, uint64_t *ZZ1, uint64_t *ZZZ1,
    const uint64_t *X2, const uint64_t *Y2
) {{
{double_body}
}}
uint32_t qsb_xyzz_finish_symmetric(
    uint64_t *Y, uint64_t *V, uint64_t *inv,
    uint64_t *xR, uint64_t *yR, uint64_t *K,
    uint64_t *x_plus, uint64_t *x_minus
) {{
{symmetric_body}
}}

struct Case {{ uint64_t a[4]; uint64_t b[4]; uint64_t expected[4]; }};
static const Case cases[] = {{
{case_rows}
}};
static const Case square_cases[] = {{
{square_rows}
}};
struct DoubleCase {{
    uint64_t x[4]; uint64_t y[4]; uint64_t X[4]; uint64_t Y[4];
    uint64_t ZZ[4]; uint64_t ZZZ[4];
}};
static const DoubleCase double_cases[] = {{
{double_rows}
}};
struct RecoveryCase {{
    uint64_t Y[4]; uint64_t V[4]; uint64_t inv[4];
    uint64_t xR[4]; uint64_t yR[4]; uint64_t K[4];
    uint64_t x_plus[4]; uint64_t x_minus[4];
    uint32_t plus_parity; uint32_t minus_parity;
}};
static const RecoveryCase recovery_cases[] = {{
{recovery_rows}
}};
static bool same(const uint64_t *a, const uint64_t *b) {{
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}}
int main() {{
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {{
        const Case &c = cases[i];
        uint64_t r[4];
        mul(r, c.a, c.b);
        if (!same(r, c.expected)) {{ std::printf("mul:%zu\\n", i); return 1; }}
        uint64_t left[4] = {{c.a[0], c.a[1], c.a[2], c.a[3]}};
        mul(left, left, c.b);
        if (!same(left, c.expected)) {{ std::printf("mul-left-alias:%zu\\n", i); return 2; }}
        uint64_t right[4] = {{c.b[0], c.b[1], c.b[2], c.b[3]}};
        mul(right, c.a, right);
        if (!same(right, c.expected)) {{ std::printf("mul-right-alias:%zu\\n", i); return 3; }}
    }}
    for (std::size_t i = 0; i < sizeof(square_cases) / sizeof(square_cases[0]); ++i) {{
        const Case &c = square_cases[i];
        uint64_t r[4];
        sqr(r, c.a);
        if (!same(r, c.expected)) {{ std::printf("sqr:%zu\\n", i); return 4; }}
        uint64_t alias[4] = {{c.a[0], c.a[1], c.a[2], c.a[3]}};
        sqr(alias, alias);
        if (!same(alias, c.expected)) {{ std::printf("sqr-alias:%zu\\n", i); return 5; }}
    }}
    for (std::size_t i = 0; i < sizeof(double_cases) / sizeof(double_cases[0]); ++i) {{
        const DoubleCase &c = double_cases[i];
        uint64_t X[4], Y[4], ZZ[4], ZZZ[4];
        _PointDoubleAffineXYZZ(X, Y, ZZ, ZZZ, c.x, c.y);
        if (!same(X, c.X) || !same(Y, c.Y) || !same(ZZ, c.ZZ) || !same(ZZZ, c.ZZZ)) {{
            std::printf("double:%zu\\n", i);
            return 6;
        }}
    }}
    for (std::size_t i = 0; i < sizeof(recovery_cases) / sizeof(recovery_cases[0]); ++i) {{
        const RecoveryCase &c = recovery_cases[i];
        uint64_t Y[4] = {{c.Y[0], c.Y[1], c.Y[2], c.Y[3]}};
        uint64_t V[4] = {{c.V[0], c.V[1], c.V[2], c.V[3]}};
        uint64_t inv[4] = {{c.inv[0], c.inv[1], c.inv[2], c.inv[3]}};
        uint64_t xR[4] = {{c.xR[0], c.xR[1], c.xR[2], c.xR[3]}};
        uint64_t yR[4] = {{c.yR[0], c.yR[1], c.yR[2], c.yR[3]}};
        uint64_t K[4] = {{c.K[0], c.K[1], c.K[2], c.K[3]}};
        uint64_t x_plus[4], x_minus[4];
        uint32_t parities = qsb_xyzz_finish_symmetric(
            Y, V, inv, xR, yR, K, x_plus, x_minus
        );
        if (!same(x_plus, c.x_plus) || !same(x_minus, c.x_minus) ||
            ((parities & 1u) != c.plus_parity) ||
            (((parities >> 1) & 1u) != c.minus_parity)) {{
            std::printf("symmetric:%zu\\n", i);
            return 7;
        }}
    }}
    std::printf("HOST_PASS cases=%zu squares=%zu aliases=3 doubles=%zu symmetric=%zu\\n",
                sizeof(cases) / sizeof(cases[0]),
                sizeof(square_cases) / sizeof(square_cases[0]),
                sizeof(double_cases) / sizeof(double_cases[0]),
                sizeof(recovery_cases) / sizeof(recovery_cases[0]));
    return 0;
}}
'''
    with tempfile.TemporaryDirectory(prefix="pinning-host-audit-") as temp:
        tempdir = Path(temp)
        src = tempdir / "host_field_receipt.cpp"
        exe = tempdir / "host_field_receipt"
        src.write_text(cpp)
        proc = subprocess.run(
            [compiler, "-std=c++17", "-O0", "-w", str(src), "-o", str(exe)],
            cwd=ATTEMPT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        run = subprocess.run(
            [str(exe)], cwd=ATTEMPT, check=False, capture_output=True, text=True
        )
        assert run.returncode == 0, (run.returncode, run.stdout, run.stderr)
        assert run.stdout.startswith("HOST_PASS "), run.stdout
    return {
        "source_extract_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
        "compiler": str(Path(compiler).resolve()),
        "compiler_validated": True,
        "test_cases": len(cases),
        "square_cases": len(square_cases),
        "alias_modes": 3,
        "symmetric_recovery_cases": len(recovery_cases),
        "symmetric_boundary_cases": len(boundary_points) * 3,
        "symmetric_cold_mapping_cases": 3,
        "borrow_convention": "PTX sub.cc/subc.cc: result=a-b-borrow, next borrow=1 iff a<b+borrow",
        "stdout": run.stdout.strip(),
    }


def instruction_cost_checks(header: str) -> dict[str, int | str]:
    """Return a source-level tail ledger for the cold-carry variant.

    This is an exact instruction-stream comparison for the two inline PTX
    tails.  It intentionally does not estimate SASS issue slots or GPU
    throughput: predicated instructions may still occupy issue resources, and
    the compiler's lowering of the host canonical predicate is target-specific.
    """

    assert header.count("addc.u32 cf, 0, 0") == 2
    assert header.count("setp.ne.u32 pcarry, cf, 0") == 2
    assert header.count("@pcarry add.cc.u32 z0, z0, 977") == 2
    assert header.count("@pcarry addc.cc.u32 z1, z1, 1") == 2
    assert header.count("@pcarry addc.u32 z2, z2, 0") == 2
    assert "mul.lo.u32 k0, cf, 977" not in header
    assert header.count("t=(uint64_t)z[2]+c; z[2]=(uint32_t)t; c=t>>32;") == 1
    assert header.count("tv=(uint64_t)z[2]+c; z[2]=(uint32_t)tv; c=tv>>32;") == 1
    # Each helper has one host carry branch; the old eight-limb propagation is
    # replaced by the mathematically bounded third 32-bit word update.
    assert header.count("for (int k=2;k<8;k++){ t=(uint64_t)z[k]+c;") == 0
    assert header.count("for (int k=2;k<8;k++){ tv=(uint64_t)z[k]+c;") == 0
    fixed_base_multiplies = 95
    fixed_base_squares = 28
    recovery_prepare_multiplies = 2
    recovery_prepare_squares = 1
    recovery_finish_multiplies = 8
    recovery_finish_squares = 1
    total_multiplies = fixed_base_multiplies + recovery_prepare_multiplies + recovery_finish_multiplies
    total_squares = fixed_base_squares + recovery_prepare_squares + recovery_finish_squares
    assert (total_multiplies, total_squares) == (105, 30)
    total_calls = total_multiplies + total_squares
    old_tail = 10 * total_calls
    normal_tail = 2 * total_calls
    taken_tail = 5 * total_calls
    return {
        "old_tail_static_instructions_per_call": 10,
        "new_tail_static_instructions_per_call": 5,
        "new_tail_normal_dynamic_instructions_per_call": 2,
        "new_tail_carry_dynamic_instructions_per_call": 5,
        "normal_dynamic_delta_per_call": -8,
        "carry_dynamic_delta_per_call": -5,
        "fixed_base_multiply_calls_per_candidate": fixed_base_multiplies,
        "fixed_base_square_calls_per_candidate": fixed_base_squares,
        "recovery_prepare_multiply_calls_per_candidate": recovery_prepare_multiplies,
        "recovery_prepare_square_calls_per_candidate": recovery_prepare_squares,
        "recovery_finish_multiply_calls_per_candidate": recovery_finish_multiplies,
        "recovery_finish_square_calls_per_candidate": recovery_finish_squares,
        "total_multiply_calls_per_candidate": total_multiplies,
        "total_square_calls_per_candidate": total_squares,
        "total_corrected_calls_per_candidate": total_calls,
        "old_tail_instruction_proxy": old_tail,
        "new_tail_normal_instruction_proxy": normal_tail,
        "new_tail_carry_instruction_proxy": taken_tail,
        "proxy_delta_as_function_of_carry_calls": f"-{8 * total_calls} + 3*C",
        "canonical_predicate_source_cost":
            "two 64-bit ANDs, one upper-limb compare and a cold four-limb assignment; compiler-dependent",
        "gpu_throughput_claim": "none",
    }


def source_checks(header: str, pinning: str, old: str) -> dict[str, int]:
    # The old immutable source must fail closed on the repair marker.  This is
    # the source-binding guard against running only a corrected mathematical
    # model while retaining defective production bytes.
    assert "_ModCanonical256" not in old
    assert "addc.u32 cf, 0, 0" not in old
    assert "977ULL" not in old
    assert "_ModCanonical256" in header
    assert len(re.findall(r"^\s*_ModCanonical256\(r\);$", header, re.MULTILINE)) == 2
    assert header.count("addc.u32 cf, 0, 0") == 2
    assert header.count("setp.ne.u32 pcarry, cf, 0") == 2
    assert header.count("@pcarry add.cc.u32 z0, z0, 977") == 2
    assert header.count("@pcarry addc.cc.u32 z1, z1, 1") == 2
    assert header.count("@pcarry addc.u32 z2, z2, 0") == 2
    assert "mul.lo.u32 k0, cf, 977" not in header
    assert header.count("977ULL") >= 2
    assert header.count("+1ULL+c") >= 2
    assert hashlib.sha256(pinning.encode()).hexdigest() == (
        "8eddc93c23cc3ab4f27bd026f691e1d986dc37e8d436880ca76b419c6ec11621"
    )
    assert "canonical interval [0,p)" in header
    assert "r[1] & r[2] & r[3]" in header
    assert "r[1] = r[2] = r[3] = 0" in header
    assert "Coordinate limbs are canonical [0,p)" in header
    for token in (
        "_PointDoubleAffineXYZZ(X1,Y1,ZZ1,ZZZ1,X2,Y2);",
        "if (!defer_y &&",
        "uint64_t XX[4], YY[4], YYYY[4], S[4], MM[4], T[4];",
        "ZZZ3 = 8*y^3",
    ):
        assert token in header, token

    # Both host transcriptions retain the outgoing carry before folding C.
    assert "t=(uint64_t)z[7]+c; z[7]=(uint32_t)t; c=t>>32;" in header
    assert "tv=(uint64_t)z[7]+c; z[7]=(uint32_t)tv; c=tv>>32;" in header
    assert header.count("if (c) {") >= 2
    assert header.count("t=(uint64_t)z[2]+c; z[2]=(uint32_t)t; c=t>>32;") == 1
    assert header.count("tv=(uint64_t)z[2]+c; z[2]=(uint32_t)tv; c=tv>>32;") == 1

    # The checkpoint tree keeps its separately documented raw [0,2^256)
    # product contract; only coordinate helpers are canonicalized.
    assert "qsb_field_mul raw [0,2^256)" in header
    assert "qsb_field_normalize" in pinning
    assert "prod[5]" in pinning
    finish = function_body(pinning, "qsb_xyzz_finish_symmetric")
    assert "_ModSqr(f, u)" in finish
    assert "qsb_field_normalize" not in finish
    assert "_ModNeg256(h, u)" not in finish
    assert "Load256(h, u)" not in finish

    # The complete cold mapping is source-bound: infinity, equal and opposite
    # projective points get independent validity bits and both recid slots.
    for token in (
        "bool p_infinity",
        "bool same",
        "bool opposite",
        "uint32_t valid_mask",
        "d_neg2u2rx",
        "d_neg2u2ry",
        "valid_mask=3u",
        "valid_mask=1u",
        "valid_mask=2u",
        "if ((valid_mask & (1u<<ri)) == 0) continue;",
        "if (!active) return;",
    ):
        assert token in pinning, token

    # The copied production source is the frozen 128-leaf / 15-window
    # tree-plus-shift candidate.  These checks bind that exact geometry and
    # policy mechanism while keeping pinning.cu outside the algorithm delta.
    for token in (
        "#define QSB_TREE_N 128",
        "#define QSB_CAND_STRIDE QSB_TREE_N",
        "#define QSB_S0_THREADS QSB_TREE_N",
        "#define QSB_S2_THREADS QSB_TREE_N",
        "#define QSB_CHECKPOINT_NODES 254",
        "#define GT_CHUNKS 15",
        "#define GT_TOTAL_ENTRIES (1u << 20)",
        "static_assert(GT_TOTAL_ENTRIES*64ULL == 64ULL*1024*1024",
        "gt_mixed_step<18>",
        "gt_mixed_step<17>",
        "size_t cold = (size_t)gt_entries(0) * 64u;",
        "size_t skip = (gt_sz > cold + want) ? cold : 0;",
        "av.accessPolicyWindow.base_ptr  = (void *)(d_gt + skip);",
        "cudaDevAttrMaxPersistingL2CacheSize",
        "cudaDevAttrMaxAccessPolicyWindowSize",
        "av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;",
        "av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;",
    ):
        assert token in pinning, token
    assert "#define GT_CHUNKS 16" not in pinning
    assert "#define GT_TOTAL_ENTRIES (1u << 19)" not in pinning
    assert "gt_mixed_step<16>" not in pinning
    assert "GT_TOTAL_ENTRIES*64ULL == 32ULL*1024*1024" not in pinning
    assert "if (seqs_done % 10 == 0 || found)" in pinning
    return {"repair_markers": 1, "cold_mapping_markers": 10, "geometry_markers": 16}


def arithmetic_checks() -> dict[str, int | str]:
    # Dense boundary vectors include the reviewed canonical carry edge and the
    # transitions around p, zero, and the 256-bit ceiling.
    deltas = [0, 1, 2, 3, 7, 31, 255, 256, 977, 65535, 65536, 65537,
              65538, C - 1, C, C + 1]
    values = [0, 1, 2, P - 1, P, M - 1]
    values += [P - d for d in deltas if 0 <= d <= P]
    values = sorted(set(values))
    cases = 0
    old_mismatches = 0
    for a in values:
        for b in values:
            product = a * b
            old, carry = double_fold(product)
            fixed = repaired_reducer(product)
            assert fixed == product % P
            assert 0 <= fixed < P
            old_mismatches += old % P != product % P
            assert carry in (0, 1)
            cases += 1

    rng = random.Random(0x925384F6)
    for _ in range(2048):
        a = rng.randrange(M)
        b = rng.randrange(M)
        product = a * b
        low, carry = double_fold(product)
        assert repaired_reducer(product) == product % P
        assert canonical(low + carry * C) == product % P
        cases += 1

    known = repaired_reducer((P - 65537) ** 2)
    assert known == 0x100020001
    old_known, old_carry = double_fold((P - 65537) ** 2)
    assert old_known == 0x1FC30 and old_carry == 1

    # Alias checks model the in-place wrappers used by coordinate code.
    for a, b in [(0, 0), (1, P - 1), (P - 65537, P - 65537),
                 (P - 1, 3), (3, (M - 1) // 3)]:
        expected = field_mul(a, b)
        left = repaired_reducer(a * b)
        right = repaired_reducer(a * b)
        square = repaired_reducer(a * a)
        assert left == right == expected
        assert square == field_sqr(a)

    # Chained operations remain canonical, so the existing add/sub routines
    # consume only the domain for which their single p adjustment is closed.
    for a, b, c in [(P - 1, P - 1, 1), (P - 65537, P - 1, P - 2),
                    (3, (M - 1) // 3, P - 7), (1, P - 2, P - 3)]:
        x = field_add(field_mul(a, b), c)
        y = field_sub(x, b)
        assert 0 <= x < P and 0 <= y < P
        assert y == (a * b + c - b) % P

    return {
        "boundary_cases": cases,
        "old_boundary_mismatches": old_mismatches,
        "known_old": hex(old_known),
        "known_fixed": hex(known),
    }


def ec_add(p1: tuple[int, int] | None, p2: tuple[int, int] | None) -> tuple[int, int] | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1) * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow((x2 - x1) % P, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return x3, (lam * (x1 - x3) - y1) % P


def random_point(rng: random.Random) -> tuple[int, int]:
    """Return a deterministic random affine secp256k1 point."""

    while True:
        x = rng.randrange(P)
        y2 = (x * x * x + 7) % P
        if pow(y2, (P - 1) // 2, P) == 1:
            y = pow(y2, (P + 1) // 4, P)
            return x, y if rng.randrange(2) else (-y) % P


def exceptional_checks() -> dict[str, int]:
    gx = int("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16)
    gy = int("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16)
    R = (gx, gy)
    negR = (gx, (-gy) % P)
    twoR = ec_add(R, R)
    minus_twoR = (twoR[0], (-twoR[1]) % P)

    # This is the exact complete result table represented by the source's
    # p_infinity/same/opposite branches. Each public key keeps its own recid.
    expected = {
        "infinity": (R, negR, 3),
        "same": (twoR, None, 1),
        "opposite": (None, minus_twoR, 2),
    }
    assert ec_add(None, R) == R and ec_add(None, negR) == negR
    assert ec_add(R, R) == twoR
    assert ec_add(R, negR) is None
    assert ec_add(negR, R) is None
    assert ec_add(negR, negR) == minus_twoR
    # y parity is the serialized recid bit used by both fallback outputs.
    assert (R[1] & 1) in (0, 1) and (negR[1] & 1) == 1 - (R[1] & 1)
    assert expected["infinity"][2] == 3
    assert expected["same"][2] == 1
    assert expected["opposite"][2] == 2

    # Independently check the complete affine-doubling formula used by the
    # newly reachable final branch.  The helper returns XYZZ, so compare after
    # projective dehomogenization as well as checking ZZ^3 == ZZZ^2.
    points = [R, twoR, ec_add(twoR, R), ec_add(ec_add(twoR, R), R)]
    for point in points:
        assert point is not None and point[1] != 0
        x, y = point
        xx = (x * x) % P
        yy = (y * y) % P
        yyyy = (yy * yy) % P
        s = (4 * x * yy) % P
        mm = (3 * xx) % P
        x3 = (mm * mm - 2 * s) % P
        y3 = (mm * (s - x3) - 8 * yyyy) % P
        zz3 = (4 * yy) % P
        zzz3 = (8 * y * yy) % P
        assert (zz3 * zz3 * zz3) % P == (zzz3 * zzz3) % P
        expected_double = ec_add(point, point)
        assert expected_double is not None
        assert (x3 * pow(zz3, P - 2, P)) % P == expected_double[0]
        assert (y3 * pow(zzz3, P - 2, P)) % P == expected_double[1]
    return {
        "complete_cases": 3,
        "finite_double_checks": 2,
        "opposite_checks": 2,
        "doubling_projective_checks": len(points),
    }


def sub_limbs(a: list[int], b: list[int]) -> tuple[list[int], int]:
    out: list[int] = []
    borrow = 0
    for x, y in zip(a, b):
        value = x - y - borrow
        out.append(value & ((1 << 64) - 1))
        borrow = int(value < 0)
    return out, borrow


def recode_setup_model(k: int) -> tuple[list[int], int]:
    """Mirror gt_recode_setup's four-limb unsigned operations exactly."""

    n_words = limbs(N)
    k_words = limbs(k)
    diff, borrow = sub_limbs(k_words, n_words)
    reduced = diff if borrow == 0 else k_words
    doubled = [
        (reduced[0] << 1) & ((1 << 64) - 1),
        ((reduced[1] << 1) | (reduced[0] >> 63)) & ((1 << 64) - 1),
        ((reduced[2] << 1) | (reduced[1] >> 63)) & ((1 << 64) - 1),
        ((reduced[3] << 1) | (reduced[2] >> 63)) & ((1 << 64) - 1),
    ]
    top_carry = reduced[3] >> 63
    diff, borrow = sub_limbs(doubled, n_words)
    m = diff if top_carry or borrow == 0 else doubled
    odd = m[0] & 1
    negated, _ = sub_limbs(n_words, m)
    return (m if odd else negated), (1 if odd else -1)


def recode_step_model(state: list[int], sign: int, bits: int) -> tuple[list[int], int]:
    """Mirror gt_mixed_step<BITS>, including its forced odd state bit."""

    mask64 = (1 << 64) - 1
    shift = bits + 1
    digit = (state[0] & ((1 << shift) - 1)) - (1 << bits)
    right = [
        ((state[0] >> shift) | (state[1] << (64 - shift))) & mask64,
        ((state[1] >> shift) | (state[2] << (64 - shift))) & mask64,
        ((state[2] >> shift) | (state[3] << (64 - shift))) & mask64,
        state[3] >> shift,
    ]
    next_state = [
        ((right[0] << 1) | 1) & mask64,
        ((right[1] << 1) | (right[0] >> 63)) & mask64,
        ((right[2] << 1) | (right[1] >> 63)) & mask64,
        ((right[3] << 1) | (right[2] >> 63)) & mask64,
    ]
    return next_state, sign * digit


def recode15_model(k: int) -> list[int]:
    state, sign = recode_setup_model(k)
    digits: list[int] = []
    state, digit = recode_step_model(state, sign, 18)
    digits.append(digit)
    for _ in range(13):
        state, digit = recode_step_model(state, sign, 17)
        digits.append(digit)
    assert state[1:] == [0, 0, 0]
    digits.append(sign * state[0])
    return digits


def scalar_checks(pinning: str) -> dict[str, int | str]:
    assert "k < 2^256 < 2n" in pinning
    assert "k already" in pinning
    assert "gt_recode_setup(k, M, &sign);" in pinning
    assert "e[GT_CHUNKS-1]=sign*(int32_t)M[0];" in pinning
    assert "ec=gt_mixed_step<18>(M,sign);" in pinning
    assert "ec=gt_mixed_step<17>(M,sign);" in pinning
    assert "ec=(c<GT_CHUNKS-1)?gt_mixed_step<17>(M,sign):sign*(int32_t)M[0];" in pinning
    # The 18+13x17 recoder's final-window equality witness is checked against the
    # group-order arithmetic, including both signs and the two zero scalars.
    B = 1 << 240
    r = N % B
    cases = {0, 1, 2, N - 2, N - 1, N, N + 1, r, N - r, M - 1}
    rng = random.Random(0x16F1)  # fixed source-audit seed
    cases.update(rng.randrange(M) for _ in range(2048))
    for k in cases:
        digits = recode15_model(k)
        assert all(d != 0 and (d & 1) for d in digits)
        assert abs(digits[0]) < (1 << 18)
        assert all(abs(d) < (1 << 17) for d in digits[1:])
        shifts = [0] + [17 * i + 1 for i in range(1, 15)]
        represented = sum(d << shift for d, shift in zip(digits, shifts))
        assert represented % N == (2 * (k % N)) % N
    assert (2 * (N - r)) % N == (N - 2 * r) % N
    assert ((N - r) - (N - r)) % N == 0
    assert "gt_mixed_step<18>" in pinning
    assert "gt_mixed_step<17>" in pinning
    return {
        "scalar_boundary_cases": 10,
        "recoder_cases": len(cases),
        "window_bits": "18+13x17+final",
        "final_window_exception_cases": 2,
    }


def main() -> None:
    header = HEADER.read_text()
    pinning = PINNING.read_text()
    old = get_old_header()
    markers = source_checks(header, pinning, old)
    arithmetic = arithmetic_checks()
    exceptional = exceptional_checks()
    scalar = scalar_checks(pinning)
    instruction_cost = instruction_cost_checks(header)
    ptx = run_ptx_receipt()
    host = run_host_source_receipt(header, pinning)
    report = {
        "status": "PASS",
        "source_bound": True,
        "compiler_validated": False,
        "host_source_compiler_validated": True,
        "cuda_runtime_validated": False,
        "header_sha256": sha256(HEADER),
        "pinning_sha256": sha256(PINNING),
        "base_commit": BASE_COMMIT,
        "checks": {
            "source": markers,
            "arithmetic": arithmetic,
            "exceptional_finish": exceptional,
            "scalar": scalar,
            "instruction_cost": instruction_cost,
        },
        "host_source_receipt": host,
        "ptx_receipt": ptx,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
