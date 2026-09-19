#pragma once

// Correctness-first GLV fixed-base accumulator.  The including translation
// unit supplies glv_split.h, gt_offset/gt_entries, and the GPUMath field/XYZZ
// primitives.  scratch is the dead prepare product tree, viewed as sixteen
// QSB_TREE_N-wide int32 digit planes.

#ifndef QSB_GLV_LANE
#define QSB_GLV_LANE ((uint32_t)threadIdx.x)
#endif

__device__ __forceinline__ bool qsb_glv_field_zero(const uint64_t a[4]) {
    const bool zero = (a[0] | a[1] | a[2] | a[3]) == 0;
    const bool prime = a[0] == 0xFFFFFFFEFFFFFC2FULL &&
                       a[1] == 0xFFFFFFFFFFFFFFFFULL &&
                       a[2] == 0xFFFFFFFFFFFFFFFFULL &&
                       a[3] == 0xFFFFFFFFFFFFFFFFULL;
    return zero || prime;
}

__device__ __forceinline__ void qsb_glv_clear_point(
    uint64_t x[4], uint64_t y[4], uint64_t zz[4], uint64_t zzz[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = y[i] = zz[i] = zzz[i] = 0;
}

__device__ __forceinline__ void qsb_glv_seed_point(
    uint64_t x[4], uint64_t y[4], uint64_t zz[4], uint64_t zzz[4],
    const uint64_t ax[4], const uint64_t ay[4]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = ax[i]; y[i] = ay[i];
        zz[i] = (i == 0); zzz[i] = (i == 0);
    }
}

// The accepted fast square/multiply can lose a rare final cold carry even for
// a canonical near-p input.  Use the exact product-tree multiplier in this
// exceptional equality/doubling path, with five-limb storage because
// qsb_field_mul writes out[4], then canonicalize every result before add/sub.
__device__ __forceinline__ void qsb_glv_mul_exact(
    uint64_t out[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t aa[5] = {a[0], a[1], a[2], a[3], 0};
    uint64_t bb[5] = {b[0], b[1], b[2], b[3], 0};
    uint64_t tmp[5];
    qsb_field_mul(tmp, aa, bb);
    qsb_field_normalize(tmp);
    #pragma unroll
    for (int i = 0; i < 4; ++i) out[i] = tmp[i];
}

__device__ __forceinline__ void qsb_glv_square_exact(
    uint64_t out[4], const uint64_t a[4]) {
    qsb_glv_mul_exact(out, a, a);
}

struct qsb_glv_xyzz_value {
    uint64_t x[4];
    uint64_t y[4];
    uint64_t zz[4];
    uint64_t zzz[4];
    uint32_t valid;
};

// Exact XYZZ doubling for y^2=x^3+7.  Every output is computed before any
// input is overwritten.  Field-zero Y maps to the identity.
__device__ __noinline__ qsb_glv_xyzz_value qsb_glv_double_xyzz_value(
    qsb_glv_xyzz_value point) {
    qsb_field_normalize(point.x);
    qsb_field_normalize(point.y);
    qsb_field_normalize(point.zz);
    qsb_field_normalize(point.zzz);
    if (qsb_glv_field_zero(point.y)) {
        qsb_glv_clear_point(point.x, point.y, point.zz, point.zzz);
        point.valid = 0;
        return point;
    }

    uint64_t a[4], b[4], c[4], d[4], e[4], f[4];
    uint64_t nx[4], ny[4], nzz[4], nzzz[4], t[4];
    qsb_glv_square_exact(a, point.x);   // A = X^2
    qsb_glv_square_exact(b, point.y);   // B = Y^2
    qsb_glv_square_exact(c, b);         // C = B^2
    _ModAdd256(t, point.x, b);
    qsb_glv_square_exact(d, t);
    _ModSub256(d, d, a);
    _ModSub256(d, d, c);
    _ModAdd256(d, d, d);                // D = 2*((X+B)^2-A-C)
    _ModAdd256(e, a, a);
    _ModAdd256(e, e, a);                // E = 3*A
    qsb_glv_square_exact(f, e);         // F = E^2
    _ModSub256(nx, f, d);
    _ModSub256(nx, nx, d);             // X3 = F-2*D
    _ModSub256(ny, d, nx);
    qsb_glv_mul_exact(ny, ny, e);       // E*(D-X3)
    _ModAdd256(t, c, c);
    _ModAdd256(t, t, t);
    _ModAdd256(t, t, t);                // 8*C
    _ModSub256(ny, ny, t);
    _ModAdd256(t, b, b);
    _ModAdd256(t, t, t);                // 4*B
    qsb_glv_mul_exact(nzz, t, point.zz);
    qsb_glv_mul_exact(t, point.y, b);
    _ModAdd256(t, t, t);
    _ModAdd256(t, t, t);
    _ModAdd256(t, t, t);                // 8*Y*B
    qsb_glv_mul_exact(nzzz, t, point.zzz);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        point.x[i] = nx[i]; point.y[i] = ny[i];
        point.zz[i] = nzz[i]; point.zzz[i] = nzzz[i];
    }
    point.valid = 1;
    return point;
}

__device__ __forceinline__ bool qsb_glv_double_xyzz(
    uint64_t x[4], uint64_t y[4], uint64_t zz[4], uint64_t zzz[4]) {
    qsb_glv_xyzz_value point;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        point.x[i] = x[i]; point.y[i] = y[i];
        point.zz[i] = zz[i]; point.zzz[i] = zzz[i];
    }
    point.valid = 1;
    point = qsb_glv_double_xyzz_value(point);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        x[i] = point.x[i]; y[i] = point.y[i];
        zz[i] = point.zz[i]; zzz[i] = point.zzz[i];
    }
    return point.valid != 0;
}

// Add an affine point to an exact XYZZ accumulator, dispatching all exceptional
// cases before calling the accepted ordinary mixed-add formula.
__device__ __forceinline__ bool qsb_glv_add_complete(
    uint64_t x[4], uint64_t y[4], uint64_t zz[4], uint64_t zzz[4],
    const uint64_t ax[4], const uint64_t ay[4]) {
    uint64_t u2[4], s2[4], h[4], r[4];
    qsb_glv_mul_exact(u2, ax, zz);
    qsb_glv_mul_exact(s2, ay, zzz);
    _ModSub256(h, u2, x);
    _ModSub256(r, s2, y);
    if (qsb_glv_field_zero(h)) {
        if (!qsb_glv_field_zero(r)) {
            qsb_glv_clear_point(x, y, zz, zzz);
            return false;
        }
        return qsb_glv_double_xyzz(x, y, zz, zzz);
    }
    const uint64_t zero[4] = {0, 0, 0, 0};
    _PointAddXYZZT<false>(x, y, zz, zzz, ax, ay, zero);
    return true;
}

__device__ __forceinline__ void qsb_glv_load_signed(
    const uint8_t *table, int segment, uint32_t magnitude, bool negative,
    uint64_t x[4], uint64_t y[4]) {
    const uint64_t *record = (const uint64_t *)(table +
        ((size_t)gt_offset(segment) + magnitude - 1u) * 64u);
    #pragma unroll
    for (int i = 0; i < 4; ++i) x[i] = record[i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) y[i] = record[4+i];
    if (negative) {
        const uint64_t p[4] = {0xFFFFFFFEFFFFFC2FULL,
                               0xFFFFFFFFFFFFFFFFULL,
                               0xFFFFFFFFFFFFFFFFULL,
                               0xFFFFFFFFFFFFFFFFULL};
        _ModSub256(y, p, y);
    }
}

__device__ __forceinline__ bool qsb_glv_fixed_base(
    uint64_t qx[4], uint64_t qy[4], uint64_t qzz[4], uint64_t qzzz[4],
    const uint64_t raw[4], const uint8_t *table, uint32_t *scratch) {
    glv_probe::U256 scalar;
    #pragma unroll
    for (int i = 0; i < 4; ++i) scalar.v[i] = raw[i];
    const glv_probe::RecodedSplit recoded = glv_probe::split_recode(scalar);
    if (!recoded.range_ok) {
        qsb_glv_clear_point(qx, qy, qzz, qzzz);
        return false;
    }

    const uint32_t lane = QSB_GLV_LANE;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        scratch[i * QSB_TREE_N + lane] = (uint32_t)recoded.first.v[i];
        scratch[(i+8) * QSB_TREE_N + lane] = (uint32_t)recoded.second.v[i];
    }

    bool have_point = false;
    qsb_glv_clear_point(qx, qy, qzz, qzzz);
    #pragma unroll 1
    for (int segment = 0; segment < 16; ++segment) {
        const int32_t digit = (int32_t)scratch[segment * QSB_TREE_N + lane];
        if (digit == 0) continue;
        const bool negative = digit < 0;
        const uint32_t magnitude = negative ? (uint32_t)(-(int64_t)digit) :
                                              (uint32_t)digit;
        if (magnitude == 0 || magnitude > gt_entries(segment)) {
            qsb_glv_clear_point(qx, qy, qzz, qzzz);
            return false;
        }
        uint64_t ax[4], ay[4];
        qsb_glv_load_signed(table, segment, magnitude, negative, ax, ay);
        if (!have_point) {
            qsb_glv_seed_point(qx, qy, qzz, qzzz, ax, ay);
            have_point = true;
        } else {
            have_point = qsb_glv_add_complete(qx, qy, qzz, qzzz, ax, ay);
        }
    }
    if (!have_point) qsb_glv_clear_point(qx, qy, qzz, qzzz);
    return true;
}
