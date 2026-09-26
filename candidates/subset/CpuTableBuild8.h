#pragma once
/* Include inside qcpu, under QCPU_VEC, after the Q8T field helpers.
 * Fixed-point vector table addition adapted from the completed 296e5e53 donor
 * (2446855b0c562d364a12701973805ab6d26b1f43). Keep the caller's layout,
 * allocation policy, collision repair, and the local fe8_inv4 implementation.
 */

struct CpuTableBuild8Scratch {
    fe8 *D = nullptr, *PRE = nullptr, *X = nullptr, *Y = nullptr;
    uint8_t *flag = nullptr;
    size_t capacity = 0;
    CpuTableBuild8Scratch() = default;
    CpuTableBuild8Scratch(const CpuTableBuild8Scratch &) = delete;
    CpuTableBuild8Scratch &operator=(const CpuTableBuild8Scratch &) = delete;
    void release() {
        free(D); free(PRE); free(X); free(Y); free(flag);
        D = PRE = X = Y = nullptr; flag = nullptr; capacity = 0;
    }
    bool alloc(size_t n) {
        if (!n || n % 32) return false;
        if (capacity >= n) return true;
        if (n / 8 > (size_t)-1 / sizeof(fe8)) return false;
        void *q[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        const size_t bytes = sizeof(fe8) * (n / 8);
        for (int i = 0; i < 5; ++i) {
            if (posix_memalign(&q[i], 64, i == 4 ? n : bytes)) {
                for (int j = 0; j < i; ++j) free(q[j]);
                return false;
            }
        }
        release();
        D = (fe8 *)q[0]; PRE = (fe8 *)q[1]; X = (fe8 *)q[2]; Y = (fe8 *)q[3];
        flag = (uint8_t *)q[4]; capacity = n;
        return true;
    }
    ~CpuTableBuild8Scratch() { release(); }
};

/* out[k] = in[k] + Q for canonical finite points. Require n>0, n%32==0,
 * n<=scratch.capacity and n/8<=INT_MAX. The actual builder uses n<=4096.
 * Equal-x lanes set flag and DO NOT write their output; the caller repairs
 * them. D=1 isolates these lanes before any chain multiplication.
 * All input coordinates and Q are consumed before output stores, so out==in
 * and adjacent disjoint ranges are supported, including Q inside the input.
 */
Q8T static void cpu_table_add8_fixed(pt *out, const pt *in, size_t n,
                                    const pt &Q, CpuTableBuild8Scratch &scratch) {
    const int G = (int)(n / 8);
    fe8 *D = scratch.D, *PRE = scratch.PRE, *PX = scratch.X, *PY = scratch.Y;
    uint8_t *flag = scratch.flag;
    fe8 QX, QY; fe8_bcast(QX, Q.x); fe8_bcast(QY, Q.y);
    fe8 run[4]; for (int c = 0; c < 4; ++c) fe8_set1(run[c]);
    const pt *rows[8];
    for (int g = 0; g < G; g += 4) {
        for (int c = 0; c < 4; ++c) {
            const int h = g + c;
            unsigned eq = 0;
            for (int j = 0; j < 8; ++j) {
                rows[j] = &in[(size_t)h * 8 + j];
                const bool same_x = fe_eq(rows[j]->x, Q.x);
                flag[h * 8 + j] = (uint8_t)same_x;
                eq |= (unsigned)same_x << j;
            }
            pt8_load(PX[h], PY[h], rows);
            fe8_sub(D[h], QX, PX[h]);
            if (eq) {
                D[h].l[0] = _mm512_mask_mov_epi64(D[h].l[0], (__mmask8)eq, _mm512_set1_epi64(1));
                for (int i = 1; i < 5; ++i)
                    D[h].l[i] = _mm512_mask_mov_epi64(D[h].l[i], (__mmask8)eq, _mm512_setzero_si512());
            }
            fe8_mov(PRE[h], run[c]);
            fe8_mul(run[c], run[c], D[h]);
        }
    }
    fe8_inv4(run);                 /* retain the local safegcd/zero-safe backend */
    for (int g = G - 4; g >= 0; g -= 4) {
        for (int c = 3; c >= 0; --c) {
            const int h = g + c;
            fe8 dinv, t, lam, x3, y3;
            fe8_mul(dinv, run[c], PRE[h]); fe8_mul(run[c], run[c], D[h]);
            fe8_sub(t, QY, PY[h]); fe8_mul(lam, t, dinv);
            fe8_sqr(x3, lam); fe8_sub2(x3, x3, PX[h], QX);
            fe8_sub(t, PX[h], x3); fe8_mul(y3, lam, t); fe8_sub(y3, y3, PY[h]);
            __m512i xw[4], yw[4]; fe8_canon_words(xw, x3); fe8_canon_words(yw, y3);
            alignas(64) uint64_t XW[4][8], YW[4][8];
            for (int i = 0; i < 4; ++i) {
                _mm512_store_si512(XW[i], xw[i]); _mm512_store_si512(YW[i], yw[i]);
            }
            for (int j = 0; j < 8; ++j) {
                if (flag[h * 8 + j]) continue;
                pt &o = out[(size_t)h * 8 + j];
                for (int i = 0; i < 4; ++i) { o.x.v[i] = XW[i][j]; o.y.v[i] = YW[i][j]; }
            }
        }
    }
}

/* Startup-only check against the original scalar batch_add, not against a
 * second vector route. The caller MUST check c.vec before calling this Q8T
 * boundary. Failure disables only the new builder, not CPU grinding.
 */
Q8T static __attribute__((noinline)) bool cpu_table_build8_selfcheck(const fe &ax, const fe &ay) {
    CpuTableBuild8Scratch scratch;
    if (!scratch.alloc(64)) return false;
    pt seed[64], shared[128], reference[64], before[64];
    fe den[64], pre[64];
    uint8_t inf[64], bad[64];
    const pt *tp[64];
    const fe zero = {{0, 0, 0, 0}};
    seed[0] = pt{ax, ay};
    /* Build 1*A..64*A using only the preexisting scalar recurrence. */
    for (int have = 1; have < 64; have *= 2) {
        for (int k = 0; k < have; ++k) {
            seed[have + k] = seed[k]; tp[k] = &seed[have - 1];
            inf[k] = bad[k] = 0;
        }
        batch_add(seed + have, tp, inf, bad, have, den, pre);
        for (int k = 0; k < have; ++k) if (bad[k]) seed[have + k] = pt_double(seed[have - 1]);
    }
    for (int n = 32; n <= 64; n *= 2) {
        for (int mode = 0; mode < 3; ++mode) {
            /* Adjacent output after input, adjacent output before input,
             * then exact in-place output. Q aliases the input's final row. */
            pt *in = shared + (mode == 1 ? n : 0);
            pt *out = mode == 0 ? shared + n : shared;
            for (int k = 0; k < n; ++k) {
                in[k] = seed[k];
                if ((k & 1) && k + 1 != n) fe_sub(in[k].y, zero, in[k].y);
            }
            const pt fixed = in[n - 1];
            in[0] = fixed;                            /* a true doubling */
            in[1] = fixed; fe_sub(in[1].y, zero, in[1].y); /* inverse pair: flag, never enter the product */
            if (out != in) {
                for (int k = 0; k < n; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        out[k].x.v[i] = 0x13579bdf2468ace0ULL;
                        out[k].y.v[i] = 0xfedcba9876543210ULL;
                    }
                }
            }
            for (int k = 0; k < n; ++k) {
                reference[k] = in[k]; before[k] = out[k]; tp[k] = &fixed;
                inf[k] = bad[k] = 0;
            }
            batch_add(reference, tp, inf, bad, n, den, pre);
            cpu_table_add8_fixed(out, in, (size_t)n, in[n - 1], scratch);
            for (int k = 0; k < n; ++k) {
                if ((unsigned)scratch.flag[k] != (unsigned)bad[k]) return false;
                if (bad[k]) {
                    if (!fe_eq(out[k].x, before[k].x) || !fe_eq(out[k].y, before[k].y)) return false;
                    /* Exact finite doubling; the inverse case preserves the
                     * scalar flagged row. The production progression never
                     * contains an inverse pair and keeps its existing repair. */
                    if (fe_eq(reference[k].y, fixed.y)) reference[k] = pt_double(fixed);
                    out[k] = reference[k];
                }
                if (!fe_eq(out[k].x, reference[k].x) || !fe_eq(out[k].y, reference[k].y)) return false;
            }
        }
    }
    return true;
}
