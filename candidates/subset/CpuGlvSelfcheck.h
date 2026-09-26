#pragma once
/* Include inside qcpu after sca_batch and (when enabled) vec_batch. This one-
 * time startup check exercises the actual host GLV tables and affine kernels;
 * it never calls gate_publish, writes candidates, or changes Ctx counters. */

/* The split is used before any field arithmetic. These directed integer probes
 * cover order reduction, the 129th residual/sum bit, and both reciprocal guards.
 * Compare with the retained full-product/320-bit implementation once at startup.
 * No candidate counters or input-enumeration state are touched. */
static bool cpu_glv_split_selfcheck() {
    static const uint64_t probes[][4] = {
        {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
        {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
        {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL, 0x0000000000000000ULL},
        {0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000000ULL},
        {0x0000000000000001ULL, 0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000000ULL},
        {0xD4EE0336677A243AULL, 0xF4C41EA894C3F46DULL, 0x10C9C97FE428B49FULL, 0x0FB25D4CEF2B917AULL},
        {0xD4EE0336677A243BULL, 0xF4C41EA894C3F46DULL, 0x10C9C97FE428B49FULL, 0x0FB25D4CEF2B917AULL},
        {0x27550555E616E622ULL, 0x7FAB6F1FB19406C5ULL, 0x81AE4FD8E985797AULL, 0x5D732272B7D41A37ULL},
        {0x27550555E616E623ULL, 0x7FAB6F1FB19406C5ULL, 0x81AE4FD8E985797AULL, 0x5D732272B7D41A37ULL},
        {0x8B04358F9FFCF192ULL, 0x8641546BF721B36AULL, 0x45365CC308FAC404ULL, 0xD9A55415B295EE32ULL},
        {0x95F4E9B49F9DB193ULL, 0xE12661ABFE98CABEULL, 0x8EB261669BFA91C8ULL, 0xF4D1D7EDCD4CA8FBULL},
        {0xBFD25E8CD0364140ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
        {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
        {0xBFD25E8CD0364142ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL},
        {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}
    };
    auto check = [](const uint64_t z[4]) -> bool {
        uint64_t r1[2], r2[2], t1[2], t2[2];
        unsigned s1, s2, u1, u2;
        cpu_glv_split(z, r1, r2, &s1, &s2);
        cpu_glv_split_reference(z, t1, t2, &u1, &u2);
        return r1[0] == t1[0] && r1[1] == t1[1] && r2[0] == t2[0] && r2[1] == t2[1] && s1 == u1 && s2 == u2;
    };
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i)
        if (!check(probes[i])) return false;
    uint64_t rng = 0x510E527FADE682D1ULL;
    for (int k = 0; k < 32; ++k) {
        uint64_t z[4];
        for (int j = 0; j < 4; ++j) {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            z[j] = rng * 0x2545F4914F6CDD1DULL;
        }
        if (!check(z)) return false;
    }
    return true;
}

struct CpuGlvSelfcheckState {
    ScaBuf sca;
#if QCPU_VEC
    VecBuf vec;
#endif
    EC_GROUP *group = nullptr;
    BN_CTX *bnctx = nullptr;
    BIGNUM *x = nullptr, *y = nullptr, *z = nullptr;
    EC_POINT *a = nullptr, *cp = nullptr, *cm = nullptr, *az = nullptr, *q = nullptr;
    ~CpuGlvSelfcheckState() {
        free(sca.X); free(sca.Y); free(sca.D); free(sca.P);
        free(sca.qx); free(sca.qp); free(sca.bad);
#if QCPU_VEC
        free(vec.X); free(vec.Y); free(vec.D); free(vec.P);
        free(vec.qx); free(vec.qp); free(vec.bad);
        free(vec.rb[0]); free(vec.rb[1]); free(vec.nb[0]); free(vec.nb[1]);
#endif
        EC_POINT_free(a); EC_POINT_free(cp); EC_POINT_free(cm);
        EC_POINT_free(az); EC_POINT_free(q);
        BN_free(x); BN_free(y); BN_free(z);
        BN_CTX_free(bnctx); EC_GROUP_free(group);
    }
};

static inline bool glv_selfcheck_bn(BIGNUM *out, const fe &v) {
    uint8_t le[32];
    for (int i = 0; i < 32; ++i) le[i] = (uint8_t)(v.v[i >> 3] >> (8 * (i & 7)));
    return BN_lebin2bn(le, 32, out) != nullptr;
}

static bool glv_selfcheck(const Ctx *c, const fe &ax, const fe &ay) {
    if (!c || !c->glv || !c->table || !c->cfold || c->nw != CPU_GLV_LOGICAL_WINDOWS) return false;
    if (!cpu_glv_split_selfcheck()) return false;
    try {
        CpuGlvSelfcheckState s;
        const int B = 32;
        uint32_t dig[B * NWMAX] = {0};
        uint8_t bad[B], scalar_be[B][32];
        uint64_t rng = 0x6A09E667F3BCC909ULL;
        int valid = 0;
        for (int k = 0; k < B; ++k) {
            uint32_t words[8];
            for (int j = 0; j < 4; ++j) {
                /* Fixed xorshift64*: unsigned wrap is defined, independent of
                 * CPU, thread, clock, or candidate enumeration state. */
                rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
                const uint64_t w = rng * 0x2545F4914F6CDD1DULL;
                words[2 * j] = (uint32_t)(w >> 32); words[2 * j + 1] = (uint32_t)w;
            }
            for (int j = 0; j < 32; ++j)
                scalar_be[k][j] = (uint8_t)(words[j >> 2] >> (24 - 8 * (j & 3)));
            bad[k] = (uint8_t)cpu_glv_digits(dig + (size_t)k * NWMAX, words);
            valid += bad[k] == 0;
        }
        if (valid < 24 || !scabuf_alloc(s.sca, B)) return false;
        sca_batch(c, dig, bad, B, s.sca);
#if QCPU_VEC
        if (c->vec) {
            if (!vecbuf_alloc(s.vec, B)) return false;
            vec_batch(c, dig, bad, B, s.vec);
        }
#else
        if (c->vec) return false;
#endif
        s.group = EC_GROUP_new_by_curve_name(NID_secp256k1);
        s.bnctx = BN_CTX_new(); s.x = BN_new(); s.y = BN_new(); s.z = BN_new();
        if (!s.group || !s.bnctx || !s.x || !s.y || !s.z) return false;
        s.a = EC_POINT_new(s.group); s.cp = EC_POINT_new(s.group); s.cm = EC_POINT_new(s.group);
        s.az = EC_POINT_new(s.group); s.q = EC_POINT_new(s.group);
        if (!s.a || !s.cp || !s.cm || !s.az || !s.q) return false;
        if (!glv_selfcheck_bn(s.x, ax) || !glv_selfcheck_bn(s.y, ay) ||
            EC_POINT_set_affine_coordinates_GFp(s.group, s.a, s.x, s.y, s.bnctx) != 1 ||
            EC_POINT_is_on_curve(s.group, s.a, s.bnctx) != 1 ||
            !glv_selfcheck_bn(s.x, c->cx) || !glv_selfcheck_bn(s.y, c->cy) ||
            EC_POINT_set_affine_coordinates_GFp(s.group, s.cp, s.x, s.y, s.bnctx) != 1 ||
            EC_POINT_is_on_curve(s.group, s.cp, s.bnctx) != 1 ||
            EC_POINT_copy(s.cm, s.cp) != 1 || EC_POINT_invert(s.group, s.cm, s.bnctx) != 1) return false;
        for (int k = 0; k < B; ++k) {
            if (bad[k]) continue;
            if (s.sca.bad[k]) return false;
#if QCPU_VEC
            if (c->vec && s.vec.bad[k]) return false;
#endif
            if (!BN_bin2bn(scalar_be[k], 32, s.z) ||
                EC_POINT_mul(s.group, s.az, nullptr, s.a, s.z, s.bnctx) != 1) return false;
            for (int ri = 0; ri < 2; ++ri) {
                if (EC_POINT_add(s.group, s.q, s.az, ri ? s.cm : s.cp, s.bnctx) != 1 ||
                    EC_POINT_is_at_infinity(s.group, s.q) != 0 ||
                    EC_POINT_get_affine_coordinates_GFp(s.group, s.q, s.x, s.y, s.bnctx) != 1 ||
                    BN_is_negative(s.x) || BN_num_bytes(s.x) > 32) return false;
                fe expected; fe_from_bn(expected, s.x);
                const unsigned parity = (unsigned)BN_is_odd(s.y);
                const size_t out = (size_t)k * 2 + ri;
                for (int j = 0; j < 4; ++j) if (s.sca.qx[out].v[j] != expected.v[j]) return false;
                if (s.sca.qp[out] != parity) return false;
#if QCPU_VEC
                if (c->vec) {
                    for (int j = 0; j < 4; ++j) if (s.vec.qx[out].v[j] != expected.v[j]) return false;
                    if (s.vec.qp[out] != parity) return false;
                }
#endif
            }
        }
        return true;
    } catch (...) {
        return false; /* the local owner releases every successful allocation */
    }
}
