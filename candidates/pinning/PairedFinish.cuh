/* Experimental pair-lane recovery on the unchanged ranked finish helpers.
 * Two physical lanes own one logical candidate. Default-off at the caller.
 */
#pragma once
#if QSB_PAIRED_FINISH
#if !(QSB_LAZY_REC && QSB_PARITY_SUM && QSB_PARITY_WINDOW && QSB_FIN_RAWS && QSB_XOUT_LAZY && QSB_NEG_Y_MAC && QSB_ROOT_V2 && QSB_PREP_STATE)
#error "paired finish requires the exact active 8d07 recovery path"
#endif
static_assert(QSB_TREE_N == 128 && QSB_S2_THREADS == 128 && QSB_STATE_PLANES == 4,
              "paired finish keeps the logical 128-candidate geometry");

__device__ __forceinline__ void qsb_pair_exchange256(
    uint64_t *peer, const uint64_t *own, uint32_t mask) {
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint32_t lo = __shfl_xor_sync(mask, (uint32_t)own[k], 1);
        uint32_t hi = __shfl_xor_sync(mask, (uint32_t)(own[k] >> 32), 1);
        peer[k] = (uint64_t)lo | ((uint64_t)hi << 32);
    }
}

/* Called only by FAST_TAIL/STAGE2 after both partners' identical active test.
 * Own input/inverse loads are selected before one common field multiply.
 */
__device__ __forceinline__ void qsb_paired_finish_ranked(
    int idx, int batch_size, ulonglong2 *saved, uint64_t *roots, uint64_t *tree,
    uint32_t *d_hit_cnt, uint32_t *d_hit_idx) {
    const uint32_t ri = (uint32_t)threadIdx.x & 1u;
    const uint32_t j = (uint32_t)threadIdx.x >> 1;
    uint32_t mask = __activemask();
    uint64_t slope[4], sum[4];
    {
        /* Even tbar planes2/3; odd vbar planes0/1. Same state addresses/hints. */
        const uint32_t plane = ri == 0u ? 2u : 0u;
        const ulonglong2 *st = saved + (uint32_t)(QSB_STATE_BLK *
            (QSB_STATE_PLANES * QSB_TREE_N) + j + plane * QSB_TREE_N);
#if QSB_L2STATE & 512
        const ulonglong2 a01 = __ldcg(st), a23 = __ldcg(st + QSB_TREE_N);
#else
        const ulonglong2 a01 = qsb_ld_v2(st), a23 = qsb_ld_v2(st + QSB_TREE_N);
#endif
        uint64_t own_input[4] = {a01.x, a01.y, a23.x, a23.y};
        const uint32_t input_nonzero =
            (own_input[0] | own_input[1] | own_input[2] | own_input[3]) != 0;
        const uint32_t peer_nonzero = __shfl_xor_sync(mask, input_nonzero, 1);
        const uint32_t usable = ri == 0u ? input_nonzero : peer_nonzero;
        mask = __ballot_sync(mask, usable != 0u);
        if (!usable) return;  /* Whole pairs exit before any field exchange. */

        const uint32_t root_count = ((uint32_t)batch_size + QSB_TREE_N - 1u) / QSB_TREE_N;
        const uint32_t root_row = (uint32_t)blockIdx.x + (ri == 0u ? root_count : 0u);
        const ulonglong2 *r2 = (const ulonglong2 *)roots;
        const ulonglong2 b01 = r2[2ull * root_row], b23 = r2[2ull * root_row + 1u];
        uint64_t own_inv[4] = {b01.x, b01.y, b23.x, b23.y};
        uint64_t own[4], peer[4];
        QSB_FIN_RAW_MUL(own, own_input, own_inv);  /* One full-mask M call. */
        qsb_pair_exchange256(peer, own, mask);

        /* Ordered u/v: even owns u, odd owns v. NEG_Y_MAC stays enabled. */
        if (ri == 0u) QSB_FIN_ADDL(slope, own, peer);
        else QSB_FIN_SUB(slope, peer, own);
        qsb_pair_exchange256(peer, slope, mask);

        /* Exact FIN_ADDL commutativity preserves every output bit, including
         * the inherited short low64 carry repair. Both lanes use own/peer.
         */
        QSB_FIN_ADDL(sum, slope, peer);
    }  /* Own inverse/input, peer product/slope are dead. */

    uint64_t t[4], s[4], x[4];
    uint64_t c[4] = {pin_recovery_c[0], pin_recovery_c[1],
                     pin_recovery_c[2], pin_recovery_c[3]};
    QSB_FIN_SUB(t, slope, c);
    QSB_FIN_RAW_MUL(s, sum, t);
    uint64_t b[4] = {pin_u2ry_words[0], pin_u2ry_words[1],
                     pin_u2ry_words[2], pin_u2ry_words[3]};
    const uint32_t parity = qsb_parity_product_window(slope, s, b, (uint32_t)(ri == 0u));
    uint64_t a[4] = {pin_u2rx_words[0], pin_u2rx_words[1],
                     pin_u2rx_words[2], pin_u2rx_words[3]};
    QSB_FIN_ADDL(x, s, a);

#if (QSB_L2STATE & 1024) && QSB_SM80_PTX && QSB_PREP_STATE
    __syncwarp(mask);
    if (((uint32_t)threadIdx.x & 15u) == 0u) {
        const ulonglong2 *dst = saved +
            (uint32_t)(QSB_STATE_BLK * (QSB_STATE_PLANES * QSB_TREE_N) + j);
        qsb_discard_l2(dst); qsb_discard_l2(dst + QSB_TREE_N);
        qsb_discard_l2(dst + 2 * QSB_TREE_N); qsb_discard_l2(dst + 3 * QSB_TREE_N);
    }
#endif

    const uint32_t x0 = (uint32_t)x[0], x1 = (uint32_t)(x[0] >> 32);
    const uint32_t x2 = (uint32_t)x[1], x3 = (uint32_t)(x[1] >> 32);
    const uint32_t x4 = (uint32_t)x[2], x5 = (uint32_t)(x[2] >> 32);
    const uint32_t x6 = (uint32_t)x[3], x7 = (uint32_t)(x[3] >> 32);
    uint32_t pb[16];
    pb[0] = __byte_perm(x7, 2u + parity, 0x4321);
    pb[1] = __byte_perm(x7, x6, 0x0765); pb[2] = __byte_perm(x6, x5, 0x0765);
    pb[3] = __byte_perm(x5, x4, 0x0765); pb[4] = __byte_perm(x4, x3, 0x0765);
    pb[5] = __byte_perm(x3, x2, 0x0765); pb[6] = __byte_perm(x2, x1, 0x0765);
#if QSB_FIN_BAL2 & 2
    pb[7] = __byte_perm(x1, x0, 0x0765);
    asm("{\n.reg .u32 f;\nld.const.u32 f,[pin_pow2+96];\nmad.lo.u32 %0,%1,f,%2;\n}"
        : "=r"(pb[8]) : "r"(x0), "r"(0x800000u));
#else
    pb[7] = __byte_perm(x1, x0, 0x0765); pb[8] = __byte_perm(x0, 0x80, 0x0456);
#endif
    const uint32_t hit = (uint32_t)gpu_bench_valid_h0(_SHA256Pubkey33H0(pb));
    const uint32_t peer_hit = __shfl_xor_sync(mask, hit, 1);
    if (hit && (ri == 0u || !peer_hit)) {
        const uint32_t pos = atomicAdd(d_hit_cnt, 1);
        if (pos < 1024u) d_hit_idx[pos] =
            ((uint32_t)idx + QSB_HIT_BASE) | (ri << 30);
    }
}
#endif
