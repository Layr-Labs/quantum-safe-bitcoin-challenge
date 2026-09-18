#pragma once
// Park the fifteen signed-odd table codes in the prepare-kernel product
// scratch, extracted by a uniform 64-bit funnel, instead of the volatile
// digit arena.
//
// qsb_prepare_scratch is a separate 8 KiB shared array
// (uint64_t[4][2*QSB_TREE_N]). HEAD's comment already records that it is
// dead during the fixed-base chain: the production packed-recovery path
// builds the cofactor tree in qsb_digit_arena, and the scalar used to
// take the scratch pointer only to (void) it. The 12 KiB digit arena is
// then used twice: first as 15 volatile per-lane code planes, then, after
// a block barrier, as the cofactor product tree. Those volatile stores
// are a scheduling fence in front of the first table __ldg. They are not
// an algebraic requirement. Chunk c's code is a function of the setup
// residue M and of the uniform bit cursor 17*c+2 (bit 1 for chunk 0).
//
// This recode writes the same idx|(neg<<31) words, with the same
// mixed-radix widths [18,17,...,17], into the idle scratch. The stores
// are ordinary, not volatile: the same thread consumes them before any
// other function aliases the scratch, so ptxas may forward the first
// codes from registers into gt_load_signed_flat_m instead of round-
// tripping through shared. The digit arena is left untouched until
// qsb_packed_prepare maps the cofactor tree onto it. The handoff
// barrier in packed prepare then only has to order the tree's own
// leaf writes, not a prior digit reload from the same bytes.
//
// The extractor is a five-limb funnel (M[4]=0) so every window, including
// the last field at bit 240, is lo>>sh | hi<<(64-sh) with no sh>46
// predicate and no per-chunk limb re-select after the mask. Not a
// sliding qsb_digit_window carried through the 13-add loop. Not a
// prefix peel that still dumps chunks 2..14 into the volatile arena.

__device__ __forceinline__ uint64_t qsb_funnel_bits(const uint64_t *M,
                                                   unsigned pos) {
    const unsigned j = pos >> 6, sh = pos & 63u;
    const uint64_t lo = M[j], hi = M[j + 1];
    if (sh == 0u)
        return lo;
    return (lo >> sh) | (hi << (64u - sh));
}

__device__ __forceinline__ uint32_t qsb_pack_window(uint64_t value,
                                                   unsigned bits, bool last,
                                                   int negative) {
    uint32_t f = (uint32_t)value & ((1u << bits) - 1u);
    int32_t tm = last ? -negative : (int32_t)(f >> (bits - 1u)) - 1;
    uint32_t idx = (f ^ (uint32_t)tm) & ((1u << (bits - 1u)) - 1u);
    return idx | ((uint32_t)(tm < 0) << 31);
}

__device__ __forceinline__ void qsb_decode_to_scratch(
    const uint64_t *k, uint64_t (*scratch)[2 * QSB_TREE_N])
{
    uint64_t M[5];
    int negative;
    qsb_signed_recode_setup(k, M, &negative);
    M[4] = 0ULL;
    uint32_t *codes = (uint32_t *)scratch;
    #pragma unroll
    for (int c = 0; c < GT_CHUNKS; c++) {
        const unsigned pos = c == 0 ? 1u : 17u * (unsigned)c + 2u;
        const unsigned bits = c == 0 ? 18u : 17u;
        codes[(size_t)c * QSB_TREE_N + threadIdx.x] =
            qsb_pack_window(qsb_funnel_bits(M, pos), bits,
                            c == GT_CHUNKS - 1, negative);
    }
}

__device__ __forceinline__ void qsb_load_decoded(
    const uint8_t *table, unsigned c, unsigned base,
    uint64_t *x, uint64_t *y, uint64_t (*scratch)[2 * QSB_TREE_N])
{
    uint32_t *codes = (uint32_t *)scratch;
    uint32_t code = codes[(size_t)c * QSB_TREE_N + threadIdx.x];
    gt_load_signed_flat_m(table, base, code & 0x1ffffu,
                          0ULL - (code >> 31), x, y);
}
