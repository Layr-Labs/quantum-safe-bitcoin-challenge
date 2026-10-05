// Incremental epoch producer. Epochs are enumerated in lexicographic order of their early
// omission set (o1<...<o6); consecutive epochs share o1..o5 in runs of up to 131. The SHA-256
// stream of an epoch is identical to that of its (o1..o5) "group" up to push o5, so the group's
// state (whole blocks + the partial block buffer) is computed once per group and each epoch only
// hashes pushes o5+1..cut-1 minus o6 (about 7 instead of 21 compressions per epoch on a ranked run).
// Output (mid, remW, early) is bit-identical to kernel_build_epochs for every epoch.
#pragma once

/* QSB_EPOCH_FAST (kill switch, default 1): two exact changes to the epoch producer, both pure
 * work removal, neither touching kernel_digest.
 *   (a) WORD EMIT. The old producers streamed the message a byte at a time into a 64-byte local
 *       array (`cur[cur_pos++]`, dynamic index -> STACK:80, one STL.U8 per byte and a
 *       `cur_pos == 64` test per byte), then bswap32'd 16 words into a scratch block for every
 *       compression. The fast path keeps a big-endian word accumulator and writes one word per
 *       4 bytes, with the boundary test once per word, and hands the big-endian words straight
 *       to _SHA256Transform (no scratch copy, no PRMT).
 *   (b) GROUP LOOKUP. kernel_build_epochs_inc used to recover its 6 omission indices with
 *       unrank_combo (6 binary searches = ~48 DEPENDENT BINOM_C loads, ~1440 SASS per epoch) and
 *       then locate its group with qsb_rank_lex (5 more loads). Every one of those is already
 *       known to the group that owns the epoch: o1..o5 are the group's own indices and o6 is
 *       determined by the epoch's offset within the group. kernel_epoch_groups now publishes
 *       (o1..o5, o5, base_rank) in the group record and scatters the group index into
 *       d_epoch_group[], so the per-epoch kernel does ONE coalesced u32 load instead.
 * Both are exact: the descriptors are bit-identical to kernel_build_epochs (see QSB_EPOCH_CHECK). */
#ifndef QSB_EPOCH_FAST
#define QSB_EPOCH_FAST 1
#endif

/* pos/acc/nb hold the partial block: `pos` is the byte position in the legacy path and the WORD
 * index in the fast path, `acc` the partial big-endian word and `nb` the bytes in it.
 * base/last/o[] carry what the per-epoch kernel used to recompute with unrank_combo. */
struct __align__(16) qsb_group_t {
    uint32_t st[8]; uint32_t w[16];
    uint32_t pos; uint32_t acc; uint32_t nb; uint32_t last;
    uint32_t base_lo; uint32_t base_hi; uint8_t o[8];
};
static_assert(sizeof(qsb_group_t)==128,"group record");
/* Lexicographic rank of sorted c[0..k-1] in C(n,k), inverse of unrank_combo (BINOM_C < 2^63 here). */
__device__ __forceinline__ uint64_t qsb_rank_lex(const uint8_t *c, int k, int n) {
    uint64_t r = 0; int prev = -1;
    for (int i = 0; i < k; i++) {
        // sum_{j=prev+1}^{c_i-1} C(n-j-1, k-i-1) = C(n-prev-1, k-i) - C(n-c_i, k-i)
        r += BINOM_C[n - prev - 1][k - i] - BINOM_C[n - c[i]][k - i];
        prev = c[i];
    }
    return r;
}

#if QSB_EPOCH_FAST
/* Big-endian word emitter. W[0..15] is the block being filled plus 2 overflow slots: a 10-byte push
 * appends at most 3 words, so wi <= 18 and the block boundary is resolved ONCE per push by the
 * caller (QSB_EMIT_FLUSH). Keeping the single _SHA256Transform call site out of the emitter is what
 * stops nvcc inlining a ~1000-instruction compression at every store. acc holds `nb` bytes,
 * right-aligned; W words are already big-endian, so the transform takes them directly. */
#define QSB_EMIT_FLUSH() do { if (wi >= 16) { \
    _SHA256Transform(state, W); W[0] = W[16]; W[1] = W[17]; wi -= 16; } } while (0)
__device__ __forceinline__ void qsb_emit_push(
    uint32_t *W, int &wi, uint32_t &acc, int &nb, const uint8_t * __restrict__ row)
{
    const unsigned short *p16 = (const unsigned short *)row;
    const uint32_t h0=__byte_perm(p16[0],0,0x4401), h1=__byte_perm(p16[1],0,0x4401);
    const uint32_t h2=__byte_perm(p16[2],0,0x4401), h3=__byte_perm(p16[3],0,0x4401);
    const uint32_t h4=__byte_perm(p16[4],0,0x4401);
    if (nb == 0) {                             /* 10 bytes -> 2 whole words + 2 carried */
        W[wi++] = (h0<<16)|h1;
        W[wi++] = (h2<<16)|h3;
        acc = h4; nb = 2;
    } else if (nb == 2) {                      /* 2 carried + 10 bytes -> 3 whole words */
        W[wi++] = (acc<<16)|h0;
        W[wi++] = (h1<<16)|h2;
        W[wi++] = (h3<<16)|h4;
        nb = 0;
    } else {                                   /* generic fallback, never taken on the ranked shape */
        #pragma unroll
        for (int b = 0; b < SIG_PUSH_SIZE; b++) {
            acc = (acc << 8) | row[b];
            if (++nb == 4) { W[wi++] = acc; nb = 0; }
        }
    }
}
#endif

__global__ void kernel_epoch_groups(
    uint64_t rank_first, uint32_t n_groups, int window_start, int s_early,
    const uint32_t * __restrict__ d_midstate,
    const uint8_t * __restrict__ d_prefix_remainder, int prefix_remainder_len,
    const uint8_t * __restrict__ d_dummy_sigs, qsb_group_t * __restrict__ d_groups
#if QSB_EPOCH_FAST
    , uint32_t * __restrict__ d_epoch_group, uint64_t epoch_base, uint64_t epoch_end
#endif
    )
{
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    const int k5 = s_early - 1;
    uint8_t o[MAX_T];
    unrank_combo(rank_first + g, window_start, k5, o);
    const int last = o[k5 - 1];
    if (last >= window_start - 1) return;          /* no room for the last omission: empty group */
    uint32_t state[8];
    for (int i = 0; i < 8; i++) state[i] = d_midstate[i];
#if QSB_EPOCH_FAST
    /* Publish what the per-epoch kernel would otherwise recompute: the group's own o1..o5, its
     * last index, and the epoch rank of its first epoch (o6 = last+1). */
    {
        uint8_t full[MAX_T];
        for (int i = 0; i < k5; i++) full[i] = o[i];
        full[k5] = (uint8_t)(last + 1);
        const uint64_t base = qsb_rank_lex(full, s_early, window_start);
        qsb_group_t *Gw = d_groups + g;
        Gw->base_lo = (uint32_t)base; Gw->base_hi = (uint32_t)(base >> 32);
        Gw->last = (uint32_t)last;
        for (int i = 0; i < 8; i++) Gw->o[i] = (i < k5) ? o[i] : 0;
        /* Scatter this group's epochs -> group index. The run is [base, base + (window_start-1-last)). */
        const uint64_t lo = base > epoch_base ? base : epoch_base;
        const uint64_t hi0 = base + (uint64_t)(window_start - 1 - last);
        const uint64_t hi = hi0 < epoch_end ? hi0 : epoch_end;
        for (uint64_t x = lo; x < hi; x++) d_epoch_group[x - epoch_base] = g;
    }
    uint32_t W[18], acc = 0; int wi = 0, nb = 0;
    for (int i = 0; i < prefix_remainder_len; i++) {
        acc = (acc << 8) | d_prefix_remainder[i];
        if (++nb == 4) { W[wi++] = acc; nb = 0; QSB_EMIT_FLUSH(); }
    }
    {
        int sel = 0;
        for (int i = 0; i <= last; i++) {
            if (sel < k5 && (int)o[sel] == i) { sel++; continue; }
            qsb_emit_push(W, wi, acc, nb, d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE);
            QSB_EMIT_FLUSH();
        }
    }
    qsb_group_t *G = d_groups + g;
    for (int i = 0; i < 8; i++) G->st[i] = state[i];
    for (int i = 0; i < 16; i++) G->w[i] = W[i];
    G->pos = (uint32_t)wi; G->acc = acc; G->nb = (uint32_t)nb;
#else
    uint32_t curW[16];
    uint8_t *cur = (uint8_t *)curW;
    int cur_pos = 0;
    for (int i = 0; i < prefix_remainder_len; i++) {
        cur[cur_pos++] = d_prefix_remainder[i];
        if (cur_pos == 64) {
            uint32_t blk[16];
            for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
            _SHA256Transform(state, blk);
            cur_pos = 0;
        }
    }
    int sel = 0;
    for (int i = 0; i <= last; i++) {
        if (sel < k5 && (int)o[sel] == i) { sel++; continue; }
        const uint8_t *row = d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE;
        for (int b = 0; b < SIG_PUSH_SIZE; b++) {
            cur[cur_pos++] = row[b];
            if (cur_pos == 64) {
                uint32_t blk[16];
                for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
                _SHA256Transform(state, blk);
                cur_pos = 0;
            }
        }
    }
    qsb_group_t *G = d_groups + g;
    for (int i = 0; i < 8; i++) G->st[i] = state[i];
    for (int i = 0; i < 16; i++) G->w[i] = curW[i];
    G->pos = (uint32_t)cur_pos;
#endif
}

__global__ void kernel_build_epochs_inc(
    uint64_t epoch_base, uint64_t n_epochs, int window_start, int s_early,
    const uint8_t * __restrict__ d_dummy_sigs, const qsb_group_t * __restrict__ d_groups,
    uint64_t rank_first, epoch_desc_t * __restrict__ d_epochs, uint32_t *d_hit_reset
#if QSB_EPOCH_FAST
    , const uint32_t * __restrict__ d_epoch_group
#endif
    )
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0) *d_hit_reset = 0;       /* runs before this launch's digest kernel on the same stream */
    const uint64_t e = epoch_base + (uint64_t)t;
    if (e >= n_epochs) return;
    uint8_t early[MAX_T];
#if QSB_EPOCH_FAST
    const qsb_group_t *G = d_groups + d_epoch_group[t];
    const int last = (int)G->last;
    const uint64_t base = ((uint64_t)G->base_hi << 32) | (uint64_t)G->base_lo;
    const int o6 = last + 1 + (int)(e - base);
    for (int i = 0; i < s_early - 1; i++) early[i] = G->o[i];
    early[s_early - 1] = (uint8_t)o6;
    uint32_t state[8], W[18];
    for (int i = 0; i < 8; i++) state[i] = G->st[i];
    for (int i = 0; i < 16; i++) W[i] = G->w[i];
    int wi = (int)G->pos, nb = (int)G->nb; uint32_t acc = G->acc;
    {
        for (int i = last + 1; i < window_start; i++) {
            if (i == o6) continue;
            qsb_emit_push(W, wi, acc, nb, d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE);
            QSB_EMIT_FLUSH();
        }
    }
    epoch_desc_t *d = d_epochs + t;
    for (int i = 0; i < 8; i++) d->mid[i] = state[i];
    d->remW[0] = W[0];
    d->remW[1] = W[1];
    for (int i = 0; i < s_early; i++) d->early[i] = early[i];
#else
    unrank_combo(e, window_start, s_early, early);
    const qsb_group_t *G = d_groups + (qsb_rank_lex(early, s_early - 1, window_start) - rank_first);
    uint32_t state[8], curW[16];
    for (int i = 0; i < 8; i++) state[i] = G->st[i];
    for (int i = 0; i < 16; i++) curW[i] = G->w[i];
    int cur_pos = (int)G->pos;
    uint8_t *cur = (uint8_t *)curW;
    const int o6 = early[s_early - 1];
    for (int i = early[s_early - 2] + 1; i < window_start; i++) {
        if (i == o6) continue;
        const uint8_t *row = d_dummy_sigs + (size_t)i * SIG_PUSH_SIZE;
        for (int b = 0; b < SIG_PUSH_SIZE; b++) {
            cur[cur_pos++] = row[b];
            if (cur_pos == 64) {
                uint32_t blk[16];
                for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
                _SHA256Transform(state, blk);
                cur_pos = 0;
            }
        }
    }
    epoch_desc_t *d = d_epochs + t;
    for (int i = 0; i < 8; i++) d->mid[i] = state[i];
    d->remW[0] = bswap32(curW[0]);
    d->remW[1] = bswap32(curW[1]);
    for (int i = 0; i < s_early; i++) d->early[i] = early[i];
#endif
}
