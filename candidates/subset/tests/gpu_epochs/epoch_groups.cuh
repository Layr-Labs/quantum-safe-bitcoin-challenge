// Incremental epoch producer. Epochs are enumerated in lexicographic order of their early
// omission set (o1<...<o6); consecutive epochs share o1..o5 in runs of up to 131. The SHA-256
// stream of an epoch is identical to that of its (o1..o5) "group" up to push o5, so the group's
// state (whole blocks + the partial block buffer) is computed once per group and each epoch only
// hashes pushes o5+1..cut-1 minus o6 (about 7 instead of 21 compressions per epoch on a ranked run).
// Output (mid, remW, early) is bit-identical to kernel_build_epochs for every epoch.
#pragma once
struct __align__(16) qsb_group_t { uint32_t st[8]; uint32_t w[16]; uint32_t pos; uint32_t pad[7]; };
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
__global__ void kernel_epoch_groups(
    uint64_t rank_first, uint32_t n_groups, int window_start, int s_early,
    const uint32_t * __restrict__ d_midstate,
    const uint8_t * __restrict__ d_prefix_remainder, int prefix_remainder_len,
    const uint8_t * __restrict__ d_dummy_sigs, qsb_group_t * __restrict__ d_groups)
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
}
__global__ void kernel_build_epochs_inc(
    uint64_t epoch_base, uint64_t n_epochs, int window_start, int s_early,
    const uint8_t * __restrict__ d_dummy_sigs, const qsb_group_t * __restrict__ d_groups,
    uint64_t rank_first, epoch_desc_t * __restrict__ d_epochs, uint32_t *d_hit_reset)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0) *d_hit_reset = 0;       /* runs before this launch's digest kernel on the same stream */
    const uint64_t e = epoch_base + (uint64_t)t;
    if (e >= n_epochs) return;
    uint8_t early[MAX_T];
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
}
