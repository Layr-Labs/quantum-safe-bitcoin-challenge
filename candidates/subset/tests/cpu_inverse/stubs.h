#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <stdint.h>
#include <stdlib.h>
#define SIG_PUSH_SIZE 10
#define QSB_SE_TWIN 3
#define QSB_SE_PER_EPOCH 128
#define QSB_ZEROS_N 24
typedef struct {
    uint32_t n, t;
    uint32_t total_preimage_len;
    uint32_t tail_section_len;
    uint32_t tx_suffix_len;
    uint32_t prefix_remainder_len;
    uint32_t midstate[8];
    uint8_t *prefix_remainder;
    uint8_t *dummy_sigs;
    uint8_t *tail_section;
    uint8_t *tx_suffix;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} digest_params_t;
typedef struct {
    EC_GROUP *grp; BN_CTX *ctx; BIGNUM *order; BIGNUM *nri; EC_POINT *Ru2;
    const digest_params_t *dp;
    uint8_t win3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
    int window_start, s_early;
} qsb_hv_t;
static uint64_t binom_u64(int, int) { abort(); }
static void qsb_host_unrank(uint64_t, int, int, uint8_t *) { abort(); }
static int qsb_hv_init(qsb_hv_t *, const digest_params_t *, const uint8_t (*)[QSB_SE_TWIN], int, int) { abort(); }
static int qsb_hv_check(const qsb_hv_t *, const uint8_t *, int) { abort(); }
