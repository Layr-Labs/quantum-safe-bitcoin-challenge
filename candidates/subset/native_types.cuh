#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
// Baseline wire layouts, used by both source roles and typed launch tags.
typedef struct {
    uint32_t mid[8];
    uint32_t remW[2];
    uint8_t early[6];
    uint8_t pad[64 - 8 * 4 - 2 * 4 - 6];
} epoch_desc_t;
static_assert(sizeof(epoch_desc_t) == 64, "epoch_desc_t must stay 64 bytes");
struct __align__(16) qsb_group_t {
    uint32_t st[8]; uint32_t w[16];
    uint32_t pos; uint32_t acc; uint32_t nb; uint32_t last;
    uint32_t base_lo; uint32_t base_hi; uint8_t o[8];
};
static_assert(sizeof(qsb_group_t)==128,"group record");
