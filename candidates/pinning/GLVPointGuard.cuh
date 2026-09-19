// SPDX-License-Identifier: GPL-3.0-only
#pragma once
__device__ __forceinline__ void qsb_field_mul(uint64_t*,uint64_t*,uint64_t*);
__device__ __forceinline__ bool q9_zero(const uint64_t *p){
    return !(p[0]|p[1]|p[2]|p[3]) ||
        ((p[1]&p[2]&p[3])==0xffffffffffffffffULL&&p[0]==0xfffffffefffffc2fULL);
}
