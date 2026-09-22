// SPDX-License-Identifier: GPL-3.0-only
// Independent exact certificate for the active C31 low-word add/sub contract.
// Caller supplies doubled=_ModAddLazy(u,u). On false, use the original full path.
#pragma once
static_assert(QSB_C31 && QSB_SHORT_CARRY && QSB_LAZY && QSB_LAZY_REC,
              "Sum certificate assumes the promoted low-64-bit correction sites");
__device__ __forceinline__ bool qsb_pair_sum2u_cert(
    const uint64_t *u,const uint64_t *v,const uint64_t *doubled) {
    const uint32_t uh=(uint32_t)(u[0]>>32),vh=(uint32_t)(v[0]>>32);
    const uint32_t dh=uh-vh,sh=uh+vh,qh=uh<<1;
    const uint32_t top=(uint32_t)(doubled[3]>>32);
    return dh>=3u && sh<0xfffffffdu &&
           (qh-2u)<0xfffffff9u && (top-1u)<0xfffffffeu;
}
