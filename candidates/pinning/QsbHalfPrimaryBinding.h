#pragma once
// Exact b59 primary kernel binding for the half dispatcher. Include only AFTER
// the original kernel_pinning_pipeline and qsb_tail_pre declarations. Selected primary GPU code remains unchanged; native validation is official-only.
#include "HalfDispatchPrototype.h"
#if !QSB_ASICBOOST || QSB_AB_K != 4 || !QSB_PK_ON
#error "half binding requires the verified b59 ASICBOOST4/PK-offload contract"
#endif
namespace qsb_root_secondary_research {
struct PrimaryArgs {
    const uint32_t *d_midstate;
    const uint8_t *d_suffix;
    int suffix_len,seq_offset,lt_offset,total_preimage_len;
    uint32_t seq_value,start_lt;
    const uint64_t *d_neg_r_inv,*d_u2rx,*d_u2ry,*d_neg2u2rx,*d_neg2u2ry;
    uint8_t *d_gt,*pk_plane;
    uint32_t *d_hit_cnt,*d_hit_idx;
    int easy_mode,single_hash;
    qsb_tail_pre tp;
};
static bool primary_available() {
    return qsb_carrier_has(QK_S0)&&qsb_carrier_has(QK_S2);
}
template<int Stage>
static cudaError_t primary(const PrimaryArgs &a,const qsb_half_schedule_research::Slice &q,
                            void *state,uint64_t *roots,cudaStream_t stream) {
    static_assert(Stage==0||Stage==2,"primary stage must be prepare or finish");
    if(!primary_available()||!state||!roots)return cudaErrorInvalidDeviceFunction;
    const uint32_t lt0=a.start_lt+q.locktime_delta;
    uint64_t *hit_base=(uint64_t*)(uintptr_t)q.hit_base;
    if(Stage==0)
        return qsb_carrier_launch(kernel_pinning_pipeline<true,0>,QK_S0,
            dim3(q.blocks),dim3(QSB_S0_THREADS),stream,
            a.d_midstate,a.d_suffix,a.suffix_len,a.seq_offset,a.lt_offset,a.total_preimage_len,
            a.seq_value,lt0,a.d_neg_r_inv,a.d_u2rx,a.d_u2ry,a.d_neg2u2rx,a.d_neg2u2ry,
            a.d_gt,a.d_hit_cnt,a.d_hit_idx,(int)q.count,a.easy_mode,a.single_hash,
            (ulonglong2*)state,roots,(uint64_t*)nullptr,a.tp);
    return qsb_carrier_launch(kernel_pinning_pipeline<true,2>,QK_S2,
        dim3(q.blocks),dim3(QSB_S2_THREADS),stream,
        a.d_midstate,a.d_suffix,a.suffix_len,a.seq_offset,a.lt_offset,a.total_preimage_len,
        a.seq_value,lt0,a.d_neg_r_inv,a.d_u2rx,a.d_u2ry,a.d_neg2u2rx,a.d_neg2u2ry,
        QSB_PK_ON?a.pk_plane:a.d_gt,a.d_hit_cnt,a.d_hit_idx,(int)q.count,a.easy_mode,a.single_hash,
        (ulonglong2*)state,roots,hit_base,a.tp);
}
static cudaError_t dispatch_primary(HalfResources &p,State &s,DispatchState &d,Route route,
                                    uint32_t batch_size,cudaEvent_t input,cudaStream_t readback,
                                    bool graph_active,bool root_serial,const PrimaryArgs &args) {
    if(!primary_available())return cudaErrorInvalidDeviceFunction;
    auto prepare=[&args](const qsb_half_schedule_research::Slice &q,void *state,
                         uint64_t *roots,cudaStream_t stream) {
        return primary<0>(args,q,state,roots,stream);
    };
    auto finish=[&args](const qsb_half_schedule_research::Slice &q,void *state,
                        uint64_t *roots,cudaStream_t stream) {
        return primary<2>(args,q,state,roots,stream);
    };
    return dispatch(p,s,d,route,batch_size,input,readback,graph_active,root_serial,prepare,finish);
}
} // namespace qsb_root_secondary_research
