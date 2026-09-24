// SPDX-License-Identifier: GPL-3.0-only
// On-device preparation timing; all calibration work is uncounted/unpublished.
#pragma once
#include <cmath>
#include <cstdio>

struct QsbPrefetchDecision {
    int enabled;
    double off_ms, on_ms;
};
inline QsbPrefetchDecision qsb_prefetch_decide(const float ms[8]) {
    const double off0=double(ms[0])+ms[3], on0=double(ms[1])+ms[2];
    const double off1=double(ms[5])+ms[6], on1=double(ms[4])+ms[7];
    QsbPrefetchDecision d={0,off0+off1,on0+on1};
    // Require two consistent mirrored blocks plus a 1% aggregate saving.
    // This is a local preparation-time selection threshold, not a score claim.
    d.enabled=(on0<off0*0.995 && on1<off1*0.995 && d.on_ms<d.off_ms*0.99);
    return d;
}

// launch() enqueues only stage 0 and returns its launch status. It cannot
// increment the searched counter, finish a key, or publish a hit. All slot
// inputs are ready before the first sample. Buffers are overwritten normally
// after calibration; no sampled batch is marked busy/completed by this helper.
template<class Launch>
static cudaError_t qsb_tune_cold_prefetch(cudaStream_t stream,Launch launch) {
    cudaEvent_t begin=nullptr,end=nullptr;
    cudaError_t err=cudaStreamSynchronize(stream);
    if(err!=cudaSuccess)return err;
    err=cudaEventCreate(&begin);if(err!=cudaSuccess)return err;
    err=cudaEventCreate(&end);
    if(err!=cudaSuccess){cudaEventDestroy(begin);return err;}
    auto close=[&](cudaError_t error)->cudaError_t {
        cudaError_t a=cudaEventDestroy(begin),b=cudaEventDestroy(end);
        if(error!=cudaSuccess)return error;
        return a!=cudaSuccess?a:b;
    };
    constexpr int repetitions=32;
    const int order[8]={0,1,1,0,1,0,0,1};
    float ms[8]={};
    for(int arm=0;arm<8;arm++) {
        err=cudaMemcpyToSymbol(pin_cold_prefetch_enabled,&order[arm],sizeof(int));
        if(err!=cudaSuccess)return close(err);
        // One untimed warmup on each switch, then 32 identical full batches.
        err=launch();if(err!=cudaSuccess)return close(err);
        err=cudaEventRecord(begin,stream);if(err!=cudaSuccess)return close(err);
        for(int i=0;i<repetitions;i++) {
            err=launch();if(err!=cudaSuccess)return close(err);
        }
        err=cudaEventRecord(end,stream);if(err!=cudaSuccess)return close(err);
        err=cudaEventSynchronize(end);if(err!=cudaSuccess)return close(err);
        err=cudaEventElapsedTime(&ms[arm],begin,end);if(err!=cudaSuccess)return close(err);
        if(!(ms[arm]>0.0f) || !std::isfinite(ms[arm]))return close(cudaErrorInvalidValue);
    }
    const QsbPrefetchDecision d=qsb_prefetch_decide(ms);
    err=cudaMemcpyToSymbol(pin_cold_prefetch_enabled,&d.enabled,sizeof(int));
    if(err!=cudaSuccess)return close(err);
    std::fprintf(stderr,"Cold prefetch calibration: off %.3f ms, on %.3f ms, selected %s\n",
                 d.off_ms,d.on_ms,d.enabled?"on":"off");
    return close(cudaSuccess);
}
