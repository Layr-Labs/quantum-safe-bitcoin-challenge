// SPDX-License-Identifier: GPL-3.0-only
// Asynchronous inversion mailboxes for the experimental affine search.
#pragma once
#include <stdint.h>
#include <assert.h>
#ifndef QSB_INVERSE_SERVICE_ASSERTS
#ifdef QSB_INVERSE_SERVICE_HOST_TEST
#define QSB_INVERSE_SERVICE_ASSERTS 1
#else
#define QSB_INVERSE_SERVICE_ASSERTS 0
#endif
#endif
#if QSB_INVERSE_SERVICE_ASSERTS
#define QSB_INV_ASSERT(x) assert(x)
#else
#define QSB_INV_ASSERT(x) ((void)0)
#endif
#ifdef QSB_INVERSE_SERVICE_HOST_TEST
#include <atomic>
#define QSB_INV_DEVICE inline
struct alignas(16) uint4 { uint32_t x,y,z,w; };
inline uint4 make_uint4(uint32_t x,uint32_t y,uint32_t z,uint32_t w) { return {x,y,z,w}; }
bool qsb_inverse_test_any(bool);
bool qsb_inverse_test_all(bool);
#else
#include <cuda_runtime.h>
#include <cuda/atomic>
#define QSB_INV_DEVICE __device__ __forceinline__
#endif

namespace qsb_inverse_service {
constexpr uint32_t STOP = 0xffffffffu;
struct alignas(64) Slot {
    uint4 root[2];
    uint4 inverse[2];
    uint32_t request;
    uint32_t response;
    uint32_t reserved[14];
};
static_assert(sizeof(Slot)==128, "mailbox must occupy two aligned cache lines");

QSB_INV_DEVICE uint32_t acquire(uint32_t &word) {
#ifdef QSB_INVERSE_SERVICE_HOST_TEST
    return std::atomic_ref<uint32_t>(word).load(std::memory_order_acquire);
#else
    return cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(word).load(cuda::memory_order_acquire);
#endif
}
QSB_INV_DEVICE void release(uint32_t &word,uint32_t value) {
#ifdef QSB_INVERSE_SERVICE_HOST_TEST
    std::atomic_ref<uint32_t>(word).store(value,std::memory_order_release);
#else
    cuda::atomic_ref<uint32_t,cuda::thread_scope_device>(word).store(value,cuda::memory_order_release);
#endif
}
QSB_INV_DEVICE void load_field(uint64_t out[4],const uint4 in[2]) {
    const uint4 a=in[0],b=in[1];
    out[0]=uint64_t(a.x)|(uint64_t(a.y)<<32);
    out[1]=uint64_t(a.z)|(uint64_t(a.w)<<32);
    out[2]=uint64_t(b.x)|(uint64_t(b.y)<<32);
    out[3]=uint64_t(b.z)|(uint64_t(b.w)<<32);
}
QSB_INV_DEVICE void store_field(uint4 out[2],const uint64_t in[4]) {
    out[0]=make_uint4(uint32_t(in[0]),uint32_t(in[0]>>32),uint32_t(in[1]),uint32_t(in[1]>>32));
    out[1]=make_uint4(uint32_t(in[2]),uint32_t(in[2]>>32),uint32_t(in[3]),uint32_t(in[3]>>32));
}
// Single producer: worker CTA lane zero. Epochs start at one and never wrap.
// The preceding inverse must have been consumed before this call.
QSB_INV_DEVICE void worker_publish(Slot &slot,uint32_t epoch,const uint64_t root[4]) {
    QSB_INV_ASSERT(epoch!=0 && epoch!=STOP);
    QSB_INV_ASSERT(acquire(slot.request)==epoch-1 && acquire(slot.response)==epoch-1);
    store_field(slot.root,root);
    release(slot.request,epoch);
}
QSB_INV_DEVICE void worker_wait(Slot &slot,uint32_t epoch,uint64_t out[4]) {
    while(acquire(slot.response)!=epoch) { }
    load_field(out,slot.inverse);
}
QSB_INV_DEVICE void worker_stop(Slot &slot) {
    QSB_INV_ASSERT(acquire(slot.request)==acquire(slot.response));
    release(slot.request,STOP);
}
// Every physical lane of each service warp must enter this function. A lane
// permanently owns service_idx, or is inactive if service_idx>=worker_count.
// No thread exits early within a warp; empty lanes pass identity to invert.
// invert(out,in) must contain no CTA-wide collective and must terminate for 1.
template<class Invert>
QSB_INV_DEVICE void service(Slot *slots,uint32_t worker_count,uint32_t service_idx,Invert invert) {
    const bool valid=service_idx<worker_count;
    uint32_t completed=0;
    for(;;) {
        const uint32_t request=valid?acquire(slots[service_idx].request):STOP;
        const bool stopped=request==STOP;
        const bool pending=valid && !stopped && request!=completed;
#ifdef QSB_INVERSE_SERVICE_HOST_TEST
        const bool all_stopped=qsb_inverse_test_all(stopped);
        const bool any_pending=qsb_inverse_test_any(pending);
#else
        const bool all_stopped=__all_sync(0xffffffffu,stopped);
        const bool any_pending=__any_sync(0xffffffffu,pending);
#endif
        if(all_stopped) return;
        if(any_pending) {
            uint64_t input[4]={1,0,0,0},output[4];
            if(pending) {
                QSB_INV_ASSERT(request==completed+1);
                load_field(input,slots[service_idx].root);
            }
            invert(output,input);
            if(pending) {
                store_field(slots[service_idx].inverse,output);
                release(slots[service_idx].response,request);
                completed=request;
            }
        }
    }
}
} // namespace qsb_inverse_service
#undef QSB_INV_DEVICE
#undef QSB_INV_ASSERT
