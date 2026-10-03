#pragma once
// Half-ring resource owner. Borrow the six existing green-context
// prepare/root/finish streams, preserving their priorities and policy windows.
#include "RootSecondaryPrototype.h"
#include "HalfSchedulePrototype.h"
namespace qsb_root_secondary_research {
struct HalfRing {
    void *state=nullptr;
    uint64_t *roots=nullptr;
    cudaEvent_t prepared=nullptr, rooted=nullptr, finished=nullptr;
};
struct HalfResources {
    HalfRing rings[8];
    cudaStream_t streams[6]={}; // borrowed, never destroyed here
    bool ready=false, busy=false, poisoned=false;
};
static cudaError_t drain(HalfResources &p) {
    cudaError_t first=cudaSuccess;
    if(p.busy) {
        for(int i=0;i<6;i++) {
            cudaError_t e=cudaStreamSynchronize(p.streams[i]);
            if(first==cudaSuccess)first=e;
        }
        if(first!=cudaSuccess){p.poisoned=true;return first;}
        p.busy=false;
    }
    return p.poisoned?cudaErrorUnknown:cudaSuccess;
}
static cudaError_t release(HalfResources &p) {
    p.ready=false;
    cudaError_t first=drain(p);
    // A failed drain cannot authorize freeing storage still reachable by queued work.
    if(first!=cudaSuccess)return first;
    for(int r=7;r>=0;r--) {
        HalfRing &x=p.rings[r];
        cudaEvent_t *events[]={&x.finished,&x.rooted,&x.prepared};
        for(auto h:events)if(*h) {
            cudaError_t e=cudaEventDestroy(*h);
            if(e==cudaSuccess)*h=nullptr;
            else {p.poisoned=true;if(first==cudaSuccess)first=e;}
        }
        if(x.roots) {
            cudaError_t e=cudaFree(x.roots);
            if(e==cudaSuccess)x.roots=nullptr;
            else {p.poisoned=true;if(first==cudaSuccess)first=e;}
        }
        if(x.state) {
            cudaError_t e=cudaFree(x.state);
            if(e==cudaSuccess)x.state=nullptr;
            else {p.poisoned=true;if(first==cudaSuccess)first=e;}
        }
    }
    return first; // cleanup failure is fatal to the owner; handles remain recorded.
}
static cudaError_t allocate(HalfResources &p,const cudaStream_t streams[6]) {
    if(p.ready||p.busy||p.poisoned||!streams)return cudaErrorInvalidValue;
    for(const auto &r:p.rings)
        if(r.state||r.roots||r.prepared||r.rooted||r.finished)return cudaErrorInvalidValue;
    for(int i=0;i<6;i++) {
        if(!streams[i])return cudaErrorInvalidValue;
        p.streams[i]=streams[i];
    }
    cudaError_t e=cudaSuccess;
    for(int r=0;r<8&&e==cudaSuccess;r++) {
        HalfRing &x=p.rings[r];
        e=cudaMalloc(&x.state,65536u*64u);
        if(e==cudaSuccess)e=cudaMalloc((void**)&x.roots,1024u*32u);
        if(e==cudaSuccess)e=cudaEventCreateWithFlags(&x.prepared,cudaEventDisableTiming);
        if(e==cudaSuccess)e=cudaEventCreateWithFlags(&x.rooted,cudaEventDisableTiming);
        if(e==cudaSuccess)e=cudaEventCreateWithFlags(&x.finished,cudaEventDisableTiming);
    }
    if(e!=cudaSuccess) {
        cudaError_t cleanup=release(p);
        // Allocation/runtime error isn't a performance-driven geometry fallback.
        p.poisoned=true;
        return cleanup==cudaSuccess?e:cleanup;
    }
    p.ready=true;
    return cudaSuccess;
}
static cudaError_t begin_work(HalfResources &p) {
    if(!p.ready||p.poisoned)return cudaErrorInvalidValue;
    p.busy=true; // set BEFORE the first enqueue, including failed/partial enqueue paths
    return cudaSuccess;
}
static cudaError_t replace_problem(HalfResources &p,State &s,bool all_primary_work_drained,
                                  const uint64_t y[4],const uint64_t iy[4],const uint64_t iu[4]) {
    // The caller must also drain slot/monolithic work before passing this proof of phase.
    if(!all_primary_work_drained||p.poisoned)return cudaErrorInvalidValue;
    cudaError_t e=drain(p);
    if(e==cudaSuccess)e=set_problem(s,y,iy,iu);
    if(e!=cudaSuccess){p.poisoned=true;s.launch_failed=true;}
    return e;
}
static cudaError_t close(HalfResources &p,State &s,bool all_primary_work_drained) {
    if(!all_primary_work_drained)return cudaErrorInvalidValue;
    cudaError_t e=release(p);
    if(e!=cudaSuccess){s.launch_failed=true;return e;}
    // No library unload while any owned root data could still be live.
    if(s.library) {
        e=cudaLibraryUnload(s.library);
        if(e!=cudaSuccess){s.launch_failed=true;return e;}
    }
    s=State{};
    return cudaSuccess;
}
} // namespace qsb_root_secondary_research
