#pragma once
// Half stream dispatcher; exact primary callbacks are bound by QsbHalfPrimaryBinding.h.
// Graph-enabled jobs retain the original pipeline; no measured performance claim.
#include "HalfResourcesPrototype.h"
#include "RootStartupPrototype.h"
#include <initializer_list>
namespace qsb_root_secondary_research {
struct DispatchState { uint64_t generation=0; bool used[8]={}; };
static cudaError_t fail_dispatch(HalfResources &p,State &s,cudaError_t e) {
    p.poisoned=true;s.launch_failed=true;return e;
}
template<class Prepare,class Finish>
static cudaError_t dispatch(HalfResources &p,State &s,DispatchState &d,Route route,
                            uint32_t batch_size,cudaEvent_t slot_input,
                            cudaStream_t slot_readback,bool graph_active,bool root_serial,
                            Prepare primary_prepare,Finish primary_finish) {
    // Choose mode BEFORE calling. An old graph cannot transparently select a second library.
    if(graph_active||!slot_input||!slot_readback||!batch_size||
       (route!=Route::HalfRegister&&route!=Route::HalfFused)||!s.constants_ready)
        return cudaErrorInvalidValue;
    cudaError_t e=begin_work(p); // set busy before every possible partial enqueue
    if(e!=cudaSuccess)return e;
    for(int i: {0,1,4,5}) {
        e=cudaStreamWaitEvent(p.streams[i],slot_input,0);
        if(e!=cudaSuccess)return fail_dispatch(p,s,e);
    }
    int last[2]={-1,-1};
    for(uint32_t off=0;off<batch_size;) {
        qsb_half_schedule_research::Slice q;
        if(!qsb_half_schedule_research::plan(batch_size,off,d.generation,q))
            return fail_dispatch(p,s,cudaErrorInvalidValue);
        HalfRing &r=p.rings[q.ring];
        cudaStream_t prepare=p.streams[q.lane],root=p.streams[2+(root_serial?0:q.lane)],finish=p.streams[4+q.lane];
        if(d.used[q.ring]) {
            e=cudaStreamWaitEvent(prepare,r.finished,0);
            if(e!=cudaSuccess)return fail_dispatch(p,s,e);
        }
        // Bind ONLY original b59 stage0 here. Callback must use q.count/q.blocks,
        // state and roots supplied here; preserve original all problem/tail arguments.
        e=primary_prepare(q,r.state,r.roots,prepare);
        if(e==cudaSuccess)e=cudaGetLastError();
        if(e==cudaSuccess)e=cudaEventRecord(r.prepared,prepare);
        if(e==cudaSuccess)e=cudaStreamWaitEvent(root,r.prepared,0);
        if(e!=cudaSuccess)return fail_dispatch(p,s,e);
        e=launch(s,route==Route::HalfRegister,r.roots,q.blocks,root);
        if(e==cudaSuccess)e=cudaGetLastError();
        if(e==cudaSuccess)e=cudaEventRecord(r.rooted,root);
        if(e==cudaSuccess)e=cudaStreamWaitEvent(finish,r.rooted,0);
        if(e!=cudaSuccess)return fail_dispatch(p,s,e);
        // Bind ONLY original b59 stage2 with the original PK plane; q.hit_base remains
        // host-batch-relative and q.locktime_delta=off/4 under the original ASICBOOST4.
        e=primary_finish(q,r.state,r.roots,finish);
        if(e==cudaSuccess)e=cudaGetLastError();
        if(e==cudaSuccess)e=cudaEventRecord(r.finished,finish);
        if(e!=cudaSuccess)return fail_dispatch(p,s,e);
        d.used[q.ring]=true;last[q.lane]=(int)q.ring;
        ++d.generation;off+=q.count;
    }
    // Snapshot the last finish event for each used lane NOW, before another host batch
    // re-records a ring event; CUDA wait binds the latest recording at enqueue time.
    for(int lane=0;lane<2;lane++)if(last[lane]>=0) {
        e=cudaStreamWaitEvent(slot_readback,p.rings[last[lane]].finished,0);
        if(e!=cudaSuccess)return fail_dispatch(p,s,e);
    }
    return cudaSuccess; // asynchronous: busy stays true until explicit successful drain
}
} // namespace qsb_root_secondary_research
