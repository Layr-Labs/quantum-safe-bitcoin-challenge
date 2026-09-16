#pragma once
#include "table_policy.cuh"

struct QsbTableTuner {
    QsbTablePolicy policy;
    bool enabled;
    bool events_open=false;
    cudaEvent_t start_event,stop_event;
    QsbTableTuner(const QsbTableTuner&)=delete;
    QsbTableTuner& operator=(const QsbTableTuner&)=delete;
    explicit QsbTableTuner(bool available):enabled(available){
        if(enabled){
            wide_cuda_require(cudaEventCreate(&start_event),"create table timing start");
            wide_cuda_require(cudaEventCreate(&stop_event),"create table timing stop");
            events_open=true;
        }
    }
    ~QsbTableTuner(){
        if(events_open){cudaEventDestroy(start_event);cudaEventDestroy(stop_event);}
    }
    bool wide()const{return enabled&&policy.wide();}
    void begin(){
        if(enabled&&policy.pending())wide_cuda_require(cudaEventRecord(start_event),"start table timing");
    }
    void end(){
        if(enabled&&policy.pending())wide_cuda_require(cudaEventRecord(stop_event),"stop table timing");
    }
    // Called after the existing device synchronization. No extra search batch
    // or synchronization, discarded hit, or repeated candidate range.
    void observe(uint64_t count){
        if(!enabled||!policy.pending())return;
        float ms=0;
        wide_cuda_require(cudaEventElapsedTime(&ms,start_event,stop_event),"read table timing");
        policy.observe(ms,count);
        if(!policy.pending()){
            wide_cuda_require(cudaEventDestroy(start_event),"destroy table timing start");
            wide_cuda_require(cudaEventDestroy(stop_event),"destroy table timing stop");
            events_open=false;
            printf("  Fixed-base table: %s; small %.3f ns/candidate, wide %.3f ns/candidate (runtime comparison)\n",
                policy.chosen_wide?"wide16GiB":"small32MiB",
                policy.candidates[0]?policy.milliseconds[0]*1e6/policy.candidates[0]:0.,
                policy.candidates[1]?policy.milliseconds[1]*1e6/policy.candidates[1]:0.);
            fflush(stdout);
        }
    }
};
