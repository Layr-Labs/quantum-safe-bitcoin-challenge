#pragma once
#include "adaptive_policy.cuh"
struct AdaptiveSearchTuner {
    AdaptivePolicy policy;
    bool enabled,events_open=false;
    cudaEvent_t start_event{},end_event{};
    AdaptiveSearchTuner(bool fast,bool wide):policy(wide),enabled(fast){
        if(enabled){
            wide_cuda_require(cudaEventCreate(&start_event),"create timing start");
            wide_cuda_require(cudaEventCreate(&end_event),"create timing end");
            events_open=true;
        }
    }
    AdaptiveSearchTuner(const AdaptiveSearchTuner&)=delete;
    AdaptiveSearchTuner& operator=(const AdaptiveSearchTuner&)=delete;
    void close(){
        if(events_open){
            wide_cuda_require(cudaEventDestroy(start_event),"destroy timing start");
            wide_cuda_require(cudaEventDestroy(end_event),"destroy timing end");
            events_open=false;
        }
    }
    ~AdaptiveSearchTuner(){close();}
    uint32_t group_limit(uint32_t normal)const{return enabled&&policy.pending()?4u:normal;}
    void begin(){
        const int mode=enabled?policy.mode():0;
        adaptive_use_wide=mode>=2;wide_use_fused=(mode&1)!=0;
        if(events_open)wide_cuda_require(cudaEventRecord(start_event),"start path timing");
    }
    void end(){if(events_open)wide_cuda_require(cudaEventRecord(end_event),"end path timing");}
    void observe(uint64_t count){
        if(!events_open)return;
        float ms=0;
        wide_cuda_require(cudaEventElapsedTime(&ms,start_event,end_event),"read path timing");
        policy.observe(ms,count);
        if(!policy.pending()){
            close();
            printf("  Adaptive mode: %d (0 compact/external, 1 compact/fused, 2 wide/external, 3 wide/fused)\n",policy.chosen);
            for(int i=0;i<policy.arms;i++)printf("    mode %d: %.3f ns/candidate\n",i,
                policy.candidates[i]?policy.milliseconds[i]*1e6/policy.candidates[i]:0.0);
            fflush(stdout);
        }
    }
};
