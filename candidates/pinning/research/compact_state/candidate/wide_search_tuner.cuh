// Compare two equivalent search schedules on distinct real candidate ranges.
// Every warmup/timing hit is retained and counted through the normal interface.
#pragma once
#include <math.h>

struct WideSchedulePolicy {
    int phase=0;
    double milliseconds[2]={0,0};
    uint64_t candidates[2]={0,0};
    bool chosen_fused=false;
    bool pending()const{return phase<6;}
    bool fused()const{
        // One warmup per mode, then balanced F,S,S,F measurement order.
        return pending()?(phase==1||phase==2||phase==5):chosen_fused;
    }
    void observe(double ms,uint64_t count){
        if(!pending())return;
        if(!(ms>0)||!isfinite(ms)||!count){
            // Retain the established split schedule if timing is unusable.
            phase=6;chosen_fused=false;return;
        }
        if(phase>=2){int mode=fused()?1:0;milliseconds[mode]+=ms;candidates[mode]+=count;}
        ++phase;
        if(!pending()){
            // Require a2% improvement to avoid choosing on timing noise.
            chosen_fused=candidates[0]&&candidates[1]&&
                milliseconds[1]/candidates[1]<0.98*milliseconds[0]/candidates[0];
        }
    }
};

struct WideSearchTuner {
    WideSchedulePolicy policy;
    bool enabled;
    cudaEvent_t begin_event,end_event;
    explicit WideSearchTuner(bool fast):enabled(fast){
        if(enabled){
            wide_cuda_require(cudaEventCreate(&begin_event),"create search timing start");
            wide_cuda_require(cudaEventCreate(&end_event),"create search timing end");
        }
    }
    uint32_t group_limit(uint32_t normal)const{
        // Bound each trial to4 batches instead of an entire locktime range.
        return enabled&&policy.pending()?4u:normal;
    }
    void begin(){
        wide_use_fused=enabled&&policy.fused();
        if(enabled&&policy.pending())wide_cuda_require(cudaEventRecord(begin_event),"start search timing");
    }
    void end(){
        if(enabled&&policy.pending())wide_cuda_require(cudaEventRecord(end_event),"end search timing");
    }
    // Caller first completes the ordinary blocking hit-count copy, so the
    // event is finished. There is no extra synchronization or discarded batch.
    void observe(uint64_t count){
        if(!enabled||!policy.pending())return;
        float ms=0;
        wide_cuda_require(cudaEventElapsedTime(&ms,begin_event,end_event),"read search timing");
        policy.observe(ms,count);
        if(!policy.pending()){
            wide_cuda_require(cudaEventDestroy(begin_event),"destroy search timing start");
            wide_cuda_require(cudaEventDestroy(end_event),"destroy search timing end");
            printf("  Wide search schedule: %s; split %.3f ns/candidate, fused %.3f ns/candidate (runtime comparison)\n",
                policy.chosen_fused?"CTA-local inverse":"external inverse",
                policy.candidates[0]?policy.milliseconds[0]*1e6/policy.candidates[0]:0.0,
                policy.candidates[1]?policy.milliseconds[1]*1e6/policy.candidates[1]:0.0);
            fflush(stdout);
        }
    }
};
