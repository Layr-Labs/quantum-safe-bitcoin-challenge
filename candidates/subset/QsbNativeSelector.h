#pragma once
#include <cmath>
// Host-only. Both archived device images and CPU arithmetic/ownership remain unchanged.
namespace qsb_native_select {
static double now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return double(ts.tv_sec)+double(ts.tv_nsec)*1e-9;
}
static bool cpu_state(uint64_t &config) {
#if QSB_CPU_GRIND
    return qcpu::fixed_state(config);
#else
    (void)config;return false;
#endif
}
static uint64_t cpu_count() {
#if QSB_CPU_GRIND
    return qcpu::candidates();
#else
    return 0;
#endif
}
struct Sample {
    uint64_t gpu=0,cpu=0;double seconds=0;
    double rate()const{return(double(gpu)+double(cpu))/seconds;}
};
struct Selector {
    bool attempted=false;
    uint64_t config=0;
    unsigned faults=0;
    double deadline=0;
    bool ready() {
        return !attempted && g_qsb_carrier.on && g_qsb_alternate.on && cpu_state(config);
    }
    bool stable()const {
        uint64_t value=0;
        return g_qsb_carrier.on && g_qsb_alternate.on && g_qsb_native_faults==faults &&
            cpu_state(value) && value==config;
    }
    bool intact()const { return stable() && now()<deadline; }
    bool needs_fallback()const { return attempted && g_qsb_use_alternate && !stable(); }
    // run(arm,batches,sample) returns 1 success, 0 invalidated, -1 fatal CUDA/publication error.
    template<class Run> int calibrate(Run run) {
        attempted=true;faults=g_qsb_native_faults;deadline=now()+180;
        printf("GPU_NATIVE cpu_fixed workers=%u windows=%u patterns=%u token=%llu\n",
            (unsigned)(config>>32),(unsigned)((config>>24)&255),(unsigned)((config>>8)&65535),
            (unsigned long long)config);fflush(stdout);
        Sample warm[2],s[4];
        const int route[6]={0,1,0,1,1,0};
        for(int phase=0;phase<6;phase++) {
            Sample &out=phase<2?warm[phase]:s[phase-2];
            const int count=phase<2?32:128;
            if(!intact()) {retain("state_before_arm");return 0;}
            g_qsb_use_alternate=route[phase]!=0;
            const int rc=run(route[phase],count,out);
            if(rc<0){retain("fatal_arm");return 1;}
            if(!rc || !intact() || !(out.seconds>0) || !std::isfinite(out.seconds)) {
                retain("invalid_arm");return 0;
            }
            printf("GPU_NATIVE phase=%d route=%s batches=%d gpu=%llu cpu=%llu seconds=%.9f joint=%.6f\n",
                phase,route[phase]?"3280":"f745",count,(unsigned long long)out.gpu,
                (unsigned long long)out.cpu,out.seconds,out.rate());fflush(stdout);
        }
        const double p0=s[1].rate()/s[0].rate()-1,p1=s[2].rate()/s[3].rate()-1;
        const double a=(double(s[0].gpu)+s[0].cpu+s[3].gpu+s[3].cpu)/(s[0].seconds+s[3].seconds);
        const double b=(double(s[1].gpu)+s[1].cpu+s[2].gpu+s[2].cpu)/(s[1].seconds+s[2].seconds);
        const double pooled=b/a-1,drift=s[3].rate()/s[0].rate()-1;
        g_qsb_use_alternate=intact() && p0>0 && p1>0 && pooled>=.003 && std::fabs(drift)<=.003;
        printf("GPU_NATIVE decision=%s pair0=%.9f pair1=%.9f pooled=%.9f control_drift=%.9f cpu_config=%llu\n",
            g_qsb_use_alternate?"3280":"f745",p0,p1,pooled,drift,(unsigned long long)config);fflush(stdout);
        return 0;
    }
    void retain(const char *why) {
        g_qsb_use_alternate=false;
        printf("GPU_NATIVE decision=%s reason=%s\n",g_qsb_carrier.on?"f745":"jit",why);fflush(stdout);
    }
};
}
