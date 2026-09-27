#pragma once
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <new>
namespace qcpu_coupled {
// The launched pool/epoch stride never changes. Only permission to start the
// next complete batch changes. No GPU work or candidate arithmetic is changed.
struct Progress {
    std::atomic<uint64_t> generation{~0ull}, epoch{0}, batches{0};
    std::atomic<int> wi{0};
};
struct Gate {
    int total=0;
    alignas(64) std::atomic<uint64_t> config{0}; // generation:32 | active:32
    std::atomic<int> ready{0}, parked{0};
    std::mutex mutex;
    std::condition_variable cv;
    std::unique_ptr<Progress[]> progress;
    bool init(int n){
        total=n;config.store((uint32_t)n);
        progress.reset(new(std::nothrow) Progress[n]);return bool(progress);
    }
    uint64_t request(int n){
        uint64_t value;
        {std::lock_guard<std::mutex> lock(mutex);
         value=((config.load()>>32)+1)<<32 | (uint32_t)n;config.store(value);}
        cv.notify_all();return value>>32;
    }
    void boundary(int tid,uint64_t epoch,int wi,uint64_t batches,uint64_t &seen){
        for(;;){
            const uint64_t value=config.load(std::memory_order_acquire);
            if(value!=seen){
                progress[tid].epoch.store(epoch,std::memory_order_relaxed);
                progress[tid].wi.store(wi,std::memory_order_relaxed);
                progress[tid].batches.store(batches,std::memory_order_relaxed);
                progress[tid].generation.store(value>>32,std::memory_order_release);
                if(seen==~0ull)ready.fetch_add(1);
                seen=value;
            }
            if(tid<(int)(uint32_t)value)return;
            std::unique_lock<std::mutex> lock(mutex);
            if(config.load()!=value)continue;
            parked.fetch_add(1);
            cv.wait(lock,[&]{return config.load()!=value;});
            parked.fetch_sub(1);
        }
    }
    bool acknowledged(uint64_t gen,int active)const{
        if(ready.load()!=total || parked.load()!=total-active)return false;
        for(int t=0;t<total;t++)if(progress[t].generation.load(std::memory_order_acquire)!=gen)return false;
        return true;
    }
};
struct Sample {uint64_t gpu=0,cpu=0,batches=0;double seconds=0;double rate()const{return(double(gpu)+double(cpu))/seconds;}};
struct Selector {
    Gate gate;
    std::atomic<int> exited{0};
    bool enabled=false,trace=true;
    int phase=0,leg=0,active=0; // warm, settle, measure, done
    uint64_t callbacks=0,mark_batch=0,start_gpu=0,start_cpu=0,last_gpu=0,last_cpu=0,generation=0;
    double warm_start=-1,mark_time=-1,last_time=-1,change_time=0;
    Sample samples[4];
    void init(int n,bool verbose=true){trace=verbose;enabled=n>=4&&gate.init(n);active=n;}
    void boundary(int t,uint64_t e,int wi,uint64_t b,uint64_t &seen){if(enabled)gate.boundary(t,e,wi,b,seen);}
    void abort(const char *reason){
        if(phase==3)return;
        if(enabled)gate.request(gate.total);active=gate.total;phase=3;
        if(trace){printf("CPU_COUPLED decision=retain reason=%s active=%d\n",reason,active);fflush(stdout);}
    }
    void begin(double now){
        active=(leg==1||leg==2)?gate.total-2:gate.total;
        generation=gate.request(active);mark_time=-1;change_time=now;phase=1;
        if(trace){printf("CPU_COUPLED transition leg=%d active=%d stride=%d generation=%llu\n",leg,active,gate.total,(unsigned long long)generation);fflush(stdout);}
    }
    void finish(){
        const double p0=samples[1].rate()/samples[0].rate()-1;
        const double p1=samples[2].rate()/samples[3].rate()-1;
        const double ar=(double(samples[0].gpu)+samples[0].cpu+samples[3].gpu+samples[3].cpu)/(samples[0].seconds+samples[3].seconds);
        const double br=(double(samples[1].gpu)+samples[1].cpu+samples[2].gpu+samples[2].cpu)/(samples[1].seconds+samples[2].seconds);
        const double pooled=br/ar-1,drift=samples[3].rate()/samples[0].rate()-1;
        const bool keep=p0>=.0025&&p1>=.0025&&pooled>=.005&&std::fabs(drift)<=.005;
        active=keep?gate.total-2:gate.total;gate.request(active);phase=3;
        if(trace){printf("CPU_COUPLED decision=%s active=%d stride=%d pair0=%.9f pair1=%.9f pooled=%.9f control_drift=%.9f\n",keep?"reduced":"retain",active,gate.total,p0,p1,pooled,drift);fflush(stdout);}
    }
    // Called by the sole GPU host thread after collect->launch->publish.
    // gpu counts completed, published batches. cpu counts finished CPU batches.
    void tick(uint64_t gpu,uint64_t cpu,double now,bool full){
        if(!enabled||phase==3)return;
        if(exited.load()){abort("worker_exit");return;}
        if(!full){abort("partial_gpu_batch");return;}
        if(gpu<last_gpu||cpu<last_cpu||(last_time>=0&&now<=last_time)){abort("counter_or_clock");return;}
        last_gpu=gpu;last_cpu=cpu;last_time=now;++callbacks;
        if(phase==0){
            if(gate.ready.load()!=gate.total)return;
            if(warm_start<0){warm_start=now;mark_batch=callbacks;}
            if(now-warm_start>=30&&callbacks-mark_batch>=32)begin(now);
            return;
        }
        if(phase==1){
            if(!gate.acknowledged(generation,active)){
                if(now-change_time>10)abort("worker_ack_timeout");return;
            }
            if(mark_time<0){
                mark_time=now;mark_batch=callbacks;
                if(trace)for(int t=0;t<gate.total;t++)printf("CPU_COUPLED_PROGRESS generation=%llu tid=%d epoch=%llu wi=%d batches=%llu\n",(unsigned long long)generation,t,(unsigned long long)gate.progress[t].epoch.load(),gate.progress[t].wi.load(),(unsigned long long)gate.progress[t].batches.load());
            }
            if(now-mark_time<2||callbacks-mark_batch<8)return;
            mark_time=now;mark_batch=callbacks;start_gpu=gpu;start_cpu=cpu;phase=2;return;
        }
        if(callbacks-mark_batch<128)return;
        Sample &s=samples[leg];s.gpu=gpu-start_gpu;s.cpu=cpu-start_cpu;s.seconds=now-mark_time;s.batches=callbacks-mark_batch;
        if(trace){printf("CPU_COUPLED leg=%d active=%d stride=%d gpu=%llu cpu=%llu seconds=%.9f batches=%llu combined=%.6f gpu_rate=%.6f cpu_rate=%.6f\n",leg,active,gate.total,(unsigned long long)s.gpu,(unsigned long long)s.cpu,s.seconds,(unsigned long long)s.batches,s.rate(),double(s.gpu)/s.seconds,double(s.cpu)/s.seconds);fflush(stdout);}
        if(++leg==4)finish();else begin(now);
    }
};
}
