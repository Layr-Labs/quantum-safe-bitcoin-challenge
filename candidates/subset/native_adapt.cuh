#pragma once
#include <stdint.h>
// Host-only controller. Warm work and measurement work are ordinary disjoint batches.
namespace qsb_adapt {
struct Slot { bool busy=false, native=true; uint64_t base=0, count=0; };
struct State {
    uint64_t capacity, next=0, published=0;
    unsigned phase=0, launched=0;
    bool open=false, partial=false, latched_native=true;
    Slot slots[2];
    double gpu_ms[4]={}, host_ms[4]={};
    explicit State(uint64_t e):capacity(e){}
    bool sampling() const { return phase<6; }
    bool backend() const { return phase==1 || phase==3 || phase==4 ? false : (phase<6 ? true : latched_native); }
    unsigned target() const { return phase<2 ? 2u : 8u; }
    bool empty() const { return !slots[0].busy && !slots[1].busy; }
    bool begin(uint64_t base) {
        if(!capacity || !sampling() || open || !empty() || base!=next || published!=next) return false;
        open=true;launched=0;partial=false;return true;
    }
    bool launch(unsigned slot,uint64_t base,uint64_t count,bool native) {
        if(slot>1 || slots[slot].busy || base!=next || !count || count>capacity || UINT64_MAX-base<count || native!=backend()) return false;
        if(sampling() && (!open || launched>=target())) return false;
        slots[slot].busy=true;slots[slot].native=native;slots[slot].base=base;slots[slot].count=count;
        next+=count;
        if(sampling()){launched++;partial|=count!=capacity;}
        return true;
    }
    bool publish(unsigned slot,uint64_t base,uint64_t count,bool native) {
        if(slot>1 || !slots[slot].busy || slots[slot].base!=published || base!=slots[slot].base || count!=slots[slot].count || native!=slots[slot].native) return false;
        published+=count;slots[slot].busy=false;return true;
    }
    bool window_full() const { return sampling() && open && !partial && launched==target(); }
    bool finish(double gpu,double host) {
        if(!window_full() || !empty() || published!=next) return false;
        if(phase>=2){gpu_ms[phase-2]=gpu;host_ms[phase-2]=host;}
        open=false;phase++;
        if(phase==6){
            bool valid=true;
            for(unsigned i=0;i<4;i++)valid &= gpu_ms[i]>0 && gpu_ms[i]<1e9 && host_ms[i]>0 && host_ms[i]<1e9;
            // N/S/S/N: both GPU pairs need >=2% source advantage; host sign must agree.
            const bool source=valid && gpu_ms[0]>=1.02*gpu_ms[1] && gpu_ms[3]>=1.02*gpu_ms[2]
                && host_ms[0]>host_ms[1] && host_ms[3]>host_ms[2];
            latched_native=!source;
        }
        return true;
    }
};
} // namespace qsb_adapt
