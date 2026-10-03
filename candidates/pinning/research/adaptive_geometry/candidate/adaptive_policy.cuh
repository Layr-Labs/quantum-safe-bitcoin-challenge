#pragma once
#include <stdint.h>
#include <math.h>

// Mode 0 compact/external, 1 compact/fused, 2 wide/external, 3 wide/fused.
// Warm each available mode once, then time reverse and forward orders on
// distinct candidate ranges. This compares whole paths, avoiding a nested
// geometry/schedule choice that can miss their interaction.
struct AdaptivePolicy {
    int phase=0, arms, chosen=0;
    double milliseconds[4]={0,0,0,0};
    uint64_t candidates[4]={0,0,0,0};
    explicit AdaptivePolicy(bool wide):arms(wide?4:2){}
    bool pending()const{return phase<3*arms;}
    int mode()const{
        if(!pending())return chosen;
        if(phase<arms)return phase;
        if(phase<2*arms)return 2*arms-1-phase;
        return phase-2*arms;
    }
    void observe(double ms,uint64_t count){
        if(!pending())return;
        if(!(ms>0)||!isfinite(ms)||!count){phase=3*arms;chosen=0;return;}
        int current=mode();
        if(phase>=arms){milliseconds[current]+=ms;candidates[current]+=count;}
        ++phase;
        if(!pending()){
            double threshold=0.98*milliseconds[0]/candidates[0];
            for(int i=1;i<arms;i++){
                double rate=milliseconds[i]/candidates[i];
                if(rate<threshold){chosen=i;threshold=rate;}
            }
        }
    }
};
