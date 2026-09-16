// Compare equivalent algorithms on distinct real candidate ranges. All hits
// remain in the normal result stream. No benchmark identity or score access.
#pragma once
#include <math.h>
#include <stdint.h>

struct QsbTablePolicy {
    int phase=0;
    double milliseconds[2]={0,0};
    uint64_t candidates[2]={0,0};
    bool chosen_wide=false;
    bool pending()const{return phase<6;}
    bool wide()const{
        // Warm small, warm wide, then balanced W,S,S,W measurement order.
        return pending()?(phase==1||phase==2||phase==5):chosen_wide;
    }
    void observe(double ms,uint64_t count){
        if(!pending())return;
        if(!(ms>0)||!isfinite(ms)||!count){phase=6;chosen_wide=false;return;}
        if(phase>=2){int mode=wide()?1:0;milliseconds[mode]+=ms;candidates[mode]+=count;}
        ++phase;
        if(!pending()){
            chosen_wide=candidates[0]&&candidates[1]&&
                milliseconds[1]/candidates[1]<0.98*milliseconds[0]/candidates[0];
        }
    }
};
