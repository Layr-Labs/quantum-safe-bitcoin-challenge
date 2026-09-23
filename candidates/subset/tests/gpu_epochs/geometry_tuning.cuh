// SPDX-License-Identifier: GPL-3.0-only
// Productive comparison inspired by Saviour1001 PR1072. Two table geometries,
// forty complete batches, equal work per arm, symmetric ordering across time.
#pragma once
#include <cmath>
#ifndef QSB_GEOMETRY_FORCE
#define QSB_GEOMETRY_FORCE -1
#endif
static_assert(QSB_GEOMETRY_FORCE>=-1 && QSB_GEOMETRY_FORCE<2,
              "geometry override must be -1, 0 (regular), or 1 (GLV)");
struct QsbGeometryTuning {
    unsigned warmup=0,phase=0,batches=0,selected=0;
    bool finished=false;
    double phase_seconds=0,seconds_per_candidate[12]={};
    uint64_t phase_candidates=0;
    double relative_time=1.0;
    bool done() const {return finished || QSB_GEOMETRY_FORCE>=0;}
    unsigned route() const {
        if(QSB_GEOMETRY_FORCE>=0)return QSB_GEOMETRY_FORCE;
        if(finished)return selected;
        if(warmup<4)return warmup/2;
        const unsigned order[4]={0,1,1,0};
        return order[phase%4];
    }
    bool observe(double seconds,uint64_t candidates){
        if(done())return false;
        if(!(seconds>0) || !std::isfinite(seconds) || !candidates){
            finished=true;selected=0;return true;
        }
        if(warmup<4){++warmup;return false;}
        phase_seconds+=seconds;phase_candidates+=candidates;
        if(++batches<3)return false;
        seconds_per_candidate[phase++]=phase_seconds/(double)phase_candidates;
        phase_seconds=0;phase_candidates=0;batches=0;
        if(phase<12)return false;
        double sum=0;bool every_round=true;
        for(unsigned round=0;round<3;round++){
            const double *t=seconds_per_candidate+4*round;
            const double ratio=(t[1]+t[2])/(t[0]+t[3]);
            sum+=ratio;every_round=every_round && ratio<1.0;
        }
        relative_time=sum/3;
        selected=every_round && relative_time<=0.99;
        finished=true;return true;
    }
};
