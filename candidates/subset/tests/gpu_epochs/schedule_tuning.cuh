// SPDX-License-Identifier: GPL-3.0-only
// Source lead: Saviour1001 PR1072's productive schedule selection. This policy
// compares our fused, full-stage, serial-tile and overlapping-tile routes.
#pragma once
#include <cmath>
#ifndef QSB_SCHEDULE_TUNE
#define QSB_SCHEDULE_TUNE (QSB_SHA_STAGE && QSB_SHA_TILE_PIPE)
#endif
#ifndef QSB_SCHEDULE_FORCE_ROUTE
#define QSB_SCHEDULE_FORCE_ROUTE -1
#endif
#if QSB_SCHEDULE_TUNE && !(QSB_SHA_STAGE && QSB_SHA_TILE_PIPE)
#error "schedule selection requires staged and tiled routes"
#endif
static_assert(QSB_SCHEDULE_FORCE_ROUTE>=-1 && QSB_SCHEDULE_FORCE_ROUTE<4,
              "route override must be -1 or a route in [0,3]");
#if QSB_SCHEDULE_TUNE
#define QSB_DIGEST_STAGED kernel_digest<true>
#define QSB_DIGEST_FUSED kernel_digest<false>
struct QsbScheduleTuning {
    unsigned warmup=0,phase=0,batches=0,selected=0;
    bool finished=false;
    double phase_seconds=0,seconds_per_candidate[24]={};
    uint64_t phase_candidates=0;
    double relative_time[4]={1,1,1,1};
    bool done() const { return finished || QSB_SCHEDULE_FORCE_ROUTE>=0; }
    unsigned route() const {
        if(QSB_SCHEDULE_FORCE_ROUTE>=0)return QSB_SCHEDULE_FORCE_ROUTE;
        if(finished)return selected;
        if(warmup<8)return warmup/2;
        const unsigned order[8]={0,1,2,3,3,2,1,0};
        return order[phase%8];
    }
    bool observe(double seconds,uint64_t candidates) {
        if(done())return false;
        if(!(seconds>0) || !std::isfinite(seconds) || !candidates) {
            finished=true;selected=0;return true;
        }
        if(warmup<8){++warmup;return false;}
        phase_seconds+=seconds;phase_candidates+=candidates;
        if(++batches<3)return false;
        seconds_per_candidate[phase++]=phase_seconds/(double)phase_candidates;
        phase_seconds=0;phase_candidates=0;batches=0;
        if(phase<24)return false;
        double best=1.0;
        for(unsigned r=1;r<4;r++) {
            double sum=0;bool each_round=true;
            for(unsigned round=0;round<3;round++) {
                const double* t=seconds_per_candidate+round*8;
                const double base=(t[0]+t[7])*0.5;
                const double ratio=((t[r]+t[7-r])*0.5)/base;
                sum+=ratio;each_round=each_round && ratio<1.0;
            }
            relative_time[r]=sum/3.0;
            if(each_round && relative_time[r]<=0.99 && relative_time[r]<best) {
                best=relative_time[r];selected=r;
            }
        }
        finished=true;return true;
    }
};
#else
#define QSB_DIGEST_STAGED kernel_digest
#define QSB_DIGEST_FUSED kernel_digest
#endif
