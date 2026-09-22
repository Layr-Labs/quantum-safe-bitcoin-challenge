// SPDX-License-Identifier: GPL-3.0-only
// Finite selection from ordinary, distinct, fully drained search sequences.
// Independent implementation inspired by Portablelle's public 46fca857 notes.
#pragma once
#ifndef QSB_FINISH_ROUTE
#define QSB_FINISH_ROUTE -1
#endif
static_assert(QSB_FINISH_ROUTE>=-1 && QSB_FINISH_ROUTE<=1,
              "Finish route: -1 productive comparison, 0 promoted, 1 guarded pair");
static int qsb_finish_route=QSB_FINISH_ROUTE>=0?QSB_FINISH_ROUTE:0;

struct qsb_finish_tournament {
    unsigned step=0;
    int winner=0;
    bool valid=true;
    double cost[8]={};
    bool active() const { return QSB_FINISH_ROUTE<0 && step<12u; }
    int route() const {
        if(QSB_FINISH_ROUTE>=0)return QSB_FINISH_ROUTE;
        if(step<4u)return step==1u || step==2u; // productive ABBA warmup
        if(step<12u) {
            const int order[8]={0,1,1,0,1,0,0,1};
            return order[step-4u];
        }
        return winner;
    }
    void observe(double seconds,uint64_t completed,bool clock_ok) {
        if(!active())return;
        if(!clock_ok || !completed || !(seconds>0.0 && seconds<1.0e100))valid=false;
        if(step>=4u)cost[step-4u]=completed?seconds/(double)completed:0.0;
        ++step;
        if(step!=12u || !valid)return;
        unsigned wins=0;
        double log_gain=0.0;
        for(unsigned pair=0;pair<4u;++pair) {
            // AB, BA, BA, AB: resolve by route, not temporal position.
            const unsigned a=2u*pair+((pair==1u || pair==2u)?1u:0u);
            const unsigned b=a^1u;
            if(!(cost[a]>0.0 && cost[b]>0.0))return;
            if(cost[a]<0.995*cost[b])return;
            wins+=cost[a]>cost[b];
            log_gain+=log(cost[a])-log(cost[b]);
        }
        if(wins>=3u && log_gain>=4.0*log(1.01))winner=1;
    }
};
