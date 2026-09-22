// SPDX-License-Identifier: GPL-3.0-only
// Adapted from our pinning adaptive candidate; not a proven GPU speedup.
// Host policy only. Timings cover real, distinct, fully drained search batches.
#pragma once
#include "subset_sha_flags.h"
#include <cmath>
#include <cstdint>

struct qsb_subset_sha_trial {
    static constexpr unsigned warm_batches=2, measured_batches=4, phases=10;
    // Warm both paths, then ABBA BAAB. Every timing is candidate-normalized.
    static constexpr bool mode(unsigned p) {
        return p==1 || p==3 || p==4 || p==6 || p==9;
    }
    unsigned phase=0, batches=0;
    uint64_t candidates=0;
    double rates[8]={}, ratios[4]={}, gain=1.0;
    bool selected=false;

    bool done() const { return phase==phases; }
    bool split() const { return done()?selected:mode(phase); }
    bool add_batch(uint64_t n) {
        if(done())return false;
        candidates+=n;
        return ++batches==(phase<2?warm_batches:measured_batches);
    }
    void finish_phase(double seconds) {
        if(done())return;
        if(phase>=2)rates[phase-2]=seconds>0?double(candidates)/seconds:0;
        phase++;batches=0;candidates=0;
        if(!done())return;
        double log_sum=0;
        unsigned clear_pairs=0;
        bool consistent=true;
        for(unsigned pair=0;pair<4;pair++) {
            unsigned a=2*pair,b=a+1;
            unsigned si=mode(a+2)?a:b,fi=mode(a+2)?b:a;
            double ratio=rates[fi]>0?rates[si]/rates[fi]:0;
            ratios[pair]=ratio;
            if(!std::isfinite(ratio)||ratio<=0) {consistent=false;continue;}
            log_sum+=std::log(ratio);
            clear_pairs+=ratio>1.02;
            consistent&=ratio>=0.995;
        }
        gain=std::exp(log_sum/4);
        // Require margin in both the aggregate and three independent pairs.
        selected=consistent && clear_pairs>=3 && gain>=1.025;
    }
};

// Compare each producer with the promoted fused route, then keep the strongest
// consistent gain. The second trial uses fresh real search batches too.
struct qsb_subset_sha_tuning {
    static constexpr unsigned trials=QSB_SUBSET_SHA_SINGLE?2:1;
    qsb_subset_sha_trial trial[trials];
    unsigned current=0, selected=0;
    double best_gain=1.0;
    bool done() const {return current==trials;}
    unsigned route() const {
        return done()?selected:(trial[current].split()?current+1:0);
    }
    bool add_batch(uint64_t n) {
        return !done() && trial[current].add_batch(n);
    }
    void finish_phase(double seconds) {
        if(done())return;
        auto &t=trial[current];
        t.finish_phase(seconds);
        if(!t.done())return;
        if(t.selected && t.gain>best_gain) {
            selected=current+1;best_gain=t.gain;
        }
        ++current;
    }
};
