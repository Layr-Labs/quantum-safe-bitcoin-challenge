// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cstdint>
namespace qsb_gate {
// Four equal-duration windows in ABBA order reduce bias from a linear
// clock/temperature drift. A is the original FMA form, B the plain form.
// Samples are completed GPU candidates and elapsed wall time, never hits.
struct Tune {
    int stage = -1;
    double begin = 0, seconds[2] = {0,0};
    uint64_t start = 0, candidates[2] = {0,0};
    unsigned selected = 1;
    bool done = false;
    static unsigned mode(int i) { return (i == 1 || i == 2) ? 0u : 1u; }
    bool due(double now) const { return !done && (stage < 0 ? now >= 120.0 : now - begin >= 30.0); }
    // Call only after draining both streams. All work before count is
    // published once, and all later work uses the returned constant.
    unsigned boundary(double now, uint64_t count) {
        if (done) return selected;
        if (stage >= 0) {
            unsigned m = mode(stage);
            if (count < start || now <= begin) { done = true; return selected = 1; }
            seconds[m] += now - begin;
            candidates[m] += count - start;
        }
        ++stage; begin = now; start = count;
        if (stage == 4) {
            done = true;
            // Keep A unless B has a measured margin of at least 0.5%.
            const double a = seconds[1] > 0 ? candidates[1] / seconds[1] : 0;
            const double b = seconds[0] > 0 ? candidates[0] / seconds[0] : 0;
            selected = a > 0 && b > a * 1.005 ? 0u : 1u;
            return selected;
        }
        return mode(stage);
    }
};
}
