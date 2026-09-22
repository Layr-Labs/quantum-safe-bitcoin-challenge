#!/usr/bin/env python3
"""Compile the actual C++11 policy and check selection, drift and finite progress."""
import os
from pathlib import Path
import subprocess
import tempfile


PROGRAM = r'''
#include "ProductiveTune.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

static unsigned checks = 0;
static void finish_warmup(qsb::ProductiveTune& p) {
    assert(!p.timed() && p.route() == 0 && !p.done());
    assert(!p.finish_cohort(0.0, 0));
    assert(!p.timed() && p.route() == 1 && !p.done());
    assert(!p.finish_cohort(std::numeric_limits<double>::quiet_NaN(), 0));
    assert(p.timed() && p.valid());
}

static qsb::ProductiveTune run(double speedup, double drift = 0.0,
                               bool uneven_work = false) {
    qsb::ProductiveTune p;
    assert(p.target_batches() == 8);
    finish_warmup(p);
    const unsigned order[] = {0,1,1,0,1,0,0,1};
    for (unsigned i = 0; i < 8; ++i) {
        assert(p.cohort_index() == i + 2 && p.route() == order[i]);
        assert(p.target_batches() == 128);
        const std::uint64_t work = uneven_work
            ? 1000000000ULL + 79190003ULL * i : 1073741824ULL;
        const double rate = 800000000.0 * (p.route() ? speedup : 1.0)
                            * std::exp(drift * i);
        assert(p.finish_cohort(static_cast<double>(work) / rate, work) == (i == 7));
    }
    assert(p.done() && !p.timed() && p.target_batches() == 0);
    assert(p.route() == p.selected() && p.cohort_index() == 10);
    const unsigned selected = p.selected();
    const double gain = p.geometric_gain();
    assert(!p.finish_cohort(1.0, 1));
    assert(p.selected() == selected && p.geometric_gain() == gain);
    ++checks;
    return p;
}

static qsb::ProductiveTune pairs(const double gains[4]) {
    qsb::ProductiveTune p;
    finish_warmup(p);
    for (unsigned i = 0; i < 8; ++i) {
        const double seconds = p.route() ? 1.0 / gains[i/2] : 1.0;
        p.finish_cohort(seconds, 1000000000ULL);
    }
    ++checks;
    return p;
}

int main() {
    assert(run(1.02).selected() == 1);
    assert(run(1.0).selected() == 0);
    assert(run(0.98).selected() == 0);
    assert(run(1.005).selected() == 0);
    assert(run(1.02, 0.002, true).selected() == 1);
    assert(run(1.02, -0.002, true).selected() == 1);
    assert(run(1.0, 0.002, true).selected() == 0);
    assert(run(1.0, -0.002, true).selected() == 0);
    assert(run(0.98, 0.002, true).selected() == 0);
    assert(std::fabs(run(1.02, 0.0, true).geometric_gain() - 1.02) < 1e-12);

    // A high average cannot compensate for a materially losing pair.
    const double severe_loss[] = {1.04,1.04,1.04,0.994};
    assert(pairs(severe_loss).selected() == 0);
    const double only_two_wins[] = {1.04,1.04,1.0,1.0};
    assert(pairs(only_two_wins).selected() == 0);
    const double three_wins[] = {1.02,1.02,1.02,0.999};
    assert(pairs(three_wins).selected() == 1);
    const double reversed_loss[] = {0.994,1.04,1.04,1.04};
    assert(pairs(reversed_loss).selected() == 0);

    // A bad sample at any timed position permanently selects the baseline.
    const double bad[] = {0.0,-1.0,std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()};
    for (unsigned kind = 0; kind < 5; ++kind) {
        for (unsigned position = 0; position < 8; ++position) {
            qsb::ProductiveTune p;
            finish_warmup(p);
            for (unsigned i = 0; i < 8; ++i) {
                const double t = i == position && kind < 4 ? bad[kind]
                    : (p.route() ? 0.5 : 1.0);
                const std::uint64_t work = i == position && kind == 4
                    ? 0 : 1000000000ULL;
                assert(p.finish_cohort(t,work) == (i == 7));
            }
            assert(p.done() && !p.valid() && p.selected() == 0);
            ++checks;
        }
    }
    // Extreme inputs must fail closed if a ratio under/overflows.
    qsb::ProductiveTune extreme;
    finish_warmup(extreme);
    for (unsigned i = 0; i < 8; ++i)
        extreme.finish_cohort(extreme.route() ? std::numeric_limits<double>::min()
                                             : std::numeric_limits<double>::max(), 1);
    assert(extreme.done() && !extreme.valid() && extreme.selected() == 0);
    ++checks;

    qsb::ProductiveTune bounded(0,0);
    for (unsigned i = 0; i < 10; ++i) {
        assert(bounded.target_batches() == 1);
        bounded.finish_cohort(1.0,UINT64_MAX);
    }
    assert(bounded.done() && bounded.valid() && bounded.selected() == 0);
    assert(bounded.ratio(4) == 0.0);
    ++checks;
    std::printf("productive tuning: %u policy scenarios passed\n",checks);
}
'''


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="qsb-productive-tune-") as tmp:
        source = Path(tmp) / "test.cpp"
        binary = Path(tmp) / "test"
        source.write_text(PROGRAM)
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++11", "-O2", "-Wall",
            "-Wextra", "-Werror", "-fsanitize=undefined,float-divide-by-zero",
            "-fno-sanitize-recover=all", "-I", str(here), str(source),
            "-o", str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
