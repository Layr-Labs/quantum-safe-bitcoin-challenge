// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cmath>
#include <cstdint>

namespace qsb {
// Compare complete productive sequences, including both slots, root kernels,
// finishing, readback and the exact host gate. Every sequence is searched once.
// The caller drains all slots at each warmup/measurement boundary BEFORE calling
// complete_sequence, and changes the dispatch policy only after that call.
class ColdStreamTrial {
 public:
  static constexpr unsigned kArms = 8;
  static constexpr unsigned kWarm = 2;
  static constexpr unsigned kMeasured = 4;
  enum Event { Continue = 0, Start = 1, Sample = 2, Selected = 3, Invalid = -1 };

  bool active() const { return arm_ < kArms; }
  bool streaming() const { return active() ? order(arm_) != 0 : selected_; }
  unsigned arm() const { return arm_; }
  bool drain_next() const {
    return active() && (sequence_ + 1 == kWarm ||
                        sequence_ + 1 == kWarm + kMeasured);
  }
  double seconds(unsigned arm) const { return seconds_[arm]; }
  double gain() const { return gain_; }

  Event complete_sequence(double now, uint64_t candidates) {
    if (!active()) return Continue;
    if (!std::isfinite(now) || now < 0 ||
        (seen_time_ && now <= last_time_) || !candidates) return fail();
    seen_time_ = true;
    last_time_ = now;
    // Sequence work must be equal, including the final partial batch.
    if (work_ && candidates != work_) return fail();
    work_ = candidates;
    ++sequence_;
    if (sequence_ == kWarm) {
      start_ = now;
      return Start;
    }
    if (sequence_ != kWarm + kMeasured) return Continue;
    seconds_[arm_] = now - start_;
    if (!(seconds_[arm_] > 0) || !std::isfinite(seconds_[arm_])) return fail();
    ++arm_;
    sequence_ = 0;
    if (active()) return Sample;
    const double a0 = seconds_[0] + seconds_[3];
    const double b0 = seconds_[1] + seconds_[2];
    const double a1 = seconds_[5] + seconds_[6];
    const double b1 = seconds_[4] + seconds_[7];
    const double a = a0 + a1, b = b0 + b1;
    if (!std::isfinite(a) || !std::isfinite(b)) return fail();
    gain_ = a / b - 1.0;
    // Require the gain to repeat in both mirrored halves. A tie, a noisy
    // disagreement, or less than 1.5% aggregate gain retains original loads.
    selected_ = a0 > b0 * 1.01 && a1 > b1 * 1.01 && a > b * 1.015;
    return Selected;
  }

 private:
  static unsigned order(unsigned arm) {
    const unsigned plan[kArms] = {0, 1, 1, 0, 1, 0, 0, 1};
    return plan[arm];
  }
  Event fail() { arm_ = kArms; selected_ = false; return Invalid; }
  unsigned arm_ = 0, sequence_ = 0;
  uint64_t work_ = 0;
  double seconds_[kArms] = {}, start_ = 0, last_time_ = 0, gain_ = 0;
  bool selected_ = false, seen_time_ = false;
};
}  // namespace qsb
