// SPDX-License-Identifier: GPL-3.0-only
#ifndef QSB_PRODUCTIVE_TUNE_H
#define QSB_PRODUCTIVE_TUNE_H

#include <cmath>
#include <cstdint>

namespace qsb {

// A finite comparison of two implementations on ordinary, distinct search
// batches. Every warmup and timed batch must retain its work count and hits.
// The caller drains all pending slots before starting a cohort's clock and
// before passing its elapsed time and completed candidate count here. This
// class reads no clock, changes no search range, and estimates no ranked score.
class ProductiveTune {
 public:
  explicit ProductiveTune(unsigned warmup_batches = 8,
                          unsigned timed_batches = 128)
      : warmup_batches_(warmup_batches ? warmup_batches : 1),
        timed_batches_(timed_batches ? timed_batches : 1),
        cohort_(0), selected_(0), valid_(true), geometric_gain_(0.0) {
    for (unsigned i = 0; i < 8; ++i) log_cost_[i] = 0.0;
    for (unsigned i = 0; i < 4; ++i) ratios_[i] = 0.0;
  }

  // Two warmup cohorts (A, B), followed by ABBA BAAB. After completion,
  // route() permanently returns the selected implementation.
  unsigned route() const {
    static const unsigned order[8] = {0, 1, 1, 0, 1, 0, 0, 1};
    return done() ? selected_ : (cohort_ < 2 ? cohort_ : order[cohort_ - 2]);
  }
  bool done() const { return cohort_ == 10; }
  bool timed() const { return cohort_ >= 2 && !done(); }
  unsigned cohort_index() const { return cohort_; }
  unsigned target_batches() const {
    return done() ? 0 : (timed() ? timed_batches_ : warmup_batches_);
  }
  unsigned selected() const { return selected_; }
  bool valid() const { return valid_; }
  double ratio(unsigned pair) const { return pair < 4 ? ratios_[pair] : 0.0; }
  double geometric_gain() const { return geometric_gain_; }

  // Returns true only on the transition to the permanent selected route.
  // Work is the actual number of completed candidates, including batch tails;
  // dividing by it prevents unequal cohort sizes from biasing the comparison.
  // Warmup measurements are deliberately excluded. Invalid timed measurements
  // disable selection of B, while the finite cohort schedule still completes.
  bool finish_cohort(double elapsed_seconds, std::uint64_t work) {
    if (done()) return false;
    if (timed()) {
      if (!(elapsed_seconds > 0.0) || !std::isfinite(elapsed_seconds) || !work) {
        valid_ = false;
      } else {
        const double cost = std::log(elapsed_seconds) -
                            std::log(static_cast<double>(work));
        if (!std::isfinite(cost)) valid_ = false;
        log_cost_[cohort_ - 2] = cost;
      }
    }
    ++cohort_;
    if (!done()) return false;
    decide();
    return true;
  }

 private:
  void decide() {
    if (!valid_) return;
    static const unsigned baseline[4] = {0, 3, 5, 6};
    static const unsigned candidate[4] = {1, 2, 4, 7};
    double sum = 0.0;
    unsigned wins = 0;
    bool bounded_loss = true;
    for (unsigned i = 0; i < 4; ++i) {
      const double gain = log_cost_[baseline[i]] - log_cost_[candidate[i]];
      ratios_[i] = std::exp(gain);
      if (!std::isfinite(ratios_[i]) || !(ratios_[i] > 0.0)) valid_ = false;
      sum += gain;
      if (gain > 0.0) ++wins;
      if (gain < std::log(0.995)) bounded_loss = false;
    }
    geometric_gain_ = std::exp(sum / 4.0);
    if (!std::isfinite(geometric_gain_) || !(geometric_gain_ > 0.0)) valid_ = false;
    if (valid_ && sum / 4.0 >= std::log(1.01) && wins >= 3 && bounded_loss)
      selected_ = 1;
  }

  unsigned warmup_batches_;
  unsigned timed_batches_;
  unsigned cohort_;
  unsigned selected_;
  bool valid_;
  double geometric_gain_;
  double log_cost_[8];
  double ratios_[4];
};

}  // namespace qsb
#endif
