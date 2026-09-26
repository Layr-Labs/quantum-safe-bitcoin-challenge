#pragma once
/* Include inside qcpu. No SIMD, allocation, Ctx dependency or epoch mutation.
 * A span owns its six early omissions. After recording, its fields never change
 * until reset; the accessor borrows only the completed batch and stable cwin.
 */
struct CpuEpochSpan {
    uint32_t begin, end, first_pattern;
    uint8_t early[6];
};

template <unsigned Capacity> class CpuEpochSpans {
    CpuEpochSpan spans_[Capacity ? Capacity : 1];
    uint32_t batch_ = 0, ncwin_ = 0, count_ = 0, covered_ = 0;
    bool enabled_ = false;
public:
    /* wi==ncwin denotes the pending next epoch and therefore starts at zero.
     * Preflight the ENTIRE batch: insufficient capacity selects eager metadata
     * before its first candidate, never after some skips have been omitted.
     * All arithmetic stays <=batch; no batch+ncwin-1 overflow is possible.
     */
    bool reset(uint32_t batch, uint32_t ncwin, uint32_t wi) {
        batch_ = batch; ncwin_ = ncwin; count_ = covered_ = 0;
        enabled_ = false;
        if (!ncwin || wi > ncwin) return false;
        const uint32_t first = wi == ncwin ? 0 : wi;
        const uint32_t remaining = ncwin - first;
        uint32_t needed = batch ? 1 : 0;
        if (batch > remaining) needed += 1 + (batch - remaining - 1) / ncwin;
        enabled_ = needed <= Capacity;
        return enabled_;
    }
    uint32_t count() const { return count_; }
    uint32_t covered() const { return covered_; }
    bool complete() const { return enabled_ && covered_ == batch_; }

    /* Contract after successful reset: call once at each batch/epoch
     * intersection, with begin==covered()<batch, first_pattern the current
     * wi<ncwin, and the current epoch's initialized early[6]. The first wi is
     * reset's wi (or zero when reset saw ncwin); subsequent spans start at zero.
     * These enumeration invariants plus reset prove count_<Capacity and a
     * positive length. There is no late failure, dropped record or allocation.
     */
    void record(uint32_t begin, uint32_t first_pattern, const uint8_t *early) {
        const uint32_t room = batch_ - begin, left = ncwin_ - first_pattern;
        CpuEpochSpan &s = spans_[count_++];
        s.begin = begin;
        s.end = begin + (room < left ? room : left);
        s.first_pattern = first_pattern;
        memcpy(s.early, early, sizeof s.early);
        covered_ = s.end;
    }

    /* Contract: complete(), q<batch, and the same stable ncwin-row cwin used
     * for enumeration. Call only after the hash prefilter and EC bad check.
     * The spans partition [0,batch), so lookup and cwin indexing are bounded.
     */
    const uint8_t *get(uint32_t q, const uint8_t (*cwin)[3], uint8_t local[9]) const {
        uint32_t i = 0;
        while (q >= spans_[i].end) ++i;
        const CpuEpochSpan &s = spans_[i];
        const uint32_t pattern = s.first_pattern + (q - s.begin);
        memcpy(local, s.early, sizeof s.early);
        memcpy(local + 6, cwin[pattern], 3);
        return local;
    }
};

/* One accessor type serves both staged H0 and fused sparse publication.
 * A null spans pointer deliberately selects the original eager array; lazy
 * construction requires spans->complete(). It does not inspect or reconstruct
 * any candidate until get(), and never returns a nullable hit authorization.
 * Reuse the returned nine bytes for both recids of the same candidate.
 */
template <unsigned Capacity> class CpuEpochSkipAccessor {
    const CpuEpochSpans<Capacity> *spans_;
    const uint8_t (*cwin_)[3];
    const uint8_t *eager_;
public:
    CpuEpochSkipAccessor(const CpuEpochSpans<Capacity> *spans,
            const uint8_t (*cwin)[3], const uint8_t *eager)
        : spans_(spans), cwin_(cwin), eager_(eager) {}
    const uint8_t *get(uint32_t q, uint8_t local[9]) const {
        return spans_ ? spans_->get(q, cwin_, local) : eager_ + (size_t)q * 9;
    }
};

/* Small portable startup check. The Python artifact also checks the full
 * default batch/epoch alignment period and prefilter/exact-gate ordering.
 * This check has no hashing, publication, counters or feature side effects.
 */
static bool cpu_epoch_spans_selfcheck() {
    CpuEpochSpans<8> spans;
    if (spans.reset(1024, 0, 0) || spans.reset(1024, 158, 159) ||
        spans.reset(1024, 1, 0) || spans.reset(1024, 127, 0) ||
        spans.reset(1024, 128, 1) || !spans.reset(1024, 128, 0)) return false;
    CpuEpochSpans<0> empty;
    if (empty.reset(1, 158, 0) || !empty.reset(0, 158, 0) || !empty.complete()) return false;
    if (!spans.reset(UINT32_MAX, UINT32_MAX, UINT32_MAX) ||
        spans.reset(UINT32_MAX, 1, 0)) return false;
    uint8_t cwin[286][3];
    for (unsigned p = 0; p < 286; ++p) {
        cwin[p][0] = (uint8_t)p;
        cwin[p][1] = (uint8_t)(p >> 8);
        cwin[p][2] = (uint8_t)(p * 17 + 31);
    }
    const uint32_t cases[][3] = {
        {1024, 158, 0}, {1024, 158, 1}, {1024, 158, 157}, {1024, 158, 158},
        {32, 7, 6}, {64, 286, 260}, {1, 1, 1}
    };
    for (const auto &t : cases) {
        const uint32_t batch = t[0], ncwin = t[1], initial = t[2] == t[1] ? 0 : t[2];
        if (!spans.reset(batch, ncwin, t[2])) return false;
        uint32_t wi = initial, epoch = 0;
        uint8_t early[6];
        while (spans.covered() < batch) {
            for (unsigned j = 0; j < 6; ++j) early[j] = (uint8_t)(epoch * 13 + j);
            spans.record(spans.covered(), wi, early);
            memset(early, 0xA5, sizeof early); // no span may borrow this mutable buffer
            wi = 0; ++epoch;
        }
        if (!spans.complete()) return false;
        CpuEpochSkipAccessor<8> access(&spans, cwin, nullptr);
        for (uint32_t q = 0; q < batch; ++q) {
            uint8_t local[9];
            const uint8_t *sk = access.get(q, local);
            const uint32_t e = (initial + q) / ncwin, p = (initial + q) % ncwin;
            if (sk != local) return false;
            for (unsigned j = 0; j < 6; ++j)
                if (sk[j] != (uint8_t)(e * 13 + j)) return false;
            if (memcmp(sk + 6, cwin[p], 3)) return false;
        }
    }
    uint8_t eager[18], local[9];
    for (unsigned j = 0; j < sizeof eager; ++j) eager[j] = (uint8_t)(j * 7);
    memset(local, 0x5A, sizeof local);
    CpuEpochSkipAccessor<8> fallback(nullptr, cwin, eager);
    if (fallback.get(1, local) != eager + 9) return false;
    for (unsigned j = 0; j < sizeof local; ++j) if (local[j] != 0x5A) return false;
    return true;
}
