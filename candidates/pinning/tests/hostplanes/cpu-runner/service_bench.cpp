// CPU-only timing shim. Includes the source-extracted A/C record entrypoints
// in the same TU so it times the actual private hash_record functions.
#include <time.h>
#include "../host_fixture.cpp"

struct pk_fixture_timing { uint64_t cpu_ns, wall_ns, reported_hits; };
static uint64_t pk_fixture_now(clockid_t c) {
    timespec t{};
    if (clock_gettime(c, &t) != 0) return 0;
    return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
}
#define TIMED_RECORD(TAG) \
extern "C" pk_fixture_timing pk_fixture_timed_record_##TAG( \
    const uint8_t *plane, uint32_t j, uint32_t batch_sz, int selected_mode, uint32_t calls) { \
    TAG##_literal::qsb_pk::Job job; \
    job.plane = plane; job.batch_sz = batch_sz; \
    TAG##_literal::qsb_pk::mode = selected_mode; \
    const uint64_t w0 = pk_fixture_now(CLOCK_MONOTONIC); \
    const uint64_t c0 = pk_fixture_now(CLOCK_THREAD_CPUTIME_ID); \
    for (uint32_t i = 0; i < calls; ++i) { \
        TAG##_literal::qsb_pk::hash_record(job, j); \
        asm volatile("" ::: "memory"); \
    } \
    const uint64_t c1 = pk_fixture_now(CLOCK_THREAD_CPUTIME_ID); \
    const uint64_t w1 = pk_fixture_now(CLOCK_MONOTONIC); \
    return {c1-c0, w1-w0, job.nhit.load(std::memory_order_relaxed)}; \
}
TIMED_RECORD(A)
TIMED_RECORD(C)
#undef TIMED_RECORD
