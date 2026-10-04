// Source-only fixture. No CUDA lifecycle or GPU code is included.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "../../cg_sha.h"
#ifndef QSB_SLOTS
#define QSB_SLOTS 5
#endif
#ifndef QSB_PK_LANES
#define QSB_PK_LANES 128
#endif
#define QSB_PK_REC ((size_t)QSB_PK_LANES * 68u)
#ifndef QSB_HOST_PKSHA
#define QSB_HOST_PKSHA 4
#endif
#ifndef QSB_FIN_BAL2
#define QSB_FIN_BAL2 3
#endif
#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
namespace A_literal {
#include "A-record.inc"
}
namespace C_literal {
#include "C-record.inc"
}

/* No vector value crosses this fixture ABI. The record entry points call the
 * exact extracted hash_record. They include Job init and hit-copy for the
 * oracle. For a record-service benchmark, call the internal hash_record on a
 * preinitialized Job and place reset/copy outside the timed interval.
 * mode globals make these fixture adapters single-thread entry points. */
extern "C" int pk_fixture_has_avx2() {
    __builtin_cpu_init(); return __builtin_cpu_supports("avx2") != 0;
}
extern "C" int pk_fixture_has_shani() {
    __builtin_cpu_init(); return (__builtin_cpu_supports("sha") != 0)
        && (__builtin_cpu_supports("sse4.1") != 0);
}
#define FIXTURE_RECORD(TAG) \
extern "C" uint32_t pk_fixture_record_##TAG(const uint8_t *plane, uint32_t j, \
                                            uint32_t batch_sz, int selected_mode, uint32_t hits[64]) { \
    TAG##_literal::qsb_pk::Job job; \
    job.plane = plane; job.batch_sz = batch_sz; \
    TAG##_literal::qsb_pk::mode = selected_mode; \
    TAG##_literal::qsb_pk::hash_record(job, j); \
    const uint32_t n = job.nhit.load(std::memory_order_relaxed); \
    for (uint32_t k = 0; k < std::min(64u, n); ++k) hits[k] = job.hits[k]; \
    return n; \
}
FIXTURE_RECORD(A)
FIXTURE_RECORD(C)
#undef FIXTURE_RECORD

/* A full, initialized eight-lane group is the H0/packing interface. It writes
 * nine variable words per key and both H0 vectors in ORIGINAL lane order.
 * Mask/tail publication must additionally use record_A/C, above. */
#if QSB_PK_AVX2_PLANES
extern "C" __attribute__((target("avx2"), noinline))
void pk_fixture_group_C(const uint8_t *rec, int l0, uint32_t words0[9][8],
                         uint32_t words1[9][8], uint32_t h0[8], uint32_t h1[8]) {
    const qcg_sha::v8u order = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const qcg_sha::v8u restore = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const qcg_sha::v8u y = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(
        (const __m256i *)(rec + 64u*QSB_PK_LANES + 4u*l0)), order);
    qcg_sha::v8u w0[9], w1[9];
    C_literal::qsb_pk::qsb_pksha_h0::pubkey_words_planes<0, ((QSB_FIN_BAL2 & 2) != 0)>(
        w0, rec, QSB_PK_LANES, l0, y, nullptr);
    C_literal::qsb_pk::qsb_pksha_h0::pubkey_words_planes<1, ((QSB_FIN_BAL2 & 2) != 0)>(
        w1, rec, QSB_PK_LANES, l0, y, nullptr);
    for (int k = 0; k < 9; k++) {
        _mm256_storeu_si256((__m256i *)words0[k], _mm256_permutevar8x32_epi32(w0[k], restore));
        _mm256_storeu_si256((__m256i *)words1[k], _mm256_permutevar8x32_epi32(w1[k], restore));
    }
    _mm256_storeu_si256((__m256i *)h0, _mm256_permutevar8x32_epi32(
        C_literal::qsb_pk::qsb_pksha_h0::pubkey_h0_planes<0, ((QSB_FIN_BAL2 & 2) != 0)>(
            rec, QSB_PK_LANES, l0, y, nullptr), restore));
    _mm256_storeu_si256((__m256i *)h1, _mm256_permutevar8x32_epi32(
        C_literal::qsb_pk::qsb_pksha_h0::pubkey_h0_planes<1, ((QSB_FIN_BAL2 & 2) != 0)>(
            rec, QSB_PK_LANES, l0, y, nullptr), restore));
}
#endif
