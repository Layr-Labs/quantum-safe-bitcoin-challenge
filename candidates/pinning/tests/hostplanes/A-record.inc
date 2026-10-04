namespace qsb_pk {
static const int NB = 2 * QSB_SLOTS;
static const uint32_t CH = 2;   /* records per claim */
struct alignas(64) Job {
    std::atomic<uint64_t> ctl{0};
    std::atomic<uint32_t> done{0};
    std::atomic<uint32_t> done_candidates{0};
    std::atomic<uint32_t> nhit{0};
    const uint8_t *plane = nullptr;
    std::atomic<uint32_t> nrec{0};
    uint32_t batch_sz = 0, seq = 0, lt = 0;
    uint32_t hits[64];
};
static Job jobs[NB];
static uint8_t *planes[NB];
/* One GPU slot can recycle this device output once its D2H/slot_done completes;
 * the independent host-plane owner may continue hashing that copied output. */
static uint8_t *device_planes[QSB_SLOTS];
static uint64_t owner[NB];   /* batch number + 1 of the posted, uncollected job; 0 = none */
static int ok = 0, mode = 0; /* mode: 2 = SHA-NI, 1 = AVX2, 0 = scalar */
#ifndef QSB_PK_DMA
#define QSB_PK_DMA 1
#endif
#ifndef QSB_PK_AUTOTUNE
#define QSB_PK_AUTOTUNE 1
#endif
static int admit_enabled=1, calibrating=0, ec_env_added=0;
static std::atomic<int> workers_parked{0};
static uint64_t gpu_finished=0;
static double quota_cpus() {
    const char *paths[]={"/sys/fs/cgroup/cpu.max","/sys/fs/cgroup/cpu/cpu.cfs_quota_us","/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us"};
    FILE *f=fopen(paths[0],"r");
    if(f){char q[64]={};long long per=0;int n=fscanf(f,"%63s %lld",q,&per);fclose(f);
        if(n==2 && strcmp(q,"max") && per>0)return (double)atoll(q)/(double)per;
        if(n>=1)return -1.;}
    for(int k=1;k<3;k++) {f=fopen(paths[k],"r");if(!f)continue;
        long long q=-1,per=0;fscanf(f,"%lld",&q);fclose(f);
        const char *pp=k==1?"/sys/fs/cgroup/cpu/cpu.cfs_period_us":"/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us";
        f=fopen(pp,"r");if(f){fscanf(f,"%lld",&per);fclose(f);}
        if(q>0&&per>0)return (double)q/(double)per;
    }return -1.;
}


static inline void message(uint32_t w[16], const uint64_t *x, uint32_t prefix) {
    uint32_t s[8];
    for (int i = 0; i < 4; i++) { s[2 * i] = (uint32_t)x[i]; s[2 * i + 1] = (uint32_t)(x[i] >> 32); }
    w[0] = (prefix << 24) | (s[7] >> 8);
    for (int i = 1; i < 8; i++) w[i] = (s[8 - i] << 24) | (s[7 - i] >> 8);
    w[8] = (s[0] << 24) | 0x800000u;
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = 0x108u;
}
static inline uint32_t prefix_of(uint32_t yp, int ri) {
#if QSB_FIN_BAL2 & 2
    return (yp >> (8 * ri)) & 0xFFu;
#else
    return 2u + ((yp >> ri) & 1u);
#endif
}
static inline int h0_hit(uint32_t h0) {
#if QSB_ZEROS_N >= 32
    return h0 == 0u;
#else
    return (h0 >> (32 - QSB_ZEROS_N)) == 0u;
#endif
}
static inline void record_hit(Job &J, uint32_t raw) {
    const uint32_t pos = J.nhit.fetch_add(1, std::memory_order_relaxed);
    if (pos < 64) J.hits[pos] = raw;
}
/* key limb i of recid ri of lane l is u64 number ((2*ri + i/2)*L + l)*2 + i%2 of the record */
static inline void lane_keys(const uint8_t *rec, int l, uint64_t x0[4], uint64_t x1[4]) {
    const uint64_t *u = (const uint64_t *)rec;
    for (int i = 0; i < 4; i++) {
        x0[i] = u[((size_t)(i >> 1) * QSB_PK_LANES + l) * 2 + (i & 1)];
        x1[i] = u[((size_t)(2 + (i >> 1)) * QSB_PK_LANES + l) * 2 + (i & 1)];
    }
}
#include "baseline-pksha_host_h0.h"
__attribute__((target("avx2"), noinline))
static void hash8_avx2(const uint32_t w0[16][8], const uint32_t w1[16][8], uint32_t h0[8], uint32_t h1[8]) {
    _mm256_storeu_si256((__m256i*)h0,qsb_pksha_h0::pubkey_h0(w0));
    _mm256_storeu_si256((__m256i*)h1,qsb_pksha_h0::pubkey_h0(w1));
}
__attribute__((target("sha,sse4.1"), noinline))
static void hash2_ni(const uint32_t m0[16], const uint32_t m1[16], uint32_t *h0, uint32_t *h1) {
    uint32_t s0[8], s1[8];
    memcpy(s0, qcg_sha::IV256, 32); memcpy(s1, qcg_sha::IV256, 32);
    qcg_sha::shani_compress2(s0, m0, s1, m1);
    *h0 = s0[0]; *h1 = s1[0];
}
static void hash_record(Job &J, uint32_t j) {
    const uint8_t *rec = J.plane + (size_t)j * QSB_PK_REC;
    const uint32_t *yp = (const uint32_t *)(rec + 64u * QSB_PK_LANES);
    const uint32_t base = (j * (uint32_t)QSB_HOST_PKSHA + (uint32_t)QSB_HOST_PKSHA - 1u) * (uint32_t)QSB_PK_LANES;
    for (int l0 = 0; l0 < QSB_PK_LANES; l0 += 8) {
        uint32_t w0[16][8], w1[16][8], h0[8], h1[8], live = 0;
        for (int t = 0; t < 8; t++) {
            const int l = l0 + t;
            uint32_t m0[16], m1[16];
            const uint32_t y = yp[l];
            h0[t] = h1[t] = ~0u;
            if (y != 0u && base + (uint32_t)l < J.batch_sz) {
                uint64_t x0[4], x1[4];
                lane_keys(rec, l, x0, x1);
                message(m0, x0, prefix_of(y, 0));
                message(m1, x1, prefix_of(y, 1));
                live |= 1u << t;
                if (mode == 2) {
                    hash2_ni(m0, m1, &h0[t], &h1[t]);
                } else if (mode == 0) {
                    uint32_t s0[8], s1[8];
                    memcpy(s0, qcg_sha::IV256, 32); memcpy(s1, qcg_sha::IV256, 32);
                    qcg_sha::sha_compress_ref(s0, m0); qcg_sha::sha_compress_ref(s1, m1);
                    h0[t] = s0[0]; h1[t] = s1[0];
                }
            } else {
                memset(m0, 0, sizeof m0); memset(m1, 0, sizeof m1);
            }
            if (mode == 1)
                for (int k = 0; k < 16; k++) { w0[k][t] = m0[k]; w1[k][t] = m1[k]; }
        }
        if (!live) continue;
        if (mode == 1) hash8_avx2(w0, w1, h0, h1);
        for (int t = 0; t < 8; t++) {
            if (!((live >> t) & 1u)) continue;
            const uint32_t idx = base + (uint32_t)(l0 + t);
            /* the GPU tests recid 0 first and stops at its hit */
            if (h0_hit(h0[t])) record_hit(J, idx);
            else if (h0_hit(h1[t])) record_hit(J, idx | (1u << 30));
        }
    }
}
} // namespace qsb_pk
