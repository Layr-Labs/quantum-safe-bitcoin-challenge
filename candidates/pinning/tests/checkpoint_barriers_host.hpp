// Host-only audit support. This file is not included by the CUDA candidate.
// Test the unchanged tree arithmetic modulo secp256k1 with OpenSSL. Device
// arithmetic, CUDA compilation and GPU memory ordering still require GPU tests.
#include <openssl/bn.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

class Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    unsigned participants, arrived = 0, generation = 0;
public:
    std::atomic<unsigned> calls{0};
    explicit Barrier(unsigned size): participants(size) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        unsigned gen = generation;
        calls++;
        if (++arrived == participants) {
            arrived = 0;
            generation++;
            cv.notify_all();
        } else {
            cv.wait(lock, [&]{ return generation != gen; });
        }
    }
};
struct Block {
    Barrier all;
    std::vector<std::unique_ptr<Barrier>> warps;
    explicit Block(unsigned size): all(size) {
        for (unsigned i=0; i<size/32; i++) warps.emplace_back(new Barrier(32));
    }
};
struct Coord { unsigned x; };
thread_local Coord threadIdx, blockIdx;
thread_local Block *current_block;
#define __device__
#define __forceinline__ inline
#define __shared__ static
inline void __syncthreads() { current_block->all.wait(); }
inline void __syncwarp() { current_block->warps[threadIdx.x/32]->wait(); }
inline void qsb_st_u64(uint64_t *p, uint64_t x) { *p=x; }
inline uint64_t qsb_ld_u64(const uint64_t *p) { return *p; }

struct Math {
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=nullptr,*a=BN_new(),*b=BN_new(),*r=BN_new();
    Math() { BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"); }
    ~Math() { BN_CTX_free(ctx); BN_free(p); BN_free(a); BN_free(b); BN_free(r); }
};
thread_local Math field;
inline void check(int ok) { if(!ok) std::abort(); }
inline void qsb_field_mul(uint64_t *out, const uint64_t *a, const uint64_t *b) {
    BN_lebin2bn((const unsigned char*)a,32,field.a);
    BN_lebin2bn((const unsigned char*)b,32,field.b);
    check(BN_mod_mul(field.r,field.a,field.b,field.p,field.ctx));
    check(BN_bn2lebinpad(field.r,(unsigned char*)out,32)==32);
    out[4]=0;
}
inline void qsb_field_normalize(uint64_t *v) {
    BN_lebin2bn((const unsigned char*)v,32,field.a);
    check(BN_nnmod(field.r,field.a,field.p,field.ctx));
    check(BN_bn2lebinpad(field.r,(unsigned char*)v,32)==32);
}
inline void reference_inverse(uint64_t *v) {
    BN_lebin2bn((const unsigned char*)v,32,field.a);
    check(BN_mod_inverse(field.r,field.a,field.p,field.ctx)!=nullptr);
    check(BN_bn2lebinpad(field.r,(unsigned char*)v,32)==32);
    v[4]=0;
}
