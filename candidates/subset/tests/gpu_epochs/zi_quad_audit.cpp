// Host lockstep audit of zi_inverse_quad_bounded (candidates/subset/tests/gpu_epochs/zinv32.cuh).
// Emulates the 4 cooperating lanes with 4 threads and a barrier-backed zi_x, then checks the
// returned value against an independent extended-Euclid inverse mod secp256k1 p.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <vector>
#include <array>

#define QSB_ROOT_MAX_BATCHES 16   // matches the tree default (ZI_ROOT_MAX_BATCHES = 2x)
#define __device__
#define __forceinline__ inline
#define __syncwarp(m) ((void)0)
static thread_local int g_lane = 0;
uint32_t zi_x(uint32_t v, int src);           // defined below, declared for the header
// The bounded quad inverse is what this audit exercises. The fallback tail of
// zinv32.cuh calls into the tree's field header, so stub its two symbols here;
// no code under test is replaced.
struct QsbInverseWords { uint64_t a,b,c,d; };
static inline QsbInverseWords qsb_root_fermat(QsbInverseWords v){ return v; }
#include "zinv32.cuh"

// ---- 4-lane barrier + exchange slots -------------------------------------
static uint32_t g_slot[4];
static std::mutex g_m; static std::condition_variable g_cv;
static int g_count = 0, g_gen = 0;
static void lane_barrier() {
    std::unique_lock<std::mutex> lk(g_m);
    int gen = g_gen;
    if (++g_count == 4) { g_count = 0; ++g_gen; g_cv.notify_all(); }
    else g_cv.wait(lk, [&]{ return g_gen != gen; });
}
uint32_t zi_x(uint32_t v, int src) {
    g_slot[g_lane] = v;
    lane_barrier();              // publish
    uint32_t r = g_slot[src & 3];
    lane_barrier();              // consume before next overwrite
    return r;
}

// ---- independent reference: extended Euclid on 256-bit ---------------------
typedef unsigned __int128 u128;
struct Big { uint64_t w[4]; };
static const uint64_t P[4] = {0xFFFFFFFEFFFFFC2FULL,0xFFFFFFFFFFFFFFFFULL,
                              0xFFFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
// reference via modular exponentiation x^(p-2) mod p using schoolbook 256-bit mulmod
static void mulmod(uint64_t*r,const uint64_t*a,const uint64_t*b){
    unsigned __int128 t[8]={0,0,0,0,0,0,0,0};
    uint64_t prod[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){ uint64_t carry=0;
        for(int j=0;j<4;j++){ u128 cur=(u128)a[i]*b[j]+prod[i+j]+carry;
            prod[i+j]=(uint64_t)cur; carry=(uint64_t)(cur>>64);} prod[i+4]+=carry; }
    (void)t;
    // reduce 512 -> 256 mod p (p = 2^256 - 0x1000003D1)
    const u128 K = 0x1000003D1ULL;
    uint64_t hi[4]={prod[4],prod[5],prod[6],prod[7]};
    uint64_t lo[4]={prod[0],prod[1],prod[2],prod[3]};
    for(int pass=0;pass<2;pass++){
        uint64_t carry=0, acc[5]={0,0,0,0,0};
        for(int i=0;i<4;i++){ u128 cur=(u128)hi[i]*K+lo[i]+carry; acc[i]=(uint64_t)cur; carry=(uint64_t)(cur>>64);} acc[4]=carry;
        lo[0]=acc[0];lo[1]=acc[1];lo[2]=acc[2];lo[3]=acc[3];
        hi[0]=acc[4];hi[1]=hi[2]=hi[3]=0;
        if(!acc[4]) break;
    }
    // final conditional subtract
    uint64_t t2[4]; unsigned __int128 br=0; int ge=1;
    for(int i=3;i>=0;i--){ if(lo[i]!=P[i]){ ge = lo[i]>P[i]; break; } }
    if(ge){ br=0; for(int i=0;i<4;i++){ u128 d=(u128)lo[i]-P[i]-(uint64_t)br; t2[i]=(uint64_t)d; br=(d>>127)&1;} memcpy(lo,t2,32);}    
    memcpy(r,lo,32);
}
static void ref_inv(uint64_t*r,const uint64_t*x){
    uint64_t e[4]={P[0]-2,P[1],P[2],P[3]};
    uint64_t base[4]; memcpy(base,x,32);
    uint64_t acc[4]={1,0,0,0};
    for(int i=0;i<256;i++){
        if((e[i>>6]>>(i&63))&1) mulmod(acc,acc,base);
        mulmod(base,base,base);
    }
    memcpy(r,acc,32);
}

int main(){
    std::mt19937_64 rng(12345);
    int trials=0, ok=0, zero_ok=0;
    std::vector<std::array<uint64_t,4>> cases;
    for(int i=0;i<200;i++){ std::array<uint64_t,4> v; for(int k=0;k<4;k++) v[k]=rng(); v[3]&=0x7FFFFFFFFFFFFFFFULL; cases.push_back(v);}    
    cases.push_back({1,0,0,0});
    cases.push_back({2,0,0,0});
    cases.push_back({P[0]-1,P[1],P[2],P[3]});
    cases.push_back({0,0,0,0});
    for(auto &c : cases){
        uint64_t Rl[4][5]; bool okf[4];
        for(int l=0;l<4;l++){ for(int k=0;k<4;k++) Rl[l][k]=c[k]; Rl[l][4]=0; }
        std::thread th[4];
        for(int l=0;l<4;l++) th[l]=std::thread([&,l]{ g_lane=l; okf[l]=zi_inverse_quad_bounded(Rl[l],l); });
        for(int l=0;l<4;l++) th[l].join();
        trials++;
        bool allsame=true;
        for(int l=1;l<4;l++) for(int k=0;k<4;k++) if(Rl[l][k]!=Rl[0][k]) allsame=false;
        if(!allsame){ printf("FAIL lane disagreement\n"); continue; }
        bool iszero = !(c[0]|c[1]|c[2]|c[3]);
        if(iszero){ bool z=!(Rl[0][0]|Rl[0][1]|Rl[0][2]|Rl[0][3]); if(z){zero_ok++;ok++;} else printf("FAIL zero case\n"); continue; }
        uint64_t ref[4]; ref_inv(ref,c.data());
        bool eq=true; for(int k=0;k<4;k++) if(ref[k]!=Rl[0][k]) eq=false;
        if(eq) ok++; else { printf("FAIL inv mismatch on case %016lx\n", (unsigned long)c[0]); }
    }
    printf("zi_inverse_quad_bounded host audit: %d/%d exact vs independent Fermat inverse (zero-case %d)\n", ok, trials, zero_ok);
    return ok==trials?0:1;
}
