// CPU integration harness for an extracted, current persistent CUDA body.
// The Python runner creates the include and compiles this in a temporary dir.
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;
struct CpuIndex { unsigned x; };
thread_local CpuIndex threadIdx{},blockIdx{};
thread_local size_t cpu_multiplications=0,cpu_inversions=0;
struct CpuBlock {
    std::barrier<> block{128};
    std::array<std::unique_ptr<std::barrier<>>,4> warp;
    uint64_t products[4][256]{},inverses[4][128]{};
    CpuBlock() { for(auto &w:warp)w=std::make_unique<std::barrier<>>(32); }
};
thread_local CpuBlock *cpu_block;
static std::barrier<> service_vote_barrier(32);
static std::array<bool,32> service_votes{};

#define __device__
#define __host__
#define __global__
#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__ static
inline void __syncthreads() { cpu_block->block.arrive_and_wait(); }
inline void __syncwarp() { cpu_block->warp[threadIdx.x/32]->arrive_and_wait(); }
static bool cpu_vote(bool value,bool all) {
    service_votes[threadIdx.x]=value;
    service_vote_barrier.arrive_and_wait();
    bool out=all;
    for(bool vote:service_votes)out=all?(out&&vote):(out||vote);
    service_vote_barrier.arrive_and_wait();
    return out;
}
bool qsb_inverse_test_any(bool value) { return cpu_vote(value,false); }
bool qsb_inverse_test_all(bool value) { return cpu_vote(value,true); }

static const cpp_int B=cpp_int(1)<<256;
static const cpp_int K=(cpp_int(1)<<32)+977;
static const cpp_int P=B-K;
static const cpp_int mask64=(cpp_int(1)<<64)-1;
static cpp_int integer(const uint64_t *a) {
    cpp_int n=0;
    for(int k=3;k>=0;--k){n<<=64;n+=a[k];}
    return n;
}
static void limbs(uint64_t *out,cpp_int n) {
    for(int k=0;k<4;++k){out[k]=(n&mask64).convert_to<uint64_t>();n>>=64;}
}
inline void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b) {
    limbs(out,(integer(a)*integer(b))%P);++cpu_multiplications;
    // Deliberately make one lane in each warp arrive at different times.
    if(threadIdx.x%32==31 && cpu_multiplications%17==0)std::this_thread::yield();
}
inline void qsb_field_mul_sc(uint64_t *,uint64_t *,uint64_t *) {
    assert(false && "The integration audit requires exact tree arithmetic");
}
inline void qsb_field_normalize(uint64_t *a) { limbs(a,integer(a)%P); }
inline void _ModInv(uint64_t *a) {
    const cpp_int n=integer(a);assert(n!=0 && n<P);
    limbs(a,boost::multiprecision::powm(n,P-2,P));++cpu_inversions;
}
#define QSB_INVERSE_SERVICE_HOST_TEST
#include "inverse_tree.cuh"
#include "inverse_service.cuh"
#include "shifted_tail.cuh"

struct CpuObservation {uint64_t x[2][4];uint32_t parities,valid;};
constexpr unsigned cpu_workers=2,cpu_tiles=2,cpu_candidates=256;
static CpuObservation every_tile[cpu_tiles][cpu_candidates]{};
template<class Result> inline void cpu_record(const Result& result,unsigned tile,unsigned candidate) {
    static_assert(sizeof(Result)==sizeof(CpuObservation));
    memcpy(&every_tile[tile][candidate],&result,sizeof(result));
}
#include QSB_CPU_PROBE_SOURCE

int main() {
    static_assert(AFFINE_N==128 && AFFINE_WIDTH==256 && AFFINE_WINDOWS==15);
    static_assert(QSB_AFFINE_TREE_EXACT==1);
    std::vector<QsbShiftedTailPoint> table(AFFINE_WIDTH*AFFINE_WINDOWS);
    std::vector<QsbShiftedTailRecord> tails(AFFINE_WIDTH);
    EC_GROUP* group=EC_GROUP_new_by_curve_name(NID_secp256k1);assert(group);
    BN_CTX* ctx=BN_CTX_new();BIGNUM* scalar=BN_new();BIGNUM* part=BN_new();BIGNUM* order=BN_new();
    EC_POINT* base=EC_POINT_new(group);EC_POINT* point=EC_POINT_new(group);
    EC_POINT* recovery=EC_POINT_new(group);EC_POINT* minus_recovery=EC_POINT_new(group);EC_POINT* sum=EC_POINT_new(group);
    assert(ctx&&scalar&&part&&order&&base&&point&&recovery&&minus_recovery&&sum);
    assert(EC_GROUP_get_order(group,order,ctx)==1 && BN_set_word(scalar,1009)==1);
    assert(EC_POINT_mul(group,recovery,scalar,nullptr,nullptr,ctx)==1);
    assert(EC_POINT_copy(minus_recovery,recovery)==1 && EC_POINT_invert(group,minus_recovery,ctx)==1);
    for(unsigned w=0;w<AFFINE_WINDOWS;w++) {
        assert(BN_one(scalar)==1 && BN_lshift(scalar,scalar,12*w)==1);
        assert(EC_POINT_mul(group,base,scalar,nullptr,nullptr,ctx)==1 && EC_POINT_copy(point,base)==1);
        for(unsigned j=0;j<AFFINE_WIDTH;j++) {
            assert(affine_encode(table[w*AFFINE_WIDTH+j],group,point,ctx));
            if(w==14)for(unsigned arm=0;arm<2;arm++) {
                assert(EC_POINT_add(group,sum,point,arm?minus_recovery:recovery,ctx)==1);
                assert(affine_encode(tails[j].arm[arm],group,sum,ctx));
            }
            assert(EC_POINT_add(group,point,point,base,ctx)==1);
        }
    }
    std::array<qsb_inverse_service::Slot,cpu_workers> slots{};
    std::array<AffineResult,cpu_candidates> final_results{};
    std::array<CpuBlock,cpu_workers> blocks;
    std::array<size_t,32+cpu_candidates> multiplications{},inversions{};
    std::vector<std::thread> threads;
    for(unsigned lane=0;lane<32;lane++)threads.emplace_back([&,lane] {
        threadIdx.x=lane;blockIdx.x=0;cpu_block=nullptr;
        qsb_persistent_affine_probe(table.data(),tails.data(),slots.data(),final_results.data(),1,cpu_workers,cpu_tiles);
        multiplications[lane]=cpu_multiplications;inversions[lane]=cpu_inversions;
    });
    for(unsigned candidate=0;candidate<cpu_candidates;candidate++)threads.emplace_back([&,candidate] {
        const unsigned worker=candidate/AFFINE_N;
        threadIdx.x=candidate%AFFINE_N;blockIdx.x=worker+1;cpu_block=&blocks[worker];
        qsb_persistent_affine_probe(table.data(),tails.data(),slots.data(),final_results.data(),1,cpu_workers,cpu_tiles);
        multiplications[32+candidate]=cpu_multiplications;inversions[32+candidate]=cpu_inversions;
    });
    for(auto &thread:threads)thread.join();
    unsigned checked=0,mismatches=0,declined=0;
    for(unsigned tile=0;tile<cpu_tiles;tile++)for(unsigned c=0;c<cpu_candidates;c++) {
        const CpuObservation& observed=every_tile[tile][c];
        if(!observed.valid){++declined;continue;}
        BN_zero(scalar);
        for(unsigned w=0;w<AFFINE_WINDOWS;w++) {
            const uint32_t code=affine_code(c,tile,w);
            assert(BN_set_word(part,(code&255u)+1u)==1 && BN_lshift(part,part,12*w)==1);
            if(code&256u)BN_set_negative(part,1);
            assert(BN_add(scalar,scalar,part)==1);
        }
        assert(BN_nnmod(scalar,scalar,order,ctx)==1 && EC_POINT_mul(group,point,scalar,nullptr,nullptr,ctx)==1);
        for(unsigned arm=0;arm<2;arm++) {
            QsbShiftedTailPoint expected{};
            assert(EC_POINT_add(group,sum,point,arm?minus_recovery:recovery,ctx)==1);
            assert(affine_encode(expected,group,sum,ctx));
            if(memcmp(expected.x,observed.x[arm],32)||((expected.y[0]&1u)!=((observed.parities>>arm)&1u)))++mismatches;
        }
        if(tile+1==cpu_tiles && memcmp(&observed,&final_results[c],sizeof(observed)))++mismatches;
        ++checked;
    }
    size_t total_multiplications=0,total_inversions=0;
    for(auto value:multiplications)total_multiplications+=value;
    for(auto value:inversions)total_inversions+=value;
    for(auto &slot:slots) {
        assert(slot.request==qsb_inverse_service::STOP);
        assert(slot.response==cpu_tiles*14);
    }
    assert(total_multiplications==cpu_tiles*cpu_workers*(14*(3*AFFINE_N-3)+13*AFFINE_N*3+AFFINE_N*9));
    printf("{\"worker_ctas\":%u,\"worker_threads\":%u,\"service_lanes\":32,\"tiles\":%u,\"root_roundtrips\":%u,\"checked_pairs\":%u,\"checked_affine_arms\":%u,\"field_multiplications\":%zu,\"service_inversions_including_identity_lanes\":%zu,\"mismatches\":%u,\"declined\":%u,\"actual_kernel_body\":true,\"actual_tree_header\":true,\"actual_service_header\":true,\"gpu_execution\":false}\n",
           cpu_workers,cpu_candidates,cpu_tiles,cpu_tiles*cpu_workers*14,checked,checked*2,total_multiplications,total_inversions,mismatches,declined);
    EC_POINT_free(base);EC_POINT_free(point);EC_POINT_free(recovery);EC_POINT_free(minus_recovery);EC_POINT_free(sum);
    BN_free(scalar);BN_free(part);BN_free(order);BN_CTX_free(ctx);EC_GROUP_free(group);
    return mismatches||declined?1:0;
}
