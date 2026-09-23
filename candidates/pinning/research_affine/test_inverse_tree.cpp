// Execute the actual CUDA tree header with CPU threads and exact field math.
// This checks indices, barriers, checkpoint bounds, aliases and operation count;
// it does not execute CUDA field PTX or establish GPU performance.
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;
struct CpuIndex { unsigned x; };
thread_local CpuIndex threadIdx{}, blockIdx{};
thread_local unsigned mul_count;
static std::barrier<> *block_barrier;
static std::barrier<> *warp_barriers[8];
#define __device__
#define __forceinline__ inline
#define __shared__ static
inline void __syncthreads() { block_barrier->arrive_and_wait(); }
inline void __syncwarp() { warp_barriers[threadIdx.x/32]->arrive_and_wait(); }
static const cpp_int B = cpp_int(1)<<256;
static const cpp_int K = (cpp_int(1)<<32)+977;
static const cpp_int P = B-K;
static const cpp_int mask64 = (cpp_int(1)<<64)-1;

static cpp_int integer(const uint64_t *a) {
    cpp_int n=0;
    for (int k=3;k>=0;--k) { n<<=64; n+=a[k]; }
    return n;
}
static void limbs(uint64_t *out, cpp_int n) {
    for (int k=0;k<4;++k) { out[k]=(n&mask64).convert_to<uint64_t>(); n>>=64; }
}
inline void qsb_field_mul(uint64_t *out,uint64_t *a,uint64_t *b) {
    const cpp_int product=(integer(a)*integer(b))%P;
    limbs(out,product); ++mul_count;
}
inline void qsb_field_mul_sc(uint64_t *,uint64_t *,uint64_t *) {
    assert(false && "This audit requires the exact tree configuration.");
}
inline void qsb_field_normalize(uint64_t *a) {
    if ((a[1]&a[2]&a[3])==UINT64_MAX && a[0]>=0xFFFFFFFEFFFFFC2FULL) {
        a[0]-=0xFFFFFFFEFFFFFC2FULL; a[1]=a[2]=a[3]=0;
    }
}
#include "inverse_tree.cuh"

struct Totals { size_t blocks=0, leaves=0, multiplies=0, masked=0; } totals;

template<int N> static void audit() {
    constexpr int rounds=14;
    constexpr uint64_t canary=0xd15ea5e5cafebabeULL;
    std::mt19937_64 random(0xaff10000+N);
    std::array<std::array<std::array<uint64_t,4>,N>,rounds> leaves{};
    std::array<std::array<std::array<uint64_t,4>,N>,rounds> expected{};
    std::array<std::array<uint64_t,4>,rounds> roots_expected{};
    for (int r=0;r<rounds;++r) {
        cpp_int root=1;
        for (int i=0;i<N;++i) {
            cpp_int v=0;
            if (r==0) v=1;
            else if (r==1) v=P-1;
            else if (r==2) v=B-1;
            else if (r==3) v=P+1+(i%976); // raw, noncanonical operands
            else {
                for (int k=0;k<4;++k) { v<<=64; v+=random(); }
                if (v%P==0) v=1;
            }
            // Caller-side inactive/singular masking; identity leaves are still
            // inverted normally, and the caller retains their validity mask.
            if (r>=4 && (i>r*N/rounds || i%19==0)) {
                v=1; ++totals.masked;
            }
            limbs(leaves[r][i].data(),v);
            limbs(expected[r][i].data(),boost::multiprecision::powm(v%P,P-2,P));
            root=(root*v)%P;
        }
        limbs(roots_expected[r].data(),root);
    }

    std::barrier<> whole(N);
    block_barrier=&whole;
    std::array<std::unique_ptr<std::barrier<>>,N/32> warps;
    for (int w=0;w<N/32;++w) {
        warps[w]=std::make_unique<std::barrier<>>(32);
        warp_barriers[w]=warps[w].get();
    }
    uint64_t products[4][2*N]{}, inverses[4][N]{};
    // Use block index 1, leaving a complete unused block as a bounds sentinel.
    std::vector<uint64_t> checkpoint(2*4*(N-2)+16,canary);
    std::array<uint64_t,8+16> roots;
    roots.fill(canary);
    std::array<unsigned,N> counts{};
    std::vector<std::thread> workers;
    for (int i=0;i<N;++i) workers.emplace_back([&,i] {
        threadIdx.x=i; blockIdx.x=1;
        for (int mode=0;mode<2;++mode) for (int r=0;r<rounds;++r) {
            uint64_t value[4], root[4]={}, inv[4]={};
            for (int k=0;k<4;++k) value[k]=leaves[r][i][k];
            mul_count=0;
            if (mode==0) qsb_affine_tree_prepare_shared<N>(value,root,products);
            else qsb_affine_tree_prepare<N>(value,roots.data()+8,checkpoint.data()+8);
            // Emulate the kernel boundary in checkpoint mode. For shared mode
            // this also makes the root check independent of thread scheduling.
            __syncthreads();
            if (i==0) {
                if (mode==1) for (int k=0;k<4;++k) root[k]=roots[12+k];
                for (int k=0;k<4;++k) assert(root[k]==roots_expected[r][k]);
                const cpp_int reciprocal=boost::multiprecision::powm(integer(root),P-2,P);
                limbs(inv,reciprocal);
                if (mode==1) for (int k=0;k<4;++k) roots[12+k]=inv[k];
            }
            if (mode==0) qsb_affine_tree_finish_shared<N>(value,inv,products,inverses);
            else {
                __syncthreads(); // inverse kernel completion
                qsb_affine_tree_finish<N>(value,roots.data()+8,checkpoint.data()+8);
            }
            for (int k=0;k<4;++k) assert(value[k]==expected[r][i][k]);
            counts[i]=mul_count;
            __syncthreads();
            if (i==0) {
                unsigned sum=0; for (unsigned count:counts) sum+=count;
                assert(sum==3*N-3);
                for (int k=0;k<8+4*(N-2);++k) assert(checkpoint[k]==canary);
                for (size_t k=8+2*4*(N-2);k<checkpoint.size();++k) assert(checkpoint[k]==canary);
                for (int k=0;k<12;++k) assert(roots[k]==canary);
                for (size_t k=16;k<roots.size();++k) assert(roots[k]==canary);
                ++totals.blocks; totals.leaves+=N; totals.multiplies+=sum;
            }
            __syncthreads(); // no reuse of shared scratch until every lane exits
        }
    });
    for (auto &t:workers) t.join();
}

int main() {
    static_assert(QSB_AFFINE_TREE_EXACT==1);
    audit<32>(); audit<64>(); audit<128>(); audit<256>();
    std::printf("{\"blocks\":%zu,\"leaf_inverses\":%zu,\"multiplications\":%zu,\"masked_fixture_leaves\":%zu,\"mismatches\":0,\"actual_header\":true,\"gpu_execution\":false}\n",
                totals.blocks,totals.leaves,totals.multiplies,totals.masked);
}
