// Host execution of the actual CUDA helper's control flow. This is a schedule
// audit over a small prime, not a substitute for the secp256k1 CUDA audit.
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class Barrier {
    std::mutex mutex;
    std::condition_variable changed;
    const int size;
    int arrived = 0, generation = 0;
public:
    explicit Barrier(int size) : size(size) {}
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const int previous = generation;
        if (++arrived == size) {
            arrived = 0;
            ++generation;
            changed.notify_all();
        } else {
            changed.wait(lock, [&] { return generation != previous; });
        }
    }
};

struct Dim { int x; };
thread_local Dim threadIdx, blockDim;
static std::unique_ptr<Barrier> block_barrier;
static std::vector<std::unique_ptr<Barrier>> warp_barriers;
static void __syncthreads() { block_barrier->wait(); }
static void __syncwarp() { warp_barriers[threadIdx.x / 32]->wait(); }

static constexpr uint64_t prime = (uint64_t(1) << 61) - 1;
static uint64_t multiply(uint64_t a, uint64_t b) {
    return (static_cast<__uint128_t>(a) * b) % prime;
}
static uint64_t inverse(uint64_t a) {
    uint64_t result = 1;
    for (uint64_t exponent = prime - 2; exponent; exponent >>= 1) {
        if (exponent & 1) result = multiply(result, a);
        a = multiply(a, a);
    }
    return result;
}
static void fill(uint64_t *out, uint64_t v) {
    // Redundant limbs catch failures to propagate any of the four words.
    for (int k = 0; k < 4; ++k) out[k] = v;
    out[4] = 0;
}
static void check(const uint64_t *v) {
    for (int k = 1; k < 4; ++k) assert(v[k] == v[0]);
    assert(v[4] == 0);
}
static void qsb_field_mul_raw(uint64_t *out, uint64_t *a, uint64_t *b) {
    check(a); check(b);
    fill(out, multiply(a[0], b[0]));
}
static void qsb_field_normalize(uint64_t *a) { check(a); }
static void _ModInv(uint64_t *a) {
    check(a);
    assert(a[0] != 0);
    fill(a, inverse(a[0]));
}

#define __device__
#define __forceinline__ inline
#define __shared__ static
#ifndef ZLAB_TREE
#define ZLAB_TREE 3
#endif
#include "tree_inverse.cuh"

int main() {
    int checked = 0;
    for (int n : {32, 64, 128, 256}) {
        for (int active : {1, n - 1, n}) {
            block_barrier.reset(new Barrier(n));
            warp_barriers.clear();
            for (int w = 0; w < n / 32; ++w)
                warp_barriers.emplace_back(new Barrier(32));
            std::vector<uint64_t> inputs(n), results(n * 5);
            for (int t = 0; t < n; ++t)
                inputs[t] = t >= active ? 1 :
                    (t % 3 == 0 ? prime - 1 : uint64_t(t + 1) * 987654321);
            std::vector<std::thread> workers;
            for (int t = 0; t < n; ++t) workers.emplace_back([&, t] {
                threadIdx.x = t;
                blockDim.x = n;
                uint64_t value[5];
                fill(value, inputs[t]);
                if (t % 7 == 0) std::this_thread::yield();
                qsb_block_inverse_tree(value);
                for (int k = 0; k < 5; ++k) results[5 * t + k] = value[k];
            });
            for (auto &worker : workers) worker.join();
            for (int t = 0; t < n; ++t) {
                check(&results[5 * t]);
                assert(multiply(inputs[t], results[5 * t]) == 1);
                ++checked;
            }
        }
    }
    printf("variant %d: %d lane inverses passed; full and partial blocks, 32..256 threads\n",
           ZLAB_TREE, checked);
}
