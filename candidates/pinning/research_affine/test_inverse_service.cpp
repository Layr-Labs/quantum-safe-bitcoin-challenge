// SPDX-License-Identifier: GPL-3.0-only
#define QSB_INVERSE_SERVICE_HOST_TEST
#include "inverse_service.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>
struct Warp {
    std::barrier<> gate{32};
    std::array<bool,32> ballot{};
};
thread_local Warp *warp;
thread_local unsigned lane;
bool vote(bool x,bool all) {
    warp->ballot[lane]=x;
    warp->gate.arrive_and_wait();
    bool result=all;
    for(bool b:warp->ballot) result=all?(result&&b):(result||b);
    warp->gate.arrive_and_wait();
    return result;
}
bool qsb_inverse_test_any(bool x) { return vote(x,false); }
bool qsb_inverse_test_all(bool x) { return vote(x,true); }
std::atomic<unsigned> callback_count{0};
struct AuditedTransform {
    void operator()(uint64_t out[4],const uint64_t in[4]) const {
        // A deterministic 256-bit transform checks transport of every bit. This
        // is deliberately not a field inversion or a throughput simulation.
        callback_count.fetch_add(1,std::memory_order_relaxed);
        for(unsigned j=0;j<4;++j) out[j]=(in[j]^UINT64_C(0xd19cf5a9738b624e)) + j;
        if((in[0]&7)==2) std::this_thread::yield();
    }
};
void exercise(unsigned workers,unsigned epochs) {
    using namespace qsb_inverse_service;
    const unsigned warps=(workers+31)/32+1; // includes an entirely unused warp
    std::vector<Slot> slots(workers);
    std::vector<std::unique_ptr<Warp>> groups;
    for(unsigned w=0;w<warps;++w) groups.emplace_back(new Warp);
    std::vector<std::thread> servers,producers;
    for(unsigned t=0;t<warps*32;++t) servers.emplace_back([&,t] {
        warp=groups[t/32].get();lane=t%32;
        service(slots.data(),workers,t,AuditedTransform{});
    });
    for(unsigned t=0;t<workers;++t) producers.emplace_back([&,t] {
        // Different finite lengths force stopped and busy lanes to coexist.
        const unsigned count=epochs==0?0:1+(t%epochs);
        for(unsigned e=1;e<=count;++e) {
            uint64_t root[4],got[4],want[4];
            for(unsigned j=0;j<4;++j)
                root[j]=(uint64_t(t+1)<<48) ^ (uint64_t(e)<<16) ^ (UINT64_C(0xf301d5878a629bc4)+j);
            if((e+t)%3==0) std::this_thread::yield();
            worker_publish(slots[t],e,root);
            worker_wait(slots[t],e,got);
            for(unsigned j=0;j<4;++j) {
                want[j]=(root[j]^UINT64_C(0xd19cf5a9738b624e))+j;
                assert(got[j]==want[j]);
            }
            if((e+t)%4==0) std::this_thread::yield();
        }
        worker_stop(slots[t]);
    });
    for(auto &t:producers)t.join();
    for(auto &t:servers)t.join();
    for(unsigned t=0;t<workers;++t) {
        assert(acquire(slots[t].request)==STOP);
        assert(acquire(slots[t].response)==(epochs==0?0:1+t%epochs));
    }
}
int main() {
    for(unsigned n:{0u,1u,3u,31u,32u,33u,63u}) exercise(n,8);
    exercise(33,0);
    printf("actual-header CPU protocol: PASS; 8 configurations; callback calls %u (includes idle identities)\n",callback_count.load());
}
