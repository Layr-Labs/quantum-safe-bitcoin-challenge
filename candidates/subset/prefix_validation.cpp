// Differential and boundary test of the exact candidate helper, plus microbench.
#include "PrefixRuns.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <stdexcept>
static constexpr size_t P = 10;
static size_t original(uint8_t *out, const uint8_t *src, int cut, int begin,
                       const uint8_t *skip, int count, int next) {
    size_t written=0;
    for(int i=begin;i<cut;++i) {
        if(next<count && skip[next]==i) { ++next; continue; }
        memcpy(out+written,src+(size_t)i*P,P); written+=P;
    }
    return written;
}
struct Case { int cut, begin, count, next; std::array<uint8_t,16> skip; };
static volatile uint64_t sink=0;
static void check(const Case &c,const std::vector<uint8_t> &src) {
    std::vector<uint8_t> before(2048,0xa7),after=before,expect=before;
    const size_t offset=37; // Preserve an existing prefix and guards on both sides.
    const size_t n0=original(before.data()+offset,src.data(),c.cut,c.begin,c.skip.data(),c.count,c.next);
    const size_t n1=qsb_prefix::copy_kept_runs<P>(after.data()+offset,src.data(),c.cut,c.begin,c.skip.data(),c.count,c.next);
    size_t at=offset;
    for(int i=c.begin;i<c.cut;++i) {
        bool omitted=false;
        for(int j=0;j<c.count;++j) if(c.skip[j]==i) omitted=true;
        if(!omitted) for(size_t byte=0;byte<P;++byte) expect[at++]=src[(size_t)i*P+byte];
    }
    if(n0!=n1 || n1!=at-offset || before!=after || after!=expect)
        throw std::runtime_error("Prefix output/length/guard mismatch");
}
int main() {
    std::mt19937 rng(20261004);
    std::vector<uint8_t> src(256*P);
    for(auto &b:src) b=(uint8_t)rng();
    uint64_t checks=0;
    // Exhaustive small omission sets and every legal incremental restart.
    for(int cut=0;cut<=12;++cut) for(unsigned mask=0;mask<(1u<<cut);++mask) {
        Case c{}; c.cut=cut;
        for(int i=0;i<cut;++i) if(mask&(1u<<i)) c.skip[c.count++]=(uint8_t)i;
        for(int start=0;start<=cut;++start) {
            c.begin=start;c.next=0;
            while(c.next<c.count && c.skip[c.next]<start) ++c.next;
            check(c,src);++checks;
        }
    }
    std::vector<Case> cases;
    for(int round=0;round<60000;++round) {
        Case c{};c.cut=1+(int)(rng()%150);c.count=std::min(c.cut,(int)(rng()%10));
        std::vector<int> indices(c.cut);std::iota(indices.begin(),indices.end(),0);
        std::shuffle(indices.begin(),indices.end(),rng);
        std::sort(indices.begin(),indices.begin()+c.count);
        for(int i=0;i<c.count;++i)c.skip[i]=(uint8_t)indices[i];
        c.begin=(int)(rng()%(c.cut+1));
        while(c.next<c.count && c.skip[c.next]<c.begin) ++c.next;
        check(c,src);++checks;
        if(c.cut==137 || cases.size()<128)cases.push_back(c);
    }
    cases.clear();
    for(int i=0;i<128;++i) {
        Case c{};c.cut=137;c.count=6;
        std::vector<int> ix(c.cut);std::iota(ix.begin(),ix.end(),0);std::shuffle(ix.begin(),ix.end(),rng);
        std::sort(ix.begin(),ix.begin()+6);for(int j=0;j<6;++j)c.skip[j]=(uint8_t)ix[j];
        c.next=i%5?5:0;c.begin=c.next?c.skip[c.next]:0;cases.push_back(c);
    }
    auto bench=[&](bool runs) {
        std::array<uint8_t,2048> out{};
        const auto start=std::chrono::steady_clock::now();
        uint64_t sum=0;
        for(int j=0;j<20000;++j) for(const auto &c:cases) {
            const size_t n=runs?qsb_prefix::copy_kept_runs<P>(out.data(),src.data(),c.cut,c.begin,c.skip.data(),c.count,c.next)
                               :original(out.data(),src.data(),c.cut,c.begin,c.skip.data(),c.count,c.next);
            sum+=n?out[(size_t)j%n]:0;
        }
        sink+=sum;
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    std::vector<double> old_times,new_times;
    for(int r=0;r<7;++r) {
        if(r%2) {new_times.push_back(bench(true));old_times.push_back(bench(false));}
        else {old_times.push_back(bench(false));new_times.push_back(bench(true));}
    }
    std::sort(old_times.begin(),old_times.end());std::sort(new_times.begin(),new_times.end());
    std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"original_seconds_median\":"<<old_times[3]
             <<",\"runs_seconds_median\":"<<new_times[3]<<",\"microbench_speedup\":"<<old_times[3]/new_times[3]
             <<",\"ranked_gpu_result\":null,\"compiler\":\"MSVC O2 x64\"}\n";
}
