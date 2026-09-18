#!/usr/bin/env python3
"""Run the actual checkpoint helper bodies in a CPU CUDA-barrier emulator.

Requires C++17, pthreads and OpenSSL development headers. Does not compile CUDA
or establish device performance. --sanitize-thread additionally checks host races.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def helper(source, name):
    marker = "template<int N>\n__device__ __forceinline__ void " + name + "("
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


MAIN = r'''
template<int N> void audit(int active, int test, std::mt19937_64 &rng) {
    using Value=std::array<uint64_t,5>;
    std::vector<Value> values(N), expected(N);
    uint64_t products[4][2*N] = {};
    // Alternate nonzero block offsets to exercise global checkpoint indexing.
    unsigned group=test&1;
    std::vector<uint64_t> roots(8,0), checkpoint(8*N,0);
    for(int i=0;i<N;i++) {
        Value v={rng(),rng(),rng(),rng(),0};
        if(i%19==0)v={0,0,0,0,0};
        if(i%23==0)v={0xFFFFFFFEFFFFFC2EULL,UINT64_MAX,UINT64_MAX,UINT64_MAX,0};
        // Same identity substitution as the production caller.
        if(i>=active || !(v[0]|v[1]|v[2]|v[3]))v={1,0,0,0,0};
        values[i]=expected[i]=v;
        reference_inverse(expected[i].data());
    }
    unsigned up_cta=0,down_cta=0,up_warp=0,down_warp=0;
    for(int phase=0;phase<2;phase++) {
        Block block(N);
        std::vector<std::thread> threads;
        for(int i=0;i<N;i++) threads.emplace_back([&,i]{
            threadIdx.x=i; blockIdx.x=group; current_block=&block;
            // Mixed yields disturb warp arrival.
            if((i+test)%3==0)std::this_thread::yield();
            if(phase==0)qsb_block_product_checkpoint<N>(values[i].data(),roots.data(),checkpoint.data(),products);
            else qsb_block_inverse_checkpoint<N>(values[i].data(),roots.data(),checkpoint.data());
        });
        for(auto &t:threads)t.join();
        unsigned cta=block.all.calls/N,warp=block.warps[0]->calls/32;
        if(phase==0) {
            up_cta=cta; up_warp=warp;
            unsigned expect_prepare = (N==64 ? 1u : (N==128 ? 2u : 3u));
            if(up_cta != expect_prepare || up_warp != 0) {
                std::fprintf(stderr,"FAIL prepare barriers width=%d got block/warp=%u/%u\n",N,up_cta,up_warp);
                std::abort();
            }
            // Compare the checkpoint root with a direct serial product.
            Value product={1,0,0,0,0};
            for(const auto &v:values)qsb_field_mul(product.data(),product.data(),v.data());
            if(std::memcmp(product.data(),roots.data()+4*group,32))std::abort();
            reference_inverse(product.data());
            std::memcpy(roots.data()+4*group,product.data(),32);
        } else {
            down_cta=cta; down_warp=warp;
            unsigned expect_finish = (N==64 ? 2u : (N==128 ? 4u : 6u));
            if(down_cta != expect_finish || down_warp != 0) {
                std::fprintf(stderr,"FAIL finish barriers width=%d got block/warp=%u/%u\n",N,down_cta,down_warp);
                std::abort();
            }
        }
    }
    for(int i=0;i<N;i++)if(values[i]!=expected[i]) {
        std::fprintf(stderr,"FAIL width=%d active=%d lane=%d\n",N,active,i);std::abort();
    }
    if(test==0)std::printf("width=%d prepare block/warp=%u/%u finish block/warp=%u/%u\n",N,up_cta,up_warp,down_cta,down_warp);
}
template<int N> int run(std::mt19937_64 &rng) {
    int total=0,test=0;
    for(int active: {0,1,31,32,33,N/2-1,N/2,N-1,N}) {
        audit<N>(active,test++,rng);total+=N;
    }
    return total;
}
int main() {
    std::mt19937_64 rng(0x7173626261727269ULL);
    int total=run<64>(rng)+run<128>(rng)+run<256>(rng);
    std::printf("PASS: 27 actual-source checkpoint cases; %d inverses match OpenSSL; widths 64/128/256; partial/identity lanes\n",total);
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize-thread", action="store_true")
    parser.add_argument("--openssl-prefix")
    parser.add_argument("--source", type=Path, default=ROOT / "pinning.cu")
    args = parser.parse_args()
    source = args.source.read_text()
    functions = [helper(source, name) for name in
                 ("qsb_block_product_checkpoint", "qsb_block_inverse_checkpoint")]
    support = (ROOT / "tests/checkpoint_barriers_host.hpp").read_text()
    prefix = args.openssl_prefix
    if not prefix and Path("/opt/homebrew/opt/openssl@3/include/openssl/bn.h").exists():
        prefix = "/opt/homebrew/opt/openssl@3"
    with tempfile.TemporaryDirectory(prefix="qsb-checkpoint-audit-") as directory:
        cpp = Path(directory) / "audit.cpp"
        binary = Path(directory) / "audit"
        cpp.write_text(support + "\n" + "\n".join(functions) + MAIN)
        command = shlex.split(os.environ.get("CXX", "c++"))
        command += ["-std=c++17", "-O1", "-pthread", "-Wno-unknown-pragmas"]
        if args.sanitize_thread:
            command += ["-fsanitize=thread", "-g"]
        if prefix:
            command += [f"-I{prefix}/include", f"-L{prefix}/lib", f"-Wl,-rpath,{prefix}/lib"]
        command += [str(cpp), "-lcrypto", "-o", str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True, timeout=120)


if __name__ == "__main__":
    main()
