// CPU OpenSSL equivalence test for host-side block folding; no CUDA required.
#include <openssl/sha.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(4319);
    for (int trial=0; trial<512; ++trial) {
        const size_t prefix_len=9920;
        const int suffix_len=(trial % 2 == 0) ? 75 : 119;
        const int seq_offset=(trial % 3 == 0) ? 60 : 31;
        const int lt_offset=67;
        const uint32_t seq = trial==0 ? 0x80000000u : rng();
        const uint32_t lt = trial==0 ? 500000000u : rng();
        std::vector<uint8_t> preimage(prefix_len+suffix_len);
        for (auto &v : preimage) v=uint8_t(rng());
        for (int j=0;j<4;j++) {
            preimage[prefix_len+seq_offset+j]=uint8_t(seq>>(8*j));
            preimage[prefix_len+lt_offset+j]=uint8_t(lt>>(8*j));
        }
        uint8_t expected[32];
        SHA256(preimage.data(), preimage.size(), expected);
        SHA256_CTX original;
        SHA256_Init(&original);
        for(size_t off=0;off<prefix_len;off+=64)
            SHA256_Transform(&original, preimage.data()+off);
        // Recreate exactly the submitted host continuation: init then assign h.
        SHA256_CTX folded;
        SHA256_Init(&folded);
        for(int j=0;j<8;j++) folded.h[j]=original.h[j];
        uint8_t block[64];
        memcpy(block, preimage.data()+prefix_len, 64);
        for(int j=0;j<4;j++) block[seq_offset+j]=uint8_t(seq>>(8*j));
        SHA256_Transform(&folded, block);
        // Model the CUDA tail construction after subtracting one block.
        std::array<uint8_t,64> tail{};
        const int remaining=suffix_len-64;
        memcpy(tail.data(), preimage.data()+prefix_len+64, remaining);
        for(int j=0;j<4;j++) tail[lt_offset-64+j]=uint8_t(lt>>(8*j));
        tail[remaining]=0x80;
        uint64_t bits=preimage.size()*8;
        for(int j=0;j<8;j++) tail[63-j]=uint8_t(bits>>(8*j));
        SHA256_Transform(&folded, tail.data());
        uint8_t actual[32];
        for(int j=0;j<8;j++) for(int b=0;b<4;b++)
            actual[4*j+b]=uint8_t(folded.h[j]>>(24-8*b));
        assert(memcmp(expected, actual, 32)==0);
    }
    puts("PASS: 512 OpenSSL full-message versus folded-prefix comparisons");
}
