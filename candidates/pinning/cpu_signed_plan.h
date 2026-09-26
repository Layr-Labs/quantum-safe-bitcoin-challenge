/* Signed-window layout for the optional Pinning host IFMA backend.
 * Exact host scalar recoding and folded-table layout. Independent adapter;
 * direction informed by promoted Subset a137e28 and public pending descriptions.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
namespace qcg_signed_plan {
static constexpr int windows = 13;
// 257 signed bits: ten 20-bit windows, two 19-bit windows, one unsigned top.
// The 256-bit input leaves the top sign bit unused; the recoding carry fits it.
static constexpr unsigned width(int j) { return j < 10 ? 20u : 19u; }
static constexpr unsigned shift(int j) { return j < 10 ? 20u*j : 200u+19u*(j-10); }
static constexpr uint32_t half(int j) { return 1u << (width(j)-1); }
// Explicit infinity row at index 0; magnitude half is a valid point.
static constexpr size_t rows(int j) { return (size_t)half(j)+1; }
static constexpr size_t offset(int j) {
    return j < 10 ? (size_t)j*((1u<<19)+1) :
        10ull*((1u<<19)+1)+(size_t)(j-10)*((1u<<18)+1);
}
static constexpr size_t lower_rows = offset(windows-1);
static constexpr size_t folded_rows = 2ull*rows(windows-1);
static constexpr size_t total_rows = lower_rows+folded_rows;
static constexpr size_t table_bytes = total_rows*64ull;
static_assert(shift(12)==238 && width(12)==19,"full scalar coverage");
static_assert(shift(12)+width(12)==257,"one bit for carry");
static_assert(table_bytes==402654080ull,"explicit zero rows included");

// Encoded as magnitude | (negative << 31), including signed zero as zero.
static inline void digits(uint32_t out[windows],const uint64_t z[4]) {
    uint32_t carry=0;
#pragma GCC unroll 13
    for(int j=0;j<windows;++j) {
        const unsigned bit=shift(j),w=width(j),word=bit>>6,s=bit&63;
        uint64_t v=z[word]>>s;
        if(s+w>64 && word<3)v|=z[word+1]<<(64-s);
        const uint32_t d=((uint32_t)v&((1u<<w)-1))+carry;
        if(j+1<windows) {
            carry=(d>half(j));
            const uint32_t mag=carry?((1u<<w)-d):d;
            out[j]=mag|((carry && mag)?0x80000000u:0u);
        } else out[j]=d; // d<=half(j), no final carry or sign
    }
}
static inline uint32_t magnitude(uint32_t d) { return d&0x7fffffffu; }
static inline bool negative(uint32_t d) { return (d>>31)!=0; }
// Last digit is unsigned. Two banks contain d*B_last+A and d*B_last-A.
// They replace the separate last-window and +/-A additions.
static inline size_t folded_index(uint32_t last,int recid) {
    return lower_rows+(size_t)recid*rows(windows-1)+last;
}
} // namespace qcg_signed_plan
