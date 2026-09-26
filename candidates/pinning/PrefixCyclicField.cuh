// SPDX-License-Identifier: GPL-3.0-only
// Python-validated integration, no native compilation or GPU execution. Four fields per warp.
// Include after CyclicField.cuh for its full-carry normalize8 helper.
#pragma once
namespace qsb_prefix_cyclic_research {
constexpr unsigned full=0xffffffffu;

// Exactly add cf*(2^32+977). cf<=1. When cf==1, the caller proves t<2^65,
// so word2<=1 and no propagation beyond word2 is possible. cf==0 is identity.
__device__ __forceinline__ uint32_t finish3(uint32_t t,uint32_t cf,unsigned lane){
    const unsigned d=lane&7u;
    const uint32_t s0=t+(d==0?977u*cf:0u);
    const uint32_t c0=__shfl_sync(full,(uint32_t)(s0<t),0,8);
    const uint32_t s1=s0+(d==1?cf+c0:0u);
    const uint32_t c1=__shfl_sync(full,(uint32_t)(s1<s0),1,8);
    return s1+(d==2?c1:0u);
}

__device__ __forceinline__ uint32_t multiply8(uint32_t a,uint32_t b,unsigned lane){
    const unsigned d=lane&7u;
    uint64_t total=0,prefix=0;
    // Low byte: total carry beyond bit63 (<=7). Next byte: captured prefix
    // carry. The low byte cannot overflow into the snapshot byte.
    uint32_t counts=0;
    #pragma unroll
    for(unsigned i=0;i<8;++i){
        const uint32_t av=__shfl_sync(full,a,i,8);
        const uint32_t bv=__shfl_sync(full,b,(d-i)&7u,8);
        const uint64_t product=(uint64_t)av*bv;
        // Read external predicate operands before changing outputs. Every lane
        // executes the sum; lane d captures its prefix precisely at i==d.
        asm("{ .reg .pred take;\n\t"
            "setp.eq.u32 take, %4, %5;\n\t"
            "add.cc.u64 %0, %0, %3;\n\t"
            "addc.u32 %1, %1, 0;\n\t"
            "@take mov.u64 %2, %0;\n\t"
            "@take bfi.b32 %1, %1, %1, 8, 8;\n\t}"
            : "+l"(total),"+r"(counts),"+l"(prefix)
            : "l"(product),"r"(d),"r"(i));
    }
    uint64_t upper;
    uint32_t upper_count;
    const uint32_t lower_count=(counts>>8)&255u,total_count=counts&255u;
    asm("{ sub.cc.u64 %0, %2, %3;\n\tsubc.u32 %1, %4, %5; }"
        : "=l"(upper),"=r"(upper_count)
        : "l"(total),"l"(prefix),"r"(total_count),"r"(lower_count));
    const uint32_t lo0=(uint32_t)prefix,lo1=(uint32_t)(prefix>>32);
    const uint32_t hi0=(uint32_t)upper,hi1=(uint32_t)(upper>>32);
    const uint32_t lp1=__shfl_up_sync(full,lo1,1,8);
    const uint32_t hp1=__shfl_up_sync(full,hi1,1,8);
    const uint32_t llast=__shfl_sync(full,lo1,7,8);
    const uint32_t lp2=__shfl_up_sync(full,lower_count,2,8);
    const uint32_t hp2=__shfl_up_sync(full,upper_count,2,8);
    const uint32_t lwrap=__shfl_sync(full,lower_count,(6u+d)&7u,8);
    // A normalized 96-bit column contributes to three adjacent radix words.
    const uint64_t low=(uint64_t)lo0+(d?lp1:0u)+(d>=2?lp2:0u);
    const uint64_t high=(uint64_t)hi0+(d?hp1:llast)+(d>=2?hp2:lwrap);
    const uint64_t hp=__shfl_up_sync(full,high,1,8);
    const uint64_t extra=__shfl_sync(full,high,7,8);
    uint64_t column=low+977ull*high+(d?hp:0ull);
    uint32_t top;
    uint32_t t=qsb_cyclic_field_research::normalize8(column,lane,top);
    const uint64_t tail=extra+(uint64_t)top;
    const uint32_t q0=(uint32_t)tail,q1=(uint32_t)(tail>>32);
    column=t;
    if(d==0)column+=977ull*q0;
    if(d==1)column+=(uint64_t)q0+977ull*q1;
    if(d==2)column+=q1;
    t=qsb_cyclic_field_research::normalize8(column,lane,top);
    return finish3(t,top,lane);
}
} // namespace qsb_prefix_cyclic_research
