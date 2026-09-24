#pragma once
#include <stdint.h>
#include <stddef.h>

// Promoted pinning FOUR_HOT GLV12 geometry only; subset's exact split and positive-Y
// field/anchor convention remain unchanged. File/API names retain native ABI.
#ifdef __CUDACC__
#define GLV10_HD __host__ __device__ __forceinline__
#else
#define GLV10_HD static inline
#endif
static constexpr unsigned GLV10_CHUNKS=6;
static constexpr unsigned GLV10_TERMS=2*GLV10_CHUNKS;
static constexpr unsigned GLV10_LOW_BITS=14;
static constexpr unsigned GLV10_LADDER=1u<<GLV10_LOW_BITS;
static constexpr unsigned GLV10_LOW_MASK=GLV10_LADDER-1;
static constexpr unsigned GLV10_TOP_CENTER=170559769;
GLV10_HD constexpr unsigned glv10_entries(int c) {
    return c<2?262144u:c<4?131072u:c==4?67108864u:85279885u;
}
GLV10_HD constexpr unsigned glv10_offset(int c) {
    return c==0?0u:c==1?262144u:c==2?524288u:c==3?655360u:
           c==4?786432u:67895296u;
}
GLV10_HD constexpr unsigned glv10_shift(int c) {
    return c==0?0u:c==1?18u:c==2?37u:c==3?55u:c==4?73u:100u;
}
GLV10_HD constexpr unsigned glv10_width(int c) {
    return c==0?18u:c==1?19u:c<4?18u:27u;
}
GLV10_HD constexpr int glv10_physical_bank(int position) {
    return position;
}
static constexpr uint64_t GLV10_RECORDS=153175181ULL;
static constexpr size_t GLV10_BYTES=(size_t)GLV10_RECORDS*64;
static constexpr size_t GLV10_DENSE_BYTES=(size_t)(glv10_entries(0)+glv10_entries(1)+glv10_entries(2)+glv10_entries(3))*64;
static constexpr unsigned GLV10_BIAS_COEFFICIENT=GLV10_TOP_CENTER+1;
static constexpr unsigned GLV10_BIAS_SHIFT=glv10_shift(GLV10_CHUNKS-1)-1;
static constexpr unsigned GLV10_BIAS_SUBTRACT=1u<<(glv10_width(0)-1);
// Keep interval decoding explicit; FOUR_HOT physical banks are 0,1,2,3,4,5.
GLV10_HD constexpr int glv10_bank_for_record(uint64_t record) {
    for(int position=0;position<(int)GLV10_CHUNKS;++position) {
        const int c=glv10_physical_bank(position);
        if(record>=glv10_offset(c) && record-glv10_offset(c)<glv10_entries(c))return c;
    }
    return -1;
}
static_assert(sizeof(size_t)>=8,"table byte addressing requires 64 bits");
static_assert(GLV10_BYTES==9803211584ULL,"GLV12 table size");
static_assert(GLV10_RECORDS<(1ULL<<31),"record index overlaps sign bit");
static_assert(GLV10_DENSE_BYTES==48ULL*1024*1024,"contiguous dense banks");
static_assert(glv10_offset(1)==glv10_entries(0) &&
              glv10_offset(2)==glv10_offset(1)+glv10_entries(1) &&
              glv10_offset(3)==glv10_offset(2)+glv10_entries(2) &&
              glv10_offset(4)==glv10_offset(3)+glv10_entries(3) &&
              glv10_offset(5)==glv10_offset(4)+glv10_entries(4) &&
              GLV10_RECORDS==glv10_offset(5)+glv10_entries(5),"physical bank intervals");
static_assert((GLV10_TOP_CENTER>>GLV10_LOW_BITS)<GLV10_LADDER &&
              (((1u<<glv10_width(4))-1)>>GLV10_LOW_BITS)<GLV10_LADDER &&
              ((glv10_entries(0)-1)>>GLV10_LOW_BITS)<GLV10_LADDER,"ladder bounds");
GLV10_HD uint32_t glv10_code(const uint64_t mag[2],unsigned sign,int c) {
    const unsigned shift=glv10_shift(c);
    uint64_t wide;
    if(shift<64u) {
        wide=mag[0]>>shift;
        if(shift)wide|=mag[1]<<(64u-shift);
    } else wide=mag[1]>>(shift-64u);
    uint32_t f=(uint32_t)wide,idx,negative;
    if(c==0) {
        idx=f&(glv10_entries(0)-1);negative=0;
    } else if(c==(int)GLV10_CHUNKS-1) {
        const int32_t d=(int32_t)(2u*f)-(int32_t)GLV10_TOP_CENTER;
        negative=(uint32_t)d>>31;
        const uint32_t magnitude=((uint32_t)d^(0u-negative))+negative;
        idx=(magnitude-1u)>>1;
    } else {
        const unsigned bits=glv10_width(c);
        f&=(1u<<bits)-1u;
        negative=1u-(f>>(bits-1u));
        idx=(f^(0u-negative))&(glv10_entries(c)-1u);
    }
    return (glv10_offset(c)+idx)|((negative^sign)<<31);
}
#undef GLV10_HD
