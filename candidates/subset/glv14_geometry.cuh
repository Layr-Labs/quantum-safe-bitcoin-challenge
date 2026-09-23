// GPL-3.0-only. GLV14 shared-table geometry; split based on bitcoin-core/secp256k1.
#pragma once
#define GT_CHUNKS 7
#define GT_GLV_TERMS 14
#define GT_TOTAL_ENTRIES 1215139u
#define GT_LO 256
#define GT_HI 2048
__host__ __device__ __forceinline__ unsigned gt_entries(int c) {
    return c<2?262144u:c<6?131072u:166563u;
}
__host__ __device__ __forceinline__ unsigned gt_offset(int c) {
    return c==0?0u:c==1?262144u:c==2?524288u:c==3?655360u:
           c==4?786432u:c==5?917504u:1048576u;
}
__host__ __device__ __forceinline__ int gt_shift(int c) {
    return c==0?0:c==1?18:c==2?37:c==3?55:c==4?73:c==5?91:109;
}
static_assert(GT_TOTAL_ENTRIES*64ULL==77768896ULL,"GLV14 table size");
