// Independently derived from public PR700 description; complete products.
// Full-width canonical multiplication is required for the new association.
// No native compilation or GPU performance claim.
#pragma once

template<int N> __device__ __forceinline__ void qsb_top16_exact(
    uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    static_assert(N>=32 && !(N&(N-1)), "top16 requires power-of-two N>=32");
    const int tid=threadIdx.x;
    uint64_t accumulated[4];
    #pragma unroll
    for(int span=8;span>=1;span>>=1) {
        // The multiplication must have one common call site per wave.
        const bool low=(tid<16 && span<8);
        const bool upper=(tid>=16 && tid<16+span);
        if(low || upper) {
            uint64_t a[5],b[5],out[5];
            const int source=2*N-4*span;
            if(low) {
                #pragma unroll
                for(int k=0;k<4;k++) {
                    a[k]=span==4 ? products[k][2*N-32+(tid^8)] : accumulated[k];
                    b[k]=products[k][source+((tid&(2*span-1))^span)];
                }
            } else {
                const int j=tid-16;
                #pragma unroll
                for(int k=0;k<4;k++) {
                    a[k]=products[k][source+j];
                    b[k]=products[k][source+j+span];
                }
            }
            a[4]=b[4]=0;
            qsb_field_mul(out,a,b);
            qsb_field_normalize(out);
            if(low) {
                #pragma unroll
                for(int k=0;k<4;k++)accumulated[k]=out[k];
            } else if(span==1) {
                #pragma unroll
                for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=out[k];
            } else {
                #pragma unroll
                for(int k=0;k<4;k++)products[k][2*N-2*span+tid-16]=out[k];
            }
        }
        __syncwarp();
    }
    if(tid<16) {
        #pragma unroll
        for(int k=0;k<4;k++)excluded[k][N-32+tid]=accumulated[k];
    }
    __syncwarp();
}
// Integration contract: original upward sweep stops at count>16; its last
// barrier publishes x[0..15] at products[2*N-32 .. 2*N-17]. Call this helper,
// then resume original exclusion sweep at count=32, offset=2*N-64. Keep its
// cross-warp barriers and final leaf expansion. No extra shared allocation.
