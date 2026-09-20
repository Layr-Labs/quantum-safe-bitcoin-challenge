// Four-epoch SHA phase derived from the promoted paired scheduler (043b650).
// Retain upstream dukemawex/terrapinelf and GPL notices. This file changes
// scheduling and storage only; field recovery and exact hit replay are unchanged.
#pragma once
#if QSB_SHA_PIPELINE && QSB_PAIR_SHARED && ZLAB_K2S3M && ZLAB_DUAL_EPOCH_SHA
__device__ __forceinline__ void qsb_scheduled_window_hash_quad(
    uint32_t *stateA, uint32_t *stateB, uint32_t *stateC, uint32_t *stateD, int lane,
    const uint32_t *firstA, const uint32_t *firstB, const uint32_t *firstC, const uint32_t *firstD) {
    const int first_slot=QSB_FIRST_CLASS[lane];
    const int slot=QSB_WINDOW_CLASS[lane];
    #pragma unroll
    for(int j=0;j<8;j++){
        stateA[j]=firstA[first_slot*8+j];
        stateB[j]=firstB[first_slot*8+j];
        stateC[j]=firstC[first_slot*8+j];
        stateD[j]=firstD[first_slot*8+j];
    }
    uint32_t a0,b0,c0,d0,e0,f0,g0,h0;
    uint32_t a1,b1,c1,d1,e1,f1,g1,h1,t1,t2;
    uint32_t a2,b2,c2,d2,e2,f2,g2,h2;
    uint32_t a3,b3,c3,d3,e3,f3,g3,h3;
#define QSB_QUAD_STATE_LOAD() do { \
    a0=stateA[0];b0=stateA[1];c0=stateA[2];d0=stateA[3]; \
    e0=stateA[4];f0=stateA[5];g0=stateA[6];h0=stateA[7]; \
    a1=stateB[0];b1=stateB[1];c1=stateB[2];d1=stateB[3]; \
    e1=stateB[4];f1=stateB[5];g1=stateB[6];h1=stateB[7]; \
    a2=stateC[0];b2=stateC[1];c2=stateC[2];d2=stateC[3]; \
    e2=stateC[4];f2=stateC[5];g2=stateC[6];h2=stateC[7]; \
    a3=stateD[0];b3=stateD[1];c3=stateD[2];d3=stateD[3]; \
    e3=stateD[4];f3=stateD[5];g3=stateD[6];h3=stateD[7]; \
} while(0)
#define QSB_QUAD_STATE_ADD() do { \
    stateA[0]+=a0;stateA[1]+=b0;stateA[2]+=c0;stateA[3]+=d0; \
    stateA[4]+=e0;stateA[5]+=f0;stateA[6]+=g0;stateA[7]+=h0; \
    stateB[0]+=a1;stateB[1]+=b1;stateB[2]+=c1;stateB[3]+=d1; \
    stateB[4]+=e1;stateB[5]+=f1;stateB[6]+=g1;stateB[7]+=h1; \
    stateC[0]+=a2;stateC[1]+=b2;stateC[2]+=c2;stateC[3]+=d2; \
    stateC[4]+=e2;stateC[5]+=f2;stateC[6]+=g2;stateC[7]+=h2; \
    stateD[0]+=a3;stateD[1]+=b3;stateD[2]+=c3;stateD[3]+=d3; \
    stateD[4]+=e3;stateD[5]+=f3;stateD[6]+=g3;stateD[7]+=h3; \
} while(0)
    QSB_QUAD_STATE_LOAD();
    #pragma unroll 1
    for(int r=0;r<64;r+=8){
        {const uint32_t w=QSB_WINDOW_SECOND[r][slot];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);S2Round(a2,b2,c2,d2,e2,f2,g2,h2,0,w);S2Round(a3,b3,c3,d3,e3,f3,g3,h3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+1][slot];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);S2Round(h2,a2,b2,c2,d2,e2,f2,g2,0,w);S2Round(h3,a3,b3,c3,d3,e3,f3,g3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+2][slot];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);S2Round(g2,h2,a2,b2,c2,d2,e2,f2,0,w);S2Round(g3,h3,a3,b3,c3,d3,e3,f3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+3][slot];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);S2Round(f2,g2,h2,a2,b2,c2,d2,e2,0,w);S2Round(f3,g3,h3,a3,b3,c3,d3,e3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+4][slot];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);S2Round(e2,f2,g2,h2,a2,b2,c2,d2,0,w);S2Round(e3,f3,g3,h3,a3,b3,c3,d3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+5][slot];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);S2Round(d2,e2,f2,g2,h2,a2,b2,c2,0,w);S2Round(d3,e3,f3,g3,h3,a3,b3,c3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+6][slot];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);S2Round(c2,d2,e2,f2,g2,h2,a2,b2,0,w);S2Round(c3,d3,e3,f3,g3,h3,a3,b3,0,w);}
        {const uint32_t w=QSB_WINDOW_SECOND[r+7][slot];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);S2Round(b2,c2,d2,e2,f2,g2,h2,a2,0,w);S2Round(b3,c3,d3,e3,f3,g3,h3,a3,0,w);}
    }
    QSB_QUAD_STATE_ADD();
#if QSB_PAIR_SHA_UNROLL_CONST
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int block=0;block<4;block++){
        QSB_QUAD_STATE_LOAD();
#if QSB_PAIR_SHA_UNROLL_CONST
        #pragma unroll
#else
        #pragma unroll 1
#endif
        for(int r=0;r<64;r+=8){
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r];S2Round(a0,b0,c0,d0,e0,f0,g0,h0,0,w);S2Round(a1,b1,c1,d1,e1,f1,g1,h1,0,w);S2Round(a2,b2,c2,d2,e2,f2,g2,h2,0,w);S2Round(a3,b3,c3,d3,e3,f3,g3,h3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+1];S2Round(h0,a0,b0,c0,d0,e0,f0,g0,0,w);S2Round(h1,a1,b1,c1,d1,e1,f1,g1,0,w);S2Round(h2,a2,b2,c2,d2,e2,f2,g2,0,w);S2Round(h3,a3,b3,c3,d3,e3,f3,g3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+2];S2Round(g0,h0,a0,b0,c0,d0,e0,f0,0,w);S2Round(g1,h1,a1,b1,c1,d1,e1,f1,0,w);S2Round(g2,h2,a2,b2,c2,d2,e2,f2,0,w);S2Round(g3,h3,a3,b3,c3,d3,e3,f3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+3];S2Round(f0,g0,h0,a0,b0,c0,d0,e0,0,w);S2Round(f1,g1,h1,a1,b1,c1,d1,e1,0,w);S2Round(f2,g2,h2,a2,b2,c2,d2,e2,0,w);S2Round(f3,g3,h3,a3,b3,c3,d3,e3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+4];S2Round(e0,f0,g0,h0,a0,b0,c0,d0,0,w);S2Round(e1,f1,g1,h1,a1,b1,c1,d1,0,w);S2Round(e2,f2,g2,h2,a2,b2,c2,d2,0,w);S2Round(e3,f3,g3,h3,a3,b3,c3,d3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+5];S2Round(d0,e0,f0,g0,h0,a0,b0,c0,0,w);S2Round(d1,e1,f1,g1,h1,a1,b1,c1,0,w);S2Round(d2,e2,f2,g2,h2,a2,b2,c2,0,w);S2Round(d3,e3,f3,g3,h3,a3,b3,c3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+6];S2Round(c0,d0,e0,f0,g0,h0,a0,b0,0,w);S2Round(c1,d1,e1,f1,g1,h1,a1,b1,0,w);S2Round(c2,d2,e2,f2,g2,h2,a2,b2,0,w);S2Round(c3,d3,e3,f3,g3,h3,a3,b3,0,w);}
            {const uint32_t w=QSB_CONST_SCHEDULE[block][r+7];S2Round(b0,c0,d0,e0,f0,g0,h0,a0,0,w);S2Round(b1,c1,d1,e1,f1,g1,h1,a1,0,w);S2Round(b2,c2,d2,e2,f2,g2,h2,a2,0,w);S2Round(b3,c3,d3,e3,f3,g3,h3,a3,0,w);}
        }
        QSB_QUAD_STATE_ADD();
    }
#undef QSB_QUAD_STATE_LOAD
#undef QSB_QUAD_STATE_ADD
}

// One lane handles the same omission lane in four epochs. No shared state or
// block barrier is needed. Scalar planes are coalesced at both producer/consumer.
__global__ void __launch_bounds__(256, 2) kernel_epoch_scalar_quad(
    const uint32_t *__restrict__ first, uint64_t *__restrict__ scalars, int epochs) {
    const unsigned ep=4u*blockIdx.x;
    if(ep>=(unsigned)epochs)return;
    const unsigned lane=threadIdx.x;
    const size_t stride=(size_t)QSB_FIRST_SLOTS*8;
    const uint32_t *f0=first+(size_t)ep*stride;
    const uint32_t *f1=(ep+1u<(unsigned)epochs)?f0+stride:f0;
    const uint32_t *f2=(ep+2u<(unsigned)epochs)?f0+2*stride:f0;
    const uint32_t *f3=(ep+3u<(unsigned)epochs)?f0+3*stride:f0;
    uint32_t s0[8],s1[8],s2[8],s3[8];
    qsb_scheduled_window_hash_quad(s0,s1,s2,s3,(int)lane,f0,f1,f2,f3);
    uint64_t z[4];
    qsb_pair_second_sha_z(s0,z);
    #pragma unroll
    for(int k=0;k<4;k++)scalars[((size_t)ep*4+k)*256+lane]=z[k];
    if(ep+1u<(unsigned)epochs){
        qsb_pair_second_sha_z(s1,z);
        #pragma unroll
        for(int k=0;k<4;k++)scalars[((size_t)(ep+1u)*4+k)*256+lane]=z[k];
    }
    if(ep+2u<(unsigned)epochs){
        qsb_pair_second_sha_z(s2,z);
        #pragma unroll
        for(int k=0;k<4;k++)scalars[((size_t)(ep+2u)*4+k)*256+lane]=z[k];
    }
    if(ep+3u<(unsigned)epochs){
        qsb_pair_second_sha_z(s3,z);
        #pragma unroll
        for(int k=0;k<4;k++)scalars[((size_t)(ep+3u)*4+k)*256+lane]=z[k];
    }
}
#endif
