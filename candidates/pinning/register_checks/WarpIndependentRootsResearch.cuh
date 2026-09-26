// SPDX-License-Identifier: GPL-3.0-only
// RESEARCH ONLY. Include after submitted BalancedWarpRoots.cuh. Never compiled locally.
// Four independent 32-leaf root trees in one 128-thread CTA; one inverse per warp.
#include "CyclicFieldResearch.cuh"
#include "PrefixCyclicFieldResearch.cuh"
template<int N>
__device__ __forceinline__ void qsb_block_inverse_independent_n(uint64_t *value){
    static_assert(N==128,"research fixed four-warp shape");
    __shared__ uint64_t products[4][2*N];
    __shared__ uint64_t inverses[4][N];
    const unsigned tid=threadIdx.x,lane=tid&31u,warp=tid>>5;
    const unsigned pb=warp*64u,ib=warp*32u;
    #pragma unroll
    for(int k=0;k<4;++k)products[k][pb+lane]=value[k];
    __syncwarp(0xffffffffu);
    unsigned offset=0;
    #pragma unroll 1
    for(unsigned count=32;count>1;count>>=1){
        unsigned half=count>>1;
        if(half<=4){
            const unsigned node=lane>>3,d=lane&7u;
            const bool live=node<half;
            uint32_t a=live?(uint32_t)(products[d>>1][pb+offset+node]>>(32*(d&1))):(uint32_t)(d==0);
            uint32_t b=live?(uint32_t)(products[d>>1][pb+offset+half+node]>>(32*(d&1))):(uint32_t)(d==0);
            uint32_t word=qsb_prefix_cyclic_research::multiply8(a,b,lane);
            uint32_t next=__shfl_down_sync(0xffffffffu,word,1,8);
            if(live && !(d&1))products[d>>1][pb+offset+count+node]=(uint64_t)word|((uint64_t)next<<32);
        }else if(lane<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;++k){a[k]=products[k][pb+offset+lane];b[k]=products[k][pb+offset+half+lane];}
            a[4]=b[4]=0;QSB_RF_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;++k)products[k][pb+offset+count+lane]=out[k];
        }
        offset+=count;__syncwarp(0xffffffffu);
    }
    uint64_t root[5];
    #pragma unroll
    for(int k=0;k<4;++k)root[k]=products[k][pb+62];
    root[4]=0;qsb_field_normalize(root);
    qsb_warp_research::qwr_inverse_scaled(root,lane);
    if(lane==0){
        #pragma unroll
        for(int k=0;k<4;++k)inverses[k][ib+30]=root[k];
    }
    __syncwarp(0xffffffffu);
    offset=60;
    #pragma unroll 1
    for(unsigned count=2;count<32;count<<=1){
        unsigned half=count>>1;
        if(count<=4){
            const unsigned node=lane>>3,d=lane&7u;
            const bool live=node<count;
            uint32_t a=live?(uint32_t)(inverses[d>>1][ib+offset+count-32+(node&(half-1))]>>(32*(d&1))):(uint32_t)(d==0);
            uint32_t b=live?(uint32_t)(products[d>>1][pb+offset+(node^half)]>>(32*(d&1))):(uint32_t)(d==0);
            uint32_t word=qsb_prefix_cyclic_research::multiply8(a,b,lane);
            uint32_t next=__shfl_down_sync(0xffffffffu,word,1,8);
            if(live && !(d&1))inverses[d>>1][ib+offset-32+node]=(uint64_t)word|((uint64_t)next<<32);
        }else if(lane<count){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;++k){a[k]=inverses[k][ib+offset+count-32+(lane&(half-1))];b[k]=products[k][pb+offset+(lane^half)];}
            a[4]=b[4]=0;QSB_RF_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;++k)inverses[k][ib+offset-32+lane]=out[k];
        }
        offset-=count<<1;__syncwarp(0xffffffffu);
    }
    uint64_t a[5],b[5];
    #pragma unroll
    for(int k=0;k<4;++k){a[k]=inverses[k][ib+(lane&15u)];b[k]=products[k][pb+(lane^16u)];}
    a[4]=b[4]=0;QSB_RF_MUL(value,a,b);qsb_field_normalize(value);
}
// Launch exactly <<<1,128>>> with 1<=count<=1024.
// Physical capacity is 2048 four-word rows even for a partial final tile.
__global__ void __launch_bounds__(128,1) qsb_root_independent_research(uint64_t *roots,int count) {
    if (count<=0 || count>1024) return; // uniform, before any block barrier
    const unsigned n=(unsigned)count;
    const unsigned lane=threadIdx.x;
    uint64_t total[5];
    {
        uint64_t q[2][5];
        #pragma unroll 1
        for(unsigned quartet=0;quartet<2;++quartet) {
            uint64_t p01[5],p23[5],a[5],b[5];
            unsigned i=lane+quartet*512u;
            qbw_root_load(a,roots,i,n);
            qbw_root_load(b,roots,i+128u,n);
            qsb_field_mul(p01,a,b);p01[4]=0;
            qbw_root_load(a,roots,i+256u,n);
            qbw_root_load(b,roots,i+384u,n);
            qsb_field_mul(p23,a,b);p23[4]=0;
            // Slots 2,3 hold first quartet pairs; 4,5 hold second pairs.
            qbw_scratch_put(roots,n,lane+(2u+2u*quartet)*128u,p01);
            qbw_scratch_put(roots,n,lane+(3u+2u*quartet)*128u,p23);
            qsb_field_mul(q[quartet],p01,p23);q[quartet][4]=0;
            qbw_scratch_put(roots,n,lane+quartet*128u,q[quartet]);
        }
        qsb_field_mul(total,q[0],q[1]);total[4]=0;
    }
    // No local subtree field is needed across this collective. Volatile
    // scratch accesses require reloading instead of forwarding old values.
    qsb_block_inverse_independent_n<128>(total);
    uint64_t iq0[5],iq1[5];
    {
        uint64_t q0[5],q1[5];
        qbw_scratch_get(q0,roots,n,lane);
        qbw_scratch_get(q1,roots,n,lane+128u);
        qsb_field_mul(iq0,total,q1);iq0[4]=0;
        qsb_field_mul(iq1,total,q0);iq1[4]=0;
    }
    // Fetch BOTH pair products before writing a quartet's weighted outputs.
    // Both quartet orders are safe with this rule. This descending order
    // consumes 4,5 before writing them, then consumes 2,3 before writing 0..3.
    #pragma unroll 1
    for(int quartet=1;quartet>=0;--quartet) {
        uint64_t p01[5],p23[5],ip01[5],ip23[5];
        qbw_scratch_get(p01,roots,n,lane+(2u+2u*quartet)*128u);
        qbw_scratch_get(p23,roots,n,lane+(3u+2u*quartet)*128u);
        uint64_t *parent=quartet?iq1:iq0;
        qsb_field_mul(ip01,parent,p23);ip01[4]=0;
        qsb_field_mul(ip23,parent,p01);ip23[4]=0;
        #pragma unroll
        for(unsigned pair=0;pair<2;++pair) {
            unsigned j=lane+(unsigned)quartet*512u+pair*256u;
            uint64_t a[5],b[5],ia[5],ib[5];
            bool na=qbw_root_load(a,roots,j,n);
            bool nb=qbw_root_load(b,roots,j+128u,n);
            uint64_t *pinv=pair?ip23:ip01;
            qsb_field_mul(ia,pinv,b);ia[4]=0;
            qsb_field_mul(ib,pinv,a);ib[4]=0;
            qbw_root_store(roots,n,j,ia,na);
            qbw_root_store(roots,n,j+128u,ib,nb);
        }
    }
}
