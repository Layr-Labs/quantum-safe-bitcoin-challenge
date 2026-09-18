/* GPL-3.0-only. Four-epoch reciprocal batching with per-block owned scratch.
 * Builds on the existing signed comb, K2 pre/post interface and block inverse.
 * Scratch capacity is bounded by physical blocks, not logical launch length.
 * Public synthetic candidates only; no cross-block synchronization or reuse.
 */
#pragma once
#define QSB_K4_EPOCHS 4
#define QSB_K4_WORDS 17
static_assert(QSB_X66_BLOCKS>0 && QSB_X66_BLOCKS<=256,"physical scratch bound");
static_assert(QSB_SE_PER_EPOCH==256,"one candidate owner per window lane");

__global__ void __launch_bounds__(256,2) kernel_digest_k4(
    const uint8_t *__restrict__ table,
    const epoch_desc_t *__restrict__ epochs,
    const uint32_t *__restrict__ first,int first_stride,
    int epoch_count,int physical_blocks,
    uint64_t *__restrict__ scratch,
    uint32_t *hit_count,uint32_t *hit_idx,uint8_t *hit_combos
) {
    const int tid=threadIdx.x;
    // Host launches exactly physical_blocks blocks, each with 256 lanes.
    const int groups=(epoch_count+QSB_K4_EPOCHS-1)/QSB_K4_EPOCHS;
    uint64_t *park=scratch+(size_t)blockIdx.x*QSB_K4_EPOCHS*QSB_K4_WORDS*256;
    uint64_t xR[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
    uint64_t yR[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};
    #pragma unroll 1
    for(int group=blockIdx.x;group<groups;group+=physical_blocks) {
        uint64_t prefix[5]={1,0,0,0,0};
        #pragma unroll 1
        for(int k=0;k<QSB_K4_EPOCHS;k++) {
            const int ep=group*QSB_K4_EPOCHS+k;
            uint64_t w[5]={1,0,0,0,0},m1[4]={},m2[4]={};
            int state=0;
            if(ep<epoch_count)
                state=qsb_k2s_front(first+(size_t)ep*first_stride,tid,table,xR,yR,w,m1,m2);
            if(state!=1){w[0]=1;w[1]=w[2]=w[3]=w[4]=0;}
            #pragma unroll
            for(int j=0;j<4;j++) {
                park[(k*QSB_K4_WORDS+j)*256+tid]=m1[j];
                park[(k*QSB_K4_WORDS+4+j)*256+tid]=m2[j];
                park[(k*QSB_K4_WORDS+8+j)*256+tid]=w[j];
                park[(k*QSB_K4_WORDS+12+j)*256+tid]=prefix[j];
            }
            park[(k*QSB_K4_WORDS+16)*256+tid]=(uint64_t)state;
            if(k==0){Load256(prefix,w);}
            else qsb_field_mul_raw(prefix,prefix,w);
        }
        qsb_block_inverse_tree(prefix);
        #pragma unroll 1
        for(int k=QSB_K4_EPOCHS-1;k>=0;k--) {
            uint64_t inv[5],before[5],w[5];
            if(k>0) {
                #pragma unroll
                for(int j=0;j<4;j++) {
                    before[j]=park[(k*QSB_K4_WORDS+12+j)*256+tid];
                    w[j]=park[(k*QSB_K4_WORDS+8+j)*256+tid];
                }
                qsb_field_mul_raw(inv,before,prefix);
                qsb_field_mul_raw(prefix,prefix,w);
            } else Load256(inv,prefix);
            int state=(int)park[(k*QSB_K4_WORDS+16)*256+tid];
            if(state) {
                uint64_t m1[4],m2[4],x1[4],x2[4];
                #pragma unroll
                for(int j=0;j<4;j++) {
                    m1[j]=park[(k*QSB_K4_WORDS+j)*256+tid];
                    m2[j]=park[(k*QSB_K4_WORDS+4+j)*256+tid];
                }
                uint32_t valid;
                uint32_t parity=qsb_k2s_finish_point(state,m1,m2,inv,xR,yR,x1,x2,&valid);
                int recid=0;
                if(qsb_k2s_gate(x1,x2,parity,&recid,valid)) {
                    const int ep=group*QSB_K4_EPOCHS+k;
                    uint32_t slot=atomicAdd(hit_count,1);
                    if(slot<1024) {
                        hit_idx[slot*4]=((uint32_t)ep*256+(uint32_t)tid)|((uint32_t)recid<<30);
                        for(int j=0;j<6;j++)hit_combos[slot*ZLAB_HIT_REC+j]=epochs[ep].early[j];
                        for(int j=0;j<3;j++)hit_combos[slot*ZLAB_HIT_REC+6+j]=WIN3[tid][j];
                    }
                }
            }
        }
        // No lane can reuse inverse-tree scratch until every lane has finished
        // this group. Global parked records are read/written only by their owner.
        __syncthreads();
    }
}
