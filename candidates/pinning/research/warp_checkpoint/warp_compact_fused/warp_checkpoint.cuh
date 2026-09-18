// GPL-3.0. Checkpoint/synchronization mechanism from Meganpark980320 PR98.
// Exact source head 2d5eba297f8d9823ca0f5a552d3d1788fbe3031c.
#pragma once
#define QSB_CHECKPOINT_FIRST 448
#define QSB_CHECKPOINT_NODES 62
#define QSB_CHECKPOINT_STRIDE 64
__device__ __forceinline__ void qsb_block_product_checkpoint(
    uint64_t *value, uint64_t *roots, uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][512];
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*QSB_CHECKPOINT_STRIDE;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

    int offset=0;
    #pragma unroll 1
    for(int count=256;count>1;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                a[k]=products[k][offset+tid];
                b[k]=products[k][offset+half+tid];
            }
            a[4]=b[4]=0;
            qsb_field_mul(out,a,b);
            int node=offset+count+tid;
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][node]=out[k];
                if(node>=QSB_CHECKPOINT_FIRST && node<510)
                    checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+node-QSB_CHECKPOINT_FIRST]=out[k];
            }
        }
        offset+=count;
        // At 32 produced nodes and below, all later consumers are warp 0.
        // The 128- and 64-node outputs still cross warps.
        if(count>64)__syncthreads();
        else if(count>2)__syncwarp();
    }

    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=products[k][510];
    }
}

__device__ __forceinline__ void qsb_block_inverse_checkpoint(
    uint64_t *value, const uint64_t *roots, const uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][512];
    __shared__ uint64_t inverses[4][256];
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*QSB_CHECKPOINT_STRIDE;

    /* Publish leaves, retained internal nodes and the external root inverse. */
    #pragma unroll
    for(int k=0;k<4;k++){
        products[k][tid]=value[k];
        if(tid<QSB_CHECKPOINT_NODES)
            products[k][QSB_CHECKPOINT_FIRST+tid]=checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+tid];
        if(tid==0)inverses[k][254]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    /* Each owner rebuilds two disjoint pairs and their parent. The pair
     * products stay in registers until their parent is ready; all three
     * nodes are immutable after the following barrier. */
    if(tid<64){
        uint64_t a[5]={},b[5]={},c[5]={};
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][tid];b[k]=products[k][128+tid];}
        qsb_field_mul(a,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][256+tid]=a[k];
        #pragma unroll
        for(int k=0;k<4;k++){b[k]=products[k][64+tid];c[k]=products[k][192+tid];}
        qsb_field_mul(b,b,c);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][320+tid]=b[k];
        qsb_field_mul(a,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][384+tid]=a[k];
    }
    __syncthreads();

    int offset=508;
    #pragma unroll 1
    for(int count=2;count<256;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-256+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-256+tid]=child_inv[k];
        }
        offset-=count<<1;
        // A 32-node output is consumed by 64 lanes in the next level.
        if(count>=32)__syncthreads();else __syncwarp();
    }

    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&127];
        sibling[k]=products[k][tid^128];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
    qsb_field_normalize(value);
}
