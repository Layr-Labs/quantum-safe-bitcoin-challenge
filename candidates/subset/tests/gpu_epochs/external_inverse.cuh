// Checkpointed inverse hierarchy adapted from promoted pinning commit 240f329.
// GPLv3, retaining the inherited candidate licensing in COPYING.
#pragma once
__device__ __forceinline__ void qsb_st_u64(uint64_t *p, uint64_t v) { *p=v; }
__device__ __forceinline__ uint64_t qsb_ld_u64(const uint64_t *p) { return *p; }
template<int N>
__device__ __forceinline__ void qsb_block_product_checkpoint(
    uint64_t *value, uint64_t *roots, uint64_t *checkpoint, uint64_t (*products)[2*N]
) {
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*N;

    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();

    int offset=0;
    #pragma unroll 1
    for(int count=N;count>1;count>>=1){
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
                if(node<2*N-2)
                    qsb_st_u64(&checkpoint[block_base+(size_t)k*N+node-N],out[k]);
            }
        }
        offset+=count;
        if(count>2)__syncthreads();
    }

    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=products[k][2*N-2];
    }
}
template<int N>
__device__ __forceinline__ void qsb_block_product_checkpoint(
    uint64_t *value, uint64_t *roots, uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][2*N];
    qsb_block_product_checkpoint<N>(value,roots,checkpoint,products);
}

template<int N>
__device__ __forceinline__ void qsb_block_inverse_checkpoint(
    uint64_t *value, const uint64_t *roots, const uint64_t *checkpoint
) {
    __shared__ uint64_t products[4][2*N];
    __shared__ uint64_t inverses[4][N];
    int tid=threadIdx.x;
    size_t block_base=(size_t)blockIdx.x*4u*N;

    /* Each lane supplies its saved W leaf and all but the last two lanes
     * restore one internal node. Lane zero also publishes the external root inverse.
     * One barrier makes both immutable inputs visible to the downward pass. */
    #pragma unroll
    for(int k=0;k<4;k++){
        products[k][tid]=value[k];
        if(tid<N-2)
            products[k][N+tid]=qsb_ld_u64(&checkpoint[block_base+(size_t)k*N+tid]);
        if(tid==0)inverses[k][N-2]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    int offset=2*N-4;
    #pragma unroll 1
    for(int count=2;count<N;count<<=1){
        int half=count>>1;
        if(tid<count){
            int local_parent=tid&(half-1);
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-N+local_parent];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-N+tid]=child_inv[k];
        }
        offset-=count<<1;
        __syncthreads();
    }

    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
    qsb_field_normalize(value);
}

/* Batch the per-search-CTA roots one level further. Groups of 256 roots use
 * the same checkpointed tree helpers, then one 256-lane CTA batch-inverts all
 * group roots. A full 16M candidate batch therefore executes one _ModInv
 * instead of 65,536 independent inversions. */
__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(
    const uint64_t *roots, int count, uint64_t *super_roots,
    uint64_t *root_checkpoint
) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_product_checkpoint<256>(r,super_roots,root_checkpoint);
}

/* One 256-lane CTA per 256 group roots; each CTA runs its own _ModInv, so
 * batches with more than 65,536 candidate trees need no third tree level. */
__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(
    uint64_t *super_roots, int count
) {
    int tid=(int)(blockIdx.x*256u+threadIdx.x);
    bool active=tid<count;
    uint64_t r[5]={active?super_roots[(size_t)tid*4u]:1ULL,
                   active?super_roots[(size_t)tid*4u+1]:0ULL,
                   active?super_roots[(size_t)tid*4u+2]:0ULL,
                   active?super_roots[(size_t)tid*4u+3]:0ULL,0};
    qsb_block_inverse_tree(r);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)super_roots[(size_t)tid*4u+k]=r[k];
    }
}

__global__ void __launch_bounds__(256,2) qsb_root_group_finish(
    uint64_t *roots, int count, const uint64_t *super_roots,
    const uint64_t *root_checkpoint
) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_inverse_checkpoint<256>(r,super_roots,root_checkpoint);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)i*4u+k]=r[k];
    }
}


