/* Split form of qsb_block_inverse_tree for an external, batched inversion.
 *
 * The prepare stage checkpoints the 254 non-root internal product nodes of a
 * 256-leaf block tree to global memory and publishes the raw root. Roots are
 * batched one level further: groups of 256 roots run the same checkpointed
 * helpers, and one 256-lane CTA inverts every group root, so a launch executes
 * one _ModInv per 256 blocks instead of one per block, and the search kernels
 * carry no inversion latency. The finish stage restores the immutable tree,
 * expands the supplied root inverse and returns the leaf inverses. Node
 * numbering, multiply schedule and synchronization match
 * qsb_block_inverse_tree exactly; leaves must be nonzero (identity for
 * inactive or unusable lanes) and blockDim.x must be 256. */
#pragma once
#define QSB_CKPT_STRIDE 256

__device__ __forceinline__ void qsb_block_product_checkpoint(
        uint64_t *value, uint64_t *roots, uint64_t *checkpoint){
    __shared__ uint64_t tree[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
    const size_t block_base=(size_t)blockIdx.x*4u*QSB_CKPT_STRIDE;
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
    __syncthreads();
    #pragma unroll 1
    for(int width=n>>1;width>0;width>>=1){
        if(tid<width){
            int node=width+tid;
            uint64_t a[5]={0,0,0,0,0},b[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
            qsb_field_mul(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][node]=a[k];
                if(node>=2)checkpoint[block_base+(size_t)k*QSB_CKPT_STRIDE+(node-2)]=a[k];
            }
        }
        if(width>32)__syncthreads();else __syncwarp();
    }
    if(tid==0){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4u+k]=tree[k][1];
    }
}

__device__ __forceinline__ void qsb_block_inverse_checkpoint(
        uint64_t *value, const uint64_t *roots, const uint64_t *checkpoint){
    __shared__ uint64_t tree[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
    const size_t block_base=(size_t)blockIdx.x*4u*QSB_CKPT_STRIDE;
    /* Each lane supplies its leaf and all but the last two lanes restore one
     * internal node; lane zero publishes the external root inverse. One
     * barrier makes every immutable input visible to the downward pass. */
    #pragma unroll
    for(int k=0;k<4;k++){
        tree[k][n+tid]=value[k];
        if(tid<n-2)tree[k][2+tid]=checkpoint[block_base+(size_t)k*QSB_CKPT_STRIDE+tid];
        if(tid==0)tree[k][1]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();
    #pragma unroll 1
    for(int width=1;width<n;width<<=1){
        if(tid<width){
            int node=width+tid;
            uint64_t parent[5]={0,0,0,0,0},left[5]={0,0,0,0,0},right[5]={0,0,0,0,0};
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];
                left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            qsb_field_mul(right,parent,right);qsb_field_mul(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
            }
        }
        if((width<<1)>32)__syncthreads();else __syncwarp();
    }
    #pragma unroll
    for(int k=0;k<4;k++)value[k]=tree[k][n+tid];
    value[4]=0;
}

/* Batch the per-block roots one level further. Groups of 256 roots reuse the
 * checkpointed helpers, then one 256-lane CTA inverts all group roots. */
__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(
        const uint64_t *roots,int count,uint64_t *super_roots,uint64_t *root_checkpoint){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_product_checkpoint(r,super_roots,root_checkpoint);
}

__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(uint64_t *super_roots,int count){
    int tid=(int)threadIdx.x;
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
        uint64_t *roots,int count,const uint64_t *super_roots,const uint64_t *root_checkpoint){
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_inverse_checkpoint(r,super_roots,root_checkpoint);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)i*4u+k]=r[k];
    }
}
