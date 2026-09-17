// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t tree[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
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
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        if(width>32)__syncthreads();else __syncwarp();
    }
    if(tid==0){
        uint64_t root[5]={0,0,0,0,0};
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=tree[k][1];
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][1]=root[k];
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

/* ------------------------------------------------------------------
 * Split (checkpointed) form of the block inverse, transplanted from the
 * promoted pinning frontier (candidates/pinning/pinning.cu:568-720, same
 * GPLv3 / VanitySearch-derived tree).
 *
 * qsb_block_inverse_tree above runs the up-sweep, then ONE _ModInv in lane 0
 * while the other 7 warps idle at a barrier, then the down-sweep. Splitting it
 * lets a whole launch share a single _ModInv: the prepare half checkpoints the
 * 254 internal non-root product-tree nodes to global memory and publishes the
 * block's raw root; a three-kernel root group inverts every block root of the
 * launch with one _ModInv; the finish half restores the immutable product tree
 * and expands the supplied root inverse into canonical leaf inverses.
 *
 * Node numbering here is the packed layout of the source, NOT the 1..511
 * heap numbering of qsb_block_inverse_tree: levels are (0,256), (256,128),
 * (384,64), ..., (508,2), (510,1), each level storing its left half before its
 * right half so both operands of a multiply are contiguous across a warp.
 * Leaves come back from the caller's saved W, and node 510 (the root) is
 * omitted from the checkpoint because the roots array owns it.
 *
 * ASSUMPTION: blockDim.x == 256 exactly. The 512-entry product array, the
 * 256-entry inverse array, QSB_CHECKPOINT_NODES and QSB_CHECKPOINT_STRIDE all
 * hard-code a 256-thread block. The ranked short-epoch launch is exactly
 * QSB_SE_PER_EPOCH == 256 threads, and the three root kernels below are
 * launched with 256 threads. Any other block size must use
 * qsb_block_inverse_tree, which derives its geometry from blockDim.x.
 *
 * NO EXTRA NORMALIZE IS NEEDED IN THIS TREE. The source pairs qsb_field_mul
 * (an exact residue in [0,2^256)) with a separate qsb_field_normalize before
 * _ModInv and on every returned leaf. This tree's qsb_field_mul already folds
 * that conditional subtraction in (tree.cu:775-779): a 256-bit product can
 * exceed p only when its upper 192 bits are all ones, and that case is
 * subtracted inline. Every value that reaches _ModInv (the tree root, and the
 * super-root of the root group) and every leaf inverse returned to a caller is
 * a qsb_field_mul output, hence already canonical (< p).
 * ------------------------------------------------------------------ */
#define QSB_CHECKPOINT_NODES 254
#define QSB_CHECKPOINT_STRIDE 256

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
                if(node<510)
                    checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+node-256]=out[k];
            }
        }
        offset+=count;
        if(count>2)__syncthreads();
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

    /* Each lane supplies its saved W leaf and all but the last two lanes
     * restore one internal node. Lane zero also publishes the external root
     * inverse. One barrier makes both immutable inputs visible downward. */
    #pragma unroll
    for(int k=0;k<4;k++){
        products[k][tid]=value[k];
        if(tid<QSB_CHECKPOINT_NODES)
            products[k][256+tid]=checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+tid];
        if(tid==0)inverses[k][254]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    /* If I=1/(L*R) then I*R=1/L and I*L=1/R: one thread per child expands all
     * 254 internal inverse levels, then the 256 leaves. */
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
        __syncthreads();
    }

    /* The leaf level has no shared destination and no following barrier: its
     * 256 child inverses go straight back to the callers, canonical because
     * qsb_field_mul reduces below p. */
    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&127];
        sibling[k]=products[k][tid^128];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul(value,parent_inv,sibling);
}

/* Batch the per-block roots one level further. Groups of 256 roots use the
 * same checkpointed helpers, then one 256-lane CTA batch-inverts all group
 * roots with a single _ModInv. A full QSB_SE_LAUNCH_BLOCKS launch therefore
 * executes ONE _ModInv instead of QSB_SE_LAUNCH_BLOCKS of them.
 *
 * Capacity: qsb_invert_super_roots is a single 256-thread block, so the launch
 * may have at most 256 root groups of 256 blocks each. */
static_assert(QSB_SE_LAUNCH_BLOCKS <= 256*256,
              "two-level root inverse holds at most 256*256 blocks per launch");

__global__ void __launch_bounds__(256,2) qsb_root_group_prepare(
    const uint64_t *roots, int count, uint64_t *super_roots,
    uint64_t *root_checkpoint
) {
    int i=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    bool active=i<count;
    /* Inactive tail lanes enter with the multiplicative identity, exactly as
     * the work kernel does for unusable candidates. */
    uint64_t r[5]={active?roots[(size_t)i*4u]:1ULL,
                   active?roots[(size_t)i*4u+1]:0ULL,
                   active?roots[(size_t)i*4u+2]:0ULL,
                   active?roots[(size_t)i*4u+3]:0ULL,0};
    qsb_block_product_checkpoint(r,super_roots,root_checkpoint);
}

__global__ void __launch_bounds__(256,1) qsb_invert_super_roots(
    uint64_t *super_roots, int count
) {
    int tid=(int)threadIdx.x;
    bool active=tid<count;
    uint64_t r[5]={active?super_roots[(size_t)tid*4u]:1ULL,
                   active?super_roots[(size_t)tid*4u+1]:0ULL,
                   active?super_roots[(size_t)tid*4u+2]:0ULL,
                   active?super_roots[(size_t)tid*4u+3]:0ULL,0};
    /* The one serial _ModInv of the whole launch lives inside here. This tree
     * already ships the monolithic helper, so reusing it costs no extra PTX
     * (the source's qsb_block_inverse was stripped from this tree as dead). */
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
    qsb_block_inverse_checkpoint(r,super_roots,root_checkpoint);
    if(active){
        #pragma unroll
        for(int k=0;k<4;k++)roots[(size_t)i*4u+k]=r[k];
    }
}
