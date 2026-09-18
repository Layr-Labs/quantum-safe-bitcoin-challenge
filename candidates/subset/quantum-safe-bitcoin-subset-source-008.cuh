// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
#include "quantum-safe-bitcoin-subset-source-009.cuh"
/* QSB_CFG_TREE (kill switch):
 *  0 = promoted heap tree: every product canonical, 18 barriers.
 *  1 = same heap layout, lazy canonicalization (internal nodes stay exact but
 *      possibly non-canonical residues in [0,2^256); only the root is
 *      normalized, right before _ModInv), the root product is computed by
 *      lane 0 immediately before its inversion and the first downward level
 *      right after it (no barriers in between), and every lane forms its own
 *      leaf inverse from its parent inverse and sibling product (no leaf write
 *      barrier). 15 barriers, 1 canonicalization per block, same 765 multiplies.
 *  2 = level-packed products/inverses (layout of the promoted pinning tree)
 *      with the same lazy/barrier schedule as 1; barrier kind chosen by the
 *      lanes that READ the next level (warp barrier only when all readers and
 *      writers sit in warp 0).
 * Leaves are returned as exact residues below 2^256, the same contract as
 * every _ModMult output that feeds the finish. */
#ifndef QSB_CFG_TREE
#define QSB_CFG_TREE 3  /* in-place packed inverse levels; cooperative root, 16 KiB shared */
#endif
#if QSB_CFG_TREE == 0
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
#elif QSB_CFG_TREE == 1
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t tree[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][n+tid]=value[k];
    __syncthreads();
    // Upward levels with at least two writers; the readers of level `width`
    // are its writers' low half, so a warp barrier suffices once width<=32.
    #pragma unroll 1
    for(int width=n>>1;width>1;width>>=1){
        if(tid<width){
            int node=width+tid;
            uint64_t a[5],b[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][2*node];b[k]=tree[k][2*node+1];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        if(width>32)__syncthreads();else __syncwarp();
    }
    if(tid==0){
        // Root product, normalization, inversion and the first downward level
        // are all lane 0's own reads and writes: no barrier in between.
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=tree[k][2];b[k]=tree[k][3];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4]=0;
        qsb_field_mul_raw(a,root,a);   /* 1/right */
        qsb_field_mul_raw(b,root,b);   /* 1/left  */
        #pragma unroll
        for(int k=0;k<4;k++){tree[k][2]=b[k];tree[k][3]=a[k];}
    }
    __syncwarp();
    // Remaining internal downward levels: level `width` is read by lanes < 2*width.
    #pragma unroll 1
    for(int width=2;width<(n>>1);width<<=1){
        if(tid<width){
            int node=width+tid;
            uint64_t parent[5],left[5],right[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent[k]=tree[k][node];
                left[k]=tree[k][2*node];right[k]=tree[k][2*node+1];
            }
            parent[4]=left[4]=right[4]=0;
            qsb_field_mul_raw(right,parent,right);qsb_field_mul_raw(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){
                tree[k][2*node]=right[k];tree[k][2*node+1]=left[k];
            }
        }
        // Level `width` is read by lanes < 2*width, except the last internal
        // level (width == n/4), which every lane reads for its own leaf.
        if((width<<1)>32 || (width<<2)==n)__syncthreads();else __syncwarp();
    }
    // Leaf level: every lane multiplies its parent inverse by its sibling's
    // (never overwritten) leaf product. No shared write, no barrier.
    {
        uint64_t parent[5],sibling[5];
        const int leaf=n+tid;
        #pragma unroll
        for(int k=0;k<4;k++){parent[k]=tree[k][leaf>>1];sibling[k]=tree[k][leaf^1];}
        parent[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent,sibling);
    }
    value[4]=0;
}
#elif QSB_CFG_TREE == 2
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t products[4][512];
    __shared__ uint64_t inverses[4][256];
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    // Level (offset,count): (0,n),(n,n/2),...,(2n-4,2). Level `count` is
    // formed by lanes < count/2 and read by lanes < count/4.
    int offset=0;
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    // offset == 2n-4: the two root children.
    // Lanes 0..3 of warp 0 form the same root product and invert it cooperatively
    // In the included inverse helper, lane 0 owns u, lane 1 v, lane 2 r, lane 3 s, one shared decision
    // stream. Lanes 0 and 1 then each form one child inverse, so the whole root stage
    // is one warp-uniform instruction stream instead of a serial lane-0 _ModInv.
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        root[4]=0;
        zi_inverse_quad(root,tid);
        if(tid<2){
            /* child = products[offset + 1 - tid]: lane 0 writes 1/a, lane 1 writes 1/b,
             * the same two values (and the same order) the serial form wrote. */
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=products[k][offset+1-tid];
            child[4]=0;
            qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
    }
    __syncwarp();
    // Level count (4..n/2): lanes < count read parent inverses written by
    // lanes < count/2 and write inverses read by lanes < 2*count.
    offset-=4;   /* level count=4 */
    #pragma unroll 1
    for(int count=4;count<n;count<<=1){
        int half=count>>1;
        if(tid<count){
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-n+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            qsb_field_mul_raw(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child_inv[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    // offset == 0 would be the leaf level; lanes form their own leaf inverse.
    {
        const int half=n>>1;
        uint64_t parent_inv[5],sibling[5];
        #pragma unroll
        for(int k=0;k<4;k++){
            parent_inv[k]=inverses[k][tid&(half-1)];
            sibling[k]=products[k][tid^half];
        }
        parent_inv[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent_inv,sibling);
    }
    value[4]=0;
}
#else
/* Level-packed in-place inverse propagation. Once every child at a level has
 * loaded its sibling product, that product level is dead and can hold child
 * inverses. Cross-warp levels synchronize after their loads before overwrite. */
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    __shared__ uint64_t nodes[4][512];
    const int tid=threadIdx.x,n=blockDim.x;
    #pragma unroll
    for(int k=0;k<4;k++)nodes[k][tid]=value[k];
    __syncthreads();
    int offset=0;
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=nodes[k][offset+tid];b[k]=nodes[k][offset+half+tid];}
            a[4]=b[4]=0;
            qsb_field_mul_raw(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)nodes[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=nodes[k][offset];b[k]=nodes[k][offset+1];}
        a[4]=b[4]=0;
        qsb_field_mul_raw(root,a,b);
        qsb_field_normalize(root);
        root[4]=0;
        zi_inverse_quad(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=nodes[k][offset+1-tid];
            child[4]=0;
            __syncwarp(0x3u);
            qsb_field_mul_raw(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)nodes[k][offset+tid]=child[k];
        }
    }
    __syncwarp();
    offset-=4;
    #pragma unroll 1
    for(int count=4;count<n;count<<=1){
        int half=count>>1;
        uint64_t parent_inv[5],sibling[5];
        if(tid<count){
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=nodes[k][offset+count+(tid&(half-1))];
                sibling[k]=nodes[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
        }
        if(count>32)__syncthreads();
        else if(tid<count){
            unsigned live=count==32 ? 0xffffffffu : ((1u<<count)-1u);
            __syncwarp(live);
        }
        if(tid<count){
            qsb_field_mul_raw(parent_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)nodes[k][offset+tid]=parent_inv[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    {
        const int half=n>>1;
        uint64_t parent_inv[5],sibling[5];
        #pragma unroll
        for(int k=0;k<4;k++){
            parent_inv[k]=nodes[k][n+(tid&(half-1))];
            sibling[k]=nodes[k][tid^half];
        }
        parent_inv[4]=sibling[4]=0;
        qsb_field_mul_raw(value,parent_inv,sibling);
    }
    value[4]=0;
}
#endif
