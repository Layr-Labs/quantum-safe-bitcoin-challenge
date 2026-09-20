// Paired two-CTA layout: separate immutable products from downward inverses.
// This restores the audited earlier tree layout, retaining current root-entry
// warp synchronization. The extra 8 KiB removes destructive-read barriers.
// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
#include "hm39_pair_inverse.cuh"
#include "hm41_quad_inverse.cuh"
#include "hm43_warp_inverse.cuh"
#include "zinv32.cuh"
/* ZLAB_TREE (kill switch):
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
#ifndef ZLAB_TREE
#define ZLAB_TREE 2  /* measured best on gpu2: +0.7% alone, part of the +1.85% bundle */
#endif
/* Carry a thread's just-produced packed-tree node across the level barrier.
 * The next level's low half otherwise reloads that exact shared-memory word.
 * This is the subset adaptation of fkiene's public pinning PR786 research;
 * setting the switch to 0 restores the promoted shared-load path. */
#ifndef QSB_TREE_REGISTER_CARRY
#define QSB_TREE_REGISTER_CARRY 1
#endif
#if ZLAB_TREE == 0
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
#elif ZLAB_TREE == 1
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
#else
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
#if QSB_TREE_REGISTER_CARRY
    uint64_t up_carry[5];
    #pragma unroll
    for(int k=0;k<4;k++)up_carry[k]=value[k];
    up_carry[4]=0;
#endif
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){
#if QSB_TREE_REGISTER_CARRY
                a[k]=up_carry[k];
#else
                a[k]=products[k][offset+tid];
#endif
                b[k]=products[k][offset+half+tid];
            }
            a[4]=b[4]=0;
            QSB_TREE_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][offset+count+tid]=out[k];
#if QSB_TREE_REGISTER_CARRY
                up_carry[k]=out[k];
#endif
            }
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    // offset == 2n-4: the two root children.
#if HM43_WARP_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
        root[4]=0;zi_inverse_quad(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;QSB_TREE_MUL(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
    }
#elif HM41_QUAD_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm41_quad_inverse(root,tid);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;QSB_TREE_MUL(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
    }
#elif HM39_PAIR_ROOT
    if(tid<2){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x00000003);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm39_pair_inverse(root,tid);
        uint64_t child[5];
        #pragma unroll
        for(int k=0;k<4;k++)child[k]=tid? a[k]:b[k];
        child[4]=0;QSB_TREE_MUL(child,root,child);
        #pragma unroll
        for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
    }
#else
    if(tid==0){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;
        QSB_TREE_MUL(root,a,b);
        qsb_field_normalize(root);
        _ModInv(root);
        root[4]=0;
        QSB_TREE_MUL(a,root,a);   /* 1/b */
        QSB_TREE_MUL(b,root,b);   /* 1/a */
        // inverse index = product index - n
        #pragma unroll
        for(int k=0;k<4;k++){inverses[k][offset-n]=b[k];inverses[k][offset-n+1]=a[k];}
    }
#endif
    __syncwarp();
    // Level count (4..n/2): lanes < count read parent inverses written by
    // lanes < count/2 and write inverses read by lanes < 2*count.
    offset-=4;   /* level count=4 */
#if QSB_TREE_REGISTER_CARRY
    uint64_t down_carry[5];
    if(tid<2){
        #pragma unroll
        for(int k=0;k<4;k++)down_carry[k]=inverses[k][offset+4-n+tid];
        down_carry[4]=0;
    }
#endif
    #pragma unroll 1
    for(int count=4;count<n;count<<=1){
        int half=count>>1;
        if(tid<count){
            uint64_t parent_inv[5],sibling[5],child_inv[5];
            #pragma unroll
            for(int k=0;k<4;k++){
#if QSB_TREE_REGISTER_CARRY
                parent_inv[k]=tid<half?down_carry[k]:inverses[k][offset+count-n+(tid&(half-1))];
#else
                parent_inv[k]=inverses[k][offset+count-n+(tid&(half-1))];
#endif
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent_inv[4]=sibling[4]=0;
            QSB_TREE_MUL(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++){
                inverses[k][offset-n+tid]=child_inv[k];
#if QSB_TREE_REGISTER_CARRY
                down_carry[k]=child_inv[k];
#endif
            }
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
#if QSB_TREE_REGISTER_CARRY
            parent_inv[k]=tid<half?down_carry[k]:inverses[k][tid&(half-1)];
#else
            parent_inv[k]=inverses[k][tid&(half-1)];
#endif
            sibling[k]=products[k][tid^half];
        }
        parent_inv[4]=sibling[4]=0;
        QSB_TREE_MUL(value,parent_inv,sibling);
    }
    value[4]=0;
}
#endif
