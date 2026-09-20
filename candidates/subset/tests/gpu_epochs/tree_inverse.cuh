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
/* Static 256-lane level schedule for the level-packed tree (three independent
 * kill switches, all default 1; each 0 restores the promoted loop verbatim).
 *
 * The promoted body walks the levels with runtime counters: `offset` is carried
 * across iterations, `count`/`half` drive every shared index, and the barrier
 * kind is chosen by a runtime compare per level.  `blockDim.x` is not a
 * compile-time constant, so none of that folds: each of the 7 upward and 7
 * downward levels pays an address chain (offset+tid, offset+half+tid,
 * offset+count+tid, offset+count-n+(tid&(half-1)), offset+(tid^half)) plus a
 * loop-carried update, and the compiler cannot hoist the shared loads of a
 * level above the previous level's barrier.
 *
 * kernel_digest launches this with exactly 256 threads per block
 * (__launch_bounds__(256,2), blockDim.x = 256 at every call site), so the whole
 * schedule is a constant.  Below, the n == 256 case is emitted from literal
 * level descriptors and the runtime loops stay as the fallback for any other
 * block size, so behaviour is unchanged for both.
 *
 * Exactness is an enumeration, not an estimate.  The upward loop visits
 * (offset,count) = (0,256),(256,128),(384,64),(448,32),(480,16),(496,8),(504,4)
 * and ends with offset = 508 = 2n-4; the barrier is __syncthreads() while
 * half = count/2 > 32, i.e. for the first two levels, and __syncwarp()
 * afterwards.  The downward loop visits count = 4,8,16,32,64,128 with
 * offset = 504,496,480,448,384,256 (offset -= 2*count after each level, having
 * started at 2n-4-4), its parent index is offset+count-n+(tid & (count/2-1)),
 * its sibling index offset+(tid ^ (count/2)), its write index offset-n+tid, and
 * its barrier is __syncthreads() once 2*count > 32, i.e. for count >= 32.  The
 * literals below are exactly those values, so the emitted sequence of
 * QSB_TREE_MUL calls, shared reads/writes and barriers is identical to the
 * loop's; every multiply sees the same operands, so every lane's leaf inverse
 * is bit-identical.
 *
 * QSB_TREE_SELF_REG additionally takes the lane's own factor at upward level 0
 * from the register it just stored instead of re-reading products[k][tid].  The
 * store and the read are the same thread's, three instructions apart, so the
 * value is identical by definition; only the sibling still comes from shared.
 */
#ifndef QSB_TREE_UP_STATIC
#define QSB_TREE_UP_STATIC 1
#endif
#ifndef QSB_TREE_DOWN_STATIC
#define QSB_TREE_DOWN_STATIC 1
#endif
#ifndef QSB_TREE_SELF_REG
#define QSB_TREE_SELF_REG 1
#endif
#define QSB_TREE_UP_LVL(OFF,CNT,SELF,BAR) {                                   \
    const int half_=(CNT)>>1;                                                 \
    if(tid<half_){                                                            \
        uint64_t a[5],b[5],out[5];                                            \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++){                                                 \
            a[k]=(SELF)?value[k]:products[k][(OFF)+tid];                      \
            b[k]=products[k][(OFF)+half_+tid];                                \
        }                                                                     \
        a[4]=b[4]=0;                                                          \
        QSB_TREE_MUL(out,a,b);                                                \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++)products[k][(OFF)+(CNT)+tid]=out[k];              \
    }                                                                         \
    if(BAR)__syncthreads();else __syncwarp();                                 \
}
#define QSB_TREE_DOWN_LVL(OFF,CNT,BAR) {                                      \
    const int half_=(CNT)>>1;                                                 \
    if(tid<(CNT)){                                                            \
        uint64_t parent_inv[5],sibling[5],child_inv[5];                       \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++){                                                 \
            parent_inv[k]=inverses[k][(OFF)+(CNT)-256+(tid&(half_-1))];       \
            sibling[k]=products[k][(OFF)+(tid^half_)];                        \
        }                                                                     \
        parent_inv[4]=sibling[4]=0;                                           \
        QSB_TREE_MUL(child_inv,parent_inv,sibling);                           \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++)inverses[k][(OFF)-256+tid]=child_inv[k];          \
    }                                                                         \
    if(BAR)__syncthreads();else __syncwarp();                                 \
}
/* QSB_TREE_DOWN_WARP (independent kill switch, default 1; 0 restores the
 * shared-memory downward chain above verbatim).
 *
 * Downward levels count = 4, 8, 16, 32 are executed entirely by lanes < 32,
 * i.e. by warp 0 alone, and each of them reads as its parent inverse exactly
 * the value the previous level wrote:
 *   level 8  reads inverses[248+(tid&3)],  level 4  wrote inverses[248+tid], tid<4
 *   level 16 reads inverses[240+(tid&7)],  level 8  wrote inverses[240+tid], tid<8
 *   level 32 reads inverses[224+(tid&15)], level 16 wrote inverses[224+tid], tid<16
 * In every case the producing lane index equals the shared index minus the
 * level base, so the value read by lane t is the register the lane
 * (t & (count/2-1)) of the SAME warp just produced.  Replacing the shared
 * round trip by __shfl_sync over the exact active mask therefore yields the
 * identical 256-bit operand, and the three shared writes at 248..251,
 * 240..247 and 224..239 become dead: no other reader exists (level 64 reads
 * 192+(tid&31), written by level 32, which still stores; the leaf block reads
 * 0..127, written by level 128).  The store at level 32 is kept because its
 * readers (lanes < 64) cross a warp boundary.
 *
 * The first shuffled level is 8: level 4 still takes its parent from shared,
 * because that operand is produced by the root block, whose lane set depends
 * on the selected root variant.  Masks are the exact active sets (0xf, 0xff,
 * 0xffff, 0xffffffff), so every lane named as a shuffle source is a
 * participant, and the barrier schedule is unchanged: __syncwarp() after the
 * three warp-local levels, __syncthreads() after the level that stores. */
#ifndef QSB_TREE_DOWN_WARP
#define QSB_TREE_DOWN_WARP 1
#endif
#define QSB_TREE_DOWN_WLVL(OFF,CNT,MASK,SHFL,STORE) {                         \
    const int half_=(CNT)>>1;                                                 \
    if(tid<(CNT)){                                                            \
        uint64_t parent_inv[5],sibling[5],child_inv[5];                       \
        if(SHFL){                                                             \
            _Pragma("unroll")                                                 \
            for(int k=0;k<4;k++)                                              \
                parent_inv[k]=(uint64_t)__shfl_sync((MASK),                   \
                    (unsigned long long)wcarry[k],tid&(half_-1));             \
        }else{                                                                \
            _Pragma("unroll")                                                 \
            for(int k=0;k<4;k++)                                              \
                parent_inv[k]=inverses[k][(OFF)+(CNT)-256+(tid&(half_-1))];   \
        }                                                                     \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++)sibling[k]=products[k][(OFF)+(tid^half_)];        \
        parent_inv[4]=sibling[4]=0;                                           \
        QSB_TREE_MUL(child_inv,parent_inv,sibling);                           \
        _Pragma("unroll")                                                     \
        for(int k=0;k<4;k++)wcarry[k]=child_inv[k];                           \
        if(STORE){                                                            \
            _Pragma("unroll")                                                 \
            for(int k=0;k<4;k++)inverses[k][(OFF)-256+tid]=child_inv[k];      \
        }                                                                     \
    }                                                                         \
}
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
#if QSB_TREE_UP_STATIC
    if(n==256){
        QSB_TREE_UP_LVL(0,256,QSB_TREE_SELF_REG,1)
        QSB_TREE_UP_LVL(256,128,0,1)
        QSB_TREE_UP_LVL(384,64,0,0)
        QSB_TREE_UP_LVL(448,32,0,0)
        QSB_TREE_UP_LVL(480,16,0,0)
        QSB_TREE_UP_LVL(496,8,0,0)
        QSB_TREE_UP_LVL(504,4,0,0)
        offset=508;   /* 2n-4 */
    }else
#endif
    {
    #pragma unroll 1
    for(int count=n;count>2;count>>=1){
        int half=count>>1;
        if(tid<half){
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;
            QSB_TREE_MUL(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
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
#if QSB_TREE_DOWN_STATIC
    if(n==256){
#if QSB_TREE_DOWN_WARP
        uint64_t wcarry[4]={0,0,0,0};
        QSB_TREE_DOWN_WLVL(504,4,0x0000000fu,0,0)
        __syncwarp();
        QSB_TREE_DOWN_WLVL(496,8,0x000000ffu,1,0)
        __syncwarp();
        QSB_TREE_DOWN_WLVL(480,16,0x0000ffffu,1,0)
        __syncwarp();
        QSB_TREE_DOWN_WLVL(448,32,0xffffffffu,1,1)
        __syncthreads();
#else
        QSB_TREE_DOWN_LVL(504,4,0)
        QSB_TREE_DOWN_LVL(496,8,0)
        QSB_TREE_DOWN_LVL(480,16,0)
        QSB_TREE_DOWN_LVL(448,32,1)
#endif
        QSB_TREE_DOWN_LVL(384,64,1)
        QSB_TREE_DOWN_LVL(256,128,1)
        offset=0;
    }else
#endif
    {
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
            QSB_TREE_MUL(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child_inv[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
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
        QSB_TREE_MUL(value,parent_inv,sibling);
    }
    value[4]=0;
}
#endif
