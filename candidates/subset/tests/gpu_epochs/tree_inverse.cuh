// Paired two-CTA layout: separate immutable products from downward inverses.
// This restores the audited earlier tree layout, retaining current root-entry
// warp synchronization. The extra 8 KiB removes destructive-read barriers.
// One work-efficient binary product tree per block. The caller supplies
// a power-of-two block size at most 256 and identity factors for inactive lanes.
#pragma once
/* QSB_LOSS_ROOTLAZY (default 0 = the text below byte for byte): the wave-top root skips qsb_field_normalize before the
 * divsteps. The lean tree product leaves the root below 2^256; it is at least p only with probability about 2^-224, and
 * the divsteps take any 256-bit g. */
#ifndef QSB_LOSS_ROOTLAZY
#define QSB_LOSS_ROOTLAZY 1
#endif
#ifndef QSB_ISO_FUSED_ROOT_SCALE
#define QSB_ISO_FUSED_ROOT_SCALE 1
#endif
#include "hm39_pair_inverse.cuh"
#include "hm41_quad_inverse.cuh"
#include "hm43_warp_inverse.cuh"
#include "zinv32.cuh"
#ifndef QSB_INVERSE_LIMBS
#define QSB_INVERSE_LIMBS 1
#endif
#if QSB_INVERSE_LIMBS
#include "inverse_limbs.cuh"
#endif
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
#ifndef QSB_ISO_ROOT_SCALE
#define QSB_ISO_ROOT_SCALE 1
#endif
#if QSB_ISO_ROOT_SCALE && !QSB_ISO_FUSED_ROOT_SCALE
__device__ __noinline__ void qsb_iso_scale_tree_inverse(uint64_t *value){
    uint64_t invu[5]={QSB_ISO_INVU[0],QSB_ISO_INVU[1],QSB_ISO_INVU[2],QSB_ISO_INVU[3],0};
    QSB_TREE_MUL(value,value,invu);
}
#define QSB_ISO_SCALE_ROOT(value) qsb_iso_scale_tree_inverse(value)
#else
#define QSB_ISO_SCALE_ROOT(value) ((void)0)
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
        QSB_ISO_SCALE_ROOT(root);
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
        QSB_ISO_SCALE_ROOT(root);
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
#ifndef QSB_SC_PARK
#define QSB_SC_PARK 1   /* 1 = the product arena is a file-scope array that kernel_digest also uses to park prodA across B's front call; on with QSB_Y_PAIR=1, 0 = the record's arena */
#endif
#if QSB_SC_PARK
#if QSB_TREE_ROW128
/* QSB_TREE_ROW128 (tree.cu): the product and inverse rows as limb pairs, so each node is two 16-byte words */
__shared__ __align__(16) ulonglong2 qsb_sc_products2[2][512];
#else
__shared__ uint64_t qsb_sc_products[4][512];
#endif
#endif
#if QSB_TREE_ROW128
#define QTR_LD4(A,col,v) do{const ulonglong2 qtr0_=(A)[0][(col)],qtr1_=(A)[1][(col)];(v)[0]=qtr0_.x;(v)[1]=qtr0_.y;(v)[2]=qtr1_.x;(v)[3]=qtr1_.y;}while(0)
#define QTR_ST4(A,col,v) do{(A)[0][(col)]=make_ulonglong2((v)[0],(v)[1]);(A)[1][(col)]=make_ulonglong2((v)[2],(v)[3]);}while(0)
#endif
/* QSB_ROOT_COMBINE (tree.cu switch block): the protocol lives in root_combine.cuh and is compiled only into the
 * native sm_89 image (sm_70+ atomics with .acquire/.release and __nanosleep). The JIT / ranked sm_52 pass keeps the
 * solo zi_inverse_limbs call below, which is the same inverse. */
#if QSB_ROOT_COMBINE && defined(QSB_CARRIER_BUILD) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define QSB_RC_ACTIVE 1
#include "root_combine.cuh"
#else
#define QSB_RC_ACTIVE 0
#endif
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_FILL
/*. LUT_ISSUED (QSB_ROOT_LUT_SMEM): 1 = the caller already issued qsb_root_lut_issue (kernel_digest
 * does it at kernel start), 0 = the tree issues it here. Idle (QSB_PRE3_ROOT): work that warps 1..n/32-1 run
 * on the wave-top branch while warp 0 runs the root, before the down-sweep barrier they wait at anyway. */
#if QSB_ROOT_FILL
struct QsbTreeNoIdle{__device__ __forceinline__ void operator()()const{} __device__ __forceinline__ void waves_done()const{}
                     __device__ __forceinline__ void after_down32()const{} __device__ __forceinline__ void root_done()const{}};
#else
struct QsbTreeNoIdle{__device__ __forceinline__ void operator()()const{}};
#endif
#if QSB_ROOT_LUT_SMEM
#define inverses qsb_tree_inverses_smem   /* file-scope rows (zinv32.cuh): the divstep LUT rides in them */
#endif
#if QSB_ROOT_FILL
/* QSB_ROOT_FILL (tree.cu): the inverse rows are file-scope, because kernel_digest parks the next unit's window-block
 * states in their dead columns (0..223 after the leaf step; 240..255, which the tree never uses) */
__shared__ uint64_t qsb_rf_inv[4][256];
#define inverses qsb_rf_inv
#endif
/* RW (QSB_ROOT_WARP, tree.cu): the warp whose lanes form the top of the tree (up levels with at
 * most 32 writers, the waves, the root, down level 32). Node and inverse indices use the lane's index in that
 * warp, so every node is the same product of the same operands as with warp 0; RW 0 is the base code. */
template<int LUT_ISSUED,class Idle,int RW=0>
__device__ __forceinline__ void qsb_block_inverse_tree_x(uint64_t *value,const Idle &idle){
#else
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
#endif
#if QSB_TREE_ROW128
    ulonglong2 (&products2)[2][512] = qsb_sc_products2;
    __shared__ __align__(16) ulonglong2 inverses2[2][256];
#else
#if QSB_SC_PARK
    uint64_t (&products)[4][512] = qsb_sc_products;
#else
    __shared__ uint64_t products[4][512];
#endif
#if !QSB_ROOT_LUT_SMEM && !QSB_ROOT_FILL
    __shared__ uint64_t inverses[4][256];
#endif
#endif
#if QSB_TREE_UNROLL
    static_assert(QSB_SE_BLOCK==256,"QSB_TREE_UNROLL (tree.cu): the tree is written out for 256-thread kernel_digest blocks");
    const int tid=threadIdx.x;constexpr int n=QSB_SE_BLOCK;
#else
    const int tid=threadIdx.x,n=blockDim.x;
#endif
#if QSB_TREE_ROW128
    QTR_ST4(products2,tid,value);
#else
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
#endif
#if QSB_ROOT_LUT_SMEM
    /* The table words land in the dead inverses rows (flat word i < 832 at ((uint64_t*)inverses)[i]);
     * this barrier publishes them to warp 0. The first tree write into these rows is the root section's
     * inverses[k][offset-n+tid], after the root's last lookup, by the same warp. */
    if(!LUT_ISSUED)qsb_root_lut_issue(tid,n);
    qsb_root_lut_wait();
#endif
    __syncthreads();
    // Level (offset,count): (0,n),(n,n/2),...,(2n-4,2). Level `count` is
    // formed by lanes < count/2 and read by lanes < count/4.
    int offset=0;
#if QSB_TREE_UNROLL
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int count=n;count>(QSB_TREE_WAVE_TOP?16:2);count>>=1){
        int half=count>>1;
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_FILL
        const int ut=(RW && half<=32)?tid-32*RW:tid;   /*: levels with <= 32 writers on warp RW */
        if(RW?(unsigned)ut<(unsigned)half:tid<half){
#else
        const int ut=tid;
        if(tid<half){
#endif
            uint64_t a[5],b[5],out[5];
#if QSB_TREE_ROW128
            QTR_LD4(products2,offset+ut,a);QTR_LD4(products2,offset+half+ut,b);
#else
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+ut];b[k]=products[k][offset+half+ut];}
#endif
            a[4]=b[4]=0;
            QSB_TREE_MUL(out,a,b);
#if QSB_TREE_ROW128
            QTR_ST4(products2,offset+count+ut,out);
#else
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+ut]=out[k];
#endif
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
    // offset == 2n-4: the two root children.
#if HM43_WARP_ROOT
#ifndef QSB_ROOT_UNIFORM_WARP
#define QSB_ROOT_UNIFORM_WARP 1
#endif
#if QSB_ROOT_UNIFORM_WARP && QSB_INVERSE_LIMBS
    /* Exact: the root runs on warp 0 only. A warp vote is warp-uniform to ptxas, so the branch
     * below is uniform and the root's shfl/ballot need no per-op WARPSYNC subroutine
     * (CALL/RET wrappers). The same threads execute the same code as before. */
    if(!QSB_TREE_WAVE_TOP && __all_sync(0xffffffffu,tid<32)){
#else
    if(tid<(QSB_INVERSE_LIMBS?32:4)){
#endif
#if !QSB_TREE_ROW128   /* QSB_TREE_ROW128 requires the wave top: this block is dead there and uses the 64-bit rows */
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(QSB_INVERSE_LIMBS?0xffffffffu:0x0000000fu);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
        root[4]=0;
#if QSB_INVERSE_LIMBS
        zi_inverse_limbs(root,tid);
#else
        zi_inverse_quad(root,tid);
#endif
        if(tid==0)QSB_ISO_SCALE_ROOT(root);
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=__shfl_sync(QSB_INVERSE_LIMBS?0xffffffffu:0x0000000fu,root[k],0);
        if(tid<2){
            uint64_t child[5];
            #pragma unroll
            for(int k=0;k<4;k++)child[k]=tid?a[k]:b[k];
            child[4]=0;QSB_TREE_MUL(child,root,child);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+tid]=child[k];
        }
#endif
    }
#if QSB_TREE_WAVE_TOP
#if !(QSB_ROOT_UNIFORM_WARP && QSB_INVERSE_LIMBS)
#error "QSB_TREE_WAVE_TOP is written for the uniform warp-0 root (QSB_ROOT_UNIFORM_WARP, QSB_INVERSE_LIMBS)"
#endif
    // QSB_TREE_WAVE_TOP (tree.cu): the base root block above is compiled out; offset == 2n-32, the
    // sixteen L16 nodes x[j]. P8, P4, P2 go to their base columns (the base's up levels 16, 8, 4);
    // c, d and E16 stay in the registers of lanes 0..15.
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_FILL
    if(__all_sync(0xffffffffu,RW?(unsigned)(tid-32*RW)<32u:tid<32)){
        const int lt=RW?tid-32*RW:tid;   /*: lane index inside the root warp RW */
#else
    if(__all_sync(0xffffffffu,tid<32)){
        const int lt=tid;
#endif
        const int l8=offset+16,l4=offset+24,l2=offset+28;
        const bool cof=lt<16;
        uint64_t a[5],b[5],r[5];
#if QSB_TREE_LANEMASK
        /* QSB_TREE_LANEMASK (tree.cu): each wave's loads and product run only on the lanes whose result is stored
         * or read later (A: 8, B: 20, C: 18, D: 17, inv16: 16); the other lanes computed unstored copies. */
        #pragma unroll
        for(int k=0;k<5;k++)r[k]=0;
#define QSB_TLM(cond) if(cond)
#else
#define QSB_TLM(cond)
#endif
#if QSB_WAVE_ROLL
#if !QSB_TREE_LANEMASK
#error "QSB_WAVE_ROLL requires the existing wave lane masks"
#endif
        /* Exactly the four existing waves, in their existing operand order.
         * A/B load the first operand; C/D keep the preceding lane product.
         * The shared products and warp barriers retain their original owners. */
        #pragma unroll 1
        for(int stage=0;stage<4;stage++){
            const int half=8>>stage;
            const int limit=stage==0?8:16+half;
            if(lt<limit){
                int ib;
                if(stage==0)ib=offset+8+(lt&7);
                else{
                    const int prev=offset+32-(32>>stage);
                    ib=cof?prev+((lt&(2*half-1))^half):prev+half+(lt&(half-1));
                }
                if(stage<2){
                    const int ia=stage==0?offset+(lt&7):cof?offset+(lt^8):l8+(lt&3);
#if QSB_TREE_ROW128
                    QTR_LD4(products2,ia,a);
#else
                    #pragma unroll
                    for(int k=0;k<4;k++)a[k]=products[k][ia];
#endif
                }else{
                    #pragma unroll
                    for(int k=0;k<4;k++)a[k]=r[k];
                }
#if QSB_TREE_ROW128
                QTR_LD4(products2,ib,b);
#else
                #pragma unroll
                for(int k=0;k<4;k++)b[k]=products[k][ib];
#endif
                a[4]=b[4]=0;QSB_TREE_MUL(r,a,b);
                const bool writer=stage==0?lt<8:(unsigned)(lt-16)<(unsigned)half;
                if(stage<3 && writer){
                    const int dst=offset+32-(16>>stage)+(lt&(half-1));
#if QSB_TREE_ROW128
                    QTR_ST4(products2,dst,r);
#else
                    #pragma unroll
                    for(int k=0;k<4;k++)products[k][dst]=r[k];
#endif
                }
            }
            if(stage<3)__syncwarp();
        }
#else
        // Wave A: P8[j] = x[j]*x[j+8] on lanes j < 8 (lanes 8..31 repeat them, unstored).
        QSB_TLM(lt<8){
#if QSB_TREE_ROW128
        QTR_LD4(products2,offset+(lt&7),a);QTR_LD4(products2,offset+8+(lt&7),b);
#else
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset+(lt&7)];b[k]=products[k][offset+8+(lt&7)];}
#endif
        a[4]=b[4]=0;QSB_TREE_MUL(r,a,b);
        if(lt<8){
#if QSB_TREE_ROW128
            QTR_ST4(products2,l8+lt,r);
#else
            #pragma unroll
            for(int k=0;k<4;k++)products[k][l8+lt]=r[k];
#endif
        }
        }
        __syncwarp();
        // Wave B: c[i] = x[i^8]*P8[(i&7)^4] on lanes i < 16; P4[j] = P8[j]*P8[j+4] on lanes 16+j, j < 4.
        QSB_TLM(lt<20){
            const int ia=cof?offset+(lt^8):l8+(lt&3);
            const int ib=cof?l8+((lt&7)^4):l8+4+(lt&3);
#if QSB_TREE_ROW128
            QTR_LD4(products2,ia,a);QTR_LD4(products2,ib,b);
#else
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][ia];b[k]=products[k][ib];}
#endif
            a[4]=b[4]=0;QSB_TREE_MUL(r,a,b);
            if((unsigned)(lt-16)<4u){
#if QSB_TREE_ROW128
                QTR_ST4(products2,l4+(lt&3),r);
#else
                #pragma unroll
                for(int k=0;k<4;k++)products[k][l4+(lt&3)]=r[k];
#endif
            }
        }
        __syncwarp();
        // Wave C: d[i] = c[i]*P4[(i&3)^2] on lanes i < 16; P2[j] = P4[j]*P4[j+2] on lanes 16+j, j < 2
        // (the first operand is every lane's own wave-B product: c[i], or P4[j] on lane 16+j).
        QSB_TLM(lt<18){
            const int ib=cof?l4+((lt&3)^2):l4+2+(lt&1);
#if QSB_TREE_ROW128
            QTR_LD4(products2,ib,b);
#else
            #pragma unroll
            for(int k=0;k<4;k++)b[k]=products[k][ib];
#endif
            b[4]=0;QSB_TREE_MUL(r,r,b);
            if((unsigned)(lt-16)<2u){
#if QSB_TREE_ROW128
                QTR_ST4(products2,l2+(lt&1),r);
#else
                #pragma unroll
                for(int k=0;k<4;k++)products[k][l2+(lt&1)]=r[k];
#endif
            }
        }
        __syncwarp();
        // Wave D: E16[i] = d[i]*P2[(i&1)^1] on lanes i < 16; the root P2[0]*P2[1] on lane 16 (own P2[0]).
        QSB_TLM(lt<17){
            const int ib=cof?l2+((lt&1)^1):l2+1;
#if QSB_TREE_ROW128
            QTR_LD4(products2,ib,b);
#else
            #pragma unroll
            for(int k=0;k<4;k++)b[k]=products[k][ib];
#endif
            b[4]=0;QSB_TREE_MUL(r,r,b);
        }
#endif /* QSB_WAVE_ROLL */
#if QSB_ROOT_FILL
        idle.waves_done();   /* QSB_ROOT_FILL: product columns 480..511 are dead from here on (bar.arrive) */
#endif
        uint64_t root[5];
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=__shfl_sync(0xffffffffu,r[k],16);
#if !QSB_LOSS_ROOTLAZY
        qsb_field_normalize(root);
#endif
        root[4]=0;
#if QSB_RC_ACTIVE
        qsb_root_combine_invert(root,lt);   /* QSB_ROOT_COMBINE: the same canonical inverse, shared with a partner block */
#else
        zi_inverse_limbs(root,lt);
#endif
        if(lt==0)QSB_ISO_SCALE_ROOT(root);
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=__shfl_sync(0xffffffffu,root[k],0);
        // inv16[i] = root^-1 * E16[i]: the inverse of x[i], at the base's level-16 inverse column.
        QSB_TLM(cof){
        r[4]=0;QSB_TREE_MUL(r,root,r);
        if(cof){
#if QSB_TREE_ROW128
            QTR_ST4(inverses2,offset-n+lt,r);
#else
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+lt]=r[k];
#endif
        }
        }
#undef QSB_TLM
#if QSB_ROOT_FILL
        idle.root_done();
#endif
    }
#if QSB_PRE3_ROOT || QSB_ROOT_FILL
    else idle();   /* warps 1..: QSB_PRE3_ROOT (tree.cu), before the down-sweep barrier below */
#endif
#endif
#elif HM41_QUAD_ROOT
    if(tid<4){
        uint64_t a[5],b[5],root[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][offset];b[k]=products[k][offset+1];}
        a[4]=b[4]=0;__syncwarp(0x0000000f);QSB_TREE_MUL(root,a,b);qsb_field_normalize(root);
        root[4]=0;hm41_quad_inverse(root,tid);
        if(tid==0)QSB_ISO_SCALE_ROOT(root);
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=__shfl_sync(0x0000000f,root[k],0);
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
        if(tid==0)QSB_ISO_SCALE_ROOT(root);
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=__shfl_sync(0x00000003,root[k],0);
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
        QSB_ISO_SCALE_ROOT(root);
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
    offset-=QSB_TREE_WAVE_TOP?32:4;   /* level count=4 (QSB_TREE_WAVE_TOP: 32, the waves wrote level 16) */
#if QSB_TREE_UNROLL
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int count=QSB_TREE_WAVE_TOP?32:4;count<n;count<<=1){
        int half=count>>1;
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_FILL
        const int ut=(RW && count<=32)?tid-32*RW:tid;   /*: down level 32 on warp RW (it reads RW's inverses) */
        if(RW?(unsigned)ut<(unsigned)count:tid<count){
#else
        const int ut=tid;
        if(tid<count){
#endif
            uint64_t parent_inv[5],sibling[5],child_inv[5];
#if QSB_TREE_ROW128
            QTR_LD4(inverses2,offset+count-n+(ut&(half-1)),parent_inv);QTR_LD4(products2,offset+(ut^half),sibling);
#else
            #pragma unroll
            for(int k=0;k<4;k++){
                parent_inv[k]=inverses[k][offset+count-n+(ut&(half-1))];
                sibling[k]=products[k][offset+(ut^half)];
            }
#endif
            parent_inv[4]=sibling[4]=0;
            QSB_TREE_MUL(child_inv,parent_inv,sibling);
#if QSB_TREE_ROW128
            QTR_ST4(inverses2,offset-n+ut,child_inv);
#else
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-n+ut]=child_inv[k];
#endif
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
#if QSB_ROOT_FILL
        if(count==32)idle.after_down32();   /* QSB_ROOT_FILL: the level-16 inverse columns are dead now */
#endif
    }
    // offset == 0 would be the leaf level; lanes form their own leaf inverse.
    {
        const int half=n>>1;
        uint64_t parent_inv[5],sibling[5];
#if QSB_TREE_ROW128
        QTR_LD4(inverses2,tid&(half-1),parent_inv);QTR_LD4(products2,tid^half,sibling);
#else
        #pragma unroll
        for(int k=0;k<4;k++){
            parent_inv[k]=inverses[k][tid&(half-1)];
            sibling[k]=products[k][tid^half];
        }
#endif
        parent_inv[4]=sibling[4]=0;
        QSB_TREE_MUL(value,parent_inv,sibling);
    }
    value[4]=0;
}
#if QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT || QSB_ROOT_FILL
#if QSB_ROOT_LUT_SMEM || QSB_ROOT_FILL
#undef inverses
#endif
__device__ __forceinline__ void qsb_block_inverse_tree(uint64_t *value){
    qsb_block_inverse_tree_x<0>(value,QsbTreeNoIdle());
}
#endif
#endif
#if (QSB_ROOT_LUT_SMEM || QSB_PRE3_ROOT) && ZLAB_TREE != 2
#error "QSB_ROOT_LUT_SMEM and QSB_PRE3_ROOT (tree.cu) are written for the level-packed tree (ZLAB_TREE 2)"
#endif
#if QSB_ROOT_LUT_SMEM && !(HM43_WARP_ROOT && QSB_INVERSE_LIMBS)
#error "QSB_ROOT_LUT_SMEM (tree.cu) is written for the warp-0 limbs root (HM43_WARP_ROOT, QSB_INVERSE_LIMBS)"
#endif
#if QSB_TREE_WAVE_TOP && (ZLAB_TREE != 2 || !HM43_WARP_ROOT)
#error "QSB_TREE_WAVE_TOP (tree.cu) is written for the level-packed tree (ZLAB_TREE 2) with the warp-0 root (HM43_WARP_ROOT)"
#endif
