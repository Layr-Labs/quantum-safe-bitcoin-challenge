// Ranked short-epoch pipeline with external, hierarchical batch inversion.
//
// Port onto the promoted subset tree (i34-9/dun999 65cd138, on alvaroborras
// d277241 and welttowelt/odinfree 106a682) of the external inversion pipeline
// that hybridnoise et al. ported to subset in 8e5cd89 (ranked_pipeline.cuh),
// which in turn derives from the pinning pipeline of alvaroborras PR24
// (6e76a74) / nullforest8200 (4d39b5f, dev 6d81454). GPL-3.0; see COPYING.
//
// What changes relative to the promoted monolithic kernel_digest: the ranked
// short-epoch launch is split into five kernels. The search CTAs no longer
// hold 255 idle lanes while lane zero runs _ModInv: prepare publishes one
// product-tree root per CTA and checkpoints the 254 internal nodes, the roots
// are grouped 256 at a time and inverted with one _ModInv per launch, and
// finish restores each tree and expands its root inverse. Hash schedule,
// 64 MiB mixed table, deferred XYZZ chain, direct recovery formulas, gate and
// packed hit records are the promoted ones, called unchanged; u2R comes from
// the promoted constant QSB_U2R.
//
// Barriers: the checkpointed trees use the same warp/CTA split as the promoted
// qsb_block_inverse_tree. A level whose next readers only read nodes written by
// lanes of their own 32-lane warp ends with __syncwarp(); every level whose
// readers cross warps ends with __syncthreads(). verify/warp_model.py proves
// the schedule lane by lane.
//
// Canonicalization is lazy, as in the promoted tree and the pinning frontier:
// internal products are exact residues below 2^256 (qsb_field_mul_raw), the
// super-root inverse normalizes its root before _ModInv, and every returned
// leaf inverse is normalized (qsb_field_normalize from tree.cu).
#pragma once

#define QSB_CHECKPOINT_NODES 254
#define QSB_CHECKPOINT_STRIDE 256

/* Split form of a packed 256-leaf product tree. Prepare checkpoints the 254
 * internal non-root nodes to global memory and publishes the root. Finish
 * restores the immutable products, expands the supplied root inverse and
 * returns canonical leaf inverses. Node numbering: leaves 0..255, level
 * (offset,count) = (256,128), (384,64), ..., (508,2), root 510. */
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
            qsb_field_mul_raw(out,a,b);
            int node=offset+count+tid;
            #pragma unroll
            for(int k=0;k<4;k++){
                products[k][node]=out[k];
                if(node<510)
                    checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+node-256]=out[k];
            }
        }
        offset+=count;
        /* Level count/2 reads nodes written by lanes tid and tid+count/4.
         * Both stay inside the reader's warp once count <= 64. After the
         * last level (count==2) lane 0 only reads its own root, so the warp
         * barrier there is a no-op kept for the record tree's code shape. */
        if(count>64)__syncthreads();
        else __syncwarp();
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

    /* Each lane supplies its saved leaf and all but the last two lanes restore
     * one internal node. Lane zero also publishes the external root inverse.
     * One barrier makes both immutable inputs visible to the downward pass. */
    #pragma unroll
    for(int k=0;k<4;k++){
        products[k][tid]=value[k];
        if(tid<QSB_CHECKPOINT_NODES)
            products[k][256+tid]=checkpoint[block_base+(size_t)k*QSB_CHECKPOINT_STRIDE+tid];
        if(tid==0)inverses[k][254]=roots[(size_t)blockIdx.x*4u+k];
    }
    __syncthreads();

    /* If I=1/(L*R) then I*R=1/L and I*L=1/R. Internal inverse index is the
     * product index minus 256. */
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
            qsb_field_mul_raw(child_inv,parent_inv,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)inverses[k][offset-256+tid]=child_inv[k];
        }
        offset-=count<<1;
        /* Level 2*count (and the leaf level after 128) reads inverses written
         * by lanes tid&(count-1): same warp only while count < 32. */
        if(count>=32)__syncthreads();
        else __syncwarp();
    }

    uint64_t parent_inv[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++){
        parent_inv[k]=inverses[k][tid&127];
        sibling[k]=products[k][tid^128];
    }
    parent_inv[4]=sibling[4]=0;
    qsb_field_mul_raw(value,parent_inv,sibling);
    qsb_field_normalize(value);
}

/* Batch the per-search-CTA roots one level further. Groups of 256 roots use
 * the same checkpointed helpers, then one 256-lane CTA batch-inverts all group
 * roots with the promoted block inverse (which normalizes its root). A full
 * 16.8M-candidate launch therefore executes one _ModInv instead of 65,536.
 * Inactive lanes carry the identity. */
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

/* Cross-kernel state: C=ZZ*d^2, Y, W=ZZ^2*d and ZZZ in eight aligned
 * ulonglong2 planes (128 bytes per candidate). The stride is the launch's
 * candidate count, identical for save and restore. */
__device__ __forceinline__ void qsb_save_state(ulonglong2 *state, int stride, int idx,
    const uint64_t *C, const uint64_t *Y, const uint64_t *W, const uint64_t *ZZZ) {
    state[(size_t)0*stride+idx]=make_ulonglong2(C[0],C[1]);
    state[(size_t)1*stride+idx]=make_ulonglong2(C[2],C[3]);
    state[(size_t)2*stride+idx]=make_ulonglong2(Y[0],Y[1]);
    state[(size_t)3*stride+idx]=make_ulonglong2(Y[2],Y[3]);
    state[(size_t)4*stride+idx]=make_ulonglong2(W[0],W[1]);
    state[(size_t)5*stride+idx]=make_ulonglong2(W[2],W[3]);
    state[(size_t)6*stride+idx]=make_ulonglong2(ZZZ[0],ZZZ[1]);
    state[(size_t)7*stride+idx]=make_ulonglong2(ZZZ[2],ZZZ[3]);
}

__device__ __forceinline__ void qsb_load_state(const ulonglong2 *state, int stride, int idx,
    uint64_t *C, uint64_t *Y, uint64_t *W, uint64_t *ZZZ) {
    ulonglong2 a=state[(size_t)0*stride+idx],b=state[(size_t)1*stride+idx];
    C[0]=a.x;C[1]=a.y;C[2]=b.x;C[3]=b.y;
    a=state[(size_t)2*stride+idx];b=state[(size_t)3*stride+idx];
    Y[0]=a.x;Y[1]=a.y;Y[2]=b.x;Y[3]=b.y;
    a=state[(size_t)4*stride+idx];b=state[(size_t)5*stride+idx];
    W[0]=a.x;W[1]=a.y;W[2]=b.x;W[3]=b.y;
    a=state[(size_t)6*stride+idx];b=state[(size_t)7*stride+idx];
    ZZZ[0]=a.x;ZZZ[1]=a.y;ZZZ[2]=b.x;ZZZ[3]=b.y;
}

/* Stage 1: the promoted short-epoch hash, SHA-256d, streamed 15-point
 * fixed-base sum and recovery preparation, exactly as kernel_digest computes
 * them; then save C/Y/W/ZZZ and publish this CTA's product tree. The host
 * launches only complete 256-candidate epochs. */
__global__ void __launch_bounds__(256,2) qsb_ranked_prepare(
    const epoch_desc_t *epochs, const uint8_t *gTable,
    ulonglong2 *state, uint64_t *roots, uint64_t *tree, int count) {
    int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    const epoch_desc_t *desc=epochs+blockIdx.x;
    uint32_t first[8];
    #pragma unroll
    for(int i=0;i<8;i++)first[i]=desc->mid[i];
    qsb_scheduled_window_hash(first,desc,threadIdx.x);
    uint32_t b2[16];
    #pragma unroll
    for(int i=0;i<8;i++)b2[i]=first[i];
    b2[8]=0x80000000;
    #pragma unroll
    for(int i=9;i<15;i++)b2[i]=0;
    b2[15]=0x00000100;
    uint32_t s2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    _SHA256Transform(s2,b2);
    uint64_t z[4];
    z[0]=((uint64_t)s2[6]<<32)|(uint64_t)s2[7];
    z[1]=((uint64_t)s2[4]<<32)|(uint64_t)s2[5];
    z[2]=((uint64_t)s2[2]<<32)|(uint64_t)s2[3];
    z[3]=((uint64_t)s2[0]<<32)|(uint64_t)s2[1];
    uint64_t C[4],Y[4],ZZ[4],ZZZ[4],W[5];
    _FixedBaseSignedXYZZStream(C,Y,ZZ,ZZZ,z,gTable);
    {
        uint64_t xR[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
        qsb_xyzz_finish_prepare(C,ZZ,xR,W);   /* C becomes d; W = ZZ^2*d */
    }
    _ModSqr(C,C); _ModMult(C,ZZ);         /* C = ZZ*d^2 */
    qsb_save_state(state,count,idx,C,Y,W,ZZZ);
    /* A singular lane (W=0) must not poison its CTA: it contributes the
     * identity here and in finish, and is skipped after the collective. */
    if(!(W[0]|W[1]|W[2]|W[3]))W[0]=1;
    qsb_block_product_checkpoint(W,roots,tree);
}

/* Stage 5: restore the tree, obtain 1/W, recover both flags with the promoted
 * direct-XYZZ formulas, hash each compressed key once (ranked single_hash
 * gate) and record hits exactly like kernel_digest's short-epoch path
 * (ZLAB_HITPATH packed records: tag at hit_idx[4p], nine indices at
 * hit_combos[ZLAB_HIT_REC*p]). */
__global__ void __launch_bounds__(256,3) qsb_ranked_finish(
    const epoch_desc_t *epochs, const ulonglong2 *state,
    const uint64_t *roots, const uint64_t *tree, int count,
    uint32_t *hit_count, uint32_t *hit_idx, uint8_t *hit_combos) {
    int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    uint64_t C[4],Y[4],W[4],ZZZ[4],prod[5];
    qsb_load_state(state,count,idx,C,Y,W,ZZZ);
    /* W is the saved, pre-substitution value, so this reproduces prepare's
     * singular-lane decision without a flag. */
    bool usable=(W[0]|W[1]|W[2]|W[3])!=0;
    #pragma unroll
    for(int limb=0;limb<4;limb++)prod[limb]=usable?W[limb]:(limb==0?1ULL:0ULL);
    prod[4]=0;
    qsb_block_inverse_checkpoint(prod,roots,tree);
    if(!usable)return; // the collective is complete before any lane returns
    uint64_t u2rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]};
    uint64_t u2ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};
    uint64_t q1x[4],q2x[4];
    uint32_t y_parities=qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,prod,u2rx,u2ry,q1x,q2x);

    /* Recid 0 first; a hit returns immediately (kernel_digest's order). */
    for(int ri=0;ri<2;ri++){
        uint64_t sx0=ri ? q2x[0] : q1x[0];
        uint64_t sx1=ri ? q2x[1] : q1x[1];
        uint64_t sx2=ri ? q2x[2] : q1x[2];
        uint64_t sx3=ri ? q2x[3] : q1x[3];
        uint32_t x0=(uint32_t)sx0, x1=(uint32_t)(sx0>>32);
        uint32_t x2=(uint32_t)sx1, x3=(uint32_t)(sx1>>32);
        uint32_t x4=(uint32_t)sx2, x5=(uint32_t)(sx2>>32);
        uint32_t x6=(uint32_t)sx3, x7=(uint32_t)(sx3>>32);
        uint32_t pb[16];
        pb[0]=__byte_perm(x7,0x2+(uint8_t)((y_parities>>ri)&1u),0x4321);
        pb[1]=__byte_perm(x7,x6,0x0765);pb[2]=__byte_perm(x6,x5,0x0765);
        pb[3]=__byte_perm(x5,x4,0x0765);pb[4]=__byte_perm(x4,x3,0x0765);
        pb[5]=__byte_perm(x3,x2,0x0765);pb[6]=__byte_perm(x2,x1,0x0765);
        pb[7]=__byte_perm(x1,x0,0x0765);pb[8]=__byte_perm(x0,0x80,0x0456);
        pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
        uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
        if(gpu_bench_valid_words(hs)){
            uint32_t p=atomicAdd(hit_count,1);
            if(p<1024){
                const epoch_desc_t *desc=epochs+blockIdx.x;
#if ZLAB_HITPATH
                hit_idx[p*4]=((uint32_t)idx)|((uint32_t)ri<<30);
                for(int i=0;i<6;i++)hit_combos[p*ZLAB_HIT_REC+i]=desc->early[i];
                for(int i=0;i<3;i++)hit_combos[p*ZLAB_HIT_REC+6+i]=WIN3[threadIdx.x][i];
#else
                hit_idx[p]=((uint32_t)idx)|((uint32_t)ri<<30);
                for(int i=0;i<6;i++)hit_combos[p*MAX_T+i]=desc->early[i];
                for(int i=0;i<3;i++)hit_combos[p*MAX_T+6+i]=WIN3[threadIdx.x][i];
#endif
            }
            return;
        }
    }
}

/* Launch order on the default stream: prepare, group prepare, super-root
 * inverse, group finish, finish. Requires complete 256-candidate blocks. */
/* QSB_U2R must already hold u2R (main() uploads it before any launch). */
static cudaError_t qsb_launch_ranked_pipeline(int blocks, int count,
    const epoch_desc_t *epochs, const uint8_t *gTable, ulonglong2 *state,
    uint64_t *roots, uint64_t *tree, uint64_t *super_roots, uint64_t *root_tree,
    uint32_t *hit_count, uint32_t *hit_idx, uint8_t *hit_combos) {
    if(blocks<1 || blocks>65536 || count!=blocks*256)return cudaErrorInvalidValue;
    int groups=(blocks+255)/256;
    cudaError_t e;
    qsb_ranked_prepare<<<blocks,256>>>(epochs,gTable,state,roots,tree,count);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_root_group_prepare<<<groups,256>>>(roots,blocks,super_roots,root_tree);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_invert_super_roots<<<1,256>>>(super_roots,groups);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_root_group_finish<<<groups,256>>>(roots,blocks,super_roots,root_tree);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_ranked_finish<<<blocks,256>>>(epochs,state,roots,tree,count,hit_count,hit_idx,hit_combos);
    return cudaGetLastError();
}
