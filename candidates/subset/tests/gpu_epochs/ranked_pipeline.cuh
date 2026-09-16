// Ranked subset external inversion pipeline.
// Derived from alvaroborras pinning PR24, commit 6e76a74fed8e6e5b8439e64ec20f586085f37d52.
// GPL-3.0; preserve repository COPYING. Port retains subset runtime epoch inputs.
#pragma once
// Runtime recovery point broadcast, adapted from alvaroborras 9c914db3.
__device__ __constant__ uint64_t QSB_U2R[8];
static int qsb_prepare_recovery_point(const uint8_t *rx,const uint8_t *ry) {
    uint64_t limbs[8];memcpy(limbs,rx,32);memcpy(limbs+4,ry,32);
    return cudaMemcpyToSymbol(QSB_U2R,limbs,sizeof(limbs))==cudaSuccess?0:1;
}


__device__ __forceinline__ void qsb_field_normalize(uint64_t *r) {
    if ((r[1] & r[2] & r[3]) == UINT64_MAX &&
        r[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        r[0] -= 0xFFFFFFFEFFFFFC2FULL;
        r[1] = r[2] = r[3] = 0;
    }
}

#define QSB_CHECKPOINT_FIRST 448
#define QSB_CHECKPOINT_NODES (510-QSB_CHECKPOINT_FIRST)
#define QSB_CHECKPOINT_STRIDE 64

/* Split form of qsb_block_inverse. Save only the upper 62 non-root nodes;
 * reconstruct the bottom two levels from the saved W leaves in the finish
 * kernel. This removes 192 field elements of checkpoint traffic per CTA in
 * each direction at the cost of 192 multiplies and one additional barrier.
 * The packed numbering is unchanged: leaves [0,256), pairs [256,384),
 * quadruples [384,448), saved nodes [448,510), root 510. */
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


/* Shared-denominator recovery directly from XYZZ coordinates.
 *
 * P has affine coordinates xP=X/ZZ and yP=Y/ZZZ, with ZZZ^2=ZZ^3.
 * For affine R=(xR,yR), let d=xR*ZZ-X=ZZ*(xR-xP). The collective
 * inverts W=ZZ^2*d. X is dead after d is formed, so overwrite it with d and
 * keep only four field elements live across the block-wide inverse. */
__device__ __forceinline__ void qsb_xyzz_finish_prepare(
    uint64_t *X_D, uint64_t *ZZ, uint64_t *xR, uint64_t *W
) {
    uint64_t t[4];
    _ModMult(t, xR, ZZ);
    _ModSub256(t, t, X_D);
    Load256(X_D, t);             /* X_D becomes d */
    _ModSqr(W, ZZ);
    _ModMult(W, X_D);            /* W = ZZ^2*d */
    W[4] = 0;
}

/* Across the kernel boundary, X_D has been replaced by C=ZZ*d^2 and ZZ by
 * W=ZZ^2*d. With inv=1/W, h=inv*ZZZ=A/(B*d) is the common slope scale and
 * delta=inv*C=d/ZZ=xR-xP. Thus xs=2*xR-delta=xP+xR. The y formulas are
 * anchored at R, avoiding reconstruction of affine yP:
 *   y1 = lambda1*(xR-x1)-yR
 *   y2 = -(m2*(xR-x2)-yR).
 * Returns the two y parities in bits 0 and 1. C, W, and ZZZ are deliberately
 * reused as delta, xs, and h. */
__device__ __forceinline__ uint32_t qsb_xyzz_finish_precomputed(
    uint64_t *C, uint64_t *Y, uint64_t *W, uint64_t *ZZZ,
    uint64_t *inv, uint64_t *xR, uint64_t *yR,
    uint64_t *x1, uint64_t *x2
) {
    uint64_t yb[4], m[4], t[4], s[4];

    _ModMult(yb, yR, ZZZ);       /* yR*B */
    _ModMult(ZZZ, inv);          /* h = B/(A^2*d) = A/(B*d) */

    _ModMult(C, inv);            /* delta = C/W = d/ZZ */
    _ModAdd256(W, xR, xR);
    _ModSub256(W, C);            /* xs = xP+xR = 2*xR-delta */

    _ModSub256(m, yb, Y);
    _ModMult(m, ZZZ);            /* lambda1 = (yR*B-Y)*h */
    _ModSqr(x1, m);
    _ModSub256(x1, W);
    _ModSub256(t, xR, x1);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    uint32_t parities = (uint32_t)(s[0] & 1ULL);

    _ModAdd256(m, yb, Y);
    _ModMult(m, ZZZ);            /* m2 = (yR*B+Y)*h = -lambda2 */
    _ModSqr(x2, m);
    _ModSub256(x2, W);
    _ModSub256(t, xR, x2);
    _ModMult(s, m, t);
    _ModSub256(s, yR);
    /* y2=-s. Since p is odd, field negation flips its parity. */
    parities |= (uint32_t)((((s[0]|s[1]|s[2]|s[3]) != 0) && !(s[0]&1ULL)) << 1);
    return parities;
}


// Eight aligned SoA vector planes; stride is this launch's candidate count.
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

__global__ void __launch_bounds__(256,2) qsb_ranked_prepare(
    const epoch_desc_t *epochs, const uint8_t *gTable,
    const uint64_t *rx, ulonglong2 *state, uint64_t *roots, uint64_t *tree,
    int count) {
    int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    // Host dispatch supplies complete 256-candidate epochs.
    uint32_t first[8];
    const epoch_desc_t *desc=epochs+blockIdx.x;
    #pragma unroll
    for(int i=0;i<8;i++)first[i]=desc->mid[i];
    qsb_scheduled_window_hash(first,desc,threadIdx.x);
    uint32_t b[16];
    #pragma unroll
    for(int i=0;i<8;i++)b[i]=first[i];
    b[8]=0x80000000;
    #pragma unroll
    for(int i=9;i<15;i++)b[i]=0;
    b[15]=0x100;
    uint32_t h[8];_SHA256Initialize(h);_SHA256Transform(h,b);
    uint64_t z[4]={((uint64_t)h[6]<<32)|h[7],((uint64_t)h[4]<<32)|h[5],
                   ((uint64_t)h[2]<<32)|h[3],((uint64_t)h[0]<<32)|h[1]};
    uint64_t C[4],Y[4],ZZ[4],ZZZ[4],W[5],xR[4];
    _FixedBaseSignedXYZZ(C,Y,ZZ,ZZZ,z,gTable);
    Load256(xR,QSB_U2R);
    qsb_xyzz_finish_prepare(C,ZZ,xR,W); // C now d; W = ZZ^2*d.
    _ModSqr(C,C); _ModMult(C,ZZ);     // C = ZZ*d^2.
    qsb_save_state(state,count,idx,C,Y,W,ZZZ);
    if(!(W[0]|W[1]|W[2]|W[3]))W[0]=1; // singular lane cannot poison its batch.
    qsb_block_product_checkpoint(W,roots,tree);
}

__device__ __forceinline__ int qsb_ranked_pubkey_hit(const uint64_t *x, uint32_t parity) {
    const uint32_t *x32=(const uint32_t*)x;
    uint32_t pb[16],prefix=2+(parity&1);
    pb[0]=__byte_perm(x32[7],prefix,0x4321);
    pb[1]=__byte_perm(x32[7],x32[6],0x0765);pb[2]=__byte_perm(x32[6],x32[5],0x0765);
    pb[3]=__byte_perm(x32[5],x32[4],0x0765);pb[4]=__byte_perm(x32[4],x32[3],0x0765);
    pb[5]=__byte_perm(x32[3],x32[2],0x0765);pb[6]=__byte_perm(x32[2],x32[1],0x0765);
    pb[7]=__byte_perm(x32[1],x32[0],0x0765);pb[8]=__byte_perm(x32[0],0x80,0x0456);
    pb[9]=0;pb[10]=0;pb[11]=0;pb[12]=0;pb[13]=0;pb[14]=0;pb[15]=0x108;
    uint32_t hs[8];_SHA256Initialize(hs);_SHA256Transform(hs,pb);
    return gpu_bench_valid_words(hs);
}

__global__ void __launch_bounds__(256,3) qsb_ranked_finish(
    const epoch_desc_t *epochs, const uint64_t *rx, const uint64_t *ry,
    const ulonglong2 *state, const uint64_t *roots, const uint64_t *tree,
    int count, uint32_t *hit_count, uint32_t *hit_idx, uint8_t *hit_combos) {
    int idx=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    uint64_t C[4],Y[4],W[4],ZZZ[4],inv[5];
    qsb_load_state(state,count,idx,C,Y,W,ZZZ);
    bool usable=(W[0]|W[1]|W[2]|W[3])!=0;
    Load256(inv,W);inv[4]=0;
    if(!usable)inv[0]=1;
    qsb_block_inverse_checkpoint(inv,roots,tree);
    if(!usable)return; // Collective is complete before any lane returns.
    uint64_t xR[4],yR[4],x1[4],x2[4];Load256(xR,QSB_U2R);Load256(yR,QSB_U2R+4);
    uint32_t parity=qsb_xyzz_finish_precomputed(C,Y,W,ZZZ,inv,xR,yR,x1,x2);
    int recid=-1;
    if(qsb_ranked_pubkey_hit(x1,parity))recid=0;
    else if(qsb_ranked_pubkey_hit(x2,parity>>1))recid=1;
    if(recid>=0){
        uint32_t p=atomicAdd(hit_count,1);
        if(p<1024){
            hit_idx[p]=(uint32_t)idx|((uint32_t)recid<<30);
            const epoch_desc_t *desc=epochs+blockIdx.x;
            for(int i=0;i<6;i++)hit_combos[p*MAX_T+i]=desc->early[i];
            for(int i=0;i<3;i++)hit_combos[p*MAX_T+6+i]=WIN3[threadIdx.x][i];
        }
    }
}

static cudaError_t qsb_launch_ranked_pipeline(int blocks, int count,
    const epoch_desc_t *epochs, const uint8_t *gTable,
    const uint64_t *rx, const uint64_t *ry, ulonglong2 *state,
    uint64_t *roots,uint64_t *tree,uint64_t *super_roots,uint64_t *root_tree,
    uint32_t *hit_count,uint32_t *hit_idx,uint8_t *hit_combos) {
    if(blocks<1 || blocks>65536 || count!=blocks*256)return cudaErrorInvalidValue;
    int groups=(blocks+255)/256;
    cudaError_t e;
    qsb_ranked_prepare<<<blocks,256>>>(epochs,gTable,rx,state,roots,tree,count);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_root_group_prepare<<<groups,256>>>(roots,blocks,super_roots,root_tree);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_invert_super_roots<<<1,256>>>(super_roots,groups);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_root_group_finish<<<groups,256>>>(roots,blocks,super_roots,root_tree);
    if((e=cudaGetLastError())!=cudaSuccess)return e;
    qsb_ranked_finish<<<blocks,256>>>(epochs,rx,ry,state,roots,tree,count,hit_count,hit_idx,hit_combos);
    return cudaGetLastError();
}
