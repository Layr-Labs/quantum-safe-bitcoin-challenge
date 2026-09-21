// Dependency-scoped barrier mechanism follows Calcutatatoraa95b1b9;
// applied here to the distinct public cofactor exclusion traversal.
// Public cofactor collective: tekkac, submission31e98e47, commit554fa24c.
#pragma once

// The caller supplies nonzero effective leaves (identity for unusable lanes).
// Preserve immutable products and accumulate exclusion products separately.
// All N lanes participate in every barrier; one block publishes one raw root.
#ifndef QSB_TREE_TOP16
#define QSB_TREE_TOP16 1   /* four merged waves over the top 16 nodes (supersedes QSB_TREE_TOP2) */
#endif
#if QSB_TREE_TOP16
// Merged top of the cofactor tree. The last 16 subtree roots x[0..15] live at
// products[2N-32 .. 2N-17]; the promoted traversal spends four up-sweep waves and three
// exclusion waves on them (QSB_TREE_TOP2 merges two of those seven into one, leaving six).
// Here each wave carries both roles, so one warp covers the region in four waves.
// The factor SET of every exclusion is the promoted one; the factors are applied bottom-up
// (x, P2, P4, P8) instead of top-down (P8, P4, P2, x), which is a different association of
// the same three short-carry multiplies -- the same per-operation bound the promoted tree
// already accepts, not a bit-identical rewrite for constructed raw inputs.
template<int N> __device__ __forceinline__ void qsb_cofactor_top16(
    uint64_t *roots, uint64_t (*products)[2*N], uint64_t (*excluded)[N]) {
    static_assert(N>=32 && !(N&(N-1)), "power-of-two tree, N>=32");
    const int tid=threadIdx.x;
    const int x=2*N-32, p2=2*N-16, p4=2*N-8, p8=2*N-4, e=N-32;
    if(tid<8) {                                    /* A: eight pair products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][x+tid];b[k]=products[k][x+tid+8];}
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][p2+tid]=o[k];
    }
    __syncwarp();
    if(tid<20) {                                   /* B: 16 exclusions + 4 quad products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=products[k][x+(tid^8)];b[k]=products[k][p2+((tid&7)^4)];}
            else{a[k]=products[k][p2+tid-16];b[k]=products[k][p2+tid-16+4];}
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){if(tid<16)excluded[k][e+tid]=o[k];else products[k][p4+tid-16]=o[k];}
    }
    __syncwarp();
    if(tid<18) {                                   /* C: extend exclusions + two half roots */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=excluded[k][e+tid];b[k]=products[k][p4+((tid&3)^2)];}
            else{a[k]=products[k][p4+tid-16];b[k]=products[k][p4+tid-16+2];}
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){if(tid<16)excluded[k][e+tid]=o[k];else products[k][p8+tid-16]=o[k];}
    }
    __syncwarp();
    if(tid<17) {                                   /* D: final exclusions + the root */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=excluded[k][e+tid];b[k]=products[k][p8+((tid&1)^1)];}
            else{a[k]=products[k][p8];b[k]=products[k][p8+1];}
        }
        a[4]=b[4]=0;qsb_field_mul_sc(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){if(tid<16)excluded[k][e+tid]=o[k];else roots[(size_t)blockIdx.x*4+k]=o[k];}
    }
    __syncwarp();
}
#endif

template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
    uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
    static_assert(N>=16 && !(N&(N-1)),"power-of-two tree");
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    int offset=0;
    #pragma unroll 1
#if QSB_TREE_TOP16
    for(int count=N;count>16;count>>=1) {  /* the top 16 nodes are finished by the merged waves */
#elif QSB_TREE_TOP2
    for(int count=N;count>2;count>>=1) {   /* stop below the root: the top pair is merged into the down-sweep */
#else
    for(int count=N;count>1;count>>=1) {
#endif
        int half=count>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=products[k][offset+tid];b[k]=products[k][offset+half+tid];}
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][offset+count+tid]=out[k];
        }
        offset+=count;
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
    }
#if QSB_TREE_TOP16
    qsb_cofactor_top16<N>(roots,products,excluded);
    offset=2*N-64;
    #pragma unroll 1
    for(int count=32;count<N;count<<=1) {
#elif QSB_TREE_TOP2
    /* P12: the top pair n0=products[2N-4], n1=products[2N-3] needs no separate root level
     * and no copy level: one warp-multiply gives the root n0*n1 (lane 4) together with the
     * four excluded products of the level below, E(c)=E(parent)*sibling with E(n0)=n1 and
     * E(n1)=n0 (lanes 0..3). Same operands in the same order as the two levels it replaces,
     * so the root and every excluded product are bit-identical. */
    if(tid<5) {
        uint64_t a[5],b[5],out[5];
        const int ia=tid<4 ? 2*N-4+((tid&1)^1) : 2*N-4;
        const int ib=tid<4 ? 2*N-8+(tid^2) : 2*N-3;
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][ia];b[k]=products[k][ib];}
        a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
        if(tid<4) {
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][N-8+tid]=out[k];
        } else {
            #pragma unroll
            for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=out[k];
        }
    }
    __syncwarp();
    offset=2*N-16;
    #pragma unroll 1
    for(int count=8;count<N;count<<=1) {
#else
    if(tid==0) {
        #pragma unroll
        for(int k=0;k<4;k++) {
            roots[(size_t)blockIdx.x*4+k]=products[k][2*N-2];
            excluded[k][N-2]=k==0?1:0;
        }
    }
    __syncwarp();
    offset=2*N-4;
    #pragma unroll 1
    for(int count=2;count<N;count<<=1) {
#endif
        int half=count>>1;
        if(tid<count) {
            uint64_t parent[5],sibling[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=excluded[k][offset+count-N+(tid&(half-1))];
                sibling[k]=products[k][offset+(tid^half)];
            }
            parent[4]=sibling[4]=0;
            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][offset-N+tid]=out[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t parent[5],sibling[5];
    #pragma unroll
    for(int k=0;k<4;k++) {
        parent[k]=excluded[k][tid&(N/2-1)];
        sibling[k]=products[k][tid^(N/2)];
    }
    parent[4]=sibling[4]=0;
    if(N==2){Load256(value,sibling);}else{qsb_field_mul_sc(value,parent,sibling);}
    value[4]=0;
}
