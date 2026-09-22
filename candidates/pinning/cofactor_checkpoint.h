// Dependency-scoped barrier mechanism follows Calcutatatoraa95b1b9;
// applied here to the distinct public cofactor exclusion traversal.
// Public cofactor collective: tekkac, submission31e98e47, commit554fa24c.
// Merged top-16 traversal (QSB_TOP16): idea and schedule from @EvanYan1024's public
// submission 58005ee5, which credits a Codex (GPT 6 Astra) session for the schedule.
// Re-derived and re-implemented here against this tree's index algebra.
#pragma once

#ifndef QSB_TOP16
#define QSB_TOP16 1           /* 1: merge the top-16 up-sweep and exclusion waves (supersedes QSB_TREE_TOP2) */
#endif

#if QSB_TOP16
/* Merged top of the cofactor tree.
 *
 * After the up-sweep stops at count > 16, the sixteen surviving subtree roots x[0..15]
 * sit at products[2N-32 .. 2N-17]. The unmerged traversal then spends SEVEN waves on
 * them - four up-sweep waves with 8, 4, 2 and 1 active lanes and three exclusion waves
 * with 4, 8 and 16 - in each of which one warp runs a dependent
 * LDS -> field multiply -> STS chain while the rest of the block waits at the barrier.
 *
 * The up-sweep lanes and the exclusion lanes of consecutive waves are independent, so
 * each wave can carry both roles and the region costs FOUR waves instead of seven:
 *
 *   wave   lanes 0..15                             lanes 16+                    active
 *   A      -                                       P2[j] = x[j]*x[j+8],  j<8      8
 *   B      E4[i]  = x[i^8]  * P2[(i&7)^4]          P4[j] = P2[j]*P2[j+4], j<4    20
 *   C      E8[i]  = E4[i]   * P4[(i&3)^2]          P8[j] = P4[j]*P4[j+2], j<2    18
 *   D      E16[i] = E8[i]   * P8[(i&1)^1]          root  = P8[0]*P8[1]           17
 *
 * The up-sweep pairings are the ones the packed traversal above produces (node j of a
 * level pairs with node j+half, and the level's products are appended after it), so
 * P2/P4/P8/root are the same values at the same indices as before.
 *
 * The exclusion factors are also the same. Unrolling the unmerged down-sweep gives
 *     excluded[N-32+i] = x[i^8] * P2[(i&7)^4] * P4[(i&3)^2] * P8[(i&1)^1],
 * built top-down as ((P8 * P4) * P2) * x; the merged form builds the same four factors
 * bottom-up as ((x * P2) * P4) * P8. Same factor set, same three multiplies per entry,
 * so the short-carry convention of the rest of the tree applies unchanged - which is why
 * QSB_TOP16_SC (the default) uses qsb_field_mul_sc, the multiply the tree already uses.
 *
 * The original down-sweep then resumes unchanged at count = 32, offset = 2N-64, which is
 * exactly where it reads excluded[N-32 + (tid & 15)].
 */
#ifndef QSB_TOP16_SC
#define QSB_TOP16_SC 1        /* 1: short-carry multiply in the merged top (matches the rest of the tree) */
#endif
#if QSB_TOP16_SC
#define QSB_TOP16_MUL qsb_field_mul_sc
#else
#define QSB_TOP16_MUL qsb_field_mul
#endif

template<int N> __device__ __forceinline__ void qsb_cofactor_top16(
    uint64_t *roots, uint64_t (*products)[2*N], uint64_t (*excluded)[N]) {
    static_assert(N>=32 && !(N&(N-1)), "power-of-two tree, N>=32");
    const int tid=threadIdx.x;
    const int x=2*N-32, p2=2*N-16, p4=2*N-8, p8=2*N-4, e=N-32;

    if(tid<8) {                                  /* A: the eight pair products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=products[k][x+tid];b[k]=products[k][x+tid+8];}
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++)products[k][p2+tid]=o[k];
    }
    __syncwarp();

    if(tid<20) {                                 /* B: 16 exclusion starts + 4 quad products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=products[k][x+(tid^8)];      b[k]=products[k][p2+((tid&7)^4)];}
            else      {a[k]=products[k][p2+(tid-16)];    b[k]=products[k][p2+(tid-16)+4];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){ if(tid<16) excluded[k][e+tid]=o[k]; else products[k][p4+(tid-16)]=o[k]; }
    }
    __syncwarp();

    if(tid<18) {                                 /* C: extend the exclusions + two half roots */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=excluded[k][e+tid];          b[k]=products[k][p4+((tid&3)^2)];}
            else      {a[k]=products[k][p4+(tid-16)];    b[k]=products[k][p4+(tid-16)+2];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){ if(tid<16) excluded[k][e+tid]=o[k]; else products[k][p8+(tid-16)]=o[k]; }
    }
    __syncwarp();

    if(tid<17) {                                 /* D: finish the exclusions + the block root */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=excluded[k][e+tid];          b[k]=products[k][p8+((tid&1)^1)];}
            else      {a[k]=products[k][p8];             b[k]=products[k][p8+1];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){ if(tid<16) excluded[k][e+tid]=o[k]; else roots[(size_t)blockIdx.x*4+k]=o[k]; }
    }
    __syncwarp();
}
#endif /* QSB_TOP16 */

/* QSB_TREE_FLAT: under the merged top-16 traversal the up-sweep stops at
 * count = 32 (always > 2), the down-sweep starts at count = 32 (never == 2)
 * and N >= 32, so the count>2 barrier test, the count==2 copy arm and the
 * N==2 leaf arm are compile-time dead. Removing them changes no value.
 * (Ported from the public ItlaStudent d1ce6f2b bundle.) */
#ifndef QSB_TREE_FLAT
#define QSB_TREE_FLAT 1
#endif

// The caller supplies nonzero effective leaves (identity for unusable lanes).
// Preserve immutable products and accumulate exclusion products separately.
// All N lanes participate in every barrier; one block publishes one raw root.
template<int N> __device__ __forceinline__ void qsb_cofactor_prepare(
    uint64_t *value,uint64_t *roots,uint64_t (*products)[2*N],uint64_t (*excluded)[N]) {
#if QSB_TOP16
    static_assert(N>=32 && !(N&(N-1)),"power-of-two tree, N>=32");
#else
    static_assert(N>=16 && !(N&(N-1)),"power-of-two tree");
#endif
    int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)products[k][tid]=value[k];
    __syncthreads();
    int offset=0;
    #pragma unroll 1
#if QSB_TOP16
    for(int count=N;count>16;count>>=1) {  /* stop at the sixteen subtree roots: the top is merged */
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
#if QSB_TOP16 && QSB_TREE_FLAT
        if(half>32)__syncthreads();else __syncwarp();
#else
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
#endif
    }
#if QSB_TOP16
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
#if QSB_TOP16 && QSB_TREE_FLAT
            qsb_field_mul_sc(out,parent,sibling);
#else
            if(count==2){Load256(out,sibling);}else{qsb_field_mul_sc(out,parent,sibling);}
#endif
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
#if QSB_TOP16 && QSB_TREE_FLAT
    qsb_field_mul_sc(value,parent,sibling);
#else
    if(N==2){Load256(value,sibling);}else{qsb_field_mul_sc(value,parent,sibling);}
#endif
    value[4]=0;
}
