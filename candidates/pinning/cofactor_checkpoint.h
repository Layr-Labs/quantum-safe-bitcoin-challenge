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
#ifndef QSB_TOP16_REG
#define QSB_TOP16_REG 1       /* 1: waves B/C/D keep their own exclusion factor in a register */
#endif
#ifndef QSB_TOP16_PEEL
#define QSB_TOP16_PEEL 1      /* 1: peel the count=32 down-sweep level; lanes 0..15 reuse wave D's register */
#endif
#if QSB_TOP16_SC
#define QSB_TOP16_MUL qsb_field_mul_sc
#else
#define QSB_TOP16_MUL qsb_field_mul
#endif

/* Wave-to-wave forwarding (QSB_TOP16_REG). Lanes 0..15 of waves B, C and D form the
 * dependent chain E4 -> E8 -> E16: each of those lanes reads back in the next wave
 * exactly the word it wrote in the previous one, and no other lane reads excluded[e+i]
 * before the merged top returns (waves C and D read only products[p4]/products[p8],
 * which are written by lanes 16+). The two intermediate exclusion planes are therefore
 * private to their lane and stay in registers: waves B and C lose 8 STS and waves C and
 * D lose 8 LDS, and - the reason it is worth doing - two shared-memory round trips come
 * off a chain that the other 96 threads of the block spend at the barrier. Only wave D
 * still publishes excluded[e+i], which the down-sweep's lanes 16..31 read. The operand
 * values and the multiply order are bit-identical either way.
 */

template<int N> __device__ __forceinline__ void qsb_cofactor_top16(
    uint64_t *roots, uint64_t (*products)[2*N], uint64_t (*excluded)[N], uint64_t *top) {
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

#if QSB_TOP16_REG
    uint64_t acc[4];
#endif
    if(tid<20) {                                 /* B: 16 exclusion starts + 4 quad products */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
            if(tid<16){a[k]=products[k][x+(tid^8)];      b[k]=products[k][p2+((tid&7)^4)];}
            else      {a[k]=products[k][p2+(tid-16)];    b[k]=products[k][p2+(tid-16)+4];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
#if QSB_TOP16_REG
        for(int k=0;k<4;k++){ if(tid<16) acc[k]=o[k];                else products[k][p4+(tid-16)]=o[k]; }
#else
        for(int k=0;k<4;k++){ if(tid<16) excluded[k][e+tid]=o[k];    else products[k][p4+(tid-16)]=o[k]; }
#endif
    }
    __syncwarp();

    if(tid<18) {                                 /* C: extend the exclusions + two half roots */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
#if QSB_TOP16_REG
            if(tid<16){a[k]=acc[k];                      b[k]=products[k][p4+((tid&3)^2)];}
#else
            if(tid<16){a[k]=excluded[k][e+tid];          b[k]=products[k][p4+((tid&3)^2)];}
#endif
            else      {a[k]=products[k][p4+(tid-16)];    b[k]=products[k][p4+(tid-16)+2];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
#if QSB_TOP16_REG
        for(int k=0;k<4;k++){ if(tid<16) acc[k]=o[k];                else products[k][p8+(tid-16)]=o[k]; }
#else
        for(int k=0;k<4;k++){ if(tid<16) excluded[k][e+tid]=o[k];    else products[k][p8+(tid-16)]=o[k]; }
#endif
    }
    __syncwarp();

    if(tid<17) {                                 /* D: finish the exclusions + the block root */
        uint64_t a[5],b[5],o[5];
        #pragma unroll
        for(int k=0;k<4;k++) {
#if QSB_TOP16_REG
            if(tid<16){a[k]=acc[k];                      b[k]=products[k][p8+((tid&1)^1)];}
#else
            if(tid<16){a[k]=excluded[k][e+tid];          b[k]=products[k][p8+((tid&1)^1)];}
#endif
            else      {a[k]=products[k][p8];             b[k]=products[k][p8+1];}
        }
        a[4]=b[4]=0;QSB_TOP16_MUL(o,a,b);
        #pragma unroll
        for(int k=0;k<4;k++){ if(tid<16){ excluded[k][e+tid]=o[k]; top[k]=o[k]; } else roots[(size_t)blockIdx.x*4+k]=o[k]; }
    }
    __syncwarp();
}
#endif /* QSB_TOP16 */

#ifndef QSB_TREE_SEEDREG
#define QSB_TREE_SEEDREG 1    /* 1: the first up-sweep level takes its low operand from the register
                               * the lane has just published instead of reading it back */
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
    uint64_t seed[4];
    #pragma unroll
    for(int k=0;k<4;k++){seed[k]=value[k];products[k][tid]=seed[k];}
    __syncthreads();
    int offset=0;
#if QSB_TREE_SEEDREG
    /* First up-sweep level, peeled (QSB_TREE_SEEDREG). Lane i < N/2 multiplies
     * products[i] by products[N/2+i]; the low operand is the leaf this very lane wrote
     * one line above, so it is still live in a register and only the high operand comes
     * from shared memory. Same pair, same order, same short-carry multiply as the loop
     * iteration it replaces, and the level's barrier is the one that iteration would
     * have executed (half = N/2 > 32 on this geometry, so __syncthreads). Four leaf
     * LDS per active lane disappear from the level that immediately follows the block's
     * only full barrier, where every lane is issuing at once. */
    {
        const int half=N>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=seed[k];b[k]=products[k][half+tid];}
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)products[k][N+tid]=out[k];
        }
        offset=N;
        if(half>32)__syncthreads();else __syncwarp();
    }
#endif
    const int cstart=QSB_TREE_SEEDREG?(N>>1):N;
    #pragma unroll 1
#if QSB_TOP16
    for(int count=cstart;count>16;count>>=1) {  /* stop at the sixteen subtree roots: the top is merged */
#elif QSB_TREE_TOP2
    for(int count=cstart;count>2;count>>=1) {   /* stop below the root: the top pair is merged into the down-sweep */
#else
    for(int count=cstart;count>1;count>>=1) {
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
#if QSB_TOP16
    uint64_t top[4];
    qsb_cofactor_top16<N>(roots,products,excluded,top);
    offset=2*N-64;
#if QSB_TOP16_PEEL
    /* Peeled first down-sweep level (QSB_TOP16_PEEL), count = 32, half = 16. This is the
     * level whose parent factors are exactly the sixteen words wave D has just written,
     * so lanes 0..15 take the parent from that register and only lanes 16..31 read it
     * back; the sibling comes from shared memory for all 32 lanes, as before. The three
     * indices are the ones the general loop computes at count = 32 with offset = 2N-64:
     * parent N-32+(tid&15), sibling 2N-64+(tid^16), child N-64+tid. The barrier and the
     * offset update are the loop's own ((count<<1) = 64 > 32, offset -= 64). Because the
     * trip count is fixed, the loop's runtime count==2 test does not appear on this level
     * at all - under QSB_TOP16 the down-sweep never runs at count 2. */
    if(N>32) {
        if(tid<32) {
            uint64_t parent[5],sibling[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k] =tid<16?top[k]:excluded[k][N-32+(tid&15)];
                sibling[k]=products[k][2*N-64+(tid^16)];
            }
            parent[4]=sibling[4]=0;qsb_field_mul_sc(out,parent,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)excluded[k][N-64+tid]=out[k];
        }
        __syncthreads();
        offset=2*N-128;
    }
    #pragma unroll 1
    for(int count=(N>32)?64:32;count<N;count<<=1) {
#else
    #pragma unroll 1
    for(int count=32;count<N;count<<=1) {
#endif
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
