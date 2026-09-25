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
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
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

#ifndef QSB_TREE_GFILL
#define QSB_TREE_GFILL 1      /* 1: fill the idle lanes of the top-16 waves with the leaf products of warps 0 and 1 */
#endif
#if QSB_TREE_GFILL
#if !QSB_TOP16
#error "QSB_TREE_GFILL needs the merged top-16 traversal (QSB_TOP16=1)"
#endif
/* QSB_TREE_GFILL: the four top-16 waves (A..D) run on warp 0 with 8, 20, 18 and 17 busy
 * lanes, so 64 of their 128 lane slots are free, while every warp later spends one full
 * multiply on its leaf cofactor. Each lane needs
 *     hc = U * cofactor = U * W[t^64] * E64[t&63]
 * (the leaf level pairs lane t with t^64; E64 is the excluded product of its pair node).
 * The base multiplies E64 * W[t^64] first and then U. This switch multiplies the other way
 * round: G = U * W[t^64] needs only the leaves, so it can run in the top waves, and then
 * hc = G * E64. The 64 G products of lanes 0..63 fill the free slots of waves A..D on warp 0
 * (24 + 12 + 14 + 14); warps 2 and 3 compute their own G after the down-sweep. Per 128-lane
 * block that is 6 warp-multiplies (G on warps 2,3; hc on four warps) in place of 8 (leaf
 * cofactor and hc on four warps each).
 *
 * The reordered products and the G-hosting waves use qsb_field_mul, the carry-complete
 * product of the root-group kernels (an exact residue in [0,2^256)), so no new truncation
 * is introduced: hc is congruent to the base value for every input, and bit-identical except
 * when the base's own short-carry cofactor or hc product dropped a carry or the residue is
 * below 2^256-p (two representatives). Every other product of the tree keeps its operands,
 * its order and qsb_field_mul_sc; the top-wave products are the same values, computed exactly.
 *
 * Shared layout: one limb-major array T[4][3N+N/2] (14 KiB for N=128, the 12 KiB arena grown
 * by 2 KiB). Product node p is column p (p < 2N), excluded node e is column 2N+e (e < N), and
 * column 3N+c holds U, then G, of lane c < N/2. Every operand has the same limb stride. Lanes
 * 0..63 park U before the first barrier; the G task for lane c reads it with W[c+64] (column
 * c+64, never written) and writes G in place; owners read it back after the final down-sweep
 * barrier. */
#define QSB_GF_P(i) (i)
#define QSB_GF_E(i) (2*N+(i))
#define QSB_GF_G(i) (3*N+(i))
#define QSB_GF_COLS (3*QSB_TREE_N+QSB_TREE_N/2)

struct qsb_gf_op { int a, b, o, kind; };   /* kind 0: idle, 1: tree node, 2: G task, 3: block root */

/* Operand columns of lane l in top wave w (0..3 = A..D). Useful products are exactly those of
 * qsb_cofactor_top16; the free lanes take G tasks c = 0..63 in order. */
template<int N>
__host__ __device__ constexpr qsb_gf_op qsb_gfill_op(int w, int l) {
    const int x=2*N-32, p2=2*N-16, p4=2*N-8, p8=2*N-4, e=N-32;
    qsb_gf_op r{0,0,0,0};
    int c=0;
    if(w==0) {                                   /* A: P2[j] = x[j]*x[j+8], j<8 */
        if(l<8){ r.a=QSB_GF_P(x+l); r.b=QSB_GF_P(x+l+8); r.o=QSB_GF_P(p2+l); r.kind=1; return r; }
        c=l-8;                                   /* 0..23 */
    } else if(w==1) {                            /* B: E4[i], i<16; P4[j], j<4 */
        if(l<16){ r.a=QSB_GF_P(x+(l^8)); r.b=QSB_GF_P(p2+((l&7)^4)); r.o=QSB_GF_E(e+l); r.kind=1; return r; }
        if(l<20){ r.a=QSB_GF_P(p2+(l-16)); r.b=QSB_GF_P(p2+(l-16)+4); r.o=QSB_GF_P(p4+(l-16)); r.kind=1; return r; }
        c=l+4;                                   /* 24..35 */
    } else if(w==2) {                            /* C: E8[i], i<16; P8[j], j<2 */
        if(l<16){ r.a=QSB_GF_E(e+l); r.b=QSB_GF_P(p4+((l&3)^2)); r.o=QSB_GF_E(e+l); r.kind=1; return r; }
        if(l<18){ r.a=QSB_GF_P(p4+(l-16)); r.b=QSB_GF_P(p4+(l-16)+2); r.o=QSB_GF_P(p8+(l-16)); r.kind=1; return r; }
        c=l+18;                                  /* 36..49 */
    } else {                                     /* D: E16[i], i<16; the block root */
        if(l<16){ r.a=QSB_GF_E(e+l); r.b=QSB_GF_P(p8+((l&1)^1)); r.o=QSB_GF_E(e+l); r.kind=1; return r; }
        if(l==16){ r.a=QSB_GF_P(p8); r.b=QSB_GF_P(p8+1); r.o=0; r.kind=3; return r; }
        if(l==31){ r.a=0; r.b=0; r.o=0; r.kind=0; return r; }
        c=l+33;                                  /* 50..63 */
    }
    r.a=QSB_GF_G(c); r.b=QSB_GF_P(c+N/2); r.o=QSB_GF_G(c); r.kind=2;
    return r;
}

/* The plan is packed per lane (a | b<<9 | o<<18 | kind<<27; columns < 3N+N/2 = 448 < 512) in a
 * compile-time table, and each wave loads its lane's word just before it runs. Computing the
 * columns inline let ptxas hoist the four waves' index arithmetic, which cost two registers
 * and re-allocated the chain loop (+5 IMAD.MOV per trip). */
struct qsb_gf_tab_t { uint32_t v[4*32]; };
template<int N> constexpr qsb_gf_tab_t qsb_gf_make_tab() {
    qsb_gf_tab_t t{};
    for(int w=0;w<4;w++) for(int l=0;l<32;l++) {
        const qsb_gf_op op=qsb_gfill_op<N>(w,l);
        t.v[w*32+l]=(uint32_t)op.a|((uint32_t)op.b<<9)|((uint32_t)op.o<<18)|((uint32_t)op.kind<<27);
    }
    return t;
}
static_assert(3*QSB_TREE_N+QSB_TREE_N/2<=512,"QSB_TREE_GFILL column packing");
__device__ const qsb_gf_tab_t qsb_gf_tab=qsb_gf_make_tab<QSB_TREE_N>();

template<int W,int N> __device__ __forceinline__ void qsb_gfill_wave(uint64_t *roots, uint64_t (*T)[3*N+N/2]) {
    const int tid=threadIdx.x;
    if(tid<32) {
        const uint32_t op=qsb_gf_tab.v[W*32+tid];
        const uint32_t kind=op>>27;
        if(kind) {
            const uint32_t ia=op&511u, ib=(op>>9)&511u, io=(op>>18)&511u;
            uint64_t a[5],b[5],o[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=T[k][ia];b[k]=T[k][ib];}
            a[4]=b[4]=0;qsb_field_mul(o,a,b);
            if(W==3 && kind==3u) {
                #pragma unroll
                for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=o[k];
            } else {
                #pragma unroll
                for(int k=0;k<4;k++)T[k][io]=o[k];
            }
        }
    }
    __syncwarp();
}

/* value: in the lane's leaf W, out hc = U * (product of the other N-1 leaves), exact.
 * roots: the block root, as in qsb_cofactor_prepare. */
template<int N> __device__ __forceinline__ void qsb_cofactor_gfill(
    uint64_t *value, const uint64_t *U, uint64_t *roots, uint64_t (*T)[3*N+N/2]) {
    static_assert(N==128,"QSB_TREE_GFILL slot plan is for 128-leaf trees");
    const int tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)T[k][QSB_GF_P(tid)]=value[k];
    if(tid<N/2) {
        #pragma unroll
        for(int k=0;k<4;k++)T[k][QSB_GF_G(tid)]=U[k];
    }
    __syncthreads();
    int offset=0;
    #pragma unroll 1
    for(int count=N;count>16;count>>=1) {        /* up-sweep: same products as qsb_cofactor_prepare */
        int half=count>>1;
        if(tid<half) {
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=T[k][QSB_GF_P(offset+tid)];b[k]=T[k][QSB_GF_P(offset+half+tid)];}
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)T[k][QSB_GF_P(offset+count+tid)]=out[k];
        }
        offset+=count;
        if(count>2){if(half>32)__syncthreads();else __syncwarp();}
    }
    qsb_gfill_wave<0,N>(roots,T);
    qsb_gfill_wave<1,N>(roots,T);
    qsb_gfill_wave<2,N>(roots,T);
    qsb_gfill_wave<3,N>(roots,T);
    offset=2*N-64;
    #pragma unroll 1
    for(int count=32;count<N;count<<=1) {        /* down-sweep: same products as qsb_cofactor_prepare */
        int half=count>>1;
        if(tid<count) {
            uint64_t parent[5],sibling[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=T[k][QSB_GF_E(offset+count-N+(tid&(half-1)))];
                sibling[k]=T[k][QSB_GF_P(offset+(tid^half))];
            }
            parent[4]=sibling[4]=0;
            qsb_field_mul_sc(out,parent,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)T[k][QSB_GF_E(offset-N+tid)]=out[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t g[5],e64[5];
    if(tid<N/2) {                                /* warps 0,1: G from the top waves */
        #pragma unroll
        for(int k=0;k<4;k++)g[k]=T[k][QSB_GF_G(tid)];
    } else {                                     /* warps 2,3: own G */
        uint64_t a[5],b[5];
        #pragma unroll
        for(int k=0;k<4;k++){a[k]=U[k];b[k]=T[k][QSB_GF_P(tid^(N/2))];}
        a[4]=b[4]=0;qsb_field_mul(g,a,b);
    }
    #pragma unroll
    for(int k=0;k<4;k++)e64[k]=T[k][QSB_GF_E(tid&(N/2-1))];
    g[4]=e64[4]=0;
    qsb_field_mul(value,g,e64);
    value[4]=0;
}

#ifndef QSB_TREE_TOP5
#define QSB_TREE_TOP5 2       /* 1: L3 wave + five-wave top, all 128 leaf products G in free lanes; 2: same, balanced over warps */
#endif
#if QSB_TREE_TOP5 != 0 && QSB_TREE_TOP5 != 1 && QSB_TREE_TOP5 != 2
#error "QSB_TREE_TOP5 must be 0, 1 or 2"
#endif
#if QSB_TREE_TOP5
/* QSB_TREE_TOP5: the slot bound of QSB_TREE_GFILL. With GFILL a 128-lane block issues 25
 * warp-multiplies (tree, G, hc, vbar, tbar) for 783 lane products, which is ceil(783/32):
 * no further packing helps unless lane products go. The merged top spends 48 products on
 * the sixteen E16 (three per entry); the level-by-level down-sweep spends 28 but needs
 * seven waves. A five-wave top in between spends 32:
 *
 *   wave  lanes (useful)                                             useful  G tasks
 *   L3    x[j]   = L2[j]*L2[j+16], j<16                                16     c 0..15
 *   A     P2[j]  = x[j]*x[j+8], j<8                                      8     c 16..39
 *   B     P4[j]  = P2[j]*P2[j+4], j<4                                    4     c 40..67
 *   C     P8[j]  = P4[j]*P4[j+2], j<2;  a[j] = P2[j^4]*P4[(j&3)^2], j<8 10     c 68..89
 *   D     root   = P8[0]*P8[1];         b[j] = a[j]*P8[(j&1)^1], j<8    9     c 90..112
 *   E     E16[i] = x[i^8]*b[i&7], i<16                                  16     c 113..127
 *
 * b[j] = P2[j^4]*P4[(j&3)^2]*P8[(j&1)^1] is the product of the seven other P2 nodes, so
 * E16[i] = x[i^8]*b[i&7] is the same factor set as the merged top's
 * x[i^8]*P2[(i&7)^4]*P4[(i&3)^2]*P8[(i&1)^1], associated differently. The six waves (the
 * up-sweep's 32-to-16 level included, whose lanes 16..31 were idle) carry 63 + 128 = 191
 * lane products in 192 slots, so the G products of all four warps fit and warps 2 and 3
 * no longer multiply their own. Per block: 2+1 (up) + 6 (waves) + 3 (down) + 4 (hc) + 8
 * (vbar, tbar) = 24 warp-multiplies for 767 lane products, in place of 25.
 *
 * Exactness: the wave products use qsb_field_mul (carry-complete, as the GFILL waves do).
 * x[j] moves from the short-carry to the carry-complete product with the same operands, so
 * it is bit-identical except where the base's short-carry product dropped a carry. P2, P4,
 * P8, the root and every G have the same operands and the same product as the base: bit-
 * identical. E16 is congruent to the base value for every input; the two builds' exact
 * products can pick different representatives only when the residue is below 2^256-p
 * (about 2^-224). The down-sweep, hc, vbar and tbar keep their code.
 *
 * Layout: the GFILL arena T[4][3N+N/2], unchanged size. U of lane c < N/2 is parked at
 * column 3N+c (as GFILL), U of lane c >= N/2 at excluded column 2N+c-N/2 (E64, which the
 * last down-sweep level writes only after every wave). The G task for lane c reads U and the
 * leaf W[c^N/2] and writes G over that leaf (column c^N/2): each leaf feeds exactly one G
 * task, and nothing else reads leaves after the up-sweep. a[j], b[j] live at excluded
 * columns 2N+N-16+j, which the tree never uses.
 *
 * Placement. With QSB_TREE_TOP5=1 the levels stay on the warps the base uses: L1 and E64 on
 * warps 0,1, everything from L2 to E32 on warp 0. Warp 0 then issues 13 of the 24 multiplies
 * (base: 12 of 25), warps 1..3 issue 5, 3, 3. If
 * warp w of every block runs on sub-partition w, the busiest sub-partition sets the pace.
 * QSB_TREE_TOP5=2 is the same plan and the same products with the levels moved:
 * L1 and E64 on warps 2,3; L2 and waves L3, A on warp 1; wave B on warp 2; wave C on warp 3;
 * waves D, E and E32 on warp 0. Per-warp multiplies 6, 6, 6, 6 (tree, hc, vbar, tbar), for
 * three extra __syncthreads per block, one at each hand-off of the serial chain. */
#define QSB_T5_U(c) ((c)<N/2 ? 3*N+(c) : 2*N+(c)-N/2)

template<int N>
__host__ __device__ constexpr qsb_gf_op qsb_t5_op(int w, int l) {
    const int l2=2*N-64, x=2*N-32, p2=2*N-16, p4=2*N-8, p8=2*N-4, e=N-32, ab=N-16;
    qsb_gf_op r{0,0,0,0};
    int c=0;
    if(w==0) {                                   /* L3: x[j] = L2[j]*L2[j+16], j<16 */
        if(l<16){ r.a=QSB_GF_P(l2+l); r.b=QSB_GF_P(l2+16+l); r.o=QSB_GF_P(x+l); r.kind=1; return r; }
        c=l-16;                                  /* 0..15 */
    } else if(w==1) {                            /* A: P2[j] = x[j]*x[j+8], j<8 */
        if(l<8){ r.a=QSB_GF_P(x+l); r.b=QSB_GF_P(x+l+8); r.o=QSB_GF_P(p2+l); r.kind=1; return r; }
        c=l+8;                                   /* 16..39 */
    } else if(w==2) {                            /* B: P4[j] = P2[j]*P2[j+4], j<4 */
        if(l<4){ r.a=QSB_GF_P(p2+l); r.b=QSB_GF_P(p2+l+4); r.o=QSB_GF_P(p4+l); r.kind=1; return r; }
        c=l+36;                                  /* 40..67 */
    } else if(w==3) {                            /* C: P8[j], j<2; a[j] = P2[j^4]*P4[(j&3)^2], j<8 */
        if(l<2){ r.a=QSB_GF_P(p4+l); r.b=QSB_GF_P(p4+l+2); r.o=QSB_GF_P(p8+l); r.kind=1; return r; }
        if(l<10){ const int j=l-2; r.a=QSB_GF_P(p2+(j^4)); r.b=QSB_GF_P(p4+((j&3)^2)); r.o=QSB_GF_E(ab+j); r.kind=1; return r; }
        c=l+58;                                  /* 68..89 */
    } else if(w==4) {                            /* D: the block root; b[j] = a[j]*P8[(j&1)^1], j<8 */
        if(l==0){ r.a=QSB_GF_P(p8); r.b=QSB_GF_P(p8+1); r.o=0; r.kind=3; return r; }
        if(l<9){ const int j=l-1; r.a=QSB_GF_E(ab+j); r.b=QSB_GF_P(p8+((j&1)^1)); r.o=QSB_GF_E(ab+j); r.kind=1; return r; }
        c=l+81;                                  /* 90..112 */
    } else {                                     /* E: E16[i] = x[i^8]*b[i&7], i<16 */
        if(l<16){ r.a=QSB_GF_P(x+(l^8)); r.b=QSB_GF_E(ab+(l&7)); r.o=QSB_GF_E(e+l); r.kind=1; return r; }
        if(l==31){ r.a=0; r.b=0; r.o=0; r.kind=0; return r; }
        c=l+97;                                  /* 113..127 */
    }
    r.a=QSB_T5_U(c); r.b=QSB_GF_P(c^(N/2)); r.o=QSB_GF_P(c^(N/2)); r.kind=2;
    return r;
}

struct qsb_t5_tab_t { uint32_t v[6*32]; };
template<int N> constexpr qsb_t5_tab_t qsb_t5_make_tab() {
    qsb_t5_tab_t t{};
    for(int w=0;w<6;w++) for(int l=0;l<32;l++) {
        const qsb_gf_op op=qsb_t5_op<N>(w,l);
        t.v[w*32+l]=(uint32_t)op.a|((uint32_t)op.b<<9)|((uint32_t)op.o<<18)|((uint32_t)op.kind<<27);
    }
    return t;
}
__device__ const qsb_t5_tab_t qsb_t5_tab=qsb_t5_make_tab<QSB_TREE_N>();

/* Wave W of the plan on warp WARP of the block (lane = tid-32*WARP). The table index keeps
 * the form tid+constant: (tid&31) made ptxas keep a new tid-derived value live across the
 * chain loop (128 registers, +6 IMAD per chain trip). */
template<int W,int N,int WARP=0> __device__ __forceinline__ void qsb_t5_wave(uint64_t *roots, uint64_t (*T)[3*N+N/2]) {
    const int tid=threadIdx.x;
    if(WARP==0 ? tid<32 : (unsigned)(tid-32*WARP)<32u) {
        const uint32_t op=qsb_t5_tab.v[W*32+tid-32*WARP];
        const uint32_t kind=op>>27;
        if(kind) {
            const uint32_t ia=op&511u, ib=(op>>9)&511u, io=(op>>18)&511u;
            uint64_t a[5],b[5],o[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=T[k][ia];b[k]=T[k][ib];}
            a[4]=b[4]=0;qsb_field_mul(o,a,b);
            if(W==4 && kind==3u) {
                #pragma unroll
                for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=o[k];
            } else {
                #pragma unroll
                for(int k=0;k<4;k++)T[k][io]=o[k];
            }
        }
    }
    __syncwarp();
}

/* value: in the lane's leaf W, out hc = U * (product of the other N-1 leaves), exact.
 * roots: the block root, as in qsb_cofactor_prepare. */
template<int N> __device__ __forceinline__ void qsb_cofactor_top5(
    uint64_t *value, const uint64_t *U, uint64_t *roots, uint64_t (*T)[3*N+N/2]) {
    static_assert(N==128,"QSB_TREE_TOP5 slot plan is for 128-leaf trees");
    const int tid=threadIdx.x;
    const int ucol=QSB_T5_U(tid);
    #pragma unroll
    for(int k=0;k<4;k++){T[k][QSB_GF_P(tid)]=value[k];T[k][ucol]=U[k];}
    __syncthreads();
    int offset=0;
    #pragma unroll 1
    for(int count=N;count>32;count>>=1) {        /* up-sweep to the 32 L2 nodes: same products as qsb_cofactor_prepare */
        int half=count>>1;
#if QSB_TREE_TOP5 == 2
        const int j=tid-half;                    /* node j: L1 on warps 2,3, L2 on warp 1 */
        if((unsigned)j<(unsigned)half) {
#else
        const int j=tid;
        if(j<half) {
#endif
            uint64_t a[5],b[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=T[k][QSB_GF_P(offset+j)];b[k]=T[k][QSB_GF_P(offset+half+j)];}
            a[4]=b[4]=0;qsb_field_mul_sc(out,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)T[k][QSB_GF_P(offset+count+j)]=out[k];
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
#if QSB_TREE_TOP5 == 2
    /* Balanced placement: the serial chain L2..E32 runs on warps 1, 2, 3, 0 in turn, with a
     * barrier at each hand-off, so each warp issues 6 of the block's 24 multiplies. */
    qsb_t5_wave<0,N,1>(roots,T);
    qsb_t5_wave<1,N,1>(roots,T);
    __syncthreads();
    qsb_t5_wave<2,N,2>(roots,T);
    __syncthreads();
    qsb_t5_wave<3,N,3>(roots,T);
    __syncthreads();
    qsb_t5_wave<4,N,0>(roots,T);
    qsb_t5_wave<5,N,0>(roots,T);
#else
    qsb_t5_wave<0,N>(roots,T);
    qsb_t5_wave<1,N>(roots,T);
    qsb_t5_wave<2,N>(roots,T);
    qsb_t5_wave<3,N>(roots,T);
    qsb_t5_wave<4,N>(roots,T);
    qsb_t5_wave<5,N>(roots,T);
#endif
    offset=2*N-64;
    #pragma unroll 1
    for(int count=32;count<N;count<<=1) {        /* down-sweep: same products as qsb_cofactor_prepare */
        int half=count>>1;
#if QSB_TREE_TOP5 == 2
        const int j=tid^(count&64);              /* E32 on warp 0, E64 on warps 2,3 */
#else
        const int j=tid;
#endif
        if(j<count) {
            uint64_t parent[5],sibling[5],out[5];
            #pragma unroll
            for(int k=0;k<4;k++) {
                parent[k]=T[k][QSB_GF_E(offset+count-N+(j&(half-1)))];
                sibling[k]=T[k][QSB_GF_P(offset+(j^half))];
            }
            parent[4]=sibling[4]=0;
            qsb_field_mul_sc(out,parent,sibling);
            #pragma unroll
            for(int k=0;k<4;k++)T[k][QSB_GF_E(offset-N+j)]=out[k];
        }
        offset-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
    uint64_t g[5],e64[5];
    #pragma unroll
    for(int k=0;k<4;k++){g[k]=T[k][QSB_GF_P(tid^(N/2))];e64[k]=T[k][QSB_GF_E(tid&(N/2-1))];}
    g[4]=e64[4]=0;
    qsb_field_mul(value,g,e64);
    value[4]=0;
}
#if (QSB_POST_GLUE & 1) && !(QSB_TREE_GFILL && QSB_TREE_TOP5 && QSB_TREE_N == 128)
#error "QSB_POST_GLUE bit 1 rewrites the 128-leaf QSB_TREE_TOP5 tree"
#endif
#if QSB_POST_GLUE & 1
/* QSB_POST_GLUE bit 1: the same tree, the same plan and the same products, with every column
 * held as two 16-byte limb pairs. TOP5 keeps limb k of column c at T[k][c] (four planes of
 * 8-byte words), so each operand is four LDS.64. Here plane h (h = 0, 1) holds the pair
 * {limb 2h, limb 2h+1} of column c at byte 16*c + h*QSB_TV_PLANE, so an operand is two
 * LDS.128. The arena is the same 14 KiB (2 x 448 x 16 = 4 x 448 x 8 bytes), now 16-byte
 * aligned. Lanes that read consecutive columns read consecutive 16-byte entries, which is
 * conflict-free for 128-bit loads. Results are still written as four 8-byte stores: a
 * 16-byte store needs its two limbs in four consecutive registers, and ptxas copies the
 * product outputs into such quads.
 * Only where a limb is kept in shared memory changes; every product reads the same limb
 * values in the same order, so every value (tree nodes, root, hc) is bit-identical to TOP5.
 *
 * Bit 2: the wave plan holds each lane's operation as four words (the byte offsets of a, b
 * and o in plane 0, and kind), loaded as one 16-byte word, so the field extractions and
 * their shifts are gone. Without bit 2 the packed TOP5 word is decoded as before.
 * Bit 8 (up-sweep) and bit 128 (down-sweep): the two-level loops are written out level by
 * level (same lanes, products and barriers), so the loop counters and their index
 * arithmetic become immediates. */
#define QSB_TV_COLS (3*QSB_TREE_N+QSB_TREE_N/2)
#define QSB_TV_PLANE (QSB_TV_COLS*16)
__device__ __forceinline__ void qsb_tv_ld(uint64_t *x, const char *T, uint32_t off) {
    const ulonglong2 lo=*(const ulonglong2 *)(T+off), hi=*(const ulonglong2 *)(T+off+QSB_TV_PLANE);
    x[0]=lo.x;x[1]=lo.y;x[2]=hi.x;x[3]=hi.y;x[4]=0;
}
__device__ __forceinline__ void qsb_tv_st(char *T, uint32_t off, const uint64_t *x) {
#ifdef __CUDA_ARCH__
    /* volatile: neither NVVM nor ptxas merges these into 16-byte stores */
    const uint32_t a=(uint32_t)__cvta_generic_to_shared(T+off);
    asm volatile("st.volatile.shared.u64 [%0], %1;\n\tst.volatile.shared.u64 [%0+8], %2;\n\t"
                 "st.volatile.shared.u64 [%0+%5], %3;\n\tst.volatile.shared.u64 [%0+%6], %4;"
                 :: "r"(a), "l"(x[0]), "l"(x[1]), "l"(x[2]), "l"(x[3]),
                    "n"(QSB_TV_PLANE), "n"(QSB_TV_PLANE+8) : "memory");
#else   /* host build: the same four words at the same offsets */
    *(uint64_t *)(T+off)=x[0];*(uint64_t *)(T+off+8)=x[1];
    *(uint64_t *)(T+off+QSB_TV_PLANE)=x[2];*(uint64_t *)(T+off+QSB_TV_PLANE+8)=x[3];
#endif
}
#if QSB_POST_GLUE & 2
struct __align__(16) qsb_t5v_ent { uint32_t a, b, o, kind; };
struct qsb_t5v_tab_t { qsb_t5v_ent v[6*32]; };
template<int N> constexpr qsb_t5v_tab_t qsb_t5v_make_tab() {
    qsb_t5v_tab_t t{};
    for(int w=0;w<6;w++) for(int l=0;l<32;l++) {
        const qsb_gf_op op=qsb_t5_op<N>(w,l);
        t.v[w*32+l]=qsb_t5v_ent{(uint32_t)op.a*16u,(uint32_t)op.b*16u,(uint32_t)op.o*16u,(uint32_t)op.kind};
    }
    return t;
}
__device__ const qsb_t5v_tab_t qsb_t5v_tab=qsb_t5v_make_tab<QSB_TREE_N>();
typedef const uint4 *qsb_t5v_tp;
#else
typedef const uint32_t *qsb_t5v_tp;
#endif

/* Wave W on warp WARP; tp points at this thread's entry of wave 0 (index tid). */
template<int W,int N,int WARP=0> __device__ __forceinline__ void qsb_t5v_wave(uint64_t *roots, char *T, qsb_t5v_tp tp) {
    const int tid=threadIdx.x;
    if(WARP==0 ? tid<32 : (unsigned)(tid-32*WARP)<32u) {
#if QSB_POST_GLUE & 2
        const uint4 op=__ldg(tp+(W*32-32*WARP));
        const uint32_t kind=op.w, ia=op.x, ib=op.y, io=op.z;
#else
        const uint32_t op=tp[W*32-32*WARP];
        const uint32_t kind=op>>27;
        const uint32_t ia=(op&511u)*16u, ib=((op>>9)&511u)*16u, io=((op>>18)&511u)*16u;
#endif
        if(kind) {
            uint64_t a[5],b[5],o[5];
            qsb_tv_ld(a,T,ia);qsb_tv_ld(b,T,ib);
            qsb_field_mul(o,a,b);
            if(W==4 && kind==3u) {
                #pragma unroll
                for(int k=0;k<4;k++)roots[(size_t)blockIdx.x*4+k]=o[k];
            } else {
                qsb_tv_st(T,io,o);
            }
        }
    }
    __syncwarp();
}

/* One up-sweep level (count nodes to count/2) and one down-sweep level, as in the loops of
 * qsb_cofactor_top5 at the iteration with this count and offset. */
template<int N,int COUNT,int OFFSET> __device__ __forceinline__ void qsb_t5v_up(char *T) {
    const int tid=threadIdx.x;
    constexpr int half=COUNT>>1;
#if QSB_TREE_TOP5 == 2
    const int j=tid-half;
    if((unsigned)j<(unsigned)half) {
#else
    const int j=tid;
    if(j<half) {
#endif
        uint64_t a[5],b[5],out[5];
        qsb_tv_ld(a,T,16u*QSB_GF_P(OFFSET+j));qsb_tv_ld(b,T,16u*QSB_GF_P(OFFSET+half+j));
        qsb_field_mul_sc(out,a,b);
        qsb_tv_st(T,16u*QSB_GF_P(OFFSET+COUNT+j),out);
    }
    if(half>32)__syncthreads();else __syncwarp();
}
template<int N,int COUNT,int OFFSET> __device__ __forceinline__ void qsb_t5v_down(char *T) {
    const int tid=threadIdx.x;
    constexpr int half=COUNT>>1;
#if QSB_TREE_TOP5 == 2
    const int j=tid^(COUNT&64);
#else
    const int j=tid;
#endif
    if(j<COUNT) {
        uint64_t parent[5],sibling[5],out[5];
        qsb_tv_ld(parent,T,16u*QSB_GF_E(OFFSET+COUNT-N+(j&(half-1))));
        qsb_tv_ld(sibling,T,16u*QSB_GF_P(OFFSET+(j^half)));
        qsb_field_mul_sc(out,parent,sibling);
        qsb_tv_st(T,16u*QSB_GF_E(OFFSET-N+j),out);
    }
    if((COUNT<<1)>32)__syncthreads();else __syncwarp();
}

/* qsb_cofactor_top5 with the paired layout: the same levels, lanes, products and barriers. */
template<int N> __device__ __forceinline__ void qsb_cofactor_top5v(
    uint64_t *value, const uint64_t *U, uint64_t *roots, char *T) {
    static_assert(N==128,"QSB_TREE_TOP5 slot plan is for 128-leaf trees");
    const int tid=threadIdx.x;
    const int ucol=QSB_T5_U(tid);
    qsb_tv_st(T,16u*QSB_GF_P(tid),value);
    qsb_tv_st(T,16u*ucol,U);
    __syncthreads();
#if QSB_POST_GLUE & 8
    qsb_t5v_up<N,N,0>(T);
    qsb_t5v_up<N,N/2,N>(T);
#else
    int offset=0;
    #pragma unroll 1
    for(int count=N;count>32;count>>=1) {
        int half=count>>1;
#if QSB_TREE_TOP5 == 2
        const int j=tid-half;
        if((unsigned)j<(unsigned)half) {
#else
        const int j=tid;
        if(j<half) {
#endif
            uint64_t a[5],b[5],out[5];
            qsb_tv_ld(a,T,16u*QSB_GF_P(offset+j));qsb_tv_ld(b,T,16u*QSB_GF_P(offset+half+j));
            qsb_field_mul_sc(out,a,b);
            qsb_tv_st(T,16u*QSB_GF_P(offset+count+j),out);
        }
        offset+=count;
        if(half>32)__syncthreads();else __syncwarp();
    }
#endif
#if QSB_POST_GLUE & 2
    const qsb_t5v_tp tp=(const uint4 *)qsb_t5v_tab.v+(unsigned)tid;
#else
    const qsb_t5v_tp tp=qsb_t5_tab.v+tid;
#endif
#if QSB_TREE_TOP5 == 2
    qsb_t5v_wave<0,N,1>(roots,T,tp);
    qsb_t5v_wave<1,N,1>(roots,T,tp);
    __syncthreads();
    qsb_t5v_wave<2,N,2>(roots,T,tp);
    __syncthreads();
    qsb_t5v_wave<3,N,3>(roots,T,tp);
    __syncthreads();
    qsb_t5v_wave<4,N,0>(roots,T,tp);
    qsb_t5v_wave<5,N,0>(roots,T,tp);
#else
    qsb_t5v_wave<0,N>(roots,T,tp);
    qsb_t5v_wave<1,N>(roots,T,tp);
    qsb_t5v_wave<2,N>(roots,T,tp);
    qsb_t5v_wave<3,N>(roots,T,tp);
    qsb_t5v_wave<4,N>(roots,T,tp);
    qsb_t5v_wave<5,N>(roots,T,tp);
#endif
#if QSB_POST_GLUE & 128
    qsb_t5v_down<N,32,2*N-64>(T);
    qsb_t5v_down<N,64,2*N-128>(T);
#else
    int offset2=2*N-64;
    #pragma unroll 1
    for(int count=32;count<N;count<<=1) {
        int half=count>>1;
#if QSB_TREE_TOP5 == 2
        const int j=tid^(count&64);
#else
        const int j=tid;
#endif
        if(j<count) {
            uint64_t parent[5],sibling[5],out[5];
            qsb_tv_ld(parent,T,16u*QSB_GF_E(offset2+count-N+(j&(half-1))));
            qsb_tv_ld(sibling,T,16u*QSB_GF_P(offset2+(j^half)));
            qsb_field_mul_sc(out,parent,sibling);
            qsb_tv_st(T,16u*QSB_GF_E(offset2-N+j),out);
        }
        offset2-=count<<1;
        if((count<<1)>32)__syncthreads();else __syncwarp();
    }
#endif
    uint64_t g[5],e64[5];
    qsb_tv_ld(g,T,16u*QSB_GF_P(tid^(N/2)));
    qsb_tv_ld(e64,T,16u*QSB_GF_E(tid&(N/2-1)));
    qsb_field_mul(value,g,e64);
    value[4]=0;
}
#endif /* QSB_POST_GLUE & 1 */
#undef QSB_T5_U
#endif /* QSB_TREE_TOP5 */
#undef QSB_GF_P
#undef QSB_GF_E
#undef QSB_GF_G
#elif defined(QSB_TREE_TOP5) && QSB_TREE_TOP5
#error "QSB_TREE_TOP5 needs QSB_TREE_GFILL=1"
#endif /* QSB_TREE_GFILL */
