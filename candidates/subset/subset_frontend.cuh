/* subset_frontend.cuh — the subset track's SHA front-end, lifted verbatim (by line range) from
 * the promoted subset frontier tree (candidates/subset/tests/gpu_epochs/tree.cu and
 * window_schedule_shared.cuh at a68c296 / cae17c2) so that pinning's EC pipeline can consume
 * subset candidates. Provenance per block is noted inline. GPLv3 (VanitySearch-derived tree). */
#pragma once
#define QSB_SE_N_INC     10
#define QSB_SE_EARLY     6
#define QSB_SE_TWIN      3
#define QSB_SE_CUT       137
#define QSB_SE_PER_EPOCH 256
#define QSB_FAST_N_CONST 69
#define MAX_T 16
#define SIG_PUSH_SIZE 10
/* tree.cu: epoch descriptor */
typedef struct {
    uint32_t mid[8];
    uint32_t remW[2];
    uint8_t early[QSB_SE_EARLY];
    uint8_t pad[64 - 8 * 4 - 2 * 4 - QSB_SE_EARLY];
} epoch_desc_t;
__device__ uint64_t BINOM_C[151][10];
__device__ __constant__ uint8_t WIN3[QSB_SE_PER_EPOCH][QSB_SE_TWIN];
__device__ __constant__ uint32_t QSB_CONST_SCHEDULE[4][64];
/* tree.cu:99-115 host constant schedule */
static uint32_t qsb_host_rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
static int qsb_prepare_constant_schedule(const uint32_t *words,int count){
 if(count!=69)return 1;
 uint32_t round_k[64],expanded[4][64];
 if(cudaMemcpyFromSymbol(round_k,K,sizeof(round_k))!=cudaSuccess)return 1;
 for(int block=0;block<4;block++){
  uint32_t *w=expanded[block];memcpy(w,words+5+block*16,64);
  for(int i=16;i<64;i++){
   uint32_t x=w[i-15],y=w[i-2];
   uint32_t lo=qsb_host_rotr(x,7)^qsb_host_rotr(x,18)^(x>>3);
   uint32_t hi=qsb_host_rotr(y,17)^qsb_host_rotr(y,19)^(y>>10);
   w[i]=w[i-16]+lo+w[i-7]+hi;
  }
  for(int i=0;i<64;i++)w[i]+=round_k[i];
 }
 return cudaMemcpyToSymbol(QSB_CONST_SCHEDULE,expanded,sizeof(expanded))==cudaSuccess?0:1;
}
/* tree.cu:186- constant blocks (rolled) */
__device__ __forceinline__ void qsb_compress_constant_rolled(uint32_t *output){
    #pragma unroll 1
    for(int block=0;block<4;block++){
        uint32_t a=output[0],b=output[1],c=output[2],d=output[3];
        uint32_t e=output[4],f=output[5],g=output[6],h=output[7],t1,t2;
        #pragma unroll 1
        for(int r=0;r<64;r+=8){
            S2Round(a,b,c,d,e,f,g,h,0,QSB_CONST_SCHEDULE[block][r]);
            S2Round(h,a,b,c,d,e,f,g,0,QSB_CONST_SCHEDULE[block][r+1]);
            S2Round(g,h,a,b,c,d,e,f,0,QSB_CONST_SCHEDULE[block][r+2]);
            S2Round(f,g,h,a,b,c,d,e,0,QSB_CONST_SCHEDULE[block][r+3]);
            S2Round(e,f,g,h,a,b,c,d,0,QSB_CONST_SCHEDULE[block][r+4]);
            S2Round(d,e,f,g,h,a,b,c,0,QSB_CONST_SCHEDULE[block][r+5]);
            S2Round(c,d,e,f,g,h,a,b,0,QSB_CONST_SCHEDULE[block][r+6]);
            S2Round(b,c,d,e,f,g,h,a,0,QSB_CONST_SCHEDULE[block][r+7]);
        }
        output[0]+=a;output[1]+=b;output[2]+=c;output[3]+=d;
        output[4]+=e;output[5]+=f;output[6]+=g;output[7]+=h;
    }
}
/* tree.cu:774- unrank_combo */
__device__ __forceinline__ void unrank_combo(uint64_t rank, int n, int t, uint8_t *out) {
    int lo = 0;
    for (int i = 0; i < t; i++) {
        int k = t - i - 1;
        int hi = n - (t - i);
        /* binary search smallest c in [lo,hi] with prefix(c) > rank */
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            /* prefix(lo..mid) = C[n-lo][k+1] - C[n-mid-1][k+1] */
            uint64_t a = BINOM_C[n - lo][k + 1];
            uint64_t b = BINOM_C[n - mid - 1][k + 1];
            uint64_t pref = (a >= b) ? (a - b) : 0;
            if (rank < pref) hi = mid;
            else { rank -= pref; lo = mid + 1; }
        }
        out[i] = (uint8_t)lo;
        lo++;
    }
}
/* window_schedule_shared.cuh (whole) */
// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
__device__ uint32_t QSB_WINDOW_FIRST[14][256];
__device__ uint32_t QSB_WINDOW_SECOND[64][256];
__device__ uint32_t QSB_WINDOW_CLASS[256];
__device__ uint32_t QSB_FIRST_CLASS[256];
__device__ uint32_t QSB_FIRST_UNIQUE[14][256];
__device__ __constant__ int QSB_FIRST_COUNT;
static int qsb_first_class_count=0;

static uint32_t qsb_window_second_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=12,n=0;i>=0 && n<5;i--)
        if(i!=w[0]-137 && i!=w[1]-137 && i!=w[2]-137){key=(key<<4)|i;n++;}
    return key;
}

static uint32_t qsb_window_first_key(const uint8_t w[3]) {
    uint32_t key=0;
    for(int i=0,n=0;i<13 && n<6;i++)
        if(i!=w[0]-137 && i!=w[1]-137 && i!=w[2]-137){key=(key<<4)|i;n++;}
    return key;
}

static int qsb_prepare_window_schedule(const uint8_t *rows,
        const uint8_t windows[256][3], const uint32_t *constant) {
    uint32_t first[14][256], second[64][256]={}, round_k[64];
    uint32_t classes[256], unique[256][16];
    uint32_t first_classes[256], first_unique[256][14], transposed[14][256]={};
    int first_distinct=0;
    int distinct=0;
    if (cudaMemcpyFromSymbol(round_k, K, sizeof(round_k)) != cudaSuccess) return 1;
    for (int lane=0; lane<256; lane++) {
        uint8_t bytes[128]={};
        int pos=8, sel=0;
        for (int i=137; i<150; i++) {
            if (sel<3 && windows[lane][sel]==i) { sel++; continue; }
            memcpy(bytes+pos, rows+10*i, 10); pos+=10;
        }
        if (pos!=108 || sel!=3) return 1;
        for (int j=0; j<5; j++)
            for (int b=0; b<4; b++) bytes[pos++]=(uint8_t)(constant[j]>>(24-8*b));
        if (pos!=128) return 1;
        uint32_t words[32], expanded[64];
        for (int j=0; j<32; j++)
            words[j]=((uint32_t)bytes[4*j]<<24)|((uint32_t)bytes[4*j+1]<<16)|
                     ((uint32_t)bytes[4*j+2]<<8)|bytes[4*j+3];
        for (int j=0; j<14; j++) first[j][lane]=words[j+2];
        int first_slot=0;
        while(first_slot<first_distinct && memcmp(first_unique[first_slot],words+2,56))first_slot++;
        if(first_slot==first_distinct){memcpy(first_unique[first_distinct],words+2,56);first_distinct++;}
        first_classes[lane]=first_slot;
        int slot=0;
        while(slot<distinct && memcmp(unique[slot],words+16,64))slot++;
        if(slot==distinct){memcpy(unique[distinct],words+16,64);distinct++;}
        classes[lane]=slot;
        for (int j=0; j<16; j++) expanded[j]=words[j+16];
        for (int j=16; j<64; j++) {
            uint32_t x=expanded[j-15], y=expanded[j-2];
            uint32_t a=qsb_host_rotr(x,7)^qsb_host_rotr(x,18)^(x>>3);
            uint32_t b=qsb_host_rotr(y,17)^qsb_host_rotr(y,19)^(y>>10);
            expanded[j]=expanded[j-16]+a+expanded[j-7]+b;
        }
        for (int j=0; j<64; j++) second[j][slot]=expanded[j]+round_k[j];
    }
    printf("Window schedule classes: first=%d second=%d of 256\n",first_distinct,distinct);
    qsb_first_class_count=first_distinct;
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(cudaMemcpyToSymbol(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_FIRST,first,sizeof(first))!=cudaSuccess) return 1;
    return cudaMemcpyToSymbol(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
}

/* PRODUCER STAGE (sub_prod2). The per-epoch first-block SHA-256 midstate of
 * first-class `cls` used to be compressed by leader lane `cls` of the digest
 * block into a __shared__ uint32_t first_states[8][256] (8192 B) behind a
 * __syncthreads(): 54 of 256 lanes worked, 202 idled, and every lane paid the
 * barrier. kernel_build_first now computes exactly these states one launch
 * earlier and parks them in global memory; the digest's consumer lanes read
 * their class's 32 bytes directly. The arithmetic below is character-for-
 * character the old leader-lane body, so every produced state is bit-identical.
 *
 * Layout: class-major, 8 words (32 B) per class, QSB_FIRST_COUNT classes per
 * epoch. Offsets are therefore always 32 B multiples, so the 16-byte vector
 * accessors below are always correctly aligned for a cudaMalloc'd base. */
__device__ __forceinline__ void qsb_first_state_class(const epoch_desc_t *epoch,
        int cls, uint32_t out[8]) {
    uint32_t W[16];
    #pragma unroll
    for(int j=0;j<8;j++)out[j]=epoch->mid[j];
    W[0]=epoch->remW[0];W[1]=epoch->remW[1];
    #pragma unroll
    for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][cls];
    _SHA256Transform(out,W);   /* mutates W; out is the epoch midstate */
}

/* 2 x 128-bit accesses instead of 8 x 32-bit: the store is a fully coalesced
 * 32 B/lane run, and one consumer warp needs 32 sectors rather than 256. */
__device__ __forceinline__ void qsb_store_first_state(uint32_t * __restrict__ dst,
        const uint32_t in[8]) {
    uint4 a,b;
    a.x=in[0];a.y=in[1];a.z=in[2];a.w=in[3];
    b.x=in[4];b.y=in[5];b.z=in[6];b.w=in[7];
    uint4 *v=reinterpret_cast<uint4*>(dst);
    v[0]=a;v[1]=b;
}
__device__ __forceinline__ void qsb_load_first_state(const uint32_t * __restrict__ src,
        uint32_t out[8]) {
    const uint4 *v=reinterpret_cast<const uint4*>(src);
    uint4 a=v[0],b=v[1];
    out[0]=a.x;out[1]=a.y;out[2]=a.z;out[3]=a.w;
    out[4]=b.x;out[5]=b.y;out[6]=b.z;out[7]=b.w;
}

/* first_epoch points at ONE EPOCH's record: d_first + ep*stride with
 * stride = 8*QSB_FIRST_COUNT words and ep the epoch-within-launch index.
 * It is NOT "this block's" record in general: with ZLAB_K2S a digest block
 * consumes QSB_K2S_MUL epochs, so ep = QSB_K2S_MUL*blockIdx.x + k and the
 * caller passes a different first_epoch for each k. */
__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const uint32_t * __restrict__ first_epoch, int lane) {
    // No barrier and no shared staging: the first-block states were produced
    // by kernel_build_first on the previous launch of the same stream.
    int first_slot=QSB_FIRST_CLASS[lane];
    qsb_load_first_state(first_epoch+(size_t)first_slot*8,state);
    int slot=QSB_WINDOW_CLASS[lane];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7],t1,t2;
    #pragma unroll 1
    for (int r=0; r<64; r+=8) {
        S2Round(a,b,c,d,e,f,g,h,0,QSB_WINDOW_SECOND[r][slot]);
        S2Round(h,a,b,c,d,e,f,g,0,QSB_WINDOW_SECOND[r+1][slot]);
        S2Round(g,h,a,b,c,d,e,f,0,QSB_WINDOW_SECOND[r+2][slot]);
        S2Round(f,g,h,a,b,c,d,e,0,QSB_WINDOW_SECOND[r+3][slot]);
        S2Round(e,f,g,h,a,b,c,d,0,QSB_WINDOW_SECOND[r+4][slot]);
        S2Round(d,e,f,g,h,a,b,c,0,QSB_WINDOW_SECOND[r+5][slot]);
        S2Round(c,d,e,f,g,h,a,b,0,QSB_WINDOW_SECOND[r+6][slot]);
        S2Round(b,c,d,e,f,g,h,a,0,QSB_WINDOW_SECOND[r+7][slot]);
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    qsb_compress_constant_rolled(state);
}
/* tree.cu:801- kernel_build_epochs (ZLAB_HITPATH arm dropped by preprocessor: define 0) */
#ifndef ZLAB_HITPATH
#define ZLAB_HITPATH 0
#endif
__global__ void kernel_build_epochs(
    uint64_t epoch_base, uint64_t n_epochs,
    int window_start, int s_early,
    const uint32_t * __restrict__ d_midstate,
    const uint8_t * __restrict__ d_prefix_remainder, int prefix_remainder_len,
    const uint8_t * __restrict__ d_dummy_sigs,
    epoch_desc_t * __restrict__ d_epochs
#if ZLAB_HITPATH
    , uint32_t *d_hit_reset
#endif
    )
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
#if ZLAB_HITPATH
    /* Runs before this launch's digest kernel on the same stream. */
    if (t == 0) *d_hit_reset = 0;
#endif
    uint64_t e = epoch_base + (uint64_t)t;
    if (e >= n_epochs) return;
    uint8_t early[MAX_T];
    unrank_combo(e, window_start, s_early, early);
    uint32_t state[8];
    for (int i = 0; i < 8; i++) state[i] = d_midstate[i];
    uint32_t curW[16];
    uint8_t *cur = (uint8_t *)curW;
    int cur_pos = 0;
    for (int i = 0; i < prefix_remainder_len; i++) {
        cur[cur_pos++] = d_prefix_remainder[i];
        if (cur_pos == 64) {
            uint32_t blk[16];
            for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
            _SHA256Transform(state, blk);
            cur_pos = 0;
        }
    }
    /* Iterate over OUTPUT pushes, not source indices. Every epoch keeps exactly
     * window_start - s_early pushes, so with this loop shape the 64-byte boundary
     * (and therefore every _SHA256Transform) falls at the same iteration in every
     * lane; the source-push loop fired the transform at a lane-dependent iteration,
     * which made the whole producer warp-divergent. Same byte stream, same
     * midstates, bit for bit -- only the schedule changes. */
    const int kept = window_start - s_early;
    for (int j = 0; j < kept; j++) {
        int src = j;
        for (int k = 0; k < s_early; k++) src += ((int)early[k] <= src) ? 1 : 0;
        const uint8_t *row = d_dummy_sigs + (size_t)src * SIG_PUSH_SIZE;
        for (int b = 0; b < SIG_PUSH_SIZE; b++) {
            cur[cur_pos++] = row[b];
            if (cur_pos == 64) {
                uint32_t blk[16];
                for (int k = 0; k < 16; k++) blk[k] = bswap32(curW[k]);
                _SHA256Transform(state, blk);
                cur_pos = 0;
            }
        }
    }
    epoch_desc_t *d = d_epochs + t;
    for (int i = 0; i < 8; i++) d->mid[i] = state[i];
    /* cur_pos is 8 for the pinned shape (1352 = 21*64 + 8): the leftover
     * staging words hold the remainder bytes in stream order. */
    d->remW[0] = bswap32(curW[0]);
    d->remW[1] = bswap32(curW[1]);
    for (int i = 0; i < s_early; i++) d->early[i] = early[i];
}
/* tree.cu:880- kernel_build_first */
__global__ void kernel_build_first(
    uint64_t epoch_base, uint64_t n_epochs, int first_count,
    const epoch_desc_t * __restrict__ d_epochs,
    uint32_t * __restrict__ d_first)
{
    const int cls = threadIdx.x;
    if (cls >= first_count) return;              /* blockDim.x == first_count */
    /* SEAM 2. ep is the EPOCH-WITHIN-LAUNCH index, not "the digest block that
     * will consume it": under ZLAB_K2S one digest block consumes two epochs.
     * ep == blockIdx.x holds only because SEAM 1 launches one block per epoch
     * (nblk * QSB_K2S_MUL), never one block per digest block. Both d_epochs and
     * d_first are indexed by ep, so the body needs no other change. */
    const int ep = blockIdx.x;
    const uint64_t e = epoch_base + (uint64_t)ep;
    if (e >= n_epochs) return;                   /* matches kernel_build_epochs */
    uint32_t out[8];
    qsb_first_state_class(d_epochs + ep, cls, out);
    qsb_store_first_state(d_first + (size_t)ep * (first_count * 8)
                                  + (size_t)cls * 8, out);
}
/* tree.cu:1932-1946 digest params struct */
typedef struct {
    uint32_t n, t;
    uint32_t total_preimage_len;
    uint32_t tail_section_len;
    uint32_t tx_suffix_len;
    uint32_t prefix_remainder_len;   /* NEW: bytes of fixed_prefix not in midstate */
    uint32_t midstate[8];
    uint8_t *prefix_remainder;       /* NEW: the up-to-63 bytes before dummy sigs */
    uint8_t *dummy_sigs;
    uint8_t *tail_section;
    uint8_t *tx_suffix;
    uint8_t neg_r_inv[32];
    uint8_t u2r_x[32];
    uint8_t u2r_y[32];
} digest_params_t;
/* tree.cu:1948- loader */
static int load_digest_params(const char *fn, digest_params_t *p) {
    FILE *f = fopen(fn, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fn); return -1; }
    if (fread(&p->n, 4, 1, f) != 1) goto err;
    if (fread(&p->t, 4, 1, f) != 1) goto err;
    if (fread(&p->total_preimage_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tail_section_len, 4, 1, f) != 1) goto err;
    if (fread(&p->tx_suffix_len, 4, 1, f) != 1) goto err;
    if (fread(&p->prefix_remainder_len, 4, 1, f) != 1) goto err;
    if (fread(p->midstate, 4, 8, f) != 8) goto err;
    for (int i=0;i<8;i++){
        uint8_t *b=(uint8_t*)&p->midstate[i];
        p->midstate[i]=((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    }
    if (p->prefix_remainder_len > 0) {
        p->prefix_remainder = (uint8_t*)malloc(p->prefix_remainder_len);
        if (fread(p->prefix_remainder, 1, p->prefix_remainder_len, f) != p->prefix_remainder_len) goto err;
    } else {
        p->prefix_remainder = NULL;
    }
    p->dummy_sigs = (uint8_t*)malloc(p->n * SIG_PUSH_SIZE);
    if (fread(p->dummy_sigs, 1, p->n * SIG_PUSH_SIZE, f) != p->n * SIG_PUSH_SIZE) goto err;
    p->tail_section = (uint8_t*)malloc(p->tail_section_len);
    if (fread(p->tail_section, 1, p->tail_section_len, f) != p->tail_section_len) goto err;
    p->tx_suffix = (uint8_t*)malloc(p->tx_suffix_len);
    if (fread(p->tx_suffix, 1, p->tx_suffix_len, f) != p->tx_suffix_len) goto err;
    if (fread(p->neg_r_inv, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_x, 1, 32, f) != 32) goto err;
    if (fread(p->u2r_y, 1, 32, f) != 32) goto err;
    fclose(f);
    printf("  Loaded: n=%u, t=%u, preimage=%u, tail=%u, suffix=%u, prefix_rem=%u\n",
           p->n, p->t, p->total_preimage_len, p->tail_section_len, p->tx_suffix_len,
           p->prefix_remainder_len);
    return 0;
err:
    fprintf(stderr, "Error reading %s\n", fn); fclose(f); return -1;
}
/* tree.cu:2002- binom_u64 */
static uint64_t binom_u64(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    __uint128_t r = 1;
    for (int i = 0; i < k; i++) {
        r = r * (uint64_t)(n - i) / (uint64_t)(i + 1);
        if (r > (__uint128_t)0xFFFFFFFFFFFFFFFFULL) return 0xFFFFFFFFFFFFFFFFULL;
    }
    return (uint64_t)r;
}
