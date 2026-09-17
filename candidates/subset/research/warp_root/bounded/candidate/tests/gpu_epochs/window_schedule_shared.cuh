// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once
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

/* Keep each second-block schedule class inside one warp.  The retained 256
 * windows form 56 classes whose sizes pack exactly into eight 32-lane bins.
 * First-fit decreasing finds that packing deterministically (26+6, five
 * 21+6+5x1 bins, then the remaining 6/1 classes).  This preserves the exact
 * candidate set while avoiding the six class splits made by a flat key sort. */
static int qsb_pack_second_classes(uint8_t windows[256][3]) {
    uint32_t keys[256];
    int counts[256]={0},nclass=0;
    for(int lane=0;lane<256;lane++){
        uint32_t key=qsb_window_second_key(windows[lane]);
        int slot=0;
        while(slot<nclass && keys[slot]!=key)slot++;
        if(slot==nclass){keys[nclass]=key;nclass++;}
        counts[slot]++;
    }
    for(int i=1;i<nclass;i++){
        uint32_t key=keys[i];int count=counts[i],j=i;
        while(j>0 && (counts[j-1]<count ||
              (counts[j-1]==count && keys[j-1]>key))){
            keys[j]=keys[j-1];counts[j]=counts[j-1];j--;
        }
        keys[j]=key;counts[j]=count;
    }
    uint8_t packed[256][3];
    int fill[8]={0};
    for(int slot=0;slot<nclass;slot++){
        int warp=0;
        while(warp<8 && fill[warp]+counts[slot]>32)warp++;
        if(warp==8)return 1;
        for(int lane=0;lane<256;lane++)
            if(qsb_window_second_key(windows[lane])==keys[slot]){
                memcpy(packed[warp*32+fill[warp]],windows[lane],3);
                fill[warp]++;
            }
    }
    for(int warp=0;warp<8;warp++)if(fill[warp]!=32)return 1;
    memcpy(windows,packed,sizeof(packed));
    return 0;
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
    // The specialist producer uses a compact 64-class shared first-state cache.
    // Capacity mechanism: PR156, anamdongparkjinhyeong, head
    // 6f6410dc7393c48c6814d016703fb404a310a3a1. Preserve packed host selection.
    if(first_distinct>64) {
        fprintf(stderr,"Window first-state classes exceed producer capacity64: %d\n",first_distinct);
        return 1;
    }
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

__device__ __forceinline__ void qsb_scheduled_window_hash_scratch(uint32_t *state,
        const epoch_desc_t *epoch, int lane, uint32_t first_states[8][256]) {
    // Every lane in this epoch's block reaches the shared-memory barrier.
    if(lane<QSB_FIRST_COUNT) {
        uint32_t initial[8],W[16];
        #pragma unroll
        for(int j=0;j<8;j++)initial[j]=epoch->mid[j];
        W[0]=epoch->remW[0];W[1]=epoch->remW[1];
        #pragma unroll
        for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][lane];
        _SHA256Transform(initial,W);
        #pragma unroll
        for(int j=0;j<8;j++)first_states[j][lane]=initial[j];
    }
    __syncthreads();
    int first_slot=QSB_FIRST_CLASS[lane];
    #pragma unroll
    for(int j=0;j<8;j++)state[j]=first_states[j][first_slot];
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

// Preserve original ABI for independent callers.
__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const epoch_desc_t *epoch, int lane) {
    __shared__ uint32_t first_states[8][256];
    qsb_scheduled_window_hash_scratch(state,epoch,lane,first_states);
}

// One complete producer warp prepares one epoch's compact first-block cache.
// The caller keeps the cache unchanged until all eight packets have consumed it.
// Unlike the legacy CTA helper, both producer APIs contain no CTA rendezvous.
__device__ __forceinline__ void qsb_producer_prepare_first(
        const epoch_desc_t *epoch, uint32_t first_states[8][64], int lane) {
    for(int slot=lane;slot<QSB_FIRST_COUNT;slot+=32) {
        uint32_t initial[8],W[16];
        #pragma unroll
        for(int j=0;j<8;j++)initial[j]=epoch->mid[j];
        W[0]=epoch->remW[0];W[1]=epoch->remW[1];
        #pragma unroll
        for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][slot];
        _SHA256Transform(initial,W);
        #pragma unroll
        for(int j=0;j<8;j++)first_states[j][slot]=initial[j];
    }
    __syncwarp(0xffffffffu);
}

// choice is the original packed window index, not the producer's physical lane.
// Return the final SHA256d digest words, including the 32-byte second hash.
__device__ __forceinline__ void qsb_producer_window_hash(uint32_t *digest,
        const uint32_t first_states[8][64], int choice) {
    uint32_t state[8];
    int first_slot=QSB_FIRST_CLASS[choice];
    #pragma unroll
    for(int j=0;j<8;j++)state[j]=first_states[j][first_slot];
    int slot=QSB_WINDOW_CLASS[choice];
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
    uint32_t b2[16]={},s2[8];
    #pragma unroll
    for(int j=0;j<8;j++)b2[j]=state[j];
    b2[8]=0x80000000u;b2[15]=256;
    _SHA256Initialize(s2);_SHA256Transform(s2,b2);
    #pragma unroll
    for(int j=0;j<8;j++)digest[j]=s2[j];
}

// Two producer warps own distinct first-state slots. The caller must join
// both warps before cache retirement/overwrite and after this publication.
// No synchronization is hidden inside the preparation helper.
__device__ __forceinline__ void qsb_producer64_prepare_first(
        const epoch_desc_t *epoch, uint32_t first_states[8][64], int producer_tid) {
    for(int slot=producer_tid;slot<QSB_FIRST_COUNT;slot+=64) {
        uint32_t initial[8],W[16];
        #pragma unroll
        for(int j=0;j<8;j++)initial[j]=epoch->mid[j];
        W[0]=epoch->remW[0];W[1]=epoch->remW[1];
        #pragma unroll
        for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][slot];
        _SHA256Transform(initial,W);
        #pragma unroll
        for(int j=0;j<8;j++)first_states[j][slot]=initial[j];
    }
}

// Two-warp reusable subgroup barrier. Physical threads0..63 participate;
// sync[0:2] is separate from EC counters and starts at zero.
__device__ __forceinline__ void qsb_producer64_barrier(uint32_t *sync){
    __syncwarp();
    if((threadIdx.x&31)==0){
        const uint32_t generation=atomicAdd(sync+1,0u);
        __threadfence_block();
        const uint32_t ticket=atomicAdd(sync,1u);
        if(ticket==1u){
            // Reset before publishing: next-generation arrivals must not be
            // erased by a late reset from the preceding generation.
            atomicExch(sync,0u);
            __threadfence_block();
            atomicAdd(sync+1,1u);
        }else{
            while(atomicAdd(sync+1,0u)==generation){}
            __threadfence_block();
        }
    }
    __syncwarp();
}
