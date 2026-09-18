// Derived from odinfree's GPU-epoch consumer in submission 0db6e203.
// Only the first block depends on the epoch remainder. The second block's
// expanded schedule is shared by every epoch with the same window choice.
#pragma once
#define QSB_FIRST_SLOTS 64   /* first-block classes per epoch in d_first */
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
    if(first_distinct>QSB_FIRST_SLOTS)return 1;
    for(int slot=0;slot<first_distinct;slot++)
        for(int j=0;j<14;j++)transposed[j][slot]=first_unique[slot][j];
    if(cudaMemcpyToSymbol(QSB_FIRST_COUNT,&first_distinct,sizeof(first_distinct))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_CLASS,first_classes,sizeof(first_classes))!=cudaSuccess)return 1;
    if(cudaMemcpyToSymbol(QSB_FIRST_UNIQUE,transposed,sizeof(transposed))!=cudaSuccess)return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_CLASS,classes,sizeof(classes))!=cudaSuccess) return 1;
    if (cudaMemcpyToSymbol(QSB_WINDOW_FIRST,first,sizeof(first))!=cudaSuccess) return 1;
    return cudaMemcpyToSymbol(QSB_WINDOW_SECOND,second,sizeof(second))==cudaSuccess?0:1;
}

/* First-block states for every (epoch, class) of a launch, computed as its own
 * producer stage: one block per epoch, one thread per first-block class. The
 * consumer used to spend a thread barrier plus one compression of block time
 * per epoch while 54 leader lanes built these states and the other lanes
 * waited; now each lane reads its class state (32 bytes). */
__global__ void kernel_build_first(const epoch_desc_t * __restrict__ d_epochs,
        uint32_t * __restrict__ d_first) {
    const epoch_desc_t *ep = d_epochs + blockIdx.x;
    const int c = threadIdx.x;
    uint32_t st[8], W[16];
    #pragma unroll
    for(int j=0;j<8;j++)st[j]=ep->mid[j];
    W[0]=ep->remW[0];W[1]=ep->remW[1];
    #pragma unroll
    for(int j=2;j<16;j++)W[j]=QSB_FIRST_UNIQUE[j-2][c];
    _SHA256Transform(st,W);
    const size_t base=((size_t)blockIdx.x*QSB_FIRST_SLOTS+(size_t)c)*8;
    #pragma unroll
    for(int j=0;j<8;j++)d_first[base+j]=st[j];
}

__device__ __forceinline__ void qsb_scheduled_window_hash(uint32_t *state,
        const epoch_desc_t *epoch, int lane, const uint32_t *first) {
    (void)epoch;
    const int first_slot=QSB_FIRST_CLASS[lane];
    #pragma unroll
    for(int j=0;j<8;j++)state[j]=first[first_slot*8+j];
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
