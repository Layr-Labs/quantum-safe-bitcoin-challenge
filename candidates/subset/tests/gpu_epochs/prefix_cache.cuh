// Per-batch SHA prefix states derived solely from the current instance.
// Cache after whole pushes: seven for one block, thirteen for two blocks.
#ifndef QSB_PREFIX_BLOCKS
#define QSB_PREFIX_BLOCKS 1
#endif
static_assert(QSB_PREFIX_BLOCKS==1 || QSB_PREFIX_BLOCKS==2,"supported cache depth");
constexpr int QSB_PREFIX_KEEP=(64*QSB_PREFIX_BLOCKS+9)/10;
constexpr int QSB_PREFIX_BITS=QSB_PREFIX_KEEP+6;
constexpr int QSB_PREFIX_ENTRIES=1<<QSB_PREFIX_BITS;
struct __align__(16) QSBPrefixRecord { uint4 lo,hi,tail; };
#if defined(QSB_PAIR_SHARED) && !QSB_PAIR_SHARED
__device__ QSBPrefixRecord QSB_PREFIX_CACHE[QSB_PREFIX_ENTRIES];
#endif

__host__ __device__ inline bool qsb_prefix_eligible(int n,int start,int t,int fast,int rem){
    return fast==QSB_FAST_N_INC && rem==0 && t>=0 && t<=6 && start>=0 && n-start>=QSB_PREFIX_BITS;
}

template<int first,int last>
__device__ __forceinline__ void qsb_emit_pushes(uint32_t *state,uint32_t *W,uint64_t &queue,int &pos){
    #pragma unroll
    for(int k=first;k<last;k++){
        while((int)(queue&255)==pos){queue>>=8;pos++;}
        const uint4 words=QSB_PUSH_WORDS[pos++];
        uint32_t A=words.x,B=words.y,C=words.z;
        const int g=(10*k)/4;
        if((10*k)%4==0){
            W[g&15]=A; if((g&15)==15)_SHA256Transform(state,W);
            W[(g+1)&15]=B; if(((g+1)&15)==15)_SHA256Transform(state,W);
            W[(g+2)&15]=C<<16;
        }else{
            W[g&15]|=A>>16; if((g&15)==15)_SHA256Transform(state,W);
            W[(g+1)&15]=(A<<16)|(B>>16);
            if(((g+1)&15)==15)_SHA256Transform(state,W);
            W[(g+2)&15]=(B<<16)|C;
            if(((g+2)&15)==15)_SHA256Transform(state,W);
        }
    }
}

#if defined(QSB_PAIR_SHARED) && !QSB_PAIR_SHARED   /* only the legacy (non-pair) host path launches it */
__global__ void qsb_prepare_prefix_cache(const uint32_t *mid,int start,int t){
    unsigned mask=blockIdx.x*blockDim.x+threadIdx.x;
    if(mask>=QSB_PREFIX_ENTRIES || __popc(mask)>t)return;
    uint64_t queue=UINT64_MAX;
    for(int i=QSB_PREFIX_BITS-1;i>=0;i--)
        if(mask&(1u<<i))queue=(queue<<8)|(uint64_t)(start+i);
    int pos=start;
    uint32_t state[8],W[16];
    #pragma unroll
    for(int k=0;k<8;k++)state[k]=mid[k];
    qsb_emit_pushes<0,QSB_PREFIX_KEEP>(state,W,queue,pos);
    QSBPrefixRecord record;
    record.lo=make_uint4(state[0],state[1],state[2],state[3]);
    record.hi=make_uint4(state[4],state[5],state[6],state[7]);
    record.tail=make_uint4(W[0],W[1],pos,0);
    QSB_PREFIX_CACHE[mask]=record;
}
#endif

#if defined(QSB_PAIR_SHARED) && !QSB_PAIR_SHARED
__device__ __forceinline__ void qsb_fast_window_hash(uint32_t *state,const uint8_t *skip,
        int t,int early,int start,bool cached,const uint32_t *constant_words){
    uint32_t W[16];
    uint64_t queue=0;
    #pragma unroll
    for(int i=7;i>=0;i--)queue=(queue<<8)|(uint64_t)(i<t?skip[early+i]:255);
    int pos=start;
    if(cached){
        unsigned mask=0;
        #pragma unroll
        for(int i=0;i<6;i++)if(i<t){
            unsigned rel=(unsigned)(skip[early+i]-start);
            if(rel<QSB_PREFIX_BITS)mask|=1u<<rel;
        }
        const QSBPrefixRecord record=QSB_PREFIX_CACHE[mask];
        state[0]=record.lo.x;state[1]=record.lo.y;state[2]=record.lo.z;state[3]=record.lo.w;
        state[4]=record.hi.x;state[5]=record.hi.y;state[6]=record.hi.z;state[7]=record.hi.w;
        W[0]=record.tail.x;W[1]=record.tail.y;pos=(int)record.tail.z;
        while((int)(queue&255)<pos)queue>>=8;
    }else{
        qsb_emit_pushes<0,QSB_PREFIX_KEEP>(state,W,queue,pos);
    }
    qsb_emit_pushes<QSB_PREFIX_KEEP,QSB_FAST_N_INC>(state,W,queue,pos);
    #pragma unroll
    for(int j=0;j<5;j++){
        const int wi=((10*QSB_FAST_N_INC)/4+j)&15;
        W[wi]=constant_words[j];
        if(wi==15)_SHA256Transform(state,W);
    }
    qsb_compress_constant_rolled(state);
}
#endif
