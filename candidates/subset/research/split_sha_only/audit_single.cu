// Exact bitwise comparison against the unchanged public paired SHA routine.
#define main qsb_original_program_main
#include "candidate_single.cu"
#undef main
#define kernel_sha_only kernel_sha_reference
#include "sha_only.cuh"
#undef kernel_sha_only
#include <vector>

#define AUDIT_CUDA(call) do { cudaError_t err=(call); if(err!=cudaSuccess){ \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(err));return 2;} } while(0)

static uint32_t audit_next(uint32_t &s){
    s^=s<<13;s^=s>>17;s^=s<<5;return s;
}

static uint32_t audit_rotr(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
static void audit_compress(uint32_t state[8],const uint32_t schedule[64]){
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
    for(unsigned r=0;r<64;r++){
        uint32_t s1=audit_rotr(e,6)^audit_rotr(e,11)^audit_rotr(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+s1+ch+schedule[r]; // Schedule already includes round K.
        uint32_t s0=audit_rotr(a,2)^audit_rotr(a,13)^audit_rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c),t2=s0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

int main(){
    const unsigned counts[]={1,2,3,4,5,7,8,9,15,16,17,31,32,33,65};
    const unsigned max_epochs=65, max_words=max_epochs*QSB_SE_WINDOWS*4+16;
    std::vector<uint32_t> first(max_epochs*QSB_FIRST_SLOTS*8);
    uint32_t *d_first=nullptr;uint64_t *d_ref=nullptr,*d_test=nullptr;
    AUDIT_CUDA(cudaMalloc(&d_first,first.size()*sizeof(uint32_t)));
    AUDIT_CUDA(cudaMalloc(&d_ref,max_words*sizeof(uint64_t)));
    AUDIT_CUDA(cudaMalloc(&d_test,max_words*sizeof(uint64_t)));
    std::vector<uint64_t> ref(max_words),test(max_words);
    uint32_t state=0x6917d531u;unsigned long long checked=0;
    for(unsigned trial=0;trial<4;trial++){
        uint32_t fc[QSB_SE_PER_EPOCH],wc[QSB_SE_PER_EPOCH];
        uint32_t schedules[64][QSB_SE_PER_EPOCH],suffix[4][64];
        for(unsigned i=0;i<QSB_SE_PER_EPOCH;i++){
            fc[i]=audit_next(state)%QSB_FIRST_SLOTS;
            wc[i]=audit_next(state)%QSB_SE_PER_EPOCH;
        }
        for(auto &row:schedules)for(auto &word:row)word=audit_next(state);
        for(auto &row:suffix)for(auto &word:row)word=audit_next(state);
        for(auto &word:first)word=trial==0?0u:trial==1?0xffffffffu:audit_next(state);
        AUDIT_CUDA(cudaMemcpyToSymbol(QSB_FIRST_CLASS,fc,sizeof(fc)));
        AUDIT_CUDA(cudaMemcpyToSymbol(QSB_WINDOW_CLASS,wc,sizeof(wc)));
        AUDIT_CUDA(cudaMemcpyToSymbol(QSB_WINDOW_SECOND,schedules,sizeof(schedules)));
        AUDIT_CUDA(cudaMemcpyToSymbol(QSB_CONST_SCHEDULE,suffix,sizeof(suffix)));
        AUDIT_CUDA(cudaMemcpy(d_first,first.data(),first.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
        for(unsigned epochs:counts){
            const unsigned n=epochs*QSB_SE_WINDOWS,words=n*4;
            AUDIT_CUDA(cudaMemset(d_ref,0xa5,max_words*sizeof(uint64_t)));
            AUDIT_CUDA(cudaMemset(d_test,0xa5,max_words*sizeof(uint64_t)));
            kernel_sha_reference<<<(epochs+QSB_PAIR_MUL-1)/QSB_PAIR_MUL,QSB_SE_BLOCK>>>(d_first,d_ref,epochs,n);
            kernel_sha_only<<<(n+255)/256,256>>>(d_first,d_test,epochs,n);
            AUDIT_CUDA(cudaGetLastError());
            AUDIT_CUDA(cudaMemcpy(ref.data(),d_ref,(words+16)*sizeof(uint64_t),cudaMemcpyDeviceToHost));
            AUDIT_CUDA(cudaMemcpy(test.data(),d_test,(words+16)*sizeof(uint64_t),cudaMemcpyDeviceToHost));
            for(unsigned j=0;j<words;j++)if(ref[j]!=test[j]){
                fprintf(stderr,"mismatch trial=%u epochs=%u word=%u\n",trial,epochs,j);return 1;
            }
            for(unsigned i=0;i<n;i++){
                unsigned epoch=i/QSB_SE_WINDOWS,lane=i%QSB_SE_WINDOWS;
                uint32_t st[8],expanded[64];
                for(unsigned j=0;j<8;j++)st[j]=first[(epoch*QSB_FIRST_SLOTS+fc[lane])*8+j];
                for(unsigned r=0;r<64;r++)expanded[r]=schedules[r][wc[lane]];
                audit_compress(st,expanded);
                for(unsigned b=0;b<4;b++)audit_compress(st,suffix[b]);
                unsigned char message[32],digest[32];
                for(unsigned j=0;j<32;j++)message[j]=(unsigned char)(st[j/4]>>(24-8*(j%4)));
                SHA256(message,32,digest);
                for(unsigned j=0;j<4;j++){
                    uint64_t expected=0;
                    for(unsigned k=0;k<8;k++)expected=(expected<<8)|digest[(3-j)*8+k];
                    if(test[j*n+i]!=expected){
                        fprintf(stderr,"CPU oracle mismatch trial=%u epochs=%u candidate=%u limb=%u\n",trial,epochs,i,j);return 1;
                    }
                }
            }
            for(unsigned j=words;j<words+16;j++)if(ref[j]!=0xa5a5a5a5a5a5a5a5ULL || test[j]!=0xa5a5a5a5a5a5a5a5ULL){
                fprintf(stderr,"guard overwritten trial=%u epochs=%u word=%u\n",trial,epochs,j);return 1;
            }
            checked+=n;
        }
    }
    AUDIT_CUDA(cudaFree(d_first));AUDIT_CUDA(cudaFree(d_ref));AUDIT_CUDA(cudaFree(d_test));
    printf("PASS: %llu 256-bit scalars match paired SHA and independent CPU/OpenSSL oracle; 60 boundary/odd-tail cases and guard regions pass.\n",checked);
    return 0;
}
