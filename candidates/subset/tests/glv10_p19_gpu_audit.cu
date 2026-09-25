// SPDX-License-Identifier: GPL-3.0-only
// Native device scalar/recoder audit with independent OpenSSL integer oracles.
// No full table allocation, table builder, or device point-chain audit occurs.
#define QSB_GLV10_P19 1
#define main qsb_benchmark_main
#include "../subset.cu"
#undef main
#include <vector>

enum { P19_AUDIT_WORDS=32 };
__device__ void p19_audit_emit(const uint64_t p[2],unsigned sp,
                              const uint64_t q[2],unsigned sq,uint32_t *out) {
    for(int j=0;j<2;j++) {
        out[2*j]=uint32_t(p[j]);out[2*j+1]=uint32_t(p[j]>>32);
        out[4+2*j]=uint32_t(q[j]);out[5+2*j]=uint32_t(q[j]>>32);
    }
    out[8]=sp;out[9]=sq;
    qsb_s3_walker walker;qsb_s3_begin(walker,p,sp,q,sq);
    for(int t=0;t<10;t++) {
        out[10+t]=qsb_s3_code(walker,t,QSB_S3_DESC[t]);
        out[20+t]=q10_p19_code(t<5?q:p,t<5?sq:sp,t%5);
    }
    uint32_t remainder=0;for(int j=0;j<8;j++)remainder|=walker.w[j];
    out[30]=remainder;out[31]=walker.signs;
}
__global__ void p19_audit_split(const uint32_t *input,uint32_t *output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t k[4],p[2],q[2];unsigned sp,sq;
    for(int j=0;j<4;j++)k[j]=uint64_t(input[i*8+2*j])|(uint64_t(input[i*8+2*j+1])<<32);
    q9_glv_split(k,p,q,&sp,&sq);
    p19_audit_emit(p,sp,q,sq,output+size_t(i)*P19_AUDIT_WORDS);
}
__global__ void p19_audit_magnitudes(const uint32_t *input,uint32_t *output,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t p[2],q[2];
    for(int j=0;j<2;j++) {
        p[j]=uint64_t(input[i*9+2*j])|(uint64_t(input[i*9+2*j+1])<<32);
        q[j]=uint64_t(input[i*9+4+2*j])|(uint64_t(input[i*9+5+2*j])<<32);
    }
    const unsigned signs=input[i*9+8];
    p19_audit_emit(p,signs&1u,q,signs>>1,output+size_t(i)*P19_AUDIT_WORDS);
}

static void p19_require(bool ok,const char *message) {
    if(!ok){fprintf(stderr,"P19 audit failure: %s\n",message);exit(1);}
}
static void p19_cuda(cudaError_t error) {
    if(error!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(error));exit(2);}
}
static BIGNUM *p19_hex(const char *hex) {
    BIGNUM *r=nullptr;p19_require(BN_hex2bn(&r,hex)>0,"BN_hex2bn");return r;
}
static void p19_from_words(BIGNUM *out,const uint32_t *words,int count) {
    unsigned char bytes[32];
    for(int j=0;j<count*4;j++)bytes[j]=static_cast<unsigned char>(words[j/4]>>(8*(j%4)));
    p19_require(BN_lebin2bn(bytes,count*4,out)!=nullptr,"BN_lebin2bn");
}
static void p19_append_words(std::vector<uint32_t> &out,const BIGNUM *value,int count) {
    unsigned char bytes[32];p19_require(BN_bn2lebinpad(value,bytes,count*4)==count*4,"BN_bn2lebinpad");
    for(int j=0;j<count;j++)out.push_back(uint32_t(bytes[4*j])|(uint32_t(bytes[4*j+1])<<8)|
                                       (uint32_t(bytes[4*j+2])<<16)|(uint32_t(bytes[4*j+3])<<24));
}

struct P19Oracle {
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *n=p19_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    BIGNUM *bound=p19_hex("A2A8918CA85BAFE22016D0B917E4DD77");
    BIGNUM *a1=p19_hex("3086D221A7D46BCDE86C90E49284EB15");
    BIGNUM *a2=p19_hex("114CA50F7A8E2F3F657C1108D9D44CFD8");
    BIGNUM *b1=p19_hex("E4437ED6010E88286F547FA90ABFE4C3");
    BIGNUM *lambda=p19_hex("5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");
    BIGNUM *g1=BN_new(),*g2=BN_new(),*half=BN_new(),*bias=BN_new();
    BIGNUM *k=BN_new(),*c1=BN_new(),*c2=BN_new(),*x=BN_new(),*y=BN_new();
    BIGNUM *r1=BN_new(),*r2=BN_new(),*magp=BN_new(),*magq=BN_new();
    P19Oracle() {
        p19_require(ctx&&g1&&g2&&half&&bias&&k&&c1&&c2&&x&&y&&r1&&r2&&magp&&magq,"BN allocations");
        p19_require(BN_rshift1(y,n)==1,"half order");
        p19_require(BN_lshift(x,a1,384)&&BN_add(x,x,y)&&BN_div(g1,nullptr,x,n,ctx),"derive g1");
        p19_require(BN_lshift(x,b1,384)&&BN_add(x,x,y)&&BN_div(g2,nullptr,x,n,ctx),"derive g2");
        p19_require(BN_one(half)&&BN_lshift(half,half,383),"coefficient rounding constant");
        p19_require(BN_set_word(bias,170559770)&&BN_lshift(bias,bias,99)&&BN_sub_word(bias,1u<<18),"P19 bias");
    }
    ~P19Oracle() {
        BIGNUM *all[]={n,bound,a1,a2,b1,lambda,g1,g2,half,bias,k,c1,c2,x,y,r1,r2,magp,magq};
        for(BIGNUM *p:all)BN_free(p);BN_CTX_free(ctx);
    }
    void split(const uint32_t *words) {
        p19_from_words(k,words,8);p19_require(BN_nnmod(k,k,n,ctx),"scalar reduction");
        p19_require(BN_mul(x,k,g1,ctx)&&BN_add(x,x,half)&&BN_rshift(c1,x,384),"coefficient1");
        p19_require(BN_mul(x,k,g2,ctx)&&BN_add(x,x,half)&&BN_rshift(c2,x,384),"coefficient2");
        p19_require(BN_mul(x,c1,a1,ctx)&&BN_mul(y,c2,a2,ctx)&&BN_add(x,x,y)&&BN_sub(r1,k,x),"residual1");
        p19_require(BN_mul(x,c1,b1,ctx)&&BN_mul(y,c2,a1,ctx)&&BN_sub(r2,x,y),"residual2");
        p19_require(BN_copy(magp,r1)&&BN_copy(magq,r2),"copy magnitudes");
        BN_set_negative(magp,0);BN_set_negative(magq,0);
        p19_require(BN_cmp(magp,bound)<0&&BN_cmp(magq,bound)<0,"oracle residual bounds");
        p19_require(BN_mul(x,r2,lambda,ctx)&&BN_add(x,x,r1)&&BN_nnmod(x,x,n,ctx)&&BN_cmp(x,k)==0,"oracle lattice identity");
    }
};

static void p19_validate(P19Oracle &o,const uint32_t *out,const BIGNUM *p,unsigned sp,
                         const BIGNUM *q,unsigned sq,int row,const char *kind) {
    const unsigned offsets[5]={0,524288,67633152,134742016,201850880};
    const unsigned entries[5]={524288,67108864,67108864,67108864,85279885};
    const unsigned shifts[5]={0,19,46,73,100},widths[5]={19,27,27,27,28};
    BN_CTX_start(o.ctx);
    BIGNUM *got=BN_CTX_get(o.ctx),*field=BN_CTX_get(o.ctx),*term=BN_CTX_get(o.ctx);
    BIGNUM *sum=BN_CTX_get(o.ctx),*expected_sum=BN_CTX_get(o.ctx);
    p19_require(expected_sum!=nullptr,"validation BN allocations");
    p19_from_words(got,out,4);bool ok=BN_cmp(got,p)==0;
    p19_from_words(got,out+4,4);ok=ok&&BN_cmp(got,q)==0;
    ok=ok&&out[8]==sp&&out[9]==sq&&out[30]==0&&out[31]==(sq|(sp<<1));
    for(int component=0;component<2;component++) {
        const BIGNUM *mag=component?p:q;const unsigned sign=component?sp:sq;
        BN_zero(sum);
        for(int bank=0;bank<5;bank++) {
            const int t=component*5+bank;
            p19_require(BN_rshift(field,mag,shifts[bank]),"field extraction");
            if(BN_num_bits(field)>int(widths[bank]))p19_require(BN_mask_bits(field,widths[bank]),"field mask");
            const int64_t f=int64_t(BN_get_word(field));
            const int64_t digit=bank==0?2*f+1:bank==4?2*f-170559769:2*f+1-(int64_t(1)<<27);
            const unsigned index=unsigned(((digit<0?-digit:digit)-1)/2);
            const unsigned negative=unsigned(digit<0)^sign;
            const uint32_t expected=(offsets[bank]+index)|(negative<<31);
            ok=ok&&index<entries[bank]&&out[10+t]==expected&&out[20+t]==expected;
            const uint32_t code=out[10+t],record=code&0x7fffffffu;
            const bool in_bank=record>=offsets[bank]&&record-offsets[bank]<entries[bank];
            ok=ok&&in_bank;
            if(!in_bank)continue;
            const unsigned actual_index=record-offsets[bank];
            if(bank==0)p19_require(BN_copy(term,o.bias)&&BN_add_word(term,actual_index),"biased coefficient");
            else p19_require(BN_set_word(term,2*actual_index+1)&&BN_lshift(term,term,shifts[bank]-1),"odd coefficient");
            BN_set_negative(term,code>>31);p19_require(BN_add(sum,sum,term),"coefficient sum");
        }
        p19_require(BN_copy(expected_sum,mag)!=nullptr,"expected sum");BN_set_negative(expected_sum,sign);
        ok=ok&&BN_cmp(sum,expected_sum)==0;
    }
    BN_CTX_end(o.ctx);
    if(!ok){fprintf(stderr,"P19 mismatch: %s row=%d\n",kind,row);exit(1);}
}

int main() {
    P19Oracle oracle;
    std::vector<uint32_t> scalars,direct;
    BIGNUM *v=BN_new(),*z=BN_new(),*q=BN_new(),*m=BN_new(),*numerator=BN_new();
    p19_require(v&&z&&q&&m&&numerator,"input BN allocations");
    auto append_scalar=[&](const BIGNUM *value) {
        if(!BN_is_negative(value)&&BN_num_bits(value)<=256)p19_append_words(scalars,value,8);
    };
    auto scalar_near=[&](const BIGNUM *value) {
        for(int delta=-1;delta<=1;delta++) {
            BN_copy(z,value);if(delta<0)BN_sub_word(z,1);else if(delta>0)BN_add_word(z,1);
            append_scalar(z);
        }
    };
    BN_zero(v);append_scalar(v);scalar_near(oracle.n);
    BN_one(v);BN_lshift(v,v,256);BN_sub_word(v,1);append_scalar(v);
    for(int bit=0;bit<256;bit++){BN_one(v);BN_lshift(v,v,bit);scalar_near(v);}
    // Scalars immediately around independently computed coefficient half steps
    // target the device reciprocal routine's rare exact-rounding fallback band.
    for(BIGNUM *g:{oracle.g1,oracle.g2})for(int bit=0;bit<128;bit++)for(int delta=-1;delta<=1;delta++) {
        BN_one(m);BN_lshift(m,m,bit);if(delta<0)BN_sub_word(m,1);else if(delta>0)BN_add_word(m,1);
        p19_require(BN_lshift(numerator,m,384)&&BN_add(numerator,numerator,oracle.half)&&
                    BN_add(numerator,numerator,g)&&BN_sub_word(numerator,1)&&
                    BN_div(v,nullptr,numerator,g,oracle.ctx),"rounding boundary scalar");
        scalar_near(v);
    }
    uint64_t random=0x5031394750554155ULL;
    auto next=[&]() -> uint32_t {
        random^=random>>12;random^=random<<25;random^=random>>27;
        return uint32_t((random*0x2545f4914f6cdd1dULL)>>32);
    };
    for(int i=0;i<10000;i++)for(int j=0;j<8;j++)scalars.push_back(next());
    auto append_direct=[&](const BIGNUM *p,const BIGNUM *qq) {
        for(unsigned signs=0;signs<4;signs++) {
            p19_append_words(direct,p,4);p19_append_words(direct,qq,4);direct.push_back(signs);
        }
    };
    auto magnitude=[&](const BIGNUM *value) {
        if(BN_is_negative(value)||BN_cmp(value,oracle.bound)>=0)return;
        BN_copy(q,oracle.bound);BN_sub_word(q,1);BN_sub(q,q,value);append_direct(value,q);
    };
    BN_zero(v);magnitude(v);BN_one(v);magnitude(v);
    BN_copy(v,oracle.bound);BN_sub_word(v,1);magnitude(v);BN_sub_word(v,1);magnitude(v);
    for(int bit=0;bit<128;bit++)for(int delta=-1;delta<=1;delta++) {
        BN_one(v);BN_lshift(v,v,bit);if(delta<0)BN_sub_word(v,1);else if(delta>0)BN_add_word(v,1);magnitude(v);
    }
    const unsigned shifts[4]={19,46,73,100};
    const unsigned fields[7]={0,1,(1u<<26)-1,1u<<26,(1u<<26)+1,(1u<<27)-1,170559768};
    for(unsigned shift:shifts)for(unsigned field:fields)for(int delta=-1;delta<=1;delta++) {
        BN_set_word(v,field);BN_lshift(v,v,shift);if(delta<0)BN_sub_word(v,1);else if(delta>0)BN_add_word(v,1);magnitude(v);
    }
    for(int i=0;i<1000;i++) {
        uint32_t words[4];for(int j=0;j<4;j++)words[j]=next();p19_from_words(v,words,4);BN_nnmod(v,v,oracle.bound,oracle.ctx);
        for(int j=0;j<4;j++)words[j]=next();p19_from_words(q,words,4);BN_nnmod(q,q,oracle.bound,oracle.ctx);
        append_direct(v,q);
    }
    const int scalar_count=int(scalars.size()/8),direct_count=int(direct.size()/9);
    const size_t output_words=size_t(scalar_count>direct_count?scalar_count:direct_count)*P19_AUDIT_WORDS;
    const size_t input_words=scalars.size()>direct.size()?scalars.size():direct.size();
    uint32_t *device_input=nullptr,*device_output=nullptr;
    p19_cuda(cudaMalloc(&device_input,input_words*sizeof(uint32_t)));
    p19_cuda(cudaMalloc(&device_output,output_words*sizeof(uint32_t)));
    std::vector<uint32_t> output(output_words);
    p19_cuda(cudaMemcpy(device_input,scalars.data(),scalars.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    p19_audit_split<<<(scalar_count+127)/128,128>>>(device_input,device_output,scalar_count);
    p19_cuda(cudaGetLastError());p19_cuda(cudaDeviceSynchronize());
    p19_cuda(cudaMemcpy(output.data(),device_output,size_t(scalar_count)*P19_AUDIT_WORDS*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    for(int i=0;i<scalar_count;i++) {
        oracle.split(scalars.data()+size_t(i)*8);
        p19_validate(oracle,output.data()+size_t(i)*P19_AUDIT_WORDS,oracle.magp,BN_is_negative(oracle.r1),
                     oracle.magq,BN_is_negative(oracle.r2),i,"raw scalar");
    }
    p19_cuda(cudaMemcpy(device_input,direct.data(),direct.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));
    p19_audit_magnitudes<<<(direct_count+127)/128,128>>>(device_input,device_output,direct_count);
    p19_cuda(cudaGetLastError());p19_cuda(cudaDeviceSynchronize());
    p19_cuda(cudaMemcpy(output.data(),device_output,size_t(direct_count)*P19_AUDIT_WORDS*sizeof(uint32_t),cudaMemcpyDeviceToHost));
    for(int i=0;i<direct_count;i++) {
        const uint32_t *in=direct.data()+size_t(i)*9;
        p19_from_words(v,in,4);p19_from_words(q,in+4,4);
        p19_validate(oracle,output.data()+size_t(i)*P19_AUDIT_WORDS,v,in[8]&1u,q,in[8]>>1,i,"direct magnitudes");
    }
    p19_cuda(cudaFree(device_input));p19_cuda(cudaFree(device_output));
    BN_free(v);BN_free(z);BN_free(q);BN_free(m);BN_free(numerator);
    printf("PASS: %d raw scalars and %d direct signed magnitude pairs; device split/walker/direct codes against OpenSSL BN\n",scalar_count,direct_count);
    printf("Not covered: full 18GB table construction, device point chain, or publication path\n");
    return 0;
}
