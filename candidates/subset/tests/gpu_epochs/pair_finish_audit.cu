// Independent OpenSSL curve oracle for the actual pre/post recovery helpers.
// Synthetic projective scales include zero and unusable equal/opposite points.
#define main qsb_candidate_main
#include "tree.cu"
#undef main
#include <vector>
struct FinishInput {uint64_t q[16],inv[4],expected[8];uint32_t parity,usable;};
struct FinishOutput {uint64_t x[8],product[4];uint32_t parity,usable;};
__global__ void check_pair_finish(const FinishInput*in,FinishOutput*out,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t q[16],W[5],rx[4],ry[4],inverse[4];
    for(int k=0;k<16;k++)q[k]=in[i].q[k];
    for(int k=0;k<4;k++){rx[k]=QSB_U2R[k];ry[k]=QSB_U2R[4+k];inverse[k]=in[i].inv[k];}
    qsb_xyzz_finish_prepare(q,q+8,q+12,rx,W);
    out[i].usable=(W[0]|W[1]|W[2]|W[3])!=0;
    if(out[i].usable){
        uint64_t m1[4],m2[4];qsb_k2s_pre(q+4,q+8,q+12,ry,m1,m2);
        out[i].parity=qsb_k2s_post(m1,m2,inverse,rx,ry,out[i].x,out[i].x+4);
        _ModMult(out[i].product,W,inverse);
    }
}
#define REQ(x) do{if(!(x)){fprintf(stderr,"Oracle failure: %s\n",#x);return 2;}}while(0)
#define CUDA(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return 2;}}while(0)
static void export_words(uint64_t*out,const BIGNUM*x){if(BN_bn2lebinpad(x,(unsigned char*)out,32)!=32)abort();}
int main(){
    constexpr int count=2053;unsigned errors=0,usable=0,unusable=0;
    BN_CTX*ctx=BN_CTX_new();EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BIGNUM *p=BN_new(),*n=BN_new(),*r=BN_new(),*k=BN_new(),*z=BN_new(),*zz=BN_new(),*zzz=BN_new();
    BIGNUM *x=BN_new(),*y=BN_new(),*rx=BN_new(),*ry=BN_new(),*tmp=BN_new(),*c=BN_new(),*W=BN_new();
    EC_POINT*R=EC_POINT_new(g),*Q=EC_POINT_new(g),*sum=EC_POINT_new(g),*negative=EC_POINT_new(g);
    REQ(EC_GROUP_get_curve(g,p,nullptr,nullptr,ctx));REQ(EC_GROUP_get_order(g,n,ctx));
    for(int run=0;run<2;run++){
        if(!run)REQ(BN_set_word(r,17));else{REQ(BN_one(r));REQ(BN_lshift(r,r,128));REQ(BN_add_word(r,31337));}
        REQ(EC_POINT_mul(g,R,r,nullptr,nullptr,ctx));REQ(EC_POINT_get_affine_coordinates(g,R,rx,ry,ctx));
        REQ(EC_POINT_copy(negative,R));REQ(EC_POINT_invert(g,negative,ctx));
        REQ(BN_mod_sqr(c,rx,p,ctx));REQ(BN_mul_word(c,3));REQ(BN_mod_add(tmp,ry,ry,p,ctx));
        REQ(BN_mod_inverse(tmp,tmp,p,ctx));REQ(BN_mod_mul(c,c,tmp,p,ctx));
        uint64_t base[8],cc[4];export_words(base,rx);export_words(base+4,ry);export_words(cc,c);
        CUDA(cudaMemcpyToSymbol(QSB_U2R,base,sizeof(base)));CUDA(cudaMemcpyToSymbol(QSB_U2R_C,cc,sizeof(cc)));
        std::vector<FinishInput>in(count);std::vector<FinishOutput>out(count+3);
        memset(out.data(),0xa5,out.size()*sizeof(FinishOutput));
        for(int i=0;i<count;i++){
            unsigned char seed[8]={(unsigned char)i,(unsigned char)(i>>8),(unsigned char)run,73,19,213,51,4},hash[32];
            SHA256(seed,sizeof(seed),hash);REQ(BN_bin2bn(hash,32,k));REQ(BN_mod(k,k,n,ctx));
            switch(i%257){case 0:BN_zero(k);break;case 1:REQ(BN_copy(k,r));break;case 2:REQ(BN_sub(k,n,r));break;default:break;}
            seed[7]^=183;SHA256(seed,sizeof(seed),hash);REQ(BN_bin2bn(hash,32,z));REQ(BN_mod(z,z,p,ctx));
            if(i%31==0)BN_zero(z);else if(i%31==1)REQ(BN_one(z));else if(i%31==2){REQ(BN_copy(z,p));REQ(BN_sub_word(z,1));}
            REQ(BN_mod_sqr(zz,z,p,ctx));REQ(BN_mod_mul(zzz,zz,z,p,ctx));
            REQ(EC_POINT_mul(g,Q,k,nullptr,nullptr,ctx));
            if(EC_POINT_is_at_infinity(g,Q)){BN_zero(x);REQ(BN_one(y));BN_zero(zz);BN_zero(zzz);}
            else{REQ(EC_POINT_get_affine_coordinates(g,Q,x,y,ctx));REQ(BN_mod_mul(x,x,zz,p,ctx));REQ(BN_mod_mul(y,y,zzz,p,ctx));}
            auto&v=in[i];export_words(v.q,x);export_words(v.q+4,y);export_words(v.q+8,zz);export_words(v.q+12,zzz);
            REQ(BN_mod_mul(W,rx,zz,p,ctx));REQ(BN_mod_sub(W,W,x,p,ctx));REQ(BN_mod_mul(W,W,zzz,p,ctx));
            v.usable=!BN_is_zero(W);
            if(v.usable){
                REQ(BN_mod_inverse(tmp,W,p,ctx));export_words(v.inv,tmp);
                for(int j=0;j<2;j++){
                    REQ(EC_POINT_add(g,sum,Q,j?negative:R,ctx));REQ(!EC_POINT_is_at_infinity(g,sum));
                    REQ(EC_POINT_get_affine_coordinates(g,sum,x,y,ctx));export_words(v.expected+4*j,x);
                    v.parity|=(unsigned)BN_is_odd(y)<<j;
                }
                usable++;
            }else unusable++;
        }
        FinishInput*di;FinishOutput*doo;CUDA(cudaMalloc(&di,in.size()*sizeof(FinishInput)));CUDA(cudaMalloc(&doo,out.size()*sizeof(FinishOutput)));
        CUDA(cudaMemcpy(di,in.data(),in.size()*sizeof(FinishInput),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(doo,out.data(),out.size()*sizeof(FinishOutput),cudaMemcpyHostToDevice));
        check_pair_finish<<<(count+127)/128,128>>>(di,doo,count);CUDA(cudaGetLastError());
        CUDA(cudaMemcpy(out.data(),doo,out.size()*sizeof(FinishOutput),cudaMemcpyDeviceToHost));
        for(int i=0;i<count;i++){
            bool bad=out[i].usable!=in[i].usable;
            if(in[i].usable){bad|=memcmp(out[i].x,in[i].expected,64)!=0||out[i].parity!=in[i].parity;
                for(int j=0;j<4;j++)bad|=out[i].product[j]!=(uint64_t)(j==0);}
            if(bad){if(errors<8)printf("Finish mismatch run=%d case=%d\n",run,i);errors++;}
        }
        const unsigned char*guards=(const unsigned char*)(out.data()+count);
        for(size_t i=0;i<3*sizeof(FinishOutput);i++)if(guards[i]!=0xa5)errors++;
        CUDA(cudaFree(di));CUDA(cudaFree(doo));
    }
    printf("Pair finish GPU/OpenSSL audit: cases=%d usable=%u unusable=%u recovered_keys=%u guard_bytes=%zu errors=%u\n",2*count,usable,unusable,2*usable,6*sizeof(FinishOutput),errors);
    EC_POINT_free(R);EC_POINT_free(Q);EC_POINT_free(sum);EC_POINT_free(negative);EC_GROUP_free(g);
    for(auto*b:{p,n,r,k,z,zz,zzz,x,y,rx,ry,tmp,c,W})BN_free(b);BN_CTX_free(ctx);
    return errors?1:0;
}
