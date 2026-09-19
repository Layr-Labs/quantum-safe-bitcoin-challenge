#define main grinder_main
#include "../pinning.cu"
#undef main
#include <vector>
struct PointCase{uint64_t k[4],x[4],y[4],mag[2][2];unsigned infinity,pad,signs[2];};
__global__ void audit_glv_points(PointCase *cases,int n,const uint8_t*table,unsigned*errors){
    int i=blockIdx.x*128+threadIdx.x;if(i>=n)return;PointCase c=cases[i];uint64_t X[5],Y[5],U[5],V[5];
    q10_decode_components(c.mag,c.signs);q10_accumulate_codes(X,Y,U,V,table);
    bool infinity=q9_zero(U)||q9_zero(V);bool bad=infinity!=(bool)c.infinity;
    if(!infinity){
        U[4]=V[4]=0;qsb_field_normalize(U);qsb_field_normalize(V);_ModInv(U);_ModInv(V);
        qsb_field_mul(X,X,U);qsb_field_mul(Y,Y,V);qsb_field_normalize(X);qsb_field_normalize(Y);
        for(int j=0;j<4;j++)bad|=X[j]!=c.x[j]||Y[j]!=c.y[j];
    }
    if(bad&&atomicAdd(errors,1)==0)printf("GLV point mismatch case=%d infinity=%d expected=%u\n",i,infinity,c.infinity);
}
int main(int argc,char**argv){
    if(argc!=2)return 2;pinning2_params_t pp;if(load_pinning2(argv[1],&pp))return 2;
    cudaDeviceSetLimit(cudaLimitStackSize,32768);const int n=16384;
    std::vector<PointCase> cases(n);
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX*ctx=BN_CTX_new();
    BIGNUM *nri=BN_lebin2bn(pp.neg_r_inv,32,nullptr),*order=BN_new(),*lambda=nullptr,*k=BN_new(),*t=BN_new(),*x=BN_new(),*y=BN_new();
    BN_hex2bn(&lambda,"5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");EC_GROUP_get_order(g,order,ctx);EC_POINT *point=EC_POINT_new(g);
    std::vector<__uint128_t> edge={0,1,2,~(__uint128_t)0};
    const int bits[]={1,16,17,18,33,34,51,63,64,65,102,118,119,127};
    for(int bit:bits){__uint128_t z=(__uint128_t)1<<bit;edge.push_back(z-1);edge.push_back(z);edge.push_back(z+1);}
    // Include nonzero lattice cancellation and a final-joint doubling case.
    edge.push_back(((__uint128_t)0x3086d221a7d46bcdULL<<64)|0xe86c90e49284eb15ULL);
    edge.push_back(((__uint128_t)0xe4437ed6010e8828ULL<<64)|0x6f547fa90abfe4c3ULL);
    __uint128_t doubling=(__uint128_t)0-((__uint128_t)1<<119);
    edge.push_back(doubling-1);edge.push_back(doubling);edge.push_back(doubling+1);
    int targeted=(int)(4*edge.size()*edge.size());if(targeted>n)targeted=n;
    uint64_t rng=0x547710badu;
    auto next=[&](){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;};
    for(int i=0;i<n;i++){
        if(i<targeted){
            __uint128_t aa=edge[(i/4)/edge.size()],bb=edge[(i/4)%edge.size()];
            cases[i].mag[0][0]=(uint64_t)aa;cases[i].mag[0][1]=(uint64_t)(aa>>64);
            cases[i].mag[1][0]=(uint64_t)bb;cases[i].mag[1][1]=(uint64_t)(bb>>64);
            cases[i].signs[0]=i&1;cases[i].signs[1]=(i>>1)&1;
        }else{
            for(int side=0;side<2;side++)for(int j=0;j<2;j++)cases[i].mag[side][j]=next();
            cases[i].signs[0]=next()&1;cases[i].signs[1]=next()&1;
        }
        BIGNUM *aa=BN_lebin2bn((const unsigned char*)cases[i].mag[0],16,nullptr);
        BIGNUM *bb=BN_lebin2bn((const unsigned char*)cases[i].mag[1],16,nullptr);
        BN_set_negative(aa,cases[i].signs[0]&&!BN_is_zero(aa));BN_set_negative(bb,cases[i].signs[1]&&!BN_is_zero(bb));
        BN_mod_mul(k,bb,lambda,order,ctx);BN_mod_add(k,k,aa,order,ctx);
        BN_bn2lebinpad(k,(unsigned char*)cases[i].k,32);BN_free(aa);BN_free(bb);
    }
    for(int i=0;i<n;i++){
        BN_lebin2bn((unsigned char*)cases[i].k,32,k);BN_mod_mul(t,k,nri,order,ctx);EC_POINT_mul(g,point,t,nullptr,nullptr,ctx);
        cases[i].infinity=EC_POINT_is_at_infinity(g,point);
        if(!cases[i].infinity){EC_POINT_get_affine_coordinates(g,point,x,y,ctx);BN_bn2lebinpad(x,(unsigned char*)cases[i].x,32);BN_bn2lebinpad(y,(unsigned char*)cases[i].y,32);}
    }
    std::vector<uint64_t> hL((size_t)GT_CHUNKS*GT_LO*8),hH((size_t)GT_CHUNKS*GT_HI*8);gt_build_ladders(hL.data(),hH.data(),pp.neg_r_inv);
    uint64_t *dL,*dH;uint8_t*table;cudaMalloc(&dL,hL.size()*8);cudaMalloc(&dH,hH.size()*8);cudaMalloc(&table,((size_t)GT_TOTAL_ENTRIES*64));
    cudaMemcpy(dL,hL.data(),hL.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(dH,hH.data(),hH.size()*8,cudaMemcpyHostToDevice);
    kernel_build_gtable<<<(GT_TOTAL_ENTRIES+127)/128,128>>>(dL,dH,table);cudaError_t ce=cudaDeviceSynchronize();if(ce!=cudaSuccess){printf("table CUDA %s\n",cudaGetErrorString(ce));return 1;}
    std::vector<uint8_t> hosttable(((size_t)GT_TOTAL_ENTRIES*64));cudaMemcpy(hosttable.data(),table,hosttable.size(),cudaMemcpyDeviceToHost);
    if(!gt_spot_check(hosttable.data(),1024,pp.neg_r_inv))return 1;
    PointCase *d;unsigned*errors,e;cudaMalloc(&d,n*sizeof(PointCase));cudaMalloc(&errors,4);cudaMemcpy(d,cases.data(),n*sizeof(PointCase),cudaMemcpyHostToDevice);cudaMemset(errors,0,4);
    audit_glv_points<<<n/128,128>>>(d,n,table,errors);ce=cudaDeviceSynchronize();cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost);
    printf("GLV table entries=%u bytes=%llu OpenSSL_spots=1024\n",GT_TOTAL_ENTRIES,((unsigned long long)GT_TOTAL_ENTRIES*64));
    printf("GLV direct-component OpenSSL cases=%d boundary_pairs=%d mismatches=%u %s\n",n,targeted,e,cudaGetErrorString(ce));return e||ce!=cudaSuccess;
}
