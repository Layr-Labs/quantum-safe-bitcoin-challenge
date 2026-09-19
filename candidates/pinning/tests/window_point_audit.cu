#define main grinder_main
#include "../pinning.cu"
#undef main
#include <vector>
struct ScalarCase{uint64_t k[4];};
struct PointCase{uint64_t k[4],x[4],y[4];unsigned infinity,pad;};
template<int WINDOWS>
__global__ void audit_window_points(PointCase *cases,int n,const uint8_t*table,unsigned*errors){
    int i=blockIdx.x*128+threadIdx.x;if(i>=n)return;PointCase c=cases[i];uint64_t X[5],Y[5],U[5],V[5];
    _FixedBaseSignedXYZZScalar<WINDOWS>(X,Y,U,V,c.k,table,nullptr);
    bool infinity=qsb_point_zero(U)||qsb_point_zero(V);bool bad=infinity!=(bool)c.infinity;
    if(!infinity){
        U[4]=V[4]=0;qsb_field_normalize(U);qsb_field_normalize(V);_ModInv(U);_ModInv(V);
        qsb_field_mul(X,X,U);qsb_field_mul(Y,Y,V);qsb_field_normalize(X);qsb_field_normalize(Y);
        for(int j=0;j<4;j++)bad|=X[j]!=c.x[j]||Y[j]!=c.y[j];
    }
    if(bad&&atomicAdd(errors,1)==0)printf("Window point mismatch case=%d infinity=%d expected=%u\n",i,infinity,c.infinity);
}
int main(int argc,char**argv){
    if(argc!=4)return 2;int chunks=atoi(argv[3]);if(chunks<11||chunks>16)return 2;unsigned entries=gt_total_entries(chunks);pinning2_params_t pp;if(load_pinning2(argv[1],&pp))return 2;
    qsb_cuda_check(cudaDeviceSetLimit(cudaLimitStackSize,2048),"Set point-audit stack limit");const int n=16384;std::vector<ScalarCase> scalars(n);FILE*f=fopen(argv[2],"rb");if(!f)return 2;
    if(fread(scalars.data(),sizeof(ScalarCase),n,f)!=(size_t)n)return 2;fclose(f);
    std::vector<PointCase> cases(n);for(int i=0;i<n;i++)memcpy(cases[i].k,scalars[i].k,32);
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX*ctx=BN_CTX_new();
    BIGNUM *nri=BN_lebin2bn(pp.neg_r_inv,32,nullptr),*order=BN_new(),*lambda=nullptr,*k=BN_new(),*t=BN_new(),*x=BN_new(),*y=BN_new();
    BN_hex2bn(&lambda,"5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");EC_GROUP_get_order(g,order,ctx);EC_POINT *point=EC_POINT_new(g);
    int at=1024;
    // Values equal to one/two table terms and their endomorphisms deliberately
    // exercise possible incomplete-add edge cases; they are not just random k.
    for(int ch=0;ch<chunks;ch++)for(int corner=0;corner<4;corner++)for(int twice=0;twice<2;twice++)for(int endo=0;endo<2;endo++)for(int negative=0;negative<2;negative++){
        unsigned indices[4]={0,1,123,gt_entries(ch,chunks)-1};BN_set_word(k,2*indices[corner]+1);BN_lshift(k,k,gt_shift(ch,chunks));
        BIGNUM* inv2=BN_dup(order);BN_add_word(inv2,1);BN_rshift1(inv2,inv2);
        BN_mod_mul(k,k,inv2,order,ctx);BN_free(inv2);
        if(twice)BN_lshift1(k,k);if(endo)BN_mod_mul(k,k,lambda,order,ctx);BN_nnmod(k,k,order,ctx);
        if(negative&&!BN_is_zero(k))BN_sub(k,order,k);BN_bn2lebinpad(k,(unsigned char*)cases[at++].k,32);
    }
    for(int i=0;i<n;i++){
        BN_lebin2bn((unsigned char*)cases[i].k,32,k);BN_mod_mul(t,k,nri,order,ctx);EC_POINT_mul(g,point,t,nullptr,nullptr,ctx);
        cases[i].infinity=EC_POINT_is_at_infinity(g,point);
        if(!cases[i].infinity){EC_POINT_get_affine_coordinates(g,point,x,y,ctx);BN_bn2lebinpad(x,(unsigned char*)cases[i].x,32);BN_bn2lebinpad(y,(unsigned char*)cases[i].y,32);}
    }
    std::vector<uint64_t> hL((size_t)chunks*GT_LO*8),hH((size_t)chunks*GT_HI*8);gt_build_ladders(hL.data(),hH.data(),pp.neg_r_inv,chunks);
    uint64_t *dL,*dH;uint8_t*table;cudaMalloc(&dL,hL.size()*8);cudaMalloc(&dH,hH.size()*8);cudaMalloc(&table,((size_t)entries*64));
    cudaMemcpy(dL,hL.data(),hL.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(dH,hH.data(),hH.size()*8,cudaMemcpyHostToDevice);
    for(unsigned first=0;first<entries;first+=(1u<<20)) {
        unsigned count=entries-first;if(count>(1u<<20))count=1u<<20;
        kernel_build_gtable<<<(count+127)/128,128>>>(dL,dH,table,chunks,first);
    }cudaError_t ce=cudaDeviceSynchronize();if(ce!=cudaSuccess){printf("table CUDA %s\n",cudaGetErrorString(ce));return 1;}
    std::vector<uint8_t> hosttable(((size_t)entries*64));cudaMemcpy(hosttable.data(),table,hosttable.size(),cudaMemcpyDeviceToHost);
    if(!gt_spot_check(hosttable.data(),1024,pp.neg_r_inv,chunks))return 1;
    PointCase *d;unsigned*errors,e;cudaMalloc(&d,n*sizeof(PointCase));cudaMalloc(&errors,4);cudaMemcpy(d,cases.data(),n*sizeof(PointCase),cudaMemcpyHostToDevice);cudaMemset(errors,0,4);
    switch(chunks){
case 11:audit_window_points<11><<<n/128,128>>>(d,n,table,errors);break;
case 12:audit_window_points<12><<<n/128,128>>>(d,n,table,errors);break;
case 13:audit_window_points<13><<<n/128,128>>>(d,n,table,errors);break;
case 14:audit_window_points<14><<<n/128,128>>>(d,n,table,errors);break;
case 15:audit_window_points<15><<<n/128,128>>>(d,n,table,errors);break;
case 16:audit_window_points<16><<<n/128,128>>>(d,n,table,errors);break;
}
ce=cudaDeviceSynchronize();cudaMemcpy(&e,errors,4,cudaMemcpyDeviceToHost);
    printf("Window table entries=%u bytes=%llu OpenSSL_spots=1024\n",entries,((unsigned long long)entries*64));
    printf("Window point OpenSSL cases=%d targeted=%d mismatches=%u %s\n",n,at-1024,e,cudaGetErrorString(ce));return e||ce!=cudaSuccess;
}
