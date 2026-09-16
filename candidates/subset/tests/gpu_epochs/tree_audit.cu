// Audit canonical multiplication and the production block inverse on the GPU.
#define main qsb_grinder_main
#include "tree.cu"
#undef main
#include <vector>

#ifdef QSB_AUDIT_SQUARE64
#define QSB_AUDIT_SQUARE
#define QSB_AUDIT_SQUARE_FN qsb_square64
#else
#define QSB_AUDIT_SQUARE_FN qsb_square32
#endif

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;} } while(0)

__global__ void audit_products(const uint64_t *inputs,uint64_t *results,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    uint64_t a[5],b[5],r[5];
    for(int k=0;k<4;k++){a[k]=inputs[i*8+k];b[k]=inputs[i*8+4+k];}
#ifdef QSB_AUDIT_SQUARE
    a[4]=b[4]=r[4]=0;
    QSB_AUDIT_SQUARE_FN(r,a);
    for(int k=0;k<5;k++)results[i*15+k]=r[k];
    QSB_AUDIT_SQUARE_FN(a,a);
    for(int k=0;k<5;k++)results[i*15+5+k]=a[k];
    for(int k=0;k<4;k++)b[k]=inputs[i*8+k];
    _ModSqr(b,b);
    for(int k=0;k<5;k++)results[i*15+10+k]=b[k];
#else
    a[4]=b[4]=r[4]=123;
    qsb_field_mul(r,a,b);
    for(int k=0;k<5;k++)results[i*15+k]=r[k];
    qsb_field_mul(a,a,b);
    for(int k=0;k<5;k++)results[i*15+5+k]=a[k];
    for(int k=0;k<4;k++)a[k]=inputs[i*8+k];
    qsb_field_mul(b,a,b);
    for(int k=0;k<5;k++)results[i*15+10+k]=b[k];
#endif
}

__global__ void __launch_bounds__(256,2) audit_inverses(const uint64_t *inputs,uint64_t *results,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    uint64_t value[5]={1,0,0,0,0};
    if(i<n)for(int k=0;k<4;k++)value[k]=inputs[i*8+k];
    qsb_block_inverse_tree(value);
    if(i<n)for(int k=0;k<5;k++)results[i*5+k]=value[k];
}

// The hot curve primitives are distinct from the canonical inverse-tree multiply.
__global__ void audit_hot_products(const uint64_t *inputs,uint64_t *results,int n,int square){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint64_t a[4],b[4],r[4];
    for(int k=0;k<4;k++){a[k]=inputs[i*8+k];b[k]=inputs[i*8+4+k];}
    if(square)_ModSqr(r,a);else _ModMultCore(r,a,b);
    for(int k=0;k<4;k++)results[i*12+k]=r[k];
    if(square)_ModSqr(a,a);else _ModMultCore(a,a,b);
    for(int k=0;k<4;k++)results[i*12+4+k]=a[k];
    for(int k=0;k<4;k++)a[k]=inputs[i*8+k];
    if(square)_ModSqr(b,a);else _ModMultCore(b,a,b);
    for(int k=0;k<4;k++)results[i*12+8+k]=b[k];
}

int main(){
    const int n=32768;
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=nullptr,*a=BN_new(),*b=BN_new(),*r=BN_new();
    BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    std::vector<uint64_t> inputs(n*8),results(n*15),expected(n*5);
    // Cartesian edge cases include p, p+1, 2^256-1 and near-boundary products.
    std::vector<BIGNUM*> edges;
    for(int j=0;j<12;j++){
        BIGNUM *v=BN_new();
        if(j<3)BN_set_word(v,j);
        else if(j<6){BN_copy(v,p);if(j==3)BN_sub_word(v,1);if(j==5)BN_add_word(v,1);}
        else if(j<8){BN_one(v);BN_lshift(v,v,j==6?255:256);if(j==7)BN_sub_word(v,1);}
        else if(j==8){BN_copy(v,p);BN_sub_word(v,65537);}
        else if(j==9){BN_copy(v,p);BIGNUM *delta=BN_new();BN_one(delta);BN_lshift(delta,delta,224);BN_sub(v,v,delta);BN_free(delta);}
        else {BN_copy(v,p);BN_sub_word(v,j==10?65535:65536);}
        edges.push_back(v);
    }
    for(int i=0;i<n;i++){
        unsigned char digest[32];
        for(int side=0;side<2;side++){
            int seed=2*i+side;SHA256((unsigned char*)&seed,sizeof(seed),digest);
            BIGNUM *v=side?b:a;BN_bin2bn(digest,32,v);
            if(i<144)BN_copy(v,edges[side?i%12:i/12]);
            BN_bn2lebinpad(v,(unsigned char*)(inputs.data()+8*i+4*side),32);
        }
#ifdef QSB_AUDIT_SQUARE
        BN_mod_sqr(r,a,p,ctx);
#else
        BN_mod_mul(r,a,b,p,ctx);
#endif
        BN_bn2lebinpad(r,(unsigned char*)(expected.data()+5*i),32);
    }
    uint64_t *di,*dr;CHECK(cudaMalloc(&di,inputs.size()*8));CHECK(cudaMalloc(&dr,results.size()*8));
    CHECK(cudaMemcpy(di,inputs.data(),inputs.size()*8,cudaMemcpyHostToDevice));
    audit_products<<<(n+255)/256,256>>>(di,dr,n);CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(results.data(),dr,results.size()*8,cudaMemcpyDeviceToHost));
    int failures=0;
    for(int i=0;i<n;i++)for(int alias=0;alias<3;alias++){
        if(memcmp(results.data()+15*i+alias*5,expected.data()+5*i,40)){
            if(failures<8)fprintf(stderr,"Product mismatch sample=%d alias=%d\n",i,alias);
            failures++;
        }
    }
#ifdef QSB_AUDIT_SQUARE
    printf("Canonical squaring: %d/%d exact matches (three call modes)\n",3*n-failures,3*n);
#else
    printf("Canonical multiplication: %d/%d exact matches (three alias modes)\n",3*n-failures,3*n);
#endif
    if(failures)return 1;
    for(int square=0;square<2;square++){
        for(int i=0;i<n;i++){
            BN_lebin2bn((unsigned char*)(inputs.data()+8*i),32,a);
            BN_lebin2bn((unsigned char*)(inputs.data()+8*i+4),32,b);
            if(square)BN_mod_sqr(r,a,p,ctx);else BN_mod_mul(r,a,b,p,ctx);
            BN_bn2lebinpad(r,(unsigned char*)(expected.data()+5*i),32);
        }
        audit_hot_products<<<(n+255)/256,256>>>(di,dr,n,square);CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(results.data(),dr,n*12*8,cudaMemcpyDeviceToHost));
        int bad=0;
        for(int i=0;i<n;i++)for(int alias=0;alias<3;alias++)
            if(memcmp(results.data()+12*i+alias*4,expected.data()+5*i,32))bad++;
        printf("Hot %s: %d/%d exact matches (three alias modes)\n",square?"square":"multiply",3*n-bad,3*n);
        failures+=bad;
    }
    // Nonzero canonical inputs, including identity factors, for every inverse.
    const int ni=8191;
    for(int i=0;i<ni;i++){
        BN_lebin2bn((unsigned char*)(inputs.data()+8*i),32,a);BN_mod(a,a,p,ctx);
        if(BN_is_zero(a))BN_one(a);
        BN_bn2lebinpad(a,(unsigned char*)(inputs.data()+8*i),32);
        BN_mod_inverse(r,a,p,ctx);BN_bn2lebinpad(r,(unsigned char*)(expected.data()+5*i),32);
    }
    CHECK(cudaMemcpy(di,inputs.data(),inputs.size()*8,cudaMemcpyHostToDevice));
    for(int threads=32;threads<=256;threads*=2){
        audit_inverses<<<(ni+threads-1)/threads,threads>>>(di,dr,ni);CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(results.data(),dr,ni*5*8,cudaMemcpyDeviceToHost));
        int bad=0;
        for(int i=0;i<ni;i++)if(memcmp(results.data()+5*i,expected.data()+5*i,40))bad++;
        printf("Block inverse (%d threads, partial tail): %d/%d exact matches\n",threads,ni-bad,ni);
        failures+=bad;
    }
    // Exercise the actual external checkpoint helpers across an incomplete root group.
    const int groups=(ni+255)/256;
    std::vector<uint64_t> root_inputs(ni*4),root_results(ni*4);
    for(int i=0;i<ni;i++)memcpy(root_inputs.data()+4*i,inputs.data()+8*i,32);
    uint64_t *roots,*super_roots,*checkpoint;
    CHECK(cudaMalloc(&roots,ni*4*8));CHECK(cudaMalloc(&super_roots,groups*4*8));
    CHECK(cudaMalloc(&checkpoint,groups*4*QSB_CHECKPOINT_STRIDE*sizeof(uint64_t)));
    CHECK(cudaMemcpy(roots,root_inputs.data(),ni*4*8,cudaMemcpyHostToDevice));
    qsb_root_group_prepare<<<groups,256>>>(roots,ni,super_roots,checkpoint);
    qsb_invert_super_roots<<<1,256>>>(super_roots,groups);
    qsb_root_group_finish<<<groups,256>>>(roots,ni,super_roots,checkpoint);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(root_results.data(),roots,ni*4*8,cudaMemcpyDeviceToHost));
    int root_bad=0;
    for(int i=0;i<ni;i++)if(memcmp(root_results.data()+4*i,expected.data()+5*i,32))root_bad++;
    printf("External root hierarchy (partial group): %d/%d exact matches\n",ni-root_bad,ni);
    failures+=root_bad;
    cudaFree(roots);cudaFree(super_roots);cudaFree(checkpoint);
    cudaFree(di);cudaFree(dr);
    for(auto v:edges)BN_free(v);
    BN_free(a);BN_free(b);BN_free(r);BN_free(p);BN_CTX_free(ctx);
    return failures?1:0;
}
