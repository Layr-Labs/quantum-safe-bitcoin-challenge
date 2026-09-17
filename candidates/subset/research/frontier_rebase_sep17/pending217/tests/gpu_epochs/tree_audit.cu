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

__global__ void audit_warp_roots(const uint64_t *inputs,uint64_t *results,int n){
    int linear=blockIdx.x*blockDim.x+threadIdx.x;
    int i=linear/32,lane=threadIdx.x&31;
    if(i>=n)return;
    uint64_t value[5]={0,0,0,0,0};
    if(i)for(int k=0;k<4;k++)value[k]=inputs[8*i+k];
    hm43_warp_inverse(value,lane);
    for(int k=0;k<5;k++)results[linear*5+k]=value[k];
}

__global__ void audit_bounded_status(const uint64_t *inputs,uint64_t *results,int n){
    int linear=blockIdx.x*blockDim.x+threadIdx.x;
    int i=linear/32,lane=threadIdx.x&31;
    if(i>=n)return;
    uint64_t value[5]={0,0,0,0,0};
    if(i)for(int k=0;k<4;k++)value[k]=inputs[8*i+k];
    bool done=hm43_warp_inverse_bounded(value,lane);
    results[linear*6]=done;
    for(int k=0;k<5;k++)results[linear*6+1+k]=value[k];
}

int main(){
    const int n=32768;
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=nullptr,*a=BN_new(),*b=BN_new(),*r=BN_new();
    BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    std::vector<uint64_t> inputs(n*8),results(n*15),expected(n*5);
    // Cartesian edge cases include p, p+1, 2^256-1 and near-boundary products.
    std::vector<BIGNUM*> edges;
    for(int j=0;j<8;j++){
        BIGNUM *v=BN_new();
        if(j<3)BN_set_word(v,j);
        else if(j<6){BN_copy(v,p);if(j==3)BN_sub_word(v,1);if(j==5)BN_add_word(v,1);}
        else {BN_one(v);BN_lshift(v,v,j==6?255:256);if(j==7)BN_sub_word(v,1);}
        edges.push_back(v);
    }
    for(int i=0;i<n;i++){
        unsigned char digest[32];
        for(int side=0;side<2;side++){
            int seed=2*i+side;SHA256((unsigned char*)&seed,sizeof(seed),digest);
            BIGNUM *v=side?b:a;BN_bin2bn(digest,32,v);
            if(i<64)BN_copy(v,edges[side?i%8:i/8]);
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
    // Nonzero canonical inputs, including identity factors, for every inverse.
    const int ni=8191;
    for(int i=0;i<ni;i++){
        BN_lebin2bn((unsigned char*)(inputs.data()+8*i),32,a);BN_mod(a,a,p,ctx);
        if(BN_is_zero(a))BN_one(a);
        BN_bn2lebinpad(a,(unsigned char*)(inputs.data()+8*i),32);
        BN_mod_inverse(r,a,p,ctx);BN_bn2lebinpad(r,(unsigned char*)(expected.data()+5*i),32);
    }
    CHECK(cudaMemcpy(di,inputs.data(),inputs.size()*8,cudaMemcpyHostToDevice));
    for(int threads=32;threads<=256;threads*=2)for(int count:{ni,1,129,255,256,257}){
        audit_inverses<<<(count+threads-1)/threads,threads>>>(di,dr,count);CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(results.data(),dr,count*5*8,cudaMemcpyDeviceToHost));
        int bad=0;
        int noncanon=0;
        for(int i=0;i<count;i++)if(memcmp(results.data()+5*i,expected.data()+5*i,40)){
            /* ZLAB_TREE>=1 returns exact residues below 2^256: accept r == expected + p. */
            BN_lebin2bn((unsigned char*)(results.data()+5*i),32,a);
            BN_lebin2bn((unsigned char*)(expected.data()+5*i),32,b);
            BN_add(r,b,p);
            if(results[5*i+4]==0 && BN_cmp(a,r)==0) noncanon++; else bad++;
        }
        printf("Block inverse (%d threads, count=%d): %d/%d exact, %d congruent non-canonical, %d wrong\n",threads,count,count-bad-noncanon,count,noncanon,bad);
        failures+=bad;
    }
    const int roots=2048;
    audit_warp_roots<<<roots/8,256>>>(di,dr,roots);CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(results.data(),dr,roots*32*5*8,cudaMemcpyDeviceToHost));
    int root_bad=0;const uint64_t zero[5]={0,0,0,0,0};
    for(int i=0;i<roots;i++)for(int lane=0;lane<32;lane++){
        const uint64_t *want=i?expected.data()+i*5:zero;
        if(memcmp(results.data()+(i*32+lane)*5,want,40))root_bad++;
    }
    printf("Distributed root (zero plus nonzero inputs, all lane outputs): %d/%d exact, %d wrong\n",roots*32-root_bad,roots*32,root_bad);
    failures+=root_bad;
    audit_bounded_status<<<roots/8,256>>>(di,dr,roots);CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(results.data(),dr,roots*32*6*8,cudaMemcpyDeviceToHost));
    int status_bad=0,declined=0;
    for(int i=0;i<roots;i++)for(int lane=0;lane<32;lane++){
        const uint64_t *out=results.data()+(i*32+lane)*6;
        if(out[0]!=results[i*32*6])status_bad++;
        if(out[0]){
            const uint64_t *want=i?expected.data()+i*5:zero;
            if(memcmp(out+1,want,40))status_bad++;
        }else{
            declined++;
            const uint64_t *want=i?inputs.data()+i*8:zero;
            if(memcmp(out+1,want,32) || out[5])status_bad++;
        }
    }
    if(QSB_ROOT_MAX_BATCHES==0 && declined!=roots*32)status_bad++;
    if(QSB_ROOT_MAX_BATCHES==1 && declined==0)status_bad++;
    printf("Bounded helper cap=%d: %d lane declines, %d status/output errors; wrapper fallback values checked above\n",QSB_ROOT_MAX_BATCHES,declined,status_bad);
    failures+=status_bad;
    cudaFree(di);cudaFree(dr);
    for(auto v:edges)BN_free(v);
    BN_free(a);BN_free(b);BN_free(r);BN_free(p);BN_CTX_free(ctx);
    return failures?1:0;
}
