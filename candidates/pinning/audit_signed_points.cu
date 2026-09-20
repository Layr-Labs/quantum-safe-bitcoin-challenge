#include <vector>
#include <string>
#define main qsb_unused_main
#include "pinning.cu"
#undef main
__device__ bool glv_zero(const uint64_t *p){return !(p[0]|p[1]|p[2]|p[3]);}

__device__ void audit_affine(uint64_t *o,uint64_t *X,uint64_t *Y,uint64_t *U,uint64_t *V) {
    if(glv_zero(U)){for(int i=0;i<9;i++)o[i]=0;return;}
    uint64_t iu[5]={U[0],U[1],U[2],U[3],0},iv[5]={V[0],V[1],V[2],V[3],0};
    qsb_field_normalize(iu);qsb_field_normalize(iv);_ModInv(iu);_ModInv(iv);
    qsb_recovery_mul(o,X,iu);qsb_recovery_mul(o+4,Y,iv);o[8]=1;
}
__global__ void audit_glv_points(const uint64_t *in,uint64_t *out,unsigned n,const uint8_t *table) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    uint64_t X[4],Y[4],U[4],V[4];
#ifdef QSB_TEST_JAC
    qsb_fixed_jacobian(X,Y,U,V,in+4ull*i,table);
#else
    _FixedBaseSignedXYZZScalar<QSB_AUDIT_SAFE_NEG>(X,Y,U,V,in+4ull*i,table,qsb_prepare_scratch());
#endif

    audit_affine(out+9ull*i,X,Y,U,V);
}
int main(int argc,char **argv) {
    if(argc<3||argc>4)return 2;FILE *f=fopen(argv[1],"rb");if(!f)return 2;unsigned n=0;
    if(fread(&n,4,1,f)!=1||!n||n>1000000)return 2;
    std::vector<uint64_t> in(4ull*n),out(9ull*n),exception(36);
    if(fread(in.data(),32,n,f)!=n)return 2;fclose(f);
    uint8_t a_le[32]={};a_le[0]=1;uint8_t *table=nullptr;
    if(argc==4)for(int j=0;j<32;j++)a_le[j]=(uint8_t)(13*j+71);
    std::vector<uint64_t> hL((size_t)GT_CHUNKS*GT_LO*8),hH((size_t)GT_CHUNKS*GT_HI*8);
    gt_build_ladders(hL.data(),hH.data(),a_le);uint64_t *dL,*dH;
    if(cudaMalloc(&table,64ull*GT_TOTAL_ENTRIES)!=cudaSuccess || cudaMalloc(&dL,hL.size()*8)!=cudaSuccess || cudaMalloc(&dH,hH.size()*8)!=cudaSuccess)return 3;
    if(cudaMemcpy(dL,hL.data(),hL.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess || cudaMemcpy(dH,hH.data(),hH.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess)return 3;
    kernel_build_gtable<<<(GT_TOTAL_ENTRIES+255)/256,256>>>(dL,dH,table);
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess)return 3;
    cudaFree(dL);cudaFree(dH);
    std::vector<uint8_t> mirror(64ull*GT_TOTAL_ENTRIES);
    if(cudaMemcpy(mirror.data(),table,mirror.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
    if(!qsb_table_y_range_ok(mirror.data(),GT_TOTAL_ENTRIES))return 5;
    fprintf(stderr,"Native point audit uses SAFE_NEG=%d; all table entries passed guard.\n",QSB_AUDIT_SAFE_NEG);

    uint64_t *di=nullptr,*doo=nullptr,*de=nullptr,*dp=nullptr;
    if(cudaMalloc(&di,in.size()*8)!=cudaSuccess||cudaMalloc(&doo,out.size()*8)!=cudaSuccess||
        cudaMalloc(&de,exception.size()*8)!=cudaSuccess||cudaMalloc(&dp,64)!=cudaSuccess)return 3;
    if(cudaMemcpy(di,in.data(),in.size()*8,cudaMemcpyHostToDevice)!=cudaSuccess)return 3;
    audit_glv_points<<<(n+127)/128,128>>>(di,doo,n,table);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)return 3;
    if(cudaMemcpy(out.data(),doo,out.size()*8,cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
    // Preserve original native coordinates before the CPU comparison can fail.
    std::string dump=std::string(argv[2])+".points.bin";
    FILE *raw=fopen(dump.c_str(),"wb");if(!raw)return 3;
    bool saved=fwrite(out.data(),72,n,raw)==n;
    saved=(fclose(raw)==0)&&saved;if(!saved)return 3;
    EC_GROUP *group=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    BIGNUM *k=BN_new(),*x=BN_new(),*y=BN_new(),*a=BN_lebin2bn(a_le,32,nullptr),*order=BN_new();EC_POINT *p=EC_POINT_new(group);
    EC_GROUP_get_order(group,order,ctx);
    uint64_t want[8],g[8];
    EC_POINT_copy(p,EC_GROUP_get0_generator(group));gt_point_to_limbs(group,p,x,y,ctx,g);
    if(cudaMemcpy(dp,g,64,cudaMemcpyHostToDevice)!=cudaSuccess)return 3;
    for(unsigned i=0;i<n;i++) {
        const uint64_t *got=i<n?out.data()+9ull*i:exception.data()+9ull*(i-n);
        if(i<n){BN_lebin2bn((const unsigned char*)(in.data()+4ull*i),32,k);BN_mod_mul(k,k,a,order,ctx);}
        else BN_set_word(k,i==n?2:i==n+1?0:1);
        if(!EC_POINT_mul(group,p,k,nullptr,nullptr,ctx))return 3;
        bool infinity=EC_POINT_is_at_infinity(group,p);
        if(infinity) {if(got[8]){fprintf(stderr,"expected infinity at %u\n",i);return 4;}}
        else {
            gt_point_to_limbs(group,p,x,y,ctx,want);
            for(int j=0;j<8;j++)if(!got[8]||got[j]!=want[j]) {
                fprintf(stderr,"point mismatch at %u limb %d: got %016llx expected %016llx\n",i,j,
                    (unsigned long long)got[j],(unsigned long long)want[j]);return 4;
            }
        }
    }
    f=fopen(argv[2],"w");if(!f)return 2;
    fprintf(f,"{\"passed\":true,\"terms\":%d,\"native_points\":%u,\"dense_base\":%s,\"exception_cases\":0}\n",GT_CHUNKS,n,argc==4?"true":"false");
    fclose(f);cudaFree(di);cudaFree(doo);cudaFree(de);cudaFree(dp);cudaFree(table);
    EC_POINT_free(p);EC_GROUP_free(group);BN_CTX_free(ctx);BN_free(k);BN_free(x);BN_free(y);BN_free(a);BN_free(order);return 0;
}
