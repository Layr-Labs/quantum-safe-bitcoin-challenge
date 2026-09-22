// Audit the actual point chain against OpenSSL EC_POINT_mul.
// Includes the parent final-doubling witness and all old scalar edge cases.
// Not a performance workload; unused record fields preserve fixture layout.
#define main qsb_candidate_main
#ifndef QSB_POINT_TREE
#define QSB_POINT_TREE "tree.cu"
#endif
#include QSB_POINT_TREE
#undef main
#include <vector>

struct point_case {uint64_t k[4],u[3],v[3];uint32_t signs[2],digits[14];};
static_assert(sizeof(point_case)==144,"fixture layout");
__global__ void audit_complete_points(const point_case *cases,const unsigned *select,unsigned count,
                                const uint8_t *table,uint64_t *out){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    uint64_t X[4],Y[4],ZZ[4],ZZZ[4];
    _FixedBaseSignedXYZZStream(X,Y,ZZ,ZZZ,cases[select[i]].k,table);
    for(int j=0;j<4;j++){
        out[i*16+j]=X[j];out[i*16+4+j]=Y[j];out[i*16+8+j]=ZZ[j];out[i*16+12+j]=ZZZ[j];
    }
}
static bool gpu_ok(cudaError_t e){if(e==cudaSuccess)return true;fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return false;}
int main(int argc,char**argv){
    if(argc!=2)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;
    unsigned count;if(fread(&count,4,1,f)!=1)return 2;
    std::vector<point_case> cases(count);
    if(fread(cases.data(),sizeof(point_case),count,f)!=count)return 2;fclose(f);
    point_case *dc;unsigned *ds;unsigned errors=0;
    if(!gpu_ok(cudaMalloc(&dc,cases.size()*sizeof(point_case))))return 2;
    if(!gpu_ok(cudaMemcpy(dc,cases.data(),cases.size()*sizeof(point_case),cudaMemcpyHostToDevice)))return 2;
    std::vector<unsigned> select;
    // All deterministic edges and crafted doubling cases, plus fresh randoms.
    for(unsigned i=0;i<count;i++)if(i<(count>100000?count-100000:count) || i+2048>=count)select.push_back(i);
    cudaMalloc(&ds,select.size()*4);cudaMemcpy(ds,select.data(),select.size()*4,cudaMemcpyHostToDevice);
    uint64_t *dout;cudaMalloc(&dout,select.size()*128);
    std::vector<uint64_t> points(select.size()*16);
    BN_CTX*ctx=BN_CTX_new();EC_GROUP*grp=EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT*want=EC_POINT_new(grp);
    BIGNUM*n=BN_new(),*p=BN_new(),*base=BN_new(),*scalar=BN_new(),*key=BN_new();
    BIGNUM*x=BN_new(),*y=BN_new(),*xx=BN_new(),*yy=BN_new(),*zz=BN_new(),*zzz=BN_new();
    BIGNUM*tmp=BN_new(),*rhs=BN_new();EC_GROUP_get_order(grp,n,ctx);EC_GROUP_get_curve_GFp(grp,p,NULL,NULL,ctx);
    unsigned infinity=0,table_checks=0;
    for(unsigned run=0;run<2;run++){
        uint8_t seed[32],nri[32];
        const uint64_t identity_scale[4]={1,0,0,0};
        for(int i=0;i<32;i++)seed[i]=(uint8_t)(i*43+run*79+11);
        SHA256(seed,32,nri);BN_bin2bn(nri,32,base);BN_mod(base,base,n,ctx);
        // The production table builder accepts little-endian neg_r_inv.
        BN_bn2lebinpad(base,nri,32);
        std::vector<uint64_t>L(GT_CHUNKS*GT_LO*8),H(GT_CHUNKS*GT_HI*8);
        gt_build_ladders(L.data(),H.data(),nri,identity_scale,identity_scale);
        uint64_t *dL,*dH;uint8_t*table;size_t bytes=(size_t)GT_TOTAL_ENTRIES*64;
        if(!gpu_ok(cudaMalloc(&dL,L.size()*8))||!gpu_ok(cudaMalloc(&dH,H.size()*8))||!gpu_ok(cudaMalloc(&table,bytes)))return 2;
        cudaMemcpy(dL,L.data(),L.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(dH,H.data(),H.size()*8,cudaMemcpyHostToDevice);
        kernel_build_gtable<<<(GT_TOTAL_ENTRIES+255)/256,256>>>(dL,dH,table);
        std::vector<uint8_t>table_host(bytes);
        if(!gpu_ok(cudaMemcpy(table_host.data(),table,bytes,cudaMemcpyDeviceToHost)))return 2;
        if(!gt_spot_check(table_host.data(),GT_CHUNKS*4+192,nri,
                          identity_scale,identity_scale,NULL))return 1;
        table_checks+=GT_CHUNKS*4+192;
        audit_complete_points<<<(select.size()+127)/128,128>>>(dc,ds,select.size(),table,dout);
        if(!gpu_ok(cudaMemcpy(points.data(),dout,points.size()*8,cudaMemcpyDeviceToHost)))return 2;
        for(unsigned i=0;i<select.size();i++){
            const point_case&c=cases[select[i]];const uint64_t*got=points.data()+16*i;
            BN_lebin2bn((const unsigned char*)c.k,32,scalar);BN_mod_mul(key,scalar,base,n,ctx);
            EC_POINT_mul(grp,want,key,NULL,NULL,ctx);
            BN_lebin2bn((const unsigned char*)got,32,xx);BN_lebin2bn((const unsigned char*)(got+4),32,yy);
            BN_lebin2bn((const unsigned char*)(got+8),32,zz);BN_lebin2bn((const unsigned char*)(got+12),32,zzz);
            bool bad=BN_cmp(xx,p)>=0||BN_cmp(yy,p)>=0||BN_cmp(zz,p)>=0||BN_cmp(zzz,p)>=0;
            if(EC_POINT_is_at_infinity(grp,want)){
                infinity++;bad|=!BN_is_zero(zz)||!BN_is_zero(zzz);
            }else{
                bad|=BN_is_zero(zz)||BN_is_zero(zzz);
                EC_POINT_get_affine_coordinates_GFp(grp,want,x,y,ctx);
                BN_mod_mul(tmp,x,zz,p,ctx);bad|=BN_cmp(tmp,xx)!=0;
                BN_mod_mul(tmp,y,zzz,p,ctx);bad|=BN_cmp(tmp,yy)!=0;
                BN_mod_sqr(tmp,zz,p,ctx);BN_mod_mul(tmp,tmp,zz,p,ctx);
                BN_mod_sqr(rhs,zzz,p,ctx);bad|=BN_cmp(tmp,rhs)!=0;
            }
            if(bad){if(errors<8)printf("Point mismatch base=%u fixture=%u\n",run,select[i]);errors++;}
        }
        cudaFree(dL);cudaFree(dH);cudaFree(table);
        printf("Point audit base %u: %zu points, cumulative errors=%u\n",run,select.size(),errors);fflush(stdout);
        if(errors)return 1;
    }
    printf("Point GPU audit PASS: chunks=%d input_cases=%u point_cases=%zu infinity_cases=%u table_checks=%u errors=%u\n",
        GT_CHUNKS,count,select.size()*2,infinity,table_checks,errors);
    return errors?1:0;
}
