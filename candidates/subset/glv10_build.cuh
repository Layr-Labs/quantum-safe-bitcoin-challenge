#pragma once
#include <vector>
#include "glv10_audit_scalars.h"
static constexpr uint64_t GLV10_RECORDS=138760484ULL;
static constexpr size_t GLV10_BYTES=(size_t)GLV10_RECORDS*64;
static constexpr unsigned GLV10_LADDER=8192;

static double glv10_seconds() {
    timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;
}
static void glv10_memory(const char *stage) {
    size_t free_bytes,total;
    qsb_native::check(cudaMemGetInfo(&free_bytes,&total),"memory telemetry");
    fprintf(stderr,"QSB stage=%s monotonic=%.6f memory_free=%zu memory_total=%zu\n",
            stage,glv10_seconds(),free_bytes,total);
}
static void glv10_require(bool ok,const char *what) {
    if(!ok){fprintf(stderr,"QSB GLV10 failure: %s\n",what);exit(2);}
}

__global__ void kernel_build_glv10(const uint64_t *L,const uint64_t *H,uint8_t *table,
                                  uint64_t start,uint64_t count,const uint32_t *indices) {
    const uint64_t out=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(out>=count)return;
    const uint64_t t=indices?indices[out]:start+out;
    const int c=t<glv10_offset(1)?0:t<glv10_offset(2)?1:
                t<glv10_offset(3)?2:t<glv10_offset(4)?3:4;
    const unsigned index=(unsigned)(t-glv10_offset(c));
    const unsigned m=c==0?index:2*index+1;
    const unsigned hi=m>>13,lo=m&8191;
    const uint64_t *lp=L+((size_t)c*GLV10_LADDER+lo)*8;
    const uint64_t *hp=H+((size_t)c*GLV10_LADDER+hi)*8;
    uint64_t x[4],y[4];
    if(!hi) {Load256(x,lp);Load256(y,lp+4);}
    else {
        uint64_t z[5]={1,0,0,0,0},qx[4],qy[4];
        Load256(x,hp);Load256(y,hp+4);Load256(qx,lp);Load256(qy,lp+4);
        // H and L are never equal/opposite; biased segment zero includes lo=0.
        _PointAddSecp256k1(x,y,z,qx,qy);
        _ModInv(z);_ModMult(x,z);_ModMult(y,z);
    }
    qsb_field_normalize(x);qsb_field_normalize(y);
    memcpy(table+(size_t)out*64,x,32);memcpy(table+(size_t)out*64+32,y,32);
}
// Exact, canonical setup-only tree. Identity padding participates in every barrier.
__device__ __forceinline__ void glv10_batch_inverse(uint64_t value[5]) {
    __shared__ uint64_t tree[4][512];
    const unsigned tid=threadIdx.x;
    #pragma unroll
    for(int k=0;k<4;k++)tree[k][256+tid]=value[k];
    __syncthreads();
    for(unsigned width=128;width;width>>=1) {
        if(tid<width) {
            const unsigned node=width+tid;
            uint64_t a[5]={},b[5]={};
            #pragma unroll
            for(int k=0;k<4;k++){a[k]=tree[k][node*2];b[k]=tree[k][node*2+1];}
            qsb_field_mul(a,a,b);
            #pragma unroll
            for(int k=0;k<4;k++)tree[k][node]=a[k];
        }
        __syncthreads();
    }
    if(tid==0) {
        uint64_t root[5]={};
        #pragma unroll
        for(int k=0;k<4;k++)root[k]=tree[k][1];
        _ModInv(root);
        #pragma unroll
        for(int k=0;k<4;k++)tree[k][1]=root[k];
    }
    __syncthreads();
    for(unsigned width=1;width<256;width<<=1) {
        if(tid<width) {
            const unsigned node=width+tid;
            uint64_t parent[5]={},left[5]={},right[5]={};
            #pragma unroll
            for(int k=0;k<4;k++){parent[k]=tree[k][node];left[k]=tree[k][node*2];right[k]=tree[k][node*2+1];}
            qsb_field_mul(right,parent,right);qsb_field_mul(left,parent,left);
            #pragma unroll
            for(int k=0;k<4;k++){tree[k][node*2]=right[k];tree[k][node*2+1]=left[k];}
        }
        __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<4;k++)value[k]=tree[k][256+tid];
    value[4]=0;
}
__global__ void kernel_build_glv10_batch(const uint64_t *L,const uint64_t *H,uint8_t *table,
                                        uint64_t start,uint64_t count,const uint32_t *indices,
                                        unsigned *failed) {
    const uint64_t out=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    const bool valid=out<count;
    unsigned hi=0;const uint64_t *lp=nullptr,*hp=nullptr;
    uint64_t inv[5]={1,0,0,0,0};
    if(valid) {
        const uint64_t t=indices?indices[out]:start+out;
        const int c=t<glv10_offset(1)?0:t<glv10_offset(2)?1:t<glv10_offset(3)?2:t<glv10_offset(4)?3:4;
        const unsigned index=(unsigned)(t-glv10_offset(c)),m=c==0?index:2*index+1;
        hi=m>>13;
        lp=L+((size_t)c*GLV10_LADDER+(m&8191))*8;
        hp=H+((size_t)c*GLV10_LADDER+hi)*8;
        if(hi) {
            uint64_t lx[4],hx[4];Load256(lx,lp);Load256(hx,hp);
            _ModSub256(inv,lx,hx);qsb_field_normalize(inv);
            if(!(inv[0]|inv[1]|inv[2]|inv[3])) {
                atomicExch(failed,1u);inv[0]=1; // Fail closed on host; protect neighboring tree leaves.
            }
        }
    }
    glv10_batch_inverse(inv);
    if(valid) {
        uint64_t x[4],y[4];
        if(!hi){Load256(x,lp);Load256(y,lp+4);}
        else {
            uint64_t lx[4],ly[4],hx[4],hy[4],slope[4];
            Load256(lx,lp);Load256(ly,lp+4);Load256(hx,hp);Load256(hy,hp+4);
            _ModSub256(slope,ly,hy);_ModMult(slope,inv);_ModSqr(x,slope);
            _ModSub256(x,x,hx);_ModSub256(x,x,lx);
            _ModSub256(y,hx,x);_ModMult(y,slope);_ModSub256(y,y,hy);
        }
        qsb_field_normalize(x);qsb_field_normalize(y);
        memcpy(table+(size_t)out*64,x,32);memcpy(table+(size_t)out*64+32,y,32);
    }
}
__global__ void kernel_gather_glv10(const uint8_t *table,const uint32_t *indices,
                                   uint64_t *out,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)memcpy(out+(size_t)i*8,table+(size_t)indices[i]*64,64);
}

__global__ void kernel_audit_glv10(const uint8_t *table,const uint64_t *scalars,
                                  uint64_t *points,unsigned *status,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    uint64_t x[4],y[4],zz[5]={},zzz[5]={};uint32_t bad=0;
    qsb_filter_chain_trial(x,y,zz,zzz,scalars+(size_t)i*4,table,bad);
    qsb_field_normalize(zz);qsb_field_normalize(zzz);
    const bool zero_zz=!(zz[0]|zz[1]|zz[2]|zz[3]);
    const bool zero_zzz=!(zzz[0]|zzz[1]|zzz[2]|zzz[3]);
    if(zero_zz||zero_zzz) {
        status[i]=zero_zz&&zero_zzz?1:2;
        #pragma unroll
        for(int j=0;j<8;j++)points[(size_t)i*8+j]=0;
        return;
    }
    _ModInv(zz);_ModInv(zzz);_ModMult(x,zz);_ModMult(y,zzz);
    qsb_field_normalize(x);qsb_field_normalize(y);status[i]=0;
    memcpy(points+(size_t)i*8,x,32);memcpy(points+(size_t)i*8+4,y,32);
}
static void glv10_chain_audit(const uint8_t *table,const uint8_t nri_bytes[32]) {
    const unsigned count=sizeof(glv10_audit_scalars)/sizeof(glv10_audit_scalars[0]);
    uint64_t *scalars=nullptr,*points=nullptr;unsigned *status=nullptr;
    qsb_native::check(cudaMalloc(&scalars,sizeof(glv10_audit_scalars)),"GLV audit scalars");
    qsb_native::check(cudaMalloc(&points,(size_t)count*64),"GLV audit points");
    qsb_native::check(cudaMalloc(&status,(size_t)count*4),"GLV audit status");
    qsb_native::check(cudaMemcpy(scalars,glv10_audit_scalars,sizeof(glv10_audit_scalars),cudaMemcpyHostToDevice),"GLV audit upload");
    QSB_LAUNCH(kernel_audit_glv10,(count+63)/64,64,table,scalars,points,status,count);
    std::vector<uint64_t> actual((size_t)count*8);std::vector<unsigned> flags(count);
    qsb_native::check(cudaMemcpy(actual.data(),points,(size_t)count*64,cudaMemcpyDeviceToHost),"GLV audit points read");
    qsb_native::check(cudaMemcpy(flags.data(),status,(size_t)count*4,cudaMemcpyDeviceToHost),"GLV audit status read");
    qsb_native::check(cudaFree(scalars),"free GLV audit scalars");qsb_native::check(cudaFree(points),"free GLV audit points");
    qsb_native::check(cudaFree(status),"free GLV audit status");
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    glv10_require(g&&ctx,"chain audit context");
    BIGNUM *nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*x=BN_new(),*y=BN_new();
    EC_POINT *point=EC_POINT_new(g);glv10_require(nri&&k&&x&&y&&point,"chain audit allocation");
    unsigned mismatches=0,invalid=0,infinities=0;
    for(unsigned i=0;i<count;i++) {
        glv10_require(BN_lebin2bn((const unsigned char*)glv10_audit_scalars[i],32,k)&&
                      BN_mul(k,k,nri,ctx)&&EC_POINT_mul(g,point,k,nullptr,nullptr,ctx),"chain audit expected");
        const bool infinity=EC_POINT_is_at_infinity(g,point);
        if(infinity)++infinities;
        if(flags[i]==2 || (flags[i]==1)!=infinity) {++invalid;continue;}
        if(infinity)continue;
        uint64_t want[8];
        glv10_require(EC_POINT_get_affine_coordinates(g,point,x,y,ctx)&&
                      BN_bn2lebinpad(x,(unsigned char*)want,32)==32&&
                      BN_bn2lebinpad(y,(unsigned char*)(want+4),32)==32,"chain audit affine");
        if(memcmp(want,actual.data()+(size_t)i*8,64)) {
            ++mismatches;fprintf(stderr,"QSB GLV10 chain mismatch scalar_index=%u\n",i);
            const uint64_t *rows[]={glv10_audit_scalars[i],want,want+4,actual.data()+(size_t)i*8,actual.data()+(size_t)i*8+4};
            const char *labels[]={"z","expected_x","expected_y","actual_x","actual_y"};
            for(int row=0;row<5;row++)fprintf(stderr,"  %s=%016llx%016llx%016llx%016llx\n",labels[row],
                (unsigned long long)rows[row][3],(unsigned long long)rows[row][2],(unsigned long long)rows[row][1],(unsigned long long)rows[row][0]);
        }
    }
    fprintf(stderr,"QSB GLV10 chain_audit samples=%u infinities=%u invalid=%u mismatches=%u\n",
            count,infinities,invalid,mismatches);
    EC_POINT_free(point);BN_free(nri);BN_free(k);BN_free(x);BN_free(y);BN_CTX_free(ctx);EC_GROUP_free(g);
    glv10_require(!invalid&&!mismatches,"chain audit");
}

// Batch-normalize each 8192-point host ladder (donor jacklightChen e582bda4).
static void glv10_ladder(EC_GROUP *group,const EC_POINT *first,const EC_POINT *step,
                         uint64_t *out,unsigned start,BN_CTX *ctx,BIGNUM *x,BIGNUM *y) {
    std::vector<EC_POINT*> points(GLV10_LADDER-start,nullptr);
    for(size_t i=0;i<points.size();i++) {
        points[i]=EC_POINT_new(group);glv10_require(points[i],"ladder allocation");
        glv10_require(i?EC_POINT_add(group,points[i],points[i-1],step,ctx):
                         EC_POINT_copy(points[i],first),"ladder addition");
    }
    glv10_require(EC_POINTs_make_affine(group,points.size(),points.data(),ctx),"batch affine");
    for(size_t i=0;i<points.size();i++) {
        glv10_require(EC_POINT_get_affine_coordinates(group,points[i],x,y,ctx),"ladder affine");
        glv10_require(BN_bn2lebinpad(x,(unsigned char*)(out+(i+start)*8),32)==32 &&
                      BN_bn2lebinpad(y,(unsigned char*)(out+(i+start)*8+4),32)==32,"ladder limbs");
        EC_POINT_free(points[i]);
    }
}
static void glv10_scalar(BIGNUM *k,int c,unsigned index,BIGNUM *tmp) {
    if(c==0) {
        glv10_require(BN_set_word(k,42639944)&&BN_lshift(k,k,101)&&
                      BN_set_word(tmp,1u<<23)&&BN_sub(k,k,tmp)&&BN_add_word(k,index),"biased scalar");
    } else glv10_require(BN_set_word(k,2*index+1)&&BN_lshift(k,k,glv10_shift(c)-1),"odd scalar");
}
static void glv10_build_ladders(uint64_t *L,uint64_t *H,const uint8_t nri_bytes[32]) {
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    glv10_require(g&&ctx,"OpenSSL context");
    BIGNUM *nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*tmp=BN_new(),*x=BN_new(),*y=BN_new();
    EC_POINT *A=EC_POINT_new(g),*base=EC_POINT_new(g),*first=EC_POINT_new(g),*step=EC_POINT_new(g);
    glv10_require(nri&&k&&tmp&&x&&y&&A&&base&&first&&step,"OpenSSL allocation");
    glv10_require(EC_POINT_mul(g,A,nri,nullptr,nullptr,ctx)&&!EC_POINT_is_at_infinity(g,A),"nonzero A");
    for(int c=0;c<5;c++) {
        glv10_require(BN_one(k)&&BN_lshift(k,k,c?glv10_shift(c)-1:0)&&
                      EC_POINT_mul(g,base,nullptr,A,k,ctx),"segment base");
        if(c==0) {
            glv10_scalar(k,0,0,tmp);
            glv10_require(EC_POINT_mul(g,first,nullptr,A,k,ctx),"bias point");
        } else glv10_require(EC_POINT_copy(first,base),"first point");
        glv10_ladder(g,first,base,L+(size_t)c*GLV10_LADDER*8,c?1:0,ctx,x,y);
        glv10_require(BN_set_word(k,GLV10_LADDER)&&EC_POINT_mul(g,step,nullptr,base,k,ctx),"high step");
        glv10_ladder(g,step,step,H+(size_t)c*GLV10_LADDER*8,1,ctx,x,y);
    }
    EC_POINT_free(A);EC_POINT_free(base);EC_POINT_free(first);EC_POINT_free(step);
    BN_free(nri);BN_free(k);BN_free(tmp);BN_free(x);BN_free(y);BN_CTX_free(ctx);EC_GROUP_free(g);
}
static void glv10_spot_check(const uint8_t *table,const uint8_t nri_bytes[32],const uint64_t *L,const uint64_t *H) {
    std::vector<uint32_t> indices;
    for(int c=0;c<5;c++) {
        const unsigned corners[]={0,1,2,4095,4096,8191,8192,glv10_entries(c)-1};
        for(unsigned index:corners)indices.push_back(glv10_offset(c)+index);
    }
    uint32_t seed=0x6d2b79f5;
    for(int i=0;i<192;i++){seed=1664525u*seed+1013904223u;indices.push_back(seed%GLV10_RECORDS);}
    uint32_t *di=nullptr;uint64_t *ds=nullptr;
    const size_t bytes=indices.size()*64;
    qsb_native::check(cudaMalloc(&di,indices.size()*4),"GLV sample indices");
    qsb_native::check(cudaMalloc(&ds,bytes),"GLV sample records");
    qsb_native::check(cudaMemcpy(di,indices.data(),indices.size()*4,cudaMemcpyHostToDevice),"GLV indices upload");
    QSB_LAUNCH(kernel_gather_glv10,1,256,table,di,ds,(unsigned)indices.size());
    std::vector<uint64_t> records(indices.size()*8);
    qsb_native::check(cudaMemcpy(records.data(),ds,bytes,cudaMemcpyDeviceToHost),"GLV gathered samples");
    uint8_t *reference=nullptr;unsigned *failed=nullptr;
    qsb_native::check(cudaMalloc(&reference,bytes),"GLV comparator");
    qsb_native::check(cudaMalloc(&failed,4),"GLV comparator status");
    qsb_native::check(cudaMemset(failed,0,4),"GLV comparator status reset");
    const double reference_start=glv10_seconds();
    QSB_LAUNCH(kernel_build_glv10,1,256,L,H,reference,0ULL,(uint64_t)indices.size(),di);
    std::vector<uint64_t> ordinary(indices.size()*8),batch(indices.size()*8);
    qsb_native::check(cudaMemcpy(ordinary.data(),reference,bytes,cudaMemcpyDeviceToHost),"GLV ordinary comparator");
    fprintf(stderr,"QSB stage=glv10-ordinary-samples elapsed=%.6f\n",glv10_seconds()-reference_start);
    QSB_LAUNCH(kernel_build_glv10_batch,1,256,L,H,reference,0ULL,(uint64_t)indices.size(),di,failed);
    qsb_native::check(cudaMemcpy(batch.data(),reference,bytes,cudaMemcpyDeviceToHost),"GLV batch comparator");
    unsigned failed_host=0;qsb_native::check(cudaMemcpy(&failed_host,failed,4,cudaMemcpyDeviceToHost),"GLV comparator status read");
    glv10_require(!failed_host,"GLV comparator zero denominator");
    glv10_require(ordinary==batch && ordinary==records,"ordinary/batch/gather equality");
    qsb_native::check(cudaFree(reference),"free GLV comparator");qsb_native::check(cudaFree(failed),"free GLV comparator status");
    qsb_native::check(cudaFree(di),"free GLV indices");qsb_native::check(cudaFree(ds),"free GLV samples");
    EC_GROUP *g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX *ctx=BN_CTX_new();
    glv10_require(g&&ctx,"sample context");
    BIGNUM *nri=BN_lebin2bn(nri_bytes,32,nullptr),*k=BN_new(),*tmp=BN_new(),*x=BN_new(),*y=BN_new();
    EC_POINT *point=EC_POINT_new(g);glv10_require(nri&&k&&tmp&&x&&y&&point,"sample allocation");
    for(size_t i=0;i<indices.size();i++) {
        int c=4;while(c && indices[i]<glv10_offset(c))--c;
        glv10_scalar(k,c,indices[i]-glv10_offset(c),tmp);
        glv10_require(BN_mul(k,k,nri,ctx)&&EC_POINT_mul(g,point,k,nullptr,nullptr,ctx)&&
                      EC_POINT_get_affine_coordinates(g,point,x,y,ctx),"sample point");
        uint64_t want[8];
        glv10_require(BN_bn2lebinpad(x,(unsigned char*)want,32)==32&&
                      BN_bn2lebinpad(y,(unsigned char*)(want+4),32)==32,"sample limbs");
        if(memcmp(want,records.data()+i*8,64)) {
            fprintf(stderr,"QSB GLV10 sample mismatch segment=%d record=%u\n",c,indices[i]);exit(2);
        }
    }
    EC_POINT_free(point);BN_free(nri);BN_free(k);BN_free(tmp);BN_free(x);BN_free(y);BN_CTX_free(ctx);EC_GROUP_free(g);
    fprintf(stderr,"QSB GLV10 spotcheck samples=%zu copied_bytes=%zu passed\n",indices.size(),bytes);
}
static uint8_t *glv10_build(const uint8_t nri[32]) {
    glv10_memory("before-glv10");const double start=glv10_seconds();
    uint8_t *table=nullptr;qsb_native::check(cudaMalloc(&table,GLV10_BYTES),"GLV10 table allocation");
    const size_t words=(size_t)5*GLV10_LADDER*8,bytes=words*8;
    std::vector<uint64_t> L(words,0),H(words,0);
    glv10_build_ladders(L.data(),H.data(),nri);
    fprintf(stderr,"QSB stage=glv10-ladders elapsed=%.6f bytes=%zu\n",glv10_seconds()-start,2*bytes);
    uint64_t *dl=nullptr,*dh=nullptr;
    qsb_native::check(cudaMalloc(&dl,bytes),"GLV L allocation");qsb_native::check(cudaMalloc(&dh,bytes),"GLV H allocation");
    qsb_native::check(cudaMemcpy(dl,L.data(),bytes,cudaMemcpyHostToDevice),"GLV L upload");
    qsb_native::check(cudaMemcpy(dh,H.data(),bytes,cudaMemcpyHostToDevice),"GLV H upload");
    const double build=glv10_seconds();
    unsigned *failed=nullptr;unsigned failed_host=0;
    qsb_native::check(cudaMalloc(&failed,4),"GLV builder status");
    qsb_native::check(cudaMemset(failed,0,4),"GLV builder status reset");
#ifndef QSB_GLV10_BATCH
#define QSB_GLV10_BATCH 1
#endif
#if QSB_GLV10_BATCH
    QSB_LAUNCH(kernel_build_glv10_batch,(unsigned)((GLV10_RECORDS+255)/256),256,dl,dh,table,0ULL,GLV10_RECORDS,(const uint32_t*)nullptr,failed);
#else
    QSB_LAUNCH(kernel_build_glv10,(unsigned)((GLV10_RECORDS+255)/256),256,dl,dh,table,0ULL,GLV10_RECORDS,(const uint32_t*)nullptr);
#endif
    qsb_native::check(cudaDeviceSynchronize(),"GLV build synchronization");
    fprintf(stderr,"QSB stage=glv10-table elapsed=%.6f records=%llu bytes=%zu\n",glv10_seconds()-build,
            (unsigned long long)GLV10_RECORDS,GLV10_BYTES);
    qsb_native::check(cudaMemcpy(&failed_host,failed,4,cudaMemcpyDeviceToHost),"GLV builder status read");
    glv10_require(!failed_host,"GLV builder zero denominator");
    qsb_native::check(cudaFree(failed),"free GLV builder status");
    const double check=glv10_seconds();glv10_spot_check(table,nri,dl,dh);
    qsb_native::check(cudaFree(dl),"free GLV L");qsb_native::check(cudaFree(dh),"free GLV H");
    fprintf(stderr,"QSB stage=glv10-check elapsed=%.6f total_setup=%.6f\n",glv10_seconds()-check,glv10_seconds()-start);
    const double audit=glv10_seconds();glv10_chain_audit(table,nri);
    fprintf(stderr,"QSB stage=glv10-chain-audit elapsed=%.6f\n",glv10_seconds()-audit);
    glv10_memory("after-glv10");return table;
}
