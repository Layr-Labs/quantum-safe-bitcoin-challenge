// SPDX-License-Identifier: GPL-3.0-only
// Standalone research probe, deliberately not wired into the ranked solver.
// GPU execution is required to validate the CUDA protocol and its throughput.
#define QSB_ISO_XR 0
#define QSB_YOFF 0
#ifndef QSB_ZEROS_N
#define QSB_ZEROS_N 24
#endif
#define main qsb_original_solver_main
#include "../pinning.cu"
#undef main
#include "inverse_tree.cuh"
#include "inverse_service.cuh"
#include "shifted_tail.cuh"
#include <vector>

#ifndef QSB_AFFINE_MIN_BLOCKS
#define QSB_AFFINE_MIN_BLOCKS 4
#endif
constexpr unsigned AFFINE_N=128, AFFINE_WIDTH=256, AFFINE_WINDOWS=15;

// Canonical reference arithmetic. Squaring deliberately uses the complete
// multiplier, and parity uses a full product: this is a correctness/resource
// probe, not the optimized 73M+15S+2W cost model.
struct AffineField {
    __device__ __forceinline__ static void sub(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]) {
        uint64_t r0,r1,r2,r3;
        asm volatile("{ .reg .u64 m,k;\n\t"
          "sub.cc.u64 %0,%4,%8; subc.cc.u64 %1,%5,%9;\n\t"
          "subc.cc.u64 %2,%6,%10; subc.cc.u64 %3,%7,%11;\n\t"
          "subc.u64 m,0,0; and.b64 k,m,0xfffffffefffffc2f;\n\t"
          "add.cc.u64 %0,%0,k; addc.cc.u64 %1,%1,m;\n\t"
          "addc.cc.u64 %2,%2,m; addc.u64 %3,%3,m; }"
          :"=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3)
          :"l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
           "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
        out[0]=r0;out[1]=r1;out[2]=r2;out[3]=r3;
    }
    __device__ __forceinline__ static void neg(uint64_t out[4],const uint64_t a[4]) {
        const uint64_t zero[4]={0,0,0,0};sub(out,zero,a);
    }
    __device__ __forceinline__ static void mul(uint64_t out[4],const uint64_t a[4],const uint64_t b[4]) {
        uint64_t aa[5],bb[5],r[5];
        #pragma unroll
        for(int i=0;i<4;i++){aa[i]=a[i];bb[i]=b[i];}
        aa[4]=bb[4]=0;qsb_field_mul(r,aa,bb);qsb_field_normalize(r);
        #pragma unroll
        for(int i=0;i<4;i++)out[i]=r[i];
    }
    __device__ __forceinline__ static void sqr(uint64_t out[4],const uint64_t a[4]) {mul(out,a,a);}
    __device__ __forceinline__ static uint32_t parity_product(
        const uint64_t a[4],const uint64_t b[4],const uint64_t beta[4],uint32_t negative) {
        uint64_t product[4],minus_beta[4];
        mul(product,a,b);neg(minus_beta,beta);sub(product,product,minus_beta);
        if(negative)neg(product,product);
        return uint32_t(product[0]&1u);
    }
};
struct AffineInvert {
    __device__ __forceinline__ void operator()(uint64_t out[4],const uint64_t in[4]) const {
        uint64_t t[5]={in[0],in[1],in[2],in[3],0};_ModInv(t);
        qsb_field_normalize(t);
        #pragma unroll
        for(int k=0;k<4;k++)out[k]=t[k];
    }
};
struct AffineResult {uint64_t x[2][4];uint32_t parities,valid;};

// Isolate the address-taking Euclidean inverse from the worker's temporaries.
// The outer kernel still accounts for the service's maximum register usage.
__device__ __noinline__ void affine_service_branch(qsb_inverse_service::Slot* slots,
    unsigned workers,unsigned service_index) {
    qsb_inverse_service::service(slots,workers,service_index,AffineInvert{});
}

__host__ __device__ static uint32_t affine_code(uint32_t candidate,unsigned tile,unsigned window) {
    uint32_t x=candidate*0x9e3779b9u+tile*0x85ebca6bu+window*0xc2b2ae35u;
    x^=x>>16;x*=0x7feb352du;x^=x>>15;return x;
}
__device__ __forceinline__ QsbShiftedTailPoint affine_load(
    const QsbShiftedTailPoint* table,unsigned window,uint32_t code) {
    QsbShiftedTailPoint p=table[window*AFFINE_WIDTH+(code&255u)];
    if(code&256u)AffineField::neg(p.y,p.y);
    return p;
}

// Reload the tail after the inversion reply and finish one arm at a time.
// This avoids keeping both ordinates and both numerators live through the tree.
__device__ __forceinline__ uint32_t affine_tail_finish_streamed(uint64_t out_x[2][4],
    const uint64_t inverse[4],const QsbShiftedTailPoint& prefix,
    const QsbShiftedTailRecord* tails,uint32_t index,uint32_t negative) {
    uint32_t parities=0;
    #pragma unroll
    for(unsigned arm=0;arm<2;arm++) {
        QsbShiftedTailPoint tail=tails[index].arm[arm^negative];
        if(negative)AffineField::neg(tail.y,tail.y);
        uint64_t other_delta[4],numerator[4],slope[4],offset[4];
        AffineField::sub(other_delta,tails[index].arm[(arm^1u)^negative].x,prefix.x);
        AffineField::sub(numerator,tail.y,prefix.y);
        AffineField::mul(slope,numerator,other_delta);AffineField::mul(slope,slope,inverse);
        AffineField::sqr(out_x[arm],slope);
        AffineField::sub(out_x[arm],out_x[arm],prefix.x);AffineField::sub(out_x[arm],out_x[arm],tail.x);
        AffineField::sub(offset,out_x[arm],tail.x);
        parities|=AffineField::parity_product(slope,offset,tail.y,1u)<<arm;
    }
    return parities;
}

__global__ __launch_bounds__(AFFINE_N,QSB_AFFINE_MIN_BLOCKS)
void qsb_persistent_affine_probe(const QsbShiftedTailPoint* table,
    const QsbShiftedTailRecord* tails,qsb_inverse_service::Slot* slots,
    AffineResult* output,unsigned service_blocks,unsigned workers,unsigned tiles) {
    // All roles share one cooperative launch and therefore have a proven
    // residency bound. No service thread participates in worker CTA barriers.
    if(blockIdx.x<service_blocks) {
        affine_service_branch(slots,workers,blockIdx.x*AFFINE_N+threadIdx.x);
        return;
    }
    const unsigned worker=blockIdx.x-service_blocks;
    const unsigned candidate=worker*AFFINE_N+threadIdx.x;
    __shared__ uint64_t products[4][2*AFFINE_N],inverses[4][AFFINE_N];
    uint32_t epoch=0;
    for(unsigned tile=0;tile<tiles;tile++) {
        QsbShiftedTailPoint point=affine_load(table,0,affine_code(candidate,tile,0));
        bool valid=true;
        #pragma unroll 1
        for(unsigned window=1;window<14;window++) {
            const QsbShiftedTailPoint addend=affine_load(table,window,affine_code(candidate,tile,window));
            uint64_t delta[4],numerator[4],root[4],inverse[4];
            AffineField::sub(delta,addend.x,point.x);
            AffineField::sub(numerator,addend.y,point.y);
            valid=valid&&!qsb_shifted_tail_zero(delta);
            if(!valid){delta[0]=1;delta[1]=delta[2]=delta[3]=0;}
            qsb_affine_tree_prepare_shared<AFFINE_N>(delta,root,products);
            ++epoch;
            if(threadIdx.x==0) {
                qsb_inverse_service::worker_publish(slots[worker],epoch,root);
                qsb_inverse_service::worker_wait(slots[worker],epoch,root);
            }
            qsb_affine_tree_finish_shared<AFFINE_N>(inverse,root,products,inverses);
            uint64_t slope[4],next_x[4],offset[4];
            AffineField::mul(slope,numerator,inverse);AffineField::sqr(next_x,slope);
            AffineField::sub(next_x,next_x,point.x);AffineField::sub(next_x,next_x,addend.x);
            // y = slope*(addend.x-next_x)-addend.y; use addend anchor.
            AffineField::sub(offset,addend.x,next_x);AffineField::mul(point.y,slope,offset);
            AffineField::sub(point.y,point.y,addend.y);
            #pragma unroll
            for(int k=0;k<4;k++)point.x[k]=next_x[k];
            // A slow warp may still be reading the previous tree's siblings.
            // The next prepare writes leaves before its first barrier.
            __syncthreads();
        }
        const uint32_t code=affine_code(candidate,tile,14);
        uint64_t denominator[4],root[4],inverse[4];
        {
            QsbShiftedTailPoint tail[2];uint64_t delta[2][4],numerator[2][4];
            qsb_shifted_tail_load<AffineField>(tail,tails,code&255u,(code>>8)&1u);
            const bool tail_ok=qsb_shifted_tail_prepare<AffineField>(denominator,delta,numerator,point,tail);
            valid=valid&&tail_ok;
        }
        if(!valid){denominator[0]=1;denominator[1]=denominator[2]=denominator[3]=0;}
        qsb_affine_tree_prepare_shared<AFFINE_N>(denominator,root,products);
        ++epoch;
        if(threadIdx.x==0) {
            qsb_inverse_service::worker_publish(slots[worker],epoch,root);
            qsb_inverse_service::worker_wait(slots[worker],epoch,root);
        }
        qsb_affine_tree_finish_shared<AFFINE_N>(inverse,root,products,inverses);
        // Only nonexceptional results are used. A ranked integration must route
        // every declined candidate to a complete cold path, not silently drop it.
        AffineResult result={};result.valid=valid;
        if(valid)result.parities=affine_tail_finish_streamed(result.x,inverse,point,tails,code&255u,(code>>8)&1u);
        // Keep final arithmetic live in every iteration, not only the last.
        asm volatile(""::"l"(result.x[0][0]),"l"(result.x[1][0]),"r"(result.parities):"memory");
        if(tile+1==tiles)output[candidate]=result;
        __syncthreads(); // no leaf consumer survives into the next tile
    }
    if(threadIdx.x==0)qsb_inverse_service::worker_stop(slots[worker]);
}

#define CUDA_OK(call) do {cudaError_t e=(call);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 2;}}while(0)
static bool affine_encode(QsbShiftedTailPoint& out,const EC_GROUP* group,const EC_POINT* p,BN_CTX* ctx) {
    BIGNUM* x=BN_new();BIGNUM* y=BN_new();
    bool ok=x&&y&&EC_POINT_get_affine_coordinates(group,p,x,y,ctx)==1&&
      BN_bn2lebinpad(x,(unsigned char*)out.x,32)==32&&BN_bn2lebinpad(y,(unsigned char*)out.y,32)==32;
    BN_free(x);BN_free(y);return ok;
}
int main(int argc,char** argv) {
    if(argc>1&&!strcmp(argv[1],"--describe")) {
        printf("reference affine probe: 128 threads/CTA, 14 roots/tile, 12288 shared bytes, min blocks %d; no ranked solver integration\n",QSB_AFFINE_MIN_BLOCKS);return 0;
    }
    const unsigned tiles=argc>1?unsigned(strtoul(argv[1],nullptr,10)):8;
    if(!tiles||tiles>1000000){fprintf(stderr,"tiles must be 1..1000000\n");return 2;}
    int cooperative=0,blocks_per_sm=0;cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop,0));
    CUDA_OK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));
    if(!cooperative){fprintf(stderr,"cooperative launch is required\n");return 2;}
    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,qsb_persistent_affine_probe,AFFINE_N,0));
    const unsigned grid=unsigned(blocks_per_sm*prop.multiProcessorCount);
    if(grid<2){fprintf(stderr,"insufficient resident blocks\n");return 2;}
    unsigned services=(grid+AFFINE_N)/(AFFINE_N+1),workers=grid-services;
    if(services*AFFINE_N<workers){fprintf(stderr,"invalid service capacity\n");return 2;}
    const unsigned candidates=workers*AFFINE_N;
    std::vector<QsbShiftedTailPoint> table(AFFINE_WIDTH*AFFINE_WINDOWS);
    std::vector<QsbShiftedTailRecord> tails(AFFINE_WIDTH);
    EC_GROUP* group=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX* ctx=BN_CTX_new();
    BIGNUM* scalar=BN_new();BIGNUM* part=BN_new();BIGNUM* order=BN_new();
    if(!group||!ctx||!scalar||!part||!order)return 2;
    EC_POINT* base=EC_POINT_new(group);EC_POINT* point=EC_POINT_new(group);
    EC_POINT* recovery=EC_POINT_new(group);EC_POINT* neg_recovery=EC_POINT_new(group);EC_POINT* sum=EC_POINT_new(group);
    if(!group||!ctx||!scalar||!part||!order||!base||!point||!recovery||!neg_recovery||!sum)return 2;
    bool ok=EC_GROUP_get_order(group,order,ctx)==1&&BN_set_word(scalar,1009)==1&&
      EC_POINT_mul(group,recovery,scalar,nullptr,nullptr,ctx)==1&&EC_POINT_copy(neg_recovery,recovery)==1&&EC_POINT_invert(group,neg_recovery,ctx)==1;
    for(unsigned w=0;w<AFFINE_WINDOWS&&ok;w++) {
        ok=BN_one(scalar)==1&&BN_lshift(scalar,scalar,12*w)==1&&EC_POINT_mul(group,base,scalar,nullptr,nullptr,ctx)==1&&EC_POINT_copy(point,base)==1;
        for(unsigned j=0;j<AFFINE_WIDTH&&ok;j++) {
            ok=affine_encode(table[w*AFFINE_WIDTH+j],group,point,ctx);
            if(w==14)for(unsigned arm=0;arm<2&&ok;arm++)ok=EC_POINT_add(group,sum,point,arm?neg_recovery:recovery,ctx)==1&&affine_encode(tails[j].arm[arm],group,sum,ctx);
            ok=ok&&EC_POINT_add(group,point,point,base,ctx)==1;
        }
    }
    if(!ok){fprintf(stderr,"OpenSSL table generation failed\n");return 2;}
    QsbShiftedTailPoint* dtable=nullptr;QsbShiftedTailRecord* dtails=nullptr;
    qsb_inverse_service::Slot* slots=nullptr;AffineResult* result=nullptr;
    CUDA_OK(cudaMalloc(&dtable,table.size()*sizeof(table[0])));CUDA_OK(cudaMalloc(&dtails,tails.size()*sizeof(tails[0])));
    CUDA_OK(cudaMalloc(&slots,workers*sizeof(*slots)));CUDA_OK(cudaMalloc(&result,candidates*sizeof(*result)));
    CUDA_OK(cudaMemcpy(dtable,table.data(),table.size()*sizeof(table[0]),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dtails,tails.data(),tails.size()*sizeof(tails[0]),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(slots,0,workers*sizeof(*slots)));
    void* args[]={&dtable,&dtails,&slots,&result,&services,&workers,(void*)&tiles};
    cudaEvent_t start,end;CUDA_OK(cudaEventCreate(&start));CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(start));
    CUDA_OK(cudaLaunchCooperativeKernel((void*)qsb_persistent_affine_probe,grid,AFFINE_N,args));
    CUDA_OK(cudaEventRecord(end));CUDA_OK(cudaEventSynchronize(end));
    float ms=0;CUDA_OK(cudaEventElapsedTime(&ms,start,end));
    std::vector<AffineResult> observed(candidates);CUDA_OK(cudaMemcpy(observed.data(),result,observed.size()*sizeof(observed[0]),cudaMemcpyDeviceToHost));
    unsigned checked=0,mismatches=0,declined=0;
    for(unsigned c=0;c<candidates;c++) {
        if(!observed[c].valid){declined++;continue;}
        if(c%AFFINE_N!=0 && c%251!=0)continue; // every worker and an independent stride
        BN_zero(scalar);
        for(unsigned w=0;w<AFFINE_WINDOWS;w++) {
            const uint32_t code=affine_code(c,tiles-1,w);
            ok=BN_set_word(part,(code&255u)+1u)==1&&BN_lshift(part,part,12*w)==1;
            if(code&256u)BN_set_negative(part,1);
            ok=ok&&BN_add(scalar,scalar,part)==1;if(!ok)return 2;
        }
        ok=BN_nnmod(scalar,scalar,order,ctx)==1&&EC_POINT_mul(group,point,scalar,nullptr,nullptr,ctx)==1;
        for(unsigned arm=0;arm<2;arm++) {
            QsbShiftedTailPoint expected{};
            ok=ok&&EC_POINT_add(group,sum,point,arm?neg_recovery:recovery,ctx)==1&&affine_encode(expected,group,sum,ctx);
            if(!ok||memcmp(expected.x,observed[c].x[arm],32)||((expected.y[0]&1u)!=((observed[c].parities>>arm)&1u)))mismatches++;
        }
        checked++;
    }
    printf("{\"reference_probe_only\":true,\"sm_count\":%d,\"blocks_per_sm\":%d,\"workers\":%u,\"service_blocks\":%u,\"tiles\":%u,\"milliseconds\":%.3f,\"checked_pairs\":%u,\"declined\":%u,\"mismatches\":%u}\n",prop.multiProcessorCount,blocks_per_sm,workers,services,tiles,ms,checked,declined,mismatches);
    cudaFree(dtable);cudaFree(dtails);cudaFree(slots);cudaFree(result);cudaEventDestroy(start);cudaEventDestroy(end);
    EC_POINT_free(base);EC_POINT_free(point);EC_POINT_free(recovery);EC_POINT_free(neg_recovery);EC_POINT_free(sum);
    BN_free(scalar);BN_free(part);BN_free(order);BN_CTX_free(ctx);EC_GROUP_free(group);
    return mismatches||declined?1:0;
}
