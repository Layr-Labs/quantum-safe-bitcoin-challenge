// CPU execution support for the ACTUAL production collective/recovery source.
// ucontext schedules all 256 CUDA lanes at each real __syncthreads call.
// Field primitives use independent OpenSSL arithmetic, not GPU PTX emulation.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <ucontext.h>
#include <openssl/bn.h>
#include "RecoveryConstant.h"

struct Dim { unsigned x; } threadIdx,blockIdx,blockDim{QSB_RECOVERY_N};
struct alignas(16) ulonglong2 {uint64_t x,y;};
static ulonglong2 make_ulonglong2(uint64_t x,uint64_t y){return {x,y};}
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __global__
#define __launch_bounds__(...)
#define Load256(r,a) do {for(int q=0;q<4;q++)(r)[q]=(a)[q];} while(0)
#define QSB_CHECKPOINT_NODES 254
#define QSB_CHECKPOINT_STRIDE 256

static BN_CTX *arith_ctx;
static BIGNUM *field;
static void init_arith(){
    if(arith_ctx)return;
    arith_ctx=BN_CTX_new(); field=BN_new();
    assert(arith_ctx&&field);
    assert(BN_set_word(field,1)&&BN_lshift(field,field,256)&&BN_sub_word(field,0x1000003D1UL));
}
static void field_op(uint64_t *out,const uint64_t *aa,const uint64_t *bb,int op){
    init_arith(); BN_CTX_start(arith_ctx);
    BIGNUM *a=BN_CTX_get(arith_ctx),*b=BN_CTX_get(arith_ctx),*r=BN_CTX_get(arith_ctx);
    assert(r&&BN_lebin2bn((const unsigned char*)aa,32,a));
    if(bb)assert(BN_lebin2bn((const unsigned char*)bb,32,b));
    bool ok=op==0 ? BN_mod_mul(r,a,b,field,arith_ctx) :
        op==1 ? BN_mod_add(r,a,b,field,arith_ctx) :
        op==2 ? BN_mod_sub(r,a,b,field,arith_ctx) :
        BN_mod_inverse(r,a,field,arith_ctx)!=nullptr;
    assert(ok&&BN_bn2lebinpad(r,(unsigned char*)out,32)==32);
    BN_CTX_end(arith_ctx);
}
static void qsb_field_mul(uint64_t *o,uint64_t *a,uint64_t *b){field_op(o,a,b,0);o[4]=0;}
static void qsb_field_normalize(uint64_t *r){
    if((r[1]&r[2]&r[3])==UINT64_MAX&&r[0]>=0xFFFFFFFEFFFFFC2FULL){
        r[0]-=0xFFFFFFFEFFFFFC2FULL;r[1]=r[2]=r[3]=0;
    }
}
static void _ModAdd256(uint64_t *r,uint64_t *a,uint64_t *b){field_op(r,a,b,1);}
static void _ModSub256(uint64_t *r,uint64_t *a,uint64_t *b){field_op(r,a,b,2);}
static void _ModSub256(uint64_t *r,uint64_t *b){field_op(r,r,b,2);}
static void _ModInv(uint64_t *r){field_op(r,r,nullptr,3);r[4]=0;}

static ucontext_t scheduler,lanes[256];
static std::vector<unsigned char> stacks(256*65536);
static bool finished[256];
static int phase[256];
static void (*lane_work)();
static void sync_lanes(){
    ++phase[threadIdx.x];
    assert(swapcontext(&lanes[threadIdx.x],&scheduler)==0);
}
#define __syncthreads() sync_lanes()
// BEGIN_EXTRACTED_ROOTS
// END_EXTRACTED_ROOTS
#include "LeafRecovery.cuh"

static void enter_lane(){lane_work();finished[threadIdx.x]=true;}
static void launch(void (*fn)(),unsigned block,unsigned seed,int threads=QSB_RECOVERY_N){
    blockIdx.x=block;blockDim.x=threads;lane_work=fn;
    for(int i=0;i<threads;i++){
        finished[i]=false;phase[i]=0;
        assert(getcontext(&lanes[i])==0);
        lanes[i].uc_stack.ss_sp=stacks.data()+i*65536;
        lanes[i].uc_stack.ss_size=65536;
        lanes[i].uc_link=&scheduler;
        makecontext(&lanes[i],enter_lane,0);
    }
    std::mt19937 rng(seed);
    std::vector<int> order(threads);for(int i=0;i<threads;i++)order[i]=i;
    for(int iteration=0;iteration<64;iteration++){
        std::shuffle(order.begin(),order.end(),rng);
        for(int i:order){
            assert(!finished[i]);threadIdx.x=i;
            assert(swapcontext(&scheduler,&lanes[i])==0);
        }
        for(int i=1;i<threads;i++){
            assert(finished[i]==finished[0]);assert(phase[i]==phase[0]);
        }
        if(finished[0])return;
    }
    assert(false&&"collective failed to finish");
}

static uint64_t pin_u2rx_words[4],pin_u2ry_words[4],pin_recovery_c[4];
static const uint64_t *inputs;
static uint64_t *results;
static ulonglong2 *saved;
static uint64_t *roots,*tree;
static int batch_size;
static uint64_t *group_roots,*group_checkpoint;
static int roots_count,groups_count;
static void root_prepare(){qsb_root_group_prepare(roots,roots_count,group_roots,group_checkpoint);}
static void root_invert(){qsb_invert_super_roots(group_roots,groups_count);}
static void root_finish(){qsb_root_group_finish(roots,roots_count,group_roots,group_checkpoint);}
static void invert_roots(int count,unsigned seed){
    roots_count=count;groups_count=(count+255)/256;
    std::vector<uint64_t> super((size_t)groups_count*4),checkpoint((size_t)groups_count*4*256);
    group_roots=super.data();group_checkpoint=checkpoint.data();
    for(int i=0;i<groups_count;i++)launch(root_prepare,i,seed+i,256);
    for(int i=0;i<(groups_count+255)/256;i++)launch(root_invert,i,seed+i+1000,256);
    for(int i=0;i<groups_count;i++)launch(root_finish,i,seed+i+2000,256);
}

// The generator replaces these markers with verbatim production kernel slices.
// BEGIN_EXTRACTED_STAGES
// END_EXTRACTED_STAGES

extern "C" int constant(uint64_t *out,const unsigned char *a,const unsigned char *b){
    return qsb_make_recovery_constant(out,a,b);
}

extern "C" int pipeline(int n,const uint64_t *in,const uint64_t *a,const uint64_t *b,
                         uint64_t *out,unsigned seed){
    assert(n>0); init_arith();
    const unsigned blocks=(n+QSB_RECOVERY_N-1)/QSB_RECOVERY_N;
    const uint64_t canary=0xbadc0ffee1234567ULL;
    std::vector<ulonglong2> storage((size_t)n*6+32,{canary,canary});
    std::vector<uint64_t> checkpoints((size_t)blocks*4*QSB_RECOVERY_N+32,canary);
    std::vector<uint64_t> root_storage((size_t)blocks*4+32,canary);
    batch_size=n;inputs=in;results=out;saved=storage.data()+16;
    roots=root_storage.data()+16;tree=checkpoints.data()+16;
    memcpy(pin_u2rx_words,a,32);memcpy(pin_u2ry_words,b,32);
    assert(qsb_make_recovery_constant(pin_recovery_c,(const uint8_t*)a,(const uint8_t*)b));
    memset(out,0,(size_t)n*9*sizeof(uint64_t));
    for(unsigned block=0;block<blocks;block++)launch(stage_zero,block,seed+block);
    invert_roots(blocks,seed+20000);
    for(unsigned block=0;block<blocks;block++)launch(stage_two,block,seed+block+10000);
    for(int i=0;i<16;i++){
        assert(storage[i].x==canary&&storage[i].y==canary);
        assert(storage[storage.size()-1-i].x==canary&&storage[storage.size()-1-i].y==canary);
        assert(checkpoints[i]==canary&&checkpoints[checkpoints.size()-1-i]==canary);
        assert(root_storage[i]==canary&&root_storage[root_storage.size()-1-i]==canary);
    }
    return 1;
}

extern "C" int root_test(int n,const uint64_t *in,uint64_t *out,unsigned seed){
    roots=out;memcpy(out,in,(size_t)n*32);invert_roots(n,seed);return 1;
}
