#!/usr/bin/env python3
"""Execute the production collective using CPU threads and exact OpenSSL field shims.
GPU scheduling and throughput are not measured. Device multiplication is inherited.
"""
from pathlib import Path
import subprocess
import tempfile
from test_parity_replay import function
HERE=Path(__file__).resolve().parent

def main():
    pin=(HERE/'pinning.cu').read_text()
    for name in ('hm39_divstep.cuh','hm41_quad_inverse.cuh'):
        original=subprocess.check_output(['git','show','7c3609b:candidates/subset/tests/gpu_epochs/'+name],cwd=HERE)
        assert (HERE/name).read_bytes()==original
    source=r'''
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
#define QSB_PIN_AFFINE 1
#define QSB_AFFINE_HOST_ORACLE 1
#define QSB_TREE_N 128
#define GT_CHUNKS 15
#define QSB_YOFF 1
#define QSB_NEG_Y_MAC 1
#define Load256(a,b) memcpy(a,b,32)
struct Index{unsigned x;};
thread_local Index threadIdx;
thread_local BN_CTX *ctx;
BIGNUM *prime,*order;
EC_GROUP *group;
uint64_t GT_ORDER_N[4]={0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
struct Barrier {
    std::mutex m;std::condition_variable cv;unsigned count=0,generation=0,total;
    Barrier(unsigned n=QSB_TREE_N):total(n){}
    void wait(){std::unique_lock<std::mutex> lock(m);unsigned old=generation;
        if(++count==total){count=0;++generation;cv.notify_all();}
        else cv.wait(lock,[&]{return old!=generation;});
    }
} barrier;
void __syncthreads(){barrier.wait();}
Barrier warp0(32),warp1(32),warp2(32),warp3(32);
void __syncwarp(){Barrier* warps[]={&warp0,&warp1,&warp2,&warp3};warps[threadIdx.x/32]->wait();}
void qsb_field_normalize(uint64_t *r){
    if((r[1]&r[2]&r[3])==UINT64_MAX&&r[0]>=0xFFFFFFFEFFFFFC2FULL){r[0]-=0xFFFFFFFEFFFFFC2FULL;r[1]=r[2]=r[3]=0;}
}
void qsb_field_mul(uint64_t *out,uint64_t*a,uint64_t*b){
    BN_CTX_start(ctx);auto aa=BN_CTX_get(ctx),bb=BN_CTX_get(ctx),z=BN_CTX_get(ctx);
    BN_lebin2bn((unsigned char*)a,32,aa);BN_lebin2bn((unsigned char*)b,32,bb);
    assert(BN_mod_mul(z,aa,bb,prime,ctx));assert(BN_bn2lebinpad(z,(unsigned char*)out,32)==32);
    out[4]=0;BN_CTX_end(ctx);
}
void _ModInv(uint64_t*r){
    BN_CTX_start(ctx);auto a=BN_CTX_get(ctx),z=BN_CTX_get(ctx);
    BN_lebin2bn((unsigned char*)r,32,a);assert(BN_mod_inverse(z,a,prime,ctx));
    assert(BN_bn2lebinpad(z,(unsigned char*)r,32)==32);r[4]=0;BN_CTX_end(ctx);
}
// The copied promoted quad inverse is the device backend; host field inverses use OpenSSL.
Barrier quad_barrier(4);
void hm41_quad_inverse(uint64_t*r,int){quad_barrier.wait();_ModInv(r);quad_barrier.wait();}
struct Point {uint64_t x[4],y[4];};
Point points[128][15],result[128];uint64_t scalars[128][4];bool result_inf[128];
uint32_t expected_codes[128][15];
void gt_load_signed_flat_m(const uint8_t*,unsigned base,unsigned index,uint64_t mask,uint64_t*x,uint64_t*y){
    unsigned c=base?(base>>16)-1:0,t=threadIdx.x;
    assert(expected_codes[t][c]==(index|(uint32_t(mask!=0)<<31)));
    Load256(x,points[t][c].x);Load256(y,points[t][c].y);
}
'''
    for marker in [
        '__host__ __device__ __forceinline__ unsigned gt_offset(',
        '__host__ __device__ __forceinline__ int gt_shift(',
        '__device__ __forceinline__ uint64_t *qsb_digit_arena(',
        '__device__ __forceinline__ void qsb_signed_recode_setup(',
        '__device__ __forceinline__ void qsb_decode_to_shared(',
        '__device__ __forceinline__ void qsb_load_decoded(',
    ]:source+=function(pin,marker)+'\n'
    source+=(HERE/'affine_block.cuh').read_text()
    source+=r'''
uint64_t seed=0xabe27bf51ULL;
uint64_t random64(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
void put_point(Point &out,const EC_POINT *p){
    BN_CTX_start(ctx);auto x=BN_CTX_get(ctx),y=BN_CTX_get(ctx),c=BN_CTX_get(ctx);
    assert(EC_POINT_get_affine_coordinates(group,p,x,y,ctx));
    BN_set_word(c,0x800001e8ULL);assert(BN_add(y,y,c));
    assert(BN_bn2lebinpad(x,(unsigned char*)out.x,32)==32);
    assert(BN_bn2lebinpad(y,(unsigned char*)out.y,32)==32);BN_CTX_end(ctx);
}
void get_point(EC_POINT *out,const Point &p){
    BN_CTX_start(ctx);auto x=BN_CTX_get(ctx),y=BN_CTX_get(ctx),c=BN_CTX_get(ctx);
    BN_lebin2bn((const unsigned char*)p.x,32,x);BN_lebin2bn((const unsigned char*)p.y,32,y);
    BN_set_word(c,0x800001e8ULL);BN_mod_sub(y,y,c,prime,ctx);
    assert(EC_POINT_set_affine_coordinates(group,out,x,y,ctx));BN_CTX_end(ctx);
}
void check_arithmetic(){
    uint64_t a[4],b[4],out[4],want[4];
    BN_CTX_start(ctx);auto aa=BN_CTX_get(ctx),bb=BN_CTX_get(ctx),z=BN_CTX_get(ctx);
    for(unsigned i=0;i<20000;i++){
        for(auto &v:a)v=random64();for(auto &v:b)v=random64();
        if(i<64){memset(a,255,32);a[0]=0xfffffffefffffc2fULL-(i%16);memset(b,0,32);b[0]=i/16;}
        qsb_field_normalize(a);qsb_field_normalize(b);
        BN_lebin2bn((unsigned char*)a,32,aa);BN_lebin2bn((unsigned char*)b,32,bb);
        BN_mod_add(z,aa,bb,prime,ctx);BN_bn2lebinpad(z,(unsigned char*)want,32);
        qsb_affine_add(out,a,b);assert(!memcmp(out,want,32));
        memcpy(out,a,32);qsb_affine_add(out,out,b);assert(!memcmp(out,want,32));
        BN_mod_sub(z,aa,bb,prime,ctx);BN_bn2lebinpad(z,(unsigned char*)want,32);
        qsb_affine_sub(out,a,b);assert(!memcmp(out,want,32));
        memcpy(out,b,32);qsb_affine_sub(out,a,out);assert(!memcmp(out,want,32));
    }BN_CTX_end(ctx);
}
int main(){
    ctx=BN_CTX_new();prime=BN_new();order=BN_new();group=EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx);EC_GROUP_get_order(group,order,ctx);
    check_arithmetic();
    auto factor=BN_new(),two=BN_new(),inv2=BN_new(),k=BN_new();BN_set_word(two,2);BN_mod_inverse(inv2,two,order,ctx);
    auto p=EC_POINT_new(group),want=EC_POINT_new(group),got=EC_POINT_new(group),sum=EC_POINT_new(group);
    unsigned checked=0;
    for(unsigned pass=0;pass<3;pass++){
        for(unsigned t=0;t<128;t++){
            for(auto&v:scalars[t])v=random64();
            if(t<5){memset(scalars[t],0,32);
                if(t==1)scalars[t][0]=1;
                if(t==2||t==3){memcpy(scalars[t],GT_ORDER_N,32);if(t==2)--scalars[t][0];}
                if(t==4)memset(scalars[t],255,32);
            }
            threadIdx.x=t;qsb_decode_to_shared(scalars[t]);auto codes=(uint32_t*)qsb_digit_arena();
            for(unsigned c=0;c<15;c++){
                auto code=codes[c*128+t];expected_codes[t][c]=code;
                BN_set_word(factor,2*(code&0x1ffffu)+1);BN_lshift(factor,factor,gt_shift(c));
                BN_mod_mul(factor,factor,inv2,order,ctx);
                assert(EC_POINT_mul(group,p,factor,nullptr,nullptr,ctx));
                if(code>>31)EC_POINT_invert(group,p,ctx);put_point(points[t][c],p);
            }
            if(pass==2){
                EC_POINT_copy(p,EC_GROUP_get0_generator(group));put_point(points[t][0],p);
                if(t%3==1)EC_POINT_invert(group,p,ctx);put_point(points[t][1],p);
                if(t%3==2){EC_POINT_dbl(group,p,p,ctx);EC_POINT_invert(group,p,ctx);put_point(points[t][2],p);}
            }
        }
        std::vector<std::thread> workers;
        for(unsigned t=0;t<128;t++)workers.emplace_back([t]{
            threadIdx.x=t;ctx=BN_CTX_new();uint64_t u[4],v[4];
            qsb_fixed_affine_block(result[t].x,result[t].y,u,v,scalars[t],nullptr);
            result_inf[t]=qsb_affine_zero(u);assert(!memcmp(u,v,32));BN_CTX_free(ctx);
        });
        for(auto &w:workers)w.join();
        for(unsigned t=0;t<128;t++){
            if(pass<2){BN_lebin2bn((unsigned char*)scalars[t],32,k);assert(EC_POINT_mul(group,want,k,nullptr,nullptr,ctx));}
            else {EC_POINT_set_to_infinity(group,sum);for(unsigned c=0;c<15;c++){get_point(p,points[t][c]);assert(EC_POINT_add(group,sum,sum,p,ctx));}EC_POINT_copy(want,sum);}
            if(result_inf[t])assert(EC_POINT_is_at_infinity(group,want));
            else {BN_CTX_start(ctx);auto x=BN_CTX_get(ctx),y=BN_CTX_get(ctx);
                BN_lebin2bn((unsigned char*)result[t].x,32,x);BN_lebin2bn((unsigned char*)result[t].y,32,y);
                BN_mod_sub(y,prime,y,prime,ctx);assert(EC_POINT_set_affine_coordinates(group,got,x,y,ctx));
                assert(EC_POINT_cmp(group,want,got,ctx)==0);BN_CTX_end(ctx);
            }++checked;
        }
    }
    printf("80000 canonical add/sub and alias comparisons; %u full collective combs match OpenSSL; doubling, opposite, infinity and scalar-order boundaries covered\n",checked);
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-affine-audit-') as tmp:
        p=Path(tmp);(p/'audit.cpp').write_text(source)
        subprocess.run(['clang++','-std=c++17','-O2','-pthread','-fsanitize=undefined,bounds',
                        '-Wno-unknown-pragmas','-Wno-deprecated-declarations',
                        '-I/opt/homebrew/opt/openssl@3/include','-L/opt/homebrew/opt/openssl@3/lib',
                        str(p/'audit.cpp'),'-lcrypto','-o',str(p/'audit')],check=True)
        subprocess.run([str(p/'audit')],check=True)
if __name__=='__main__':main()
