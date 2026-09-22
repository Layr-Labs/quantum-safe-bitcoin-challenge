// SPDX-License-Identifier: GPL-3.0-only
// Execute the production affine control flow with OpenSSL field operations
// and 256 host lanes. This checks algebra/barrier participation, not GPU PTX.
// g++ -O2 -std=c++20 -pthread affine_chain_host_audit.cpp -lcrypto -o /tmp/affine-audit
#include <array>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>

constexpr int GT_CHUNKS=15, LANES=256;
using Words=std::array<uint64_t,4>;
using Point=std::array<uint64_t,8>;
static std::barrier barrier(LANES);
static thread_local int lane;
static Point points[LANES][GT_CHUNKS];
static Words denominators[LANES], inverses[LANES], scalars[LANES];
static std::array<uint64_t,16> outputs[LANES];
static unsigned expected_idx[LANES][GT_CHUNKS];
static uint64_t expected_neg[LANES][GT_CHUNKS];
static bool synthetic;

struct Field {
    BN_CTX *ctx=BN_CTX_new();
    BIGNUM *p=BN_new(),*n=BN_new(),*a=BN_new(),*b=BN_new(),*r=BN_new();
    Field() {
        BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        BN_hex2bn(&n,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    }
    ~Field(){BN_free(p);BN_free(n);BN_free(a);BN_free(b);BN_free(r);BN_CTX_free(ctx);}
};
static thread_local Field field;
static void read_bn(BIGNUM *b,const uint64_t *w){BN_lebin2bn((const unsigned char*)w,32,b);}
static void write_bn(uint64_t *w,const BIGNUM *b){if(BN_bn2lebinpad(b,(unsigned char*)w,32)!=32)std::abort();}
static void _ModAdd256(uint64_t*r,const uint64_t*a,const uint64_t*b){
    read_bn(field.a,a);read_bn(field.b,b);BN_mod_add(field.r,field.a,field.b,field.p,field.ctx);write_bn(r,field.r);
}
static void _ModSub256(uint64_t*r,const uint64_t*a,const uint64_t*b){
    read_bn(field.a,a);read_bn(field.b,b);BN_mod_sub(field.r,field.a,field.b,field.p,field.ctx);write_bn(r,field.r);
}
static void _ModMult(uint64_t*r,const uint64_t*a,const uint64_t*b){
    read_bn(field.a,a);read_bn(field.b,b);BN_mod_mul(field.r,field.a,field.b,field.p,field.ctx);write_bn(r,field.r);
}
static void _ModSqr(uint64_t*r,const uint64_t*a){_ModMult(r,a,a);}
static void Load256(uint64_t*r,const uint64_t*a){std::memcpy(r,a,32);}
static void qsb_field_normalize(uint64_t*r){read_bn(field.a,r);BN_nnmod(field.r,field.a,field.p,field.ctx);write_bn(r,field.r);}
static void __syncthreads(){barrier.arrive_and_wait();}
static int gt_shift(int c){return c?17*c+1:0;}
static unsigned gt_width(int c){return c==GT_CHUNKS-1?17:gt_shift(c+1)-gt_shift(c);}
static void gt_recode_setup(const uint64_t*k,uint64_t*M,int*sign){
    read_bn(field.a,k);BN_mod_add(field.a,field.a,field.a,field.n,field.ctx);
    *sign=BN_is_odd(field.a)?1:-1;
    if(*sign<0)BN_sub(field.a,field.n,field.a);
    write_bn(M,field.a);
}
static void gt_direct_digit(const uint64_t*M,uint64_t sf,unsigned pos,unsigned w,bool last,uint32_t*idx,uint64_t*neg){
    unsigned f=0;
    for(unsigned j=0;j<w;j++)if(pos+j<256)f|=((M[(pos+j)/64]>>((pos+j)%64))&1u)<<j;
    unsigned t=f>>(w-1);
    *idx=(last?f:(f^(t-1u)))&((1u<<(w-1))-1);
    *neg=(last?0u:(t^1u))^sf;
}
static void gt_load_signed(const uint8_t*,int c,uint32_t idx,uint64_t neg,uint64_t*x,uint64_t*y){
    if(!synthetic && (idx!=expected_idx[lane][c] || neg!=expected_neg[lane][c]))std::abort();
    Load256(x,points[lane][c].data());Load256(y,points[lane][c].data()+4);
}
static void qsb_block_inverse_tree(uint64_t*v){
    Load256(denominators[lane].data(),v);
    __syncthreads();
    if(lane==0){
        Words prefix[LANES],acc={1,0,0,0};
        for(int i=0;i<LANES;i++){
            prefix[i]=acc;
            _ModMult(acc.data(),acc.data(),denominators[i].data());
        }
        read_bn(field.a,acc.data());
        if(!BN_mod_inverse(field.r,field.a,field.p,field.ctx))std::abort();
        write_bn(acc.data(),field.r);
        for(int i=LANES-1;i>=0;i--){
            _ModMult(inverses[i].data(),acc.data(),prefix[i].data());
            _ModMult(acc.data(),acc.data(),denominators[i].data());
        }
    }
    __syncthreads();
    Load256(v,inverses[lane].data());v[4]=0;
}
#define __device__
#define __forceinline__ inline
#include "affine_chain.cuh"

int main(){
    EC_GROUP*group=EC_GROUP_new_by_curve_name(NID_secp256k1);
    EC_POINT*point=EC_POINT_new(group),*want=EC_POINT_new(group);
    BIGNUM *base=BN_new(),*half=BN_new(),*factor=BN_new(),*x=BN_new(),*y=BN_new(),*sum=BN_new();
    unsigned checked=0, infinity_count=0;
    for(unsigned batch=0;batch<4;batch++){
        synthetic=batch==3;
        BN_set_word(base,17+batch*7919);
        BN_mod_inverse(half,BN_value_one(),field.n,field.ctx);
        BN_copy(half,field.n);BN_add_word(half,1);BN_rshift1(half,half);
        for(int i=0;i<LANES;i++){
            uint32_t seed[3]={0xaff1ce,batch,(unsigned)i};unsigned char hash[32];
            SHA256((unsigned char*)seed,sizeof(seed),hash);
            BN_bin2bn(hash,32,field.a);BN_nnmod(field.a,field.a,field.n,field.ctx);
            if(i<4){BN_set_word(field.a,i);}
            if(i==4)BN_copy(field.a,field.n);
            if(i==5){BN_copy(field.a,field.n);BN_sub_word(field.a,1);}
            write_bn(scalars[i].data(),field.a);
            uint64_t M[4];int sign;gt_recode_setup(scalars[i].data(),M,&sign);
            BN_zero(sum);
            for(int c=0;c<GT_CHUNKS;c++){
                uint32_t idx;uint64_t neg;
                gt_direct_digit(M,sign<0,gt_shift(c)+1,gt_width(c),c==GT_CHUNKS-1,&idx,&neg);
                expected_idx[i][c]=idx;expected_neg[i][c]=neg;
                if(synthetic){
                    // P+P, P+(-P)+P, and repeated infinity transitions.
                    BN_set_word(factor,1);
                    if((i%3==1 && c==1) || (i%3==2 && c%2==1))BN_sub(factor,field.n,factor);
                }else{
                    BN_set_word(factor,2*idx+1);BN_lshift(factor,factor,gt_shift(c));
                    BN_mod_mul(factor,factor,half,field.n,field.ctx);
                    if(neg)BN_sub(factor,field.n,factor);
                }
                BN_mod_add(sum,sum,factor,field.n,field.ctx);
                BN_mod_mul(field.r,factor,base,field.n,field.ctx);
                EC_POINT_mul(group,point,field.r,nullptr,nullptr,field.ctx);
                EC_POINT_get_affine_coordinates(group,point,x,y,field.ctx);
                write_bn(points[i][c].data(),x);write_bn(points[i][c].data()+4,y);
            }
            if(synthetic)write_bn(scalars[i].data(),sum);
        }
        std::vector<std::thread> threads;
        for(int i=0;i<LANES;i++)threads.emplace_back([i]{
            lane=i;auto &v=outputs[i];
            qsb_affine_chain(v.data(),v.data()+4,v.data()+8,v.data()+12,scalars[i].data(),nullptr);
        });
        for(auto&t:threads)t.join();
        for(int i=0;i<LANES;i++){
            read_bn(field.a,scalars[i].data());BN_mod_mul(field.r,field.a,base,field.n,field.ctx);
            EC_POINT_mul(group,want,field.r,nullptr,nullptr,field.ctx);
            auto &v=outputs[i];bool inf=EC_POINT_is_at_infinity(group,want);
            if(inf){
                for(int j=8;j<16;j++)if(v[j])std::abort();
                infinity_count++;
            }else{
                if(v[8]!=1 || v[12]!=1)std::abort();
                for(int j=9;j<16;j++)if(j!=12 && v[j])std::abort();
                EC_POINT_get_affine_coordinates(group,want,x,y,field.ctx);
                read_bn(field.a,v.data());read_bn(field.b,v.data()+4);
                if(BN_cmp(x,field.a)||BN_cmp(y,field.b)){
                    std::fprintf(stderr,"Mismatch batch=%u lane=%d\n",batch,i);return 1;
                }
            }
            checked++;
        }
    }
    std::printf("PASS: %u points, %u final infinities, 256 concurrent lanes; actual affine header vs OpenSSL\n",checked,infinity_count);
    EC_POINT_free(point);EC_POINT_free(want);EC_GROUP_free(group);
    BN_free(base);BN_free(half);BN_free(factor);BN_free(x);BN_free(y);BN_free(sum);
}
