#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/sha.h>

#ifdef NDEBUG
#error "This audit requires assertions enabled"
#endif

static BN_CTX *ctx;
static BIGNUM *prime;
static unsigned multiplies, squares, inverses;

static BIGNUM *decode(const uint64_t *value) {
    unsigned char bytes[32];
    for (int i=0;i<32;i++) bytes[i]=(unsigned char)(value[i/8]>>(8*(i%8)));
    BIGNUM *out=BN_CTX_get(ctx);
    assert(out && BN_lebin2bn(bytes,32,out));
    return out;
}

static void encode(uint64_t *out,const BIGNUM *value) {
    unsigned char bytes[32];
    assert(BN_bn2lebinpad(value,bytes,32)==32);
    for (int i=0;i<4;i++) {
        out[i]=0;
        for (int j=0;j<8;j++) out[i]|=(uint64_t)bytes[i*8+j]<<(8*j);
    }
}

static void binary_op(uint64_t *out,const uint64_t *a,const uint64_t *b,int op) {
    BN_CTX_start(ctx);
    BIGNUM *aa=decode(a),*bb=decode(b),*result=BN_CTX_get(ctx);
    assert(result);
    int ok=op==0?BN_mod_mul(result,aa,bb,prime,ctx):
           op==1?BN_mod_add(result,aa,bb,prime,ctx):BN_mod_sub(result,aa,bb,prime,ctx);
    assert(ok);
    encode(out,result);
    BN_CTX_end(ctx);
}

static void _ModMult(uint64_t *out,uint64_t *a,uint64_t *b) {
    ++multiplies;
    binary_op(out,a,b,0);
}
static void _ModAdd256(uint64_t *out,uint64_t *a,uint64_t *b) { binary_op(out,a,b,1); }
static void _ModSub256(uint64_t *out,uint64_t *a,uint64_t *b) { binary_op(out,a,b,2); }
static void _ModSqr(uint64_t *out,const uint64_t *a) {
    ++squares;
    binary_op(out,a,a,0);
}
static void _ModInv(uint64_t *value) {
    ++inverses;
    assert(value[4]==0);
    BN_CTX_start(ctx);
    BIGNUM *a=decode(value),*out=BN_CTX_get(ctx);
    assert(out && BN_mod_inverse(out,a,prime,ctx));
    encode(value,out);
    value[4]=0;
    BN_CTX_end(ctx);
}

#define __device__
#define __forceinline__ inline
#include "shared_finish.cuh"
#undef __forceinline__
#undef __device__

static void point_words(const EC_GROUP *group,const EC_POINT *point,uint64_t *x,uint64_t *y) {
    BN_CTX_start(ctx);
    BIGNUM *xx=BN_CTX_get(ctx),*yy=BN_CTX_get(ctx);
    assert(yy && EC_POINT_get_affine_coordinates(group,point,xx,yy,ctx));
    encode(x,xx); encode(y,yy);
    BN_CTX_end(ctx);
}

static void check_point(const EC_GROUP *group,const EC_POINT *expected,const uint64_t *x,const uint64_t *y) {
    uint64_t ex[4],ey[4];
    point_words(group,expected,ex,ey);
    assert(std::memcmp(ex,x,32)==0 && std::memcmp(ey,y,32)==0);
    unsigned char actual[33],reference[33],a_hash[32],r_hash[32];
    actual[0]=2+(y[0]&1);
    for(int i=0;i<32;i++) actual[32-i]=(unsigned char)(x[i/8]>>(8*(i%8)));
    assert(EC_POINT_point2oct(group,expected,POINT_CONVERSION_COMPRESSED,reference,33,ctx)==33);
    assert(std::memcmp(actual,reference,33)==0);
    SHA256(actual,33,a_hash); SHA256(reference,33,r_hash);
    assert(std::memcmp(a_hash,r_hash,32)==0);
}

static void scalar(BIGNUM *out,int index,int tag,const BIGNUM *order) {
    char label[64];
    int n=std::snprintf(label,sizeof(label),"qsb-shared-finish-%d-%d",index,tag);
    unsigned char digest[32];
    SHA256((const unsigned char*)label,n,digest);
    assert(BN_bin2bn(digest,32,out));
    assert(BN_nnmod(out,out,order,ctx));
    if(BN_is_zero(out)) assert(BN_one(out));
}

int main() {
    ctx=BN_CTX_new(); prime=BN_new();
    EC_GROUP *group=EC_GROUP_new_by_curve_name(NID_secp256k1);
    BIGNUM *order=BN_new(),*k=BN_new(),*r=BN_new(),*scale=BN_new();
    assert(ctx && prime && group && order && k && r && scale);
    assert(EC_GROUP_get_curve(group,prime,nullptr,nullptr,ctx));
    assert(EC_GROUP_get_order(group,order,ctx));
    EC_POINT *p=EC_POINT_new(group),*rp=EC_POINT_new(group),*neg=EC_POINT_new(group);
    EC_POINT *plus=EC_POINT_new(group),*minus=EC_POINT_new(group);
    assert(p && rp && neg && plus && minus);
    const int count=4096;
    for(int i=0;i<count;i++) {
        scalar(k,i,0,order); scalar(r,i,1,order); scalar(scale,i,2,prime);
        if(i<8) {
            if(i<4) assert(BN_set_word(k,i+1));
            else { assert(BN_copy(k,order)); assert(BN_sub_word(k,i-3)); }
        }
        if(i%4==0) assert(BN_one(scale));
        if(i%4==1) { assert(BN_copy(scale,prime)); assert(BN_sub_word(scale,1)); }
        if(i%4==2) assert(BN_set_word(scale,2));
        assert(EC_POINT_mul(group,p,k,nullptr,nullptr,ctx));
        assert(EC_POINT_mul(group,rp,r,nullptr,nullptr,ctx));
        assert(EC_POINT_copy(neg,rp) && EC_POINT_invert(group,neg,ctx));
        assert(EC_POINT_add(group,plus,p,rp,ctx));
        assert(EC_POINT_add(group,minus,p,neg,ctx));
        uint64_t px[4],py[4],rx[4],ry[4],z[5]={0},x[4],y[4];
        uint64_t x1[4],y1[4],x2[4],y2[4];
        point_words(group,p,px,py); point_words(group,rp,rx,ry); encode(z,scale);
        _ModMult(x,px,z); _ModMult(y,py,z);
        multiplies=squares=inverses=0;
        assert(qsb_recover_pair_shared(x,y,z,rx,ry,x1,y1,x2,y2));
        assert(multiplies==11 && squares==2 && inverses==1);
        check_point(group,plus,x1,y1); check_point(group,minus,x2,y2);
        assert(z[4]==0);
    }
    uint64_t rx[4],ry[4],z[5]={1,0,0,0,0},x[4],y[4],outputs[16];
    point_words(group,rp,rx,ry);
    for(int kind=0;kind<4;kind++) {
        std::memcpy(x,rx,32); std::memcpy(y,ry,32);
        z[0]=kind<2?1:0;
        if(kind==1) {
            uint64_t zero[4]={0};
            _ModSub256(y,zero,ry);
        }
        if(kind==3) encode(z,prime);
        std::memset(outputs,0xA5,sizeof(outputs));
        uint64_t before[16]; std::memcpy(before,outputs,sizeof(outputs));
        multiplies=squares=inverses=0;
        assert(!qsb_recover_pair_shared(x,y,z,rx,ry,outputs,outputs+4,outputs+8,outputs+12));
        assert(inverses==0 && std::memcmp(before,outputs,sizeof(outputs))==0);
    }
    std::printf("PASS: %d projective cases, %d recovered points and compressed-key hashes; 4 fallback cases\n",count,2*count);
    std::printf("Common-path operation count: 11 multiplications, 2 squares, 1 inversion\n");
    EC_POINT_free(p); EC_POINT_free(rp); EC_POINT_free(neg); EC_POINT_free(plus); EC_POINT_free(minus);
    EC_GROUP_free(group); BN_free(order); BN_free(k); BN_free(r); BN_free(scale); BN_free(prime); BN_CTX_free(ctx);
}
