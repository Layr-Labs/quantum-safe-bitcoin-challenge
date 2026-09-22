#!/usr/bin/env python3
"""Exercise extracted seed, chain update and final resolve against OpenSSL.

Field primitives use exact CPU arithmetic here. test_negymac_ptx.py separately
executes the device point assembly and its integer MAC seed.
"""
from pathlib import Path
import subprocess
import tempfile
from test_scalar_producer import function

HERE = Path(__file__).resolve().parent


def main():
    field = (HERE / 'hit_filter_field_sc.cuh').read_text()
    tree = (HERE / 'tests/gpu_epochs/tree.cu').read_text()
    pieces = function(field, 'template<bool DEFER_Y>\n__device__ __forceinline__ void qsb_filter_point_add(')
    pieces += '\n' + function(field, '__device__ void qsb_filter_point_seed(')
    pieces += '\n' + function(tree, '__device__ __forceinline__ void qsb_filter_last_add(')
    shim = r'''
#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __device__
#define __forceinline__ inline
#define QSB_CHAIN_ANCHOR_UPDATE 1
#define QSB_SPEC_LAST_RESOLVE 1
static BN_CTX* ctx;
static BIGNUM* p;
static BIGNUM* read(const uint64_t*a){auto*b=BN_CTX_get(ctx);assert(BN_lebin2bn((const unsigned char*)a,32,b));return b;}
static void write(uint64_t*a,const BIGNUM*b){assert(BN_bn2lebinpad(b,(unsigned char*)a,32)==32);}
static void operation(uint64_t*out,const uint64_t*a,const uint64_t*b,char op){
  BN_CTX_start(ctx);auto*aa=read(a);auto*bb=read(b);auto*z=BN_CTX_get(ctx);
  if(op=='*')assert(BN_mod_mul(z,aa,bb,p,ctx));
  if(op=='-')assert(BN_mod_sub(z,aa,bb,p,ctx));
  if(op=='+')assert(BN_mod_add(z,aa,bb,p,ctx));
  write(out,z);BN_CTX_end(ctx);
}
static void _ModSub256(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'-');}
static void _ModAdd256(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'+');}
static void qsb_filter_mul(uint64_t*out,const uint64_t*a,const uint64_t*b,uint32_t&){operation(out,a,b,'*');}
static void qsb_filter_mul(uint64_t*out,const uint64_t*a,uint32_t&bad){qsb_filter_mul(out,out,a,bad);}
static void qsb_filter_sqr(uint64_t*out,const uint64_t*a,uint32_t&bad){qsb_filter_mul(out,a,a,bad);}
static void qsb_filter_add(uint64_t*out,const uint64_t*a,const uint64_t*b,uint32_t&){operation(out,a,b,'+');}
static void qsb_filter_seed_x3(uint64_t*out,const uint64_t*a,const uint64_t*b,const uint64_t*c,uint32_t&){
  _ModSub256(out,a,b);_ModSub256(out,out,c);_ModSub256(out,out,c);
}
static void Load256(uint64_t*out,const uint64_t*a){memcpy(out,a,32);}
#define QSB_FSUB _ModSub256
'''
    source = shim
    for flag, name in [(0, 'control'), (1, 'variant')]:
        source += f'\n#undef QSB_SUBSET_NEG_Y_MAC\n#define QSB_SUBSET_NEG_Y_MAC {flag}\nnamespace {name} {{\n{pieces}\n}}\n'
    source += r'''
uint64_t random_state=0x785441744848ULL;
uint64_t random64(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
int main(){
  ctx=BN_CTX_new();p=BN_new();assert(ctx&&p);
  assert(BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
  auto*group=EC_GROUP_new_by_curve_name(NID_secp256k1);
  auto*point=EC_POINT_new(group);auto*expected=EC_POINT_new(group);auto*got=EC_POINT_new(group);
  auto*k=BN_new();auto*x=BN_new();auto*y=BN_new();auto*den=BN_new();auto*inv=BN_new();
  uint64_t pool[256][8],scalars[256];
  for(unsigned j=0;j<256;j++){
    scalars[j]=random64();
    assert(BN_set_word(k,scalars[j]));assert(EC_POINT_mul(group,point,k,nullptr,nullptr,ctx));
    assert(EC_POINT_get_affine_coordinates(group,point,x,y,ctx));write(pool[j],x);write(pool[j]+4,y);
  }
  for(unsigned trial=0;trial<1024;trial++){
    unsigned indices[15];for(auto&i:indices)i=random64()%256;
    while(indices[0]==indices[1])indices[1]=random64()%256;
    BN_zero(k);for(auto i:indices)assert(BN_add_word(k,scalars[i]));
    assert(EC_POINT_mul(group,expected,k,nullptr,nullptr,ctx));
    uint64_t old[4][4],fresh[4][4],off_old[4],off_new[4];uint32_t bad=0;
    auto*a=pool[indices[0]];auto*b=pool[indices[1]];
    Load256(off_old,a+4);Load256(off_new,a+4);
    control::qsb_filter_point_seed(old[0],old[1],old[2],old[3],a,a+4,b,b+4,bad);
    variant::qsb_filter_point_seed(fresh[0],fresh[1],fresh[2],fresh[3],a,a+4,b,b+4,bad);
    for(unsigned j=2;j<14;j++){
      auto*c=pool[indices[j]];
      control::qsb_filter_point_add<true>(old[0],old[1],old[2],old[3],c,c+4,off_old,bad);
      variant::qsb_filter_point_add<true>(fresh[0],fresh[1],fresh[2],fresh[3],c,c+4,off_new,bad);
      assert(memcmp(off_old,c+4,32)==0&&memcmp(off_new,c+4,32)==0);
    }
    auto*c=pool[indices[14]];
    control::qsb_filter_last_add(old[0],old[1],old[2],old[3],c,c+4,off_old,bad);
    variant::qsb_filter_last_add(fresh[0],fresh[1],fresh[2],fresh[3],c,c+4,off_new,bad);
    assert(memcmp(old,fresh,sizeof(old))==0);
    assert(BN_lebin2bn((const unsigned char*)fresh[0],32,x));
    assert(BN_lebin2bn((const unsigned char*)fresh[2],32,den));assert(BN_mod_inverse(inv,den,p,ctx));
    assert(BN_mod_mul(x,x,inv,p,ctx));
    assert(BN_lebin2bn((const unsigned char*)fresh[1],32,y));
    assert(BN_lebin2bn((const unsigned char*)fresh[3],32,den));assert(BN_mod_inverse(inv,den,p,ctx));
    assert(BN_mod_mul(y,y,inv,p,ctx));
    assert(EC_POINT_set_affine_coordinates(group,got,x,y,ctx));assert(EC_POINT_cmp(group,got,expected,ctx)==0);
  }
  printf("{\"source_chains\":2048,\"openssl_points\":1024,\"mismatches\":0}\n");
}
'''
    with tempfile.TemporaryDirectory(prefix='qsb-subset-neg-source-') as tmp:
        cpp, binary = Path(tmp) / 'audit.cpp', Path(tmp) / 'audit'
        cpp.write_text(source)
        subprocess.run(['clang++', '-std=c++17', '-O2', '-fsanitize=undefined,bounds',
                        '-I/opt/homebrew/opt/openssl@3/include', '-L/opt/homebrew/opt/openssl@3/lib',
                        str(cpp), '-lcrypto', '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
