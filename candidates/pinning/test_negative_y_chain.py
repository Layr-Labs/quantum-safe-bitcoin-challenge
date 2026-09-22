#!/usr/bin/env python3
"""Exercise extracted chain source in both representations against OpenSSL.

Field operations are exact CPU arithmetic here. The device MAC's actual PTX is
checked separately by test_negative_y_ptx.py. No GPU throughput is measured.
"""
from pathlib import Path
import subprocess
import tempfile
from test_parity_replay import function

HERE=Path(__file__).resolve().parent

def main():
 math=(HERE/'GPUMath.h').read_text();pin=(HERE/'pinning.cu').read_text()
 needle='template<bool DEFER_Y>\n__device__ __forceinline__ void _PointAddXYZZT('
 body=function(math[math.index(needle)+len(needle):],needle)
 seed=function(math,'__device__ void _PointAddXYZZ_mm(')
 negate=function(math,'__device__ __forceinline__ void qsb_negate_residue(')
 chain=function(pin,'__device__ void _FixedBaseSignedXYZZScalar(')
 packed=(HERE/'PackedRecovery.cuh').read_text()
 finish=packed[packed.index('template<bool EXACT_PARITY=true>'):]
 header=r'''
#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#define __device__
#define __forceinline__ inline
#define QSB_YOFF 1
#define QSB_LAZY 1
#define QSB_FUSE_SQRADDSUB2 1
#define GT_CHUNKS 15
#define QSB_LAZY_REC 1
#define QSB_RAW_X 0
#define QSB_PARITY_SUM 1
#define QSB_PARITY_WINDOW 0
static BN_CTX *ctx;
static BIGNUM *p;
static uint64_t points[15][8];
static uint64_t random_state=0xf375918351abcULL;
static uint64_t random64(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
static BIGNUM *read(const uint64_t *a){BIGNUM *b=BN_CTX_get(ctx);assert(BN_lebin2bn((const unsigned char*)a,32,b));return b;}
static void write(uint64_t *a,const BIGNUM*b){assert(BN_bn2lebinpad(b,(unsigned char*)a,32)==32);}
static void operation(uint64_t *out,const uint64_t*a,const uint64_t*b,char op){
 BN_CTX_start(ctx);auto aa=read(a),bb=read(b),z=BN_CTX_get(ctx);
 if(op=='*')assert(BN_mod_mul(z,aa,bb,p,ctx));
 if(op=='+')assert(BN_mod_add(z,aa,bb,p,ctx));
 if(op=='-')assert(BN_mod_sub(z,aa,bb,p,ctx));
 write(out,z);BN_CTX_end(ctx);
}
static void _ModMult(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'*');}
static void _ModMult(uint64_t*out,const uint64_t*b){_ModMult(out,out,b);}
static void _ModSub256(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'-');}
static void _ModSub256(uint64_t*out,const uint64_t*b){_ModSub256(out,out,b);}
static void _ModAdd256(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'+');}
static void _ModAddLazy(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'+');}
static void qsb_packed_raw_mul(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'*');}
static void qsb_recovery_mul(uint64_t*out,const uint64_t*a,const uint64_t*b){operation(out,a,b,'*');}
static uint32_t qsb_sum_parity(const uint64_t*a,const uint64_t*b,uint32_t neg){uint64_t tmp[4];_ModAdd256(tmp,a,b);return (tmp[0]&1u)^((tmp[0]|tmp[1]|tmp[2]|tmp[3])?neg:0);}
static void _ModNeg256(uint64_t*out){
 BN_CTX_start(ctx);auto aa=read(out),z=BN_CTX_get(ctx);assert(BN_sub(z,p,aa));assert(!BN_is_negative(z));write(out,z);BN_CTX_end(ctx);
}
static void _ModSqr(uint64_t*out,const uint64_t*a){_ModMult(out,a,a);}
static void Load256(uint64_t*out,const uint64_t*a){memcpy(out,a,32);}
static void _ModSqrAddSub2(uint64_t*out,const uint64_t*r,const uint64_t*e,const uint64_t*q){
 uint64_t tmp[4];_ModSqr(tmp,r);_ModAdd256(tmp,tmp,e);_ModSub256(tmp,q);_ModSub256(out,tmp,q);
}
static void _ModAddLazyOff(uint64_t*out,const uint64_t*a,const uint64_t*b){
 uint64_t c[4]={0x1000003d0ULL,0,0,0};_ModAdd256(out,a,b);_ModSub256(out,c);
}
static void qsb_muladd_seed(uint64_t*out,const uint64_t*a,const uint64_t*b,const uint64_t*c){uint64_t tmp[4];_ModMult(tmp,a,b);_ModAdd256(out,tmp,c);}
static void qsb_yoff_to_y(uint64_t*out){uint64_t c[4]={0x800001e8ULL,0,0,0};_ModSub256(out,c);}
static void qsb_decode_to_shared(const uint64_t*){}
static unsigned gt_offset(unsigned c){return c;}
static void qsb_load_decoded(const uint8_t*,unsigned c,unsigned,uint64_t*x,uint64_t*y){memcpy(x,points[c],32);memcpy(y,points[c]+4,32);}
'''
 driver=r'''
int main(){
 ctx=BN_CTX_new();p=BN_new();assert(ctx&&p);assert(BN_hex2bn(&p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"));
 EC_GROUP *group=EC_GROUP_new_by_curve_name(NID_secp256k1);EC_POINT *point=EC_POINT_new(group),*sum=EC_POINT_new(group),*got=EC_POINT_new(group);
 BIGNUM *k=BN_new(),*x=BN_new(),*y=BN_new(),*zz=BN_new(),*zzz=BN_new(),*inv=BN_new();
 uint64_t edge[5][4]={{0},{1},{0xFFFFFFFEFFFFFC2FULL,UINT64_MAX,UINT64_MAX,UINT64_MAX},
 {0xFFFFFFFEFFFFFC30ULL,UINT64_MAX,UINT64_MAX,UINT64_MAX},{UINT64_MAX,UINT64_MAX,UINT64_MAX,UINT64_MAX}};
 for(auto &r:edge){
  uint64_t expected[4],zero[4]={0};_ModSub256(expected,zero,r);variant::qsb_negate_residue(r);
  BN_CTX_start(ctx);auto aa=read(r),ee=read(expected);assert(BN_nnmod(aa,aa,p,ctx));assert(BN_cmp(aa,ee)==0);BN_CTX_end(ctx);
 }
 for(unsigned test=0;test<2048;test++){
  assert(EC_POINT_set_to_infinity(group,sum));
  for(unsigned c=0;c<15;c++){
   uint64_t scalar[4]={random64(),random64(),random64(),random64()};
   assert(BN_lebin2bn((const unsigned char*)scalar,32,k));assert(EC_POINT_mul(group,point,k,nullptr,nullptr,ctx));
   assert(EC_POINT_add(group,sum,sum,point,ctx));assert(EC_POINT_get_affine_coordinates(group,point,x,y,ctx));
   assert(BN_add_word(y,0x800001e8ULL));write(points[c],x);write(points[c]+4,y);
  }
  uint64_t old[4][4],fresh[4][4],dummy[4]={0};
  control::_FixedBaseSignedXYZZScalar(old[0],old[1],old[2],old[3],dummy,nullptr);
  variant::_FixedBaseSignedXYZZScalar(fresh[0],fresh[1],fresh[2],fresh[3],dummy,nullptr);
  uint64_t negative_y[4];memcpy(negative_y,fresh[1],32);
  uint64_t zero[4]={0};_ModSub256(fresh[1],zero,fresh[1]);
  assert(memcmp(old,fresh,sizeof(old))==0);
  assert(BN_lebin2bn((const unsigned char*)fresh[2],32,zz));assert(BN_mod_inverse(inv,zz,p,ctx));
  assert(BN_lebin2bn((const unsigned char*)fresh[0],32,x));assert(BN_mod_mul(x,x,inv,p,ctx));
  assert(BN_lebin2bn((const unsigned char*)fresh[3],32,zzz));assert(BN_mod_inverse(inv,zzz,p,ctx));
  assert(BN_lebin2bn((const unsigned char*)fresh[1],32,y));assert(BN_mod_mul(y,y,inv,p,ctx));
  assert(EC_POINT_set_affine_coordinates(group,got,x,y,ctx));assert(EC_POINT_cmp(group,got,sum,ctx)==0);
  // Exercise the exact production checkpoint algebra and finish for both signs.
  assert(EC_POINT_get_affine_coordinates(group,EC_GROUP_get0_generator(group),x,y,ctx));
  uint64_t a[4],b[4],c[4],two_b[4],den[4],root_inv[4],weighted[4],tmp[4],vbar[4],negative_vbar[4],tbar[4],cp[4],cm[4],vp[4],vm[4];
  write(a,x);write(b,y);_ModMult(tmp,a,a);_ModAdd256(c,tmp,tmp);_ModAdd256(c,c,tmp);
  _ModAdd256(two_b,b,b);assert(BN_lebin2bn((const unsigned char*)two_b,32,zz));assert(BN_mod_inverse(inv,zz,p,ctx));write(tmp,inv);_ModMult(c,tmp);
  _ModMult(den,a,fresh[2]);_ModSub256(den,fresh[0]);_ModMult(den,fresh[3]);
  assert(BN_lebin2bn((const unsigned char*)den,32,zz));assert(BN_mod_inverse(inv,zz,p,ctx));write(root_inv,inv);_ModMult(weighted,root_inv,b);
  _ModMult(vbar,old[1],old[2]);_ModMult(negative_vbar,negative_y,fresh[2]);_ModMult(tbar,fresh[3],fresh[2]);
  auto old_parity=control::qsb_packed_finish(vbar,tbar,root_inv,weighted,a,b,c,cp,cm);
  auto new_parity=variant::qsb_packed_finish(negative_vbar,tbar,root_inv,weighted,a,b,c,vp,vm);
  assert(old_parity==new_parity&&memcmp(cp,vp,32)==0&&memcmp(cm,vm,32)==0);
  for(unsigned recid=0;recid<2;recid++){
   assert(EC_POINT_copy(point,EC_GROUP_get0_generator(group)));if(recid)assert(EC_POINT_invert(group,point,ctx));
   assert(EC_POINT_add(group,got,sum,point,ctx));assert(EC_POINT_get_affine_coordinates(group,got,x,y,ctx));
   write(tmp,x);assert(memcmp(tmp,recid?vm:vp,32)==0);assert(unsigned(BN_is_odd(y))==((new_parity>>recid)&1u));
  }
 }
 printf("{\"extracted_chain_comparisons\":2048,\"OpenSSL_point_addends\":30720,\"recovered_compressed_keys\":4096,\"mismatches\":0}\n");
 BN_free(k);BN_free(x);BN_free(y);BN_free(zz);BN_free(zzz);BN_free(inv);EC_POINT_free(point);EC_POINT_free(sum);EC_POINT_free(got);EC_GROUP_free(group);BN_free(p);BN_CTX_free(ctx);
}
'''
 source=header
 for name,flag in [('control',0),('variant',1)]:
  source+=f'\n#undef QSB_NEG_Y_MAC\n#define QSB_NEG_Y_MAC {flag}\nnamespace {name} {{\n'+negate+'\n'+body+'\n'+seed+'\n'+chain+'\n'+finish+'\n}\n'
 source+=driver
 with tempfile.TemporaryDirectory(prefix='qsb-negative-y-') as tmp:
  tmp=Path(tmp);src=tmp/'audit.cpp';binary=tmp/'audit';src.write_text(source)
  cmd=['clang++','-O2','-std=c++17',str(src),'-o',str(binary),'-lcrypto']
  ssl=Path('/opt/homebrew/opt/openssl@3')
  if ssl.exists():cmd+=['-I'+str(ssl/'include'),'-L'+str(ssl/'lib')]
  subprocess.run(cmd,check=True);subprocess.run([str(binary)],check=True)

if __name__=='__main__':main()
