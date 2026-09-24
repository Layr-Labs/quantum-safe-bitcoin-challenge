// Compile-only in the macOS task environment. Runtime requires an NVIDIA GPU.
#define main qsb_production_main
#include "../gpu_epochs/tree.cu"
#undef main
// Scalar fixtures come from the actual production glv10_build.cuh include.
#include "exact_z_extracted.cuh"
struct AuditOut {uint64_t raw[16],leaf[5],park[12],split[4]; unsigned signs[2]; int front,exact;};
#include "expected_splits.cuh"
__global__ void audit_chain(const uint64_t *zs,unsigned count,const uint8_t *table,const uint8_t *exact,AuditOut *out) {
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 const uint64_t *z=zs+4*i;AuditOut a;uint32_t bad=0;
 q9_glv_split(z,a.split,a.split+2,&a.signs[0],&a.signs[1]);
 qsb_filter_chain_trial(a.raw,a.raw+4,a.raw+8,a.raw+12,z,table,bad);
 uint64_t rx[4]={QSB_U2R[0],QSB_U2R[1],QSB_U2R[2],QSB_U2R[3]},ry[4]={QSB_U2R[4],QSB_U2R[5],QSB_U2R[6],QSB_U2R[7]};
 a.front=qsb_k2s_front3_z(z,table,rx,ry,a.leaf,a.park);
 a.exact=audit_verify_z(z,exact);out[i]=a;
}
__global__ void audit_poison(int *out) {
 unsigned t=threadIdx.x;if(t>=2)return;
 uint64_t prod[5]={0,0,0,0,0},n[12];for(int j=0;j<12;j++)n[j]=0x1234;
 if(t){prod[0]=0xfffffffefffffc2fULL;prod[1]=prod[2]=prod[3]=~0ULL;}
 bool changed=qsb_glv_exception(prod,n);
 bool ok=changed&&prod[0]==1&&!(prod[1]|prod[2]|prod[3]|prod[4]);
 for(int j=0;j<12;j++)ok=ok&&!n[j];out[t]=ok;
}
static void must(bool b,const char *what){if(!b){fprintf(stderr,"audit failure: %s\n",what);exit(3);}}
#include "builder_audit.cuh"
static uint8_t *exact_table(const uint8_t *a) {
 const size_t lb=(size_t)GT_CHUNKS*GT_LO*64,hb=(size_t)GT_CHUNKS*GT_HI*64;
 std::vector<uint64_t>L(lb/8),H(hb/8);gt_build_ladders(L.data(),H.data(),a);
 uint64_t *dl,*dh;uint8_t *table;
 qsb_native::check(cudaMalloc(&dl,lb),"audit exact L");qsb_native::check(cudaMalloc(&dh,hb),"audit exact H");
 qsb_native::check(cudaMalloc(&table,(size_t)GT_TOTAL_ENTRIES*64),"audit exact table");
 qsb_native::check(cudaMemcpy(dl,L.data(),lb,cudaMemcpyHostToDevice),"audit L upload");qsb_native::check(cudaMemcpy(dh,H.data(),hb,cudaMemcpyHostToDevice),"audit H upload");
 kernel_build_gtable<<<(GT_TOTAL_ENTRIES+255)/256,256>>>(dl,dh,table);
 qsb_native::check(cudaDeviceSynchronize(),"audit exact build");cudaFree(dl);cudaFree(dh);return table;
}
static int expected_gate(EC_GROUP*g,const EC_POINT *point,const EC_POINT*C,BN_CTX*ctx) {
 EC_POINT *candidate=EC_POINT_new(g),*offset=EC_POINT_dup(C,g);int result=0;
 for(int r=0;r<2;r++){
  if(r)must(EC_POINT_invert(g,offset,ctx),"invert C");
  must(EC_POINT_add(g,candidate,point,offset,ctx),"expected recovery");
  if(EC_POINT_is_at_infinity(g,candidate))continue;
  unsigned char key[33],hash[32];must(EC_POINT_point2oct(g,candidate,POINT_CONVERSION_COMPRESSED,key,33,ctx)==33,"serialize");
  SHA256(key,33,hash);unsigned bits=0;for(int j=0;j<32;j++){if(!hash[j])bits+=8;else{unsigned v=hash[j];while(!(v&128)){bits++;v<<=1;}break;}}
  if(bits>=QSB_ZEROS_N){result=r+1;break;}
 }
 EC_POINT_free(candidate);EC_POINT_free(offset);return result;
}
int main(int argc,char **argv) {
 if(argc!=2){fprintf(stderr,"Usage: %s <fresh digest_params.bin>\n",argv[0]);return 2;}
 digest_params_t dp;must(load_digest_params(argv[1],&dp)==0,"load problem");
 qsb_native::check(cudaSetDevice(0),"audit device");
 audit_builder(dp.neg_r_inv);
 // One current-instance GLV table, plus retained old exact table. No extra synthetic GLV tables.
 uint8_t *gt=glv10_build(dp.neg_r_inv),*exact=exact_table(dp.neg_r_inv);
 EC_GROUP*g=EC_GROUP_new_by_curve_name(NID_secp256k1);BN_CTX*ctx=BN_CTX_new();
 BIGNUM *a=BN_lebin2bn(dp.neg_r_inv,32,NULL),*k=BN_new(),*n=BN_new(),*p=BN_new(),*x=BN_new(),*y=BN_new(),*zz=BN_new(),*zzz=BN_new(),*tmp=BN_new();
 EC_POINT *point=EC_POINT_new(g),*C=EC_POINT_new(g);EC_GROUP_get_order(g,n,ctx);EC_GROUP_get_curve(g,p,NULL,NULL,ctx);
 // Synthetic C=17*A permits exact +/-C finish cases while retaining the current instance A/table.
 BN_set_word(k,17);BN_mod_mul(k,k,a,n,ctx);must(EC_POINT_mul(g,C,k,NULL,NULL,ctx),"C=17A");
 must(EC_POINT_get_affine_coordinates(g,C,x,y,ctx),"C affine");
 uint64_t cxy[8],slope[4];BN_bn2lebinpad(x,(unsigned char*)cxy,32);BN_bn2lebinpad(y,(unsigned char*)(cxy+4),32);
 BN_mod_add(y,y,y,p,ctx);must(BN_mod_inverse(y,y,p,ctx)!=NULL,"C inverse");BN_mod_sqr(x,x,p,ctx);BN_set_word(tmp,3);BN_mod_mul(x,x,tmp,p,ctx);BN_mod_mul(x,x,y,p,ctx);BN_bn2lebinpad(x,(unsigned char*)slope,32);
 qsb_native::check(cudaMemcpyToSymbol(QSB_U2R,cxy,64),"audit C");qsb_native::check(cudaMemcpyToSymbol(QSB_U2R_C,slope,32),"audit slope");
 std::vector<uint64_t>zs(&glv10_audit_scalars[0][0],&glv10_audit_scalars[0][0]+4*glv10_audit_scalar_count);
 uint64_t extra[4]={17,0,0,0};zs.insert(zs.end(),extra,extra+4);BN_copy(k,n);BN_sub_word(k,17);BN_bn2lebinpad(k,(unsigned char*)extra,32);zs.insert(zs.end(),extra,extra+4);
 unsigned count=zs.size()/4;uint64_t *dz;AuditOut *dout;int *dpoison;
 qsb_native::check(cudaMalloc(&dz,zs.size()*8),"audit z");qsb_native::check(cudaMalloc(&dout,count*sizeof(AuditOut)),"audit out");qsb_native::check(cudaMalloc(&dpoison,8),"audit poison");
 qsb_native::check(cudaMemcpy(dz,zs.data(),zs.size()*8,cudaMemcpyHostToDevice),"audit z upload");
 audit_chain<<<(count+63)/64,64>>>(dz,count,gt,exact,dout);audit_poison<<<1,2>>>(dpoison);
 qsb_native::check(cudaDeviceSynchronize(),"audit kernels");
 std::vector<AuditOut>out(count);int poison[2];qsb_native::check(cudaMemcpy(out.data(),dout,count*sizeof(AuditOut),cudaMemcpyDeviceToHost),"audit outputs");qsb_native::check(cudaMemcpy(poison,dpoison,8,cudaMemcpyDeviceToHost),"audit poison outputs");must(poison[0]&&poison[1],"zero/raw-p isolation");
 unsigned point_bad=0,replay_bad=0,zero_bad=0;
 for(unsigned i=0;i<count;i++){
  BN_lebin2bn((unsigned char*)(zs.data()+4*i),32,k);BN_mod_mul(k,k,a,n,ctx);must(EC_POINT_mul(g,point,k,NULL,NULL,ctx),"oracle product");
  bool infinity=EC_POINT_is_at_infinity(g,point);AuditOut &o=out[i];
  if(i<glv10_audit_scalar_count)must(!memcmp(o.split,audit_expected_split[i].limbs,32)&&o.signs[0]==audit_expected_split[i].signs[0]&&o.signs[1]==audit_expected_split[i].signs[1],"actual q9_glv_split");
  BN_lebin2bn((unsigned char*)(o.raw+8),32,zz);BN_nnmod(zz,zz,p,ctx);BN_lebin2bn((unsigned char*)(o.raw+12),32,zzz);BN_nnmod(zzz,zzz,p,ctx);
  if(infinity){if(!BN_is_zero(zz)||!BN_is_zero(zzz))zero_bad++;}
  else if(BN_is_zero(zz)||BN_is_zero(zzz)){zero_bad++;}
  else {
   // Exact host affine normalization of actual GPU output, never invert zero.
   must(BN_mod_inverse(zz,zz,p,ctx)&&BN_mod_inverse(zzz,zzz,p,ctx),"nonzero inverse");
   BN_lebin2bn((unsigned char*)o.raw,32,x);BN_lebin2bn((unsigned char*)(o.raw+4),32,y);BN_mod_mul(x,x,zz,p,ctx);BN_mod_mul(y,y,zzz,p,ctx);
   uint64_t got[8],want[8];BN_bn2lebinpad(x,(unsigned char*)got,32);BN_bn2lebinpad(y,(unsigned char*)(got+4),32);EC_POINT_get_affine_coordinates(g,point,x,y,ctx);BN_bn2lebinpad(x,(unsigned char*)want,32);BN_bn2lebinpad(y,(unsigned char*)(want+4),32);
   if(memcmp(got,want,64)){point_bad++;fprintf(stderr,"chain mismatch case=%u\n",i);}
  }
  int wanted=expected_gate(g,point,C,ctx);if(o.exact!=wanted){replay_bad++;fprintf(stderr,"replay mismatch case=%u got=%d want=%d\n",i,o.exact,wanted);}
  if(infinity||i>=count-2){if(o.leaf[0]!=1||o.leaf[1]||o.leaf[2]||o.leaf[3]||!o.front||o.park[8]||o.park[9]||o.park[10]||o.park[11])zero_bad++;}
 }
 printf("audit cases=%u chain_mismatches=%u replay_mismatches=%u zero_failures=%u poison_cases=2\n",count,point_bad,replay_bad,zero_bad);
 return (point_bad||replay_bad||zero_bad)?1:0;
}
