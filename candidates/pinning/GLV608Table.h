// Exact, instance-specific host construction of the GLV608 dual-x table.
#pragma once
#define G14_ENTRIES 785209u
#define G14_BYTES (75380064ull)
static void g14_require(int ok,const char *what) {
 if(!ok){fprintf(stderr,"GLV608 table failure: %s\n",what);exit(2);}
}
struct g14_host {
 EC_GROUP *group;BN_CTX *ctx;
 BIGNUM *x,*y,*bx,*yo,*p,*n,*beta,*lambda,*scale,*nri,*k,*a,*b,*stepk;
 EC_POINT *base,*step,*point[608],*check;
};
static void g14_linear_scalar(g14_host &h,int a,int b,BIGNUM *out) {
 g14_require(BN_set_word(h.a,(BN_ULONG)(a<0?-a:a)),"set a");if(a<0)g14_require(BN_sub(h.a,h.n,h.a),"negative a");
 g14_require(BN_set_word(h.b,(BN_ULONG)(b<0?-b:b)),"set b");if(b<0)g14_require(BN_sub(h.b,h.n,h.b),"negative b");
 g14_require(BN_mod_mul(out,h.b,h.lambda,h.n,h.ctx),"lambda coefficient");
 g14_require(BN_mod_add(out,out,h.a,h.n,h.ctx),"linear coefficient");
}
static void g14_record(g14_host &h,const EC_POINT *point,uint8_t *out) {
 if(EC_POINT_is_at_infinity(h.group,point)){memset(out,0,96);return;}
 g14_require(EC_POINT_get_affine_coordinates_GFp(h.group,point,h.x,h.y,h.ctx),"affine extraction");
 g14_require(BN_mod_mul(h.bx,h.x,h.beta,h.p,h.ctx),"beta*x");
 g14_require(BN_copy(h.yo,h.y)!=NULL && BN_add_word(h.yo,0x800001E8UL),"offset y");
 g14_require(BN_bn2lebinpad(h.x,out,32)==32 && BN_bn2lebinpad(h.bx,out+32,32)==32 &&
             BN_bn2lebinpad(h.yo,out+64,32)==32,"little-endian record");
}
// Emit one consecutive row starting a+b*lambda, incrementing da+db*lambda.
static void g14_row(g14_host &h,uint8_t *table,uint32_t start,int count,int a,int b,int da,int db) {
 g14_require(count>0 && count<=608 && (uint64_t)start+count<=G14_ENTRIES,"row bounds");
 g14_linear_scalar(h,a,b,h.k);g14_linear_scalar(h,da,db,h.stepk);
 g14_require(EC_POINT_mul(h.group,h.point[0],NULL,h.base,h.k,h.ctx),"row initial");
 g14_require(EC_POINT_mul(h.group,h.step,NULL,h.base,h.stepk,h.ctx),"row increment");
 for(int i=1;i<count;++i)g14_require(EC_POINT_add(h.group,h.point[i],h.point[i-1],h.step,h.ctx),"row addition");
 g14_require(EC_POINTs_make_affine(h.group,(size_t)count,h.point,h.ctx),"row batch normalization");
 for(int i=0;i<count;++i)g14_record(h,h.point[i],table+((size_t)start+i)*96);
 // Independently scalar-multiply the last point from G, including scale and instance nri.
 g14_linear_scalar(h,a+(count-1)*da,b+(count-1)*db,h.k);
 g14_require(BN_mod_mul(h.k,h.k,h.scale,h.n,h.ctx) && BN_mod_mul(h.k,h.k,h.nri,h.n,h.ctx),"spot scalar");
 g14_require(EC_POINT_mul(h.group,h.check,h.k,NULL,NULL,h.ctx),"spot multiplication");
 uint8_t want[96];g14_record(h,h.check,want);
 g14_require(memcmp(want,table+((size_t)start+count-1)*96,96)==0,"row independent spot check");
}
static void g14_build_table(uint8_t *table,const uint8_t neg_r_inv[32]) {
 g14_host h={};h.group=EC_GROUP_new_by_curve_name(NID_secp256k1);h.ctx=BN_CTX_new();
 g14_require(h.group && h.ctx,"context allocation");
 BIGNUM **nums[]={&h.x,&h.y,&h.bx,&h.yo,&h.p,&h.n,&h.beta,&h.lambda,&h.scale,&h.nri,&h.k,&h.a,&h.b,&h.stepk};
 for(auto v:nums){*v=BN_new();g14_require(*v!=NULL,"BN allocation");}
 h.base=EC_POINT_new(h.group);h.step=EC_POINT_new(h.group);h.check=EC_POINT_new(h.group);
 g14_require(h.base && h.step && h.check,"point allocation");
 for(int i=0;i<608;++i){h.point[i]=EC_POINT_new(h.group);g14_require(h.point[i]!=NULL,"row allocation");}
 g14_require(EC_GROUP_get_order(h.group,h.n,h.ctx),"order");
 g14_require(BN_hex2bn(&h.p,"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F"),"field");
 g14_require(BN_hex2bn(&h.beta,"7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE"),"beta");
 g14_require(BN_hex2bn(&h.lambda,"5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72"),"lambda");
 g14_require(BN_lebin2bn(neg_r_inv,32,h.nri)!=NULL && BN_nnmod(h.nri,h.nri,h.n,h.ctx),"instance scalar");
 g14_require(!BN_is_zero(h.nri),"nonzero recovery base");
 g14_require(BN_one(h.scale),"initial scale");
 // Each chunk's zero rank is the point at infinity; no field coordinates are read for it.
 memset(table,0,G14_BYTES);
 for(int ch=0;ch<14;++ch){
  g14_require(BN_mod_mul(h.k,h.scale,h.nri,h.n,h.ctx),"scaled recovery base");
  g14_require(EC_POINT_mul(h.group,h.base,h.k,NULL,NULL,h.ctx),"A times scale");
  uint32_t base=qsb_glv_base(ch);
  if(ch<13){
   const int m=ch<4?512:608;
   g14_row(h,table,base+1,m/2,0,1,0,1);
   for(int a=1;a<=(m-1)/3;++a){
    int low=2*a,high=m-a-1,cut=(m+a)/2;
    uint32_t rank=m/2+1+(a-1)*m-3*(a-1)*a/2;
    int end=high<cut?high:cut;
    if(low<=end)g14_row(h,table,base+rank,end-low+1,a,low,0,1);
    int begin=low>cut+1?low:cut+1;
    if(begin<=high)g14_row(h,table,base+rank+(begin-low),high-begin+1,a,begin-m,0,1);
   }
   g14_require(BN_mul_word(h.scale,(BN_ULONG)m),"radix scale");
  }else{
   for(int v=1;v<=236;++v)g14_row(h,table,base+1+(v-1)*237,237,-v,-v,-1,0);
  }
 }
 for(int i=0;i<608;++i)EC_POINT_free(h.point[i]);
 EC_POINT_free(h.base);EC_POINT_free(h.step);EC_POINT_free(h.check);
 for(auto v:nums)BN_free(*v);
 EC_GROUP_free(h.group);BN_CTX_free(h.ctx);
}
