#define DEV __attribute__((device))
#define NOINLINE __attribute__((noinline))
typedef unsigned long long u64;

#define UADDO(c,a,b) asm volatile("add.cc.u64 %0,%1,%2;":"=l"(c):"l"(a),"l"(b):"memory")
#define UADDC(c,a,b) asm volatile("addc.cc.u64 %0,%1,%2;":"=l"(c):"l"(a),"l"(b):"memory")
#define UADD(c,a,b) asm volatile("addc.u64 %0,%1,%2;":"=l"(c):"l"(a),"l"(b))
#define USUBO1(c,a) asm volatile("sub.cc.u64 %0,%0,%1;":"+l"(c):"l"(a):"memory")
#define USUBC1(c,a) asm volatile("subc.cc.u64 %0,%0,%1;":"+l"(c):"l"(a):"memory")
#define USUB1(c,a) asm volatile("subc.u64 %0,%0,%1;":"+l"(c):"l"(a))
#define UADDO1(c,a) asm volatile("add.cc.u64 %0,%0,%1;":"+l"(c):"l"(a):"memory")
#define UADDC1(c,a) asm volatile("addc.cc.u64 %0,%0,%1;":"+l"(c):"l"(a):"memory")
#define UADD1(c,a) asm volatile("addc.u64 %0,%0,%1;":"+l"(c):"l"(a))

DEV NOINLINE void modadd_old(u64 *r,const u64 *a,const u64 *b){
  u64 x[5];
  UADDO(x[0],a[0],b[0]); UADDC(x[1],a[1],b[1]);
  UADDC(x[2],a[2],b[2]); UADDC(x[3],a[3],b[3]); UADD(x[4],0,0);
  r[0]=x[0];r[1]=x[1];r[2]=x[2];r[3]=x[3];
  USUBO1(x[0],0xFFFFFFFEFFFFFC2FULL); USUBC1(x[1],~0ULL);
  USUBC1(x[2],~0ULL); USUBC1(x[3],~0ULL); USUB1(x[4],0);
  if((long long)x[4]>=0){r[0]=x[0];r[1]=x[1];r[2]=x[2];r[3]=x[3];}
}

DEV NOINLINE void modadd_snapshot(u64 *r,const u64 *a,const u64 *b){
  u64 x[5];
  UADDO(x[0],a[0],b[0]); UADDC(x[1],a[1],b[1]);
  UADDC(x[2],a[2],b[2]); UADDC(x[3],a[3],b[3]); UADD(x[4],0,0);
  u64 o0=x[0],o1=x[1],o2=x[2],o3=x[3];
  USUBO1(x[0],0xFFFFFFFEFFFFFC2FULL); USUBC1(x[1],~0ULL);
  USUBC1(x[2],~0ULL); USUBC1(x[3],~0ULL); USUB1(x[4],0);
  u64 ge=~((u64)((long long)x[4]>>63));
  r[0]=(x[0]&ge)|(o0&~ge); r[1]=(x[1]&ge)|(o1&~ge);
  r[2]=(x[2]&ge)|(o2&~ge); r[3]=(x[3]&ge)|(o3&~ge);
}

DEV NOINLINE void modadd_addback(u64 *r,const u64 *a,const u64 *b){
  u64 x[5];
  UADDO(x[0],a[0],b[0]); UADDC(x[1],a[1],b[1]);
  UADDC(x[2],a[2],b[2]); UADDC(x[3],a[3],b[3]); UADD(x[4],0,0);
  USUBO1(x[0],0xFFFFFFFEFFFFFC2FULL); USUBC1(x[1],~0ULL);
  USUBC1(x[2],~0ULL); USUBC1(x[3],~0ULL); USUB1(x[4],0);
  u64 restore=(u64)((long long)x[4]>>63);
  u64 p0=0xFFFFFFFEFFFFFC2FULL&restore,p1=~0ULL&restore;
  UADDO1(x[0],p0); UADDC1(x[1],p1); UADDC1(x[2],p1); UADD1(x[3],p1);
  r[0]=x[0];r[1]=x[1];r[2]=x[2];r[3]=x[3];
}
